import os
import torch
import argparse
import numpy as np
import trimesh
# from stabledelight.utils.mesh import calc_vertex_normals  
from stabledelight.utils.camera import make_round_views
import nvdiffrast.torch as dr


def save_rgb_image(image, save_path, index):
    """Save a single RGB image in [0,1] as PNG with index."""
    filename = f"{index:03d}.png"
    full_path = os.path.join(save_path, filename)
    image_np = image.clamp(0, 1).detach().cpu().numpy()

    # to uint8
    image_np = (image_np * 255).astype(np.uint8)

    import cv2
    cv2.imwrite(full_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))


def save_camera_matrix(matrix, save_path, index):
    filename = f"{index:03d}.npy"
    full_path = os.path.join(save_path, filename)
    np.save(full_path, matrix.detach().cpu().numpy())


def extract_rgb_sources(mesh: trimesh.Trimesh, device: torch.device):
    """
    Try to get RGB from the mesh in this priority:
    1) Textured mesh (UV + image)
    2) Vertex colors
    3) Fallback constant gray

    Returns:
        mode: str in {"texture", "vertex", "constant"}
        payload: dict with keys depending on mode
            - texture: {"uv": (V,2) float32 tensor [0,1], "tex": (1,H,W,3) float32 tensor [0,1]}
            - vertex: {"colors": (V,3) float32 tensor [0,1]}
            - constant: {"color": (3,) float32 tensor [0,1]}
    """
    # 1) Texture
    try:
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uv = mesh.visual.uv.astype(np.float32)
            # nvdiffrast expects [0,1] range for UV with boundary_mode control
            uv_t = torch.from_numpy(uv).to(device)

            # image
            if hasattr(mesh.visual, "material") and getattr(mesh.visual.material, "image", None) is not None:
                import PIL.Image as Image
                img = mesh.visual.material.image
                if not isinstance(img, Image.Image):
                    # Some loaders store path-like, ensure it's PIL
                    img = Image.fromarray(np.array(img))
                img = img.convert("RGB")
                tex = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)  # (H,W,3)
                tex = tex.unsqueeze(0)  # (1,H,W,3)
                return "texture", {"uv": uv_t, "tex": tex.to(device)}
    except Exception:
        pass

    # 2) Vertex colors
    try:
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            vc = mesh.visual.vertex_colors
            if vc.shape[-1] == 4:
                vc = vc[:, :3]
            colors = torch.from_numpy(vc.astype(np.float32) / 255.0).to(device)
            return "vertex", {"colors": colors}
    except Exception:
        pass

    # 3) Fallback constant
    return "constant", {"color": torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device=device)}


def process_model_file(model_path, output_dir, scale, res, num_round_views, num_elevations, min_elev, max_elev, space, gpu_id):
    try:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # Load mesh
        mesh = trimesh.load(model_path, process=False)
        vertices = torch.tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
        faces = torch.tensor(np.asarray(mesh.faces, dtype=np.int32), device=device)

        # (Optional) normals for future lighting
        # Keep calculation here; not used for flat color/texture rendering
        # _ = calc_vertex_normals(vertices, faces.to(torch.int64))

        # Camera views
        additional_elevations = np.random.uniform(min_elev, max_elev, num_elevations)
        mean_elev = (min_elev + max_elev) / 2.0
        additional_elevations = np.array([min_elev, mean_elev, max_elev])
        mv, proj, ele = make_round_views(num_round_views, additional_elevations, scale)
        glctx = dr.RasterizeGLContext()
        mvp = proj @ mv  # (C,4,4)

        # Homogeneous clip coordinates
        V = vertices.shape[0]
        vert_hom = torch.cat([vertices, torch.ones(V, 1, device=device)], dim=-1)  # (V,4)
        vertices_clip = vert_hom @ mvp.transpose(-2, -1)  # (C,V,4)

        image_size = [res, res]
        rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=False)  # (C,H,W,4)
        alpha = torch.clamp(rast_out[..., -1:], max=1.0)

        # Decide RGB source
        mode, payload = extract_rgb_sources(mesh, device)

        if mode == "texture":
            # Interpolate UV and sample texture
            uv = payload["uv"]  # (V,2) in [0,1]
            uv = payload["uv"].clone()  # (V,2)
            uv[:, 1] = 1.0 - uv[:, 1]   # correct upside-down texture issue
            tex = payload["tex"]  # (1,H,W,3)
            uvw, _ = dr.interpolate(uv, rast_out, faces)  # (C,H,W,2)
            # nvdiffrast.texture expects (C,H,W,2/3) texcoords; provide (u,v)
            rgb = dr.texture(tex, uvw, filter_mode="linear", boundary_mode="wrap")[..., :3]  # (C,H,W,3)
        elif mode == "vertex":
            colors = payload["colors"]  # (V,3)
            rgb, _ = dr.interpolate(colors, rast_out, faces)  # (C,H,W,3)
        else:
            # constant color
            c = payload["color"][None, None, None, :].expand(rast_out.shape[0], image_size[0], image_size[1], 3)
            rgb = c.contiguous()

        # Compose with alpha (black background) and anti-alias
        rgba = torch.cat([rgb, alpha], dim=-1)
        rgba_aa = dr.antialias(rgba, rast_out, vertices_clip, faces)
        rgb_aa = rgba_aa[..., :3] * rgba_aa[..., 3:4]

        # Optional: transform normals to camera/world space if doing lighting (skipped here)
        # Save
        os.makedirs(output_dir, exist_ok=True)
        num_all_views = mv.shape[0]
        for i in range(num_all_views):
            save_rgb_image(rgb_aa[i], output_dir, i + 1)
            save_camera_matrix(mv[i], output_dir, i + 1)

        print(f"Processed {model_path} successfully. Mode: {mode}")
    except Exception as e:
        print(f"Error processing {model_path}: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render RGB images for a single .model file.")
    parser.add_argument('--model_path', '-m', type=str, required=True, help="Path to the .model file to process.")
    parser.add_argument('--output_dir', '-o', type=str, required=True, help="Path to the output directory.")
    parser.add_argument('--scale', '-s', type=float, default=2.5, help="Scale for rendering.")
    parser.add_argument('--res', '-r', type=int, default=512, help="Resolution of the rendered image.")
    parser.add_argument('--num_round_views', '-nrv', type=int, default=16, help="Number of round views.")
    parser.add_argument('--num_elevations', '-ne', type=int, default=4, help="Number of elevation angles.")
    parser.add_argument('--min_elev', '-min', type=float, default=-20, help="Minimum elevation angle.")
    parser.add_argument('--max_elev', '-max', type=float, default=40, help="Maximum elevation angle.")
    parser.add_argument('--space', '-sp', type=str, choices=['camera', 'world'], default='camera',
                        help="(Kept for compatibility; not used in RGB pass).")
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help="GPU ID to use for processing. Default is 0.")

    args = parser.parse_args()
    process_model_file(
        args.model_path, args.output_dir, args.scale, args.res,
        args.num_round_views, args.num_elevations, args.min_elev,
        args.max_elev, args.space, args.gpu_id
    )
