#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
#
# https://github.com/apple/ml-direct2.5/blob/main/util/camera_utils.py


import numpy as np
import torch
import json
from scipy.spatial.transform import Rotation as R

def fov_to_focal(fov, size):
    # convert fov angle in degree to focal
    return size / np.tan(fov * np.pi / 180.0 / 2.0) / 2.0


def focal_to_fov(focal, size):
    # convert focal to fov angle in degree
    return 2.0 * np.arctan(size / (2.0 * focal)) * 180.0 / np.pi
    

def convert_blender_extrinsics_to_opencv(extrinsic_bcam):
    R_bcam2cv = np.array([[1, 0,  0], 
                          [0, -1, 0],
                          [0, 0, -1]], np.float32)
    R, t = R_bcam2cv @ extrinsic_bcam[:3, :3], R_bcam2cv @ extrinsic_bcam[:3, 3]
    extrinsic_cv = np.concatenate([R, t[..., None]], axis=1)
    
    return extrinsic_cv


def camera_intrinsic_to_opengl_projection(intrinsic,w=512,h=512,n=0,f=5,flip_y=False):
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    
    proj = np.array([
        [2.*fx/w,   0,    1-2*cx/w,           0],
        [0,    2*fy/h,   -1+2*cy/h,           0],
        [0,         0,(-f-n)/(f-n),-2*f*n/(f-n)],
        [0,         0,          -1,           0]
        ])
        
    if flip_y:
        proj[1,:] *= -1

    return proj


def build_volumes_projections(extrinsics, intrinsic, resolution=64, size=0.65):
    # build volumes in [-size, size], and compute projections on each images
    x_coords, y_coords, z_coords = np.meshgrid(np.arange(resolution), np.arange(resolution), np.arange(resolution))
    volume_coords = np.stack([y_coords, x_coords, z_coords], axis=-1).reshape(-1, 3).T   # [3, res^3]
    volume_coords = (volume_coords + 0.5) / resolution * 2 * size - size
    num_cams = extrinsics.shape[0]
    us, vs, in_regions = [], [], []
    for i in range(num_cams):
        # project to image coords
        ext = extrinsics[i]
        R, T = ext[:3, :3], ext[:3, 3]
        proj_coords = intrinsic @ (R @ volume_coords + T[:, None])
        proj_coords = proj_coords[:2] / proj_coords[2]
        u, v = np.round(proj_coords[0]).astype(np.int32), np.round(proj_coords[1]).astype(np.int32)
        in_region = np.logical_and(
            np.logical_and(u > 0, u < 256 - 1),
            np.logical_and(v > 0, v < 256 - 1))
        us.append(u[None])
        vs.append(v[None])
        in_regions.append(in_region[None])
    us = np.concatenate(us, axis=0)
    vs = np.concatenate(vs, axis=0)
    in_regions = np.concatenate(in_regions, axis=0)
    
    return us, vs, in_regions


def get_cameras_from_json(json_path):
    print("Loading camera data from {}".format(json_path))
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    intrinsics = []
    extrinsics = []
    intrinsics_np = np.array(camera_data["intrinsics"])
    intrinsics = torch.from_numpy(intrinsics_np).reshape(3, 3)
    intrinsics = intrinsics.repeat(4, 1, 1)
    image_size = camera_data.get("image_size", [256, 256])
    for extrinsics_list in camera_data["extrinsics"]:
        extrinsics_np = np.array(extrinsics_list)
        extrinsics.append(torch.from_numpy(extrinsics_np))
    extrinsics = torch.stack(extrinsics, dim=0)
    return intrinsics, extrinsics, image_size


def get_ortho_projection_matrix(left, right, bottom, top, near, far):
    projection_matrix = np.zeros((4, 4), dtype=np.float32)

    projection_matrix[0, 0] = 2.0 / (right - left)
    projection_matrix[1, 1] = -2.0 / (top - bottom) # add a negative sign here as the y axis is flipped in nvdiffrast output
    projection_matrix[2, 2] = -2.0 / (far - near)

    projection_matrix[0, 3] = -(right + left) / (right - left)
    projection_matrix[1, 3] = -(top + bottom) / (top - bottom)
    projection_matrix[2, 3] = -(far + near) / (far - near)
    projection_matrix[3, 3] = 1.0

    return projection_matrix
    
    
def make_round_views(view_nums, additional_elevations, scale=2., device='cuda'):
    elevations = []
    w2c = []
    ortho_scale = scale/2
    projection = get_ortho_projection_matrix(-ortho_scale, ortho_scale, -ortho_scale, ortho_scale, 0.1, 100)
    
    for i in reversed(range(view_nums)):
        
        phi_y = 360 / view_nums * (i+1) # 360 ~ 22.5
        tmp = np.eye(4)
        rot = R.from_euler('xyz', [0,  phi_y, 0], degrees=True).as_matrix()
        rot[:, 2] *= -1
        tmp[:3, :3] = rot
        tmp[2, 3] = -1.3
        w2c.append(tmp) 
        elevations.append(0)
        for elev in additional_elevations:

            tmp = np.eye(4)
            rot = R.from_euler('yzx', [phi_y, 0, elev], degrees=True).as_matrix() # up front right 
            rot[:, 2] *= -1
            tmp[:3, :3] = rot
            tmp[2, 3] = -1.3
            w2c.append(tmp)
            elevations.append(elev)
            
    w2c = torch.from_numpy(np.stack(w2c, 0)).float().to(device=device)
    projection = torch.from_numpy(projection).float().to(device=device)

    return w2c, projection, elevations