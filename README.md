# StableDelight: Revealing Hidden Textures by Removing Specular Reflections
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/Stable-X/StableDelight)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-green)](https://huggingface.co/Stable-X/yoso-delight-v0-4-base)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

![17541724684116_ pic_hd](https://github.com/user-attachments/assets/26713cf5-4f2c-40a3-b6e7-61309b2e175a)

StableDelight is a cutting-edge solution for specular reflection removal from textured surfaces. Building upon the success of [StableNormal](https://github.com/Stable-X/StableNormal), which focused on enhancing stability in monocular normal estimation, StableDelight takes this concept further by applying it to the challenging task of reflection removal. The training data include [Hypersim](https://github.com/apple/ml-hypersim), [Lumos](https://research.nvidia.com/labs/dir/lumos/), and various Specular Highlight Removal datasets from [TSHRNet](https://github.com/fu123456/TSHRNet). In addition, we've integrated a multi-scale SSIM loss and random conditional scales technique into our diffusion training process to improve sharpness in one-step diffusion prediction.

## Background
StableDelight is inspired by our previous work, [StableNormal](https://github.com/Stable-X/StableNormal), which introduced a novel approach to tailoring diffusion priors for monocular normal estimation. The key innovation of StableNormal was its focus on enhancing estimation stability by reducing the inherent stochasticity of diffusion models (such as Stable Diffusion). This resulted in "Stable-and-Sharp" normal estimation that outperformed multiple baselines.

## Installation:

Please run following commands to build package:
```
git clone https://github.com/Stable-X/StableDelight.git
cd StableDelight
pip install -r requirements.txt
pip install -e .
```
or directly build package:
```
pip install git+https://github.com/Stable-X/StableDelight.git
```

## Torch Hub Loader 🚀
To use the StableDelight pipeline, you can instantiate the model and apply it to an image as follows:

```python
import torch
from PIL import Image

# Load an image
input_image = Image.open("path/to/your/image.jpg")

# Create predictor instance
predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)

# Apply the model to the image
delight_image = predictor(input_image)

# Save or display the result
delight_image.save("output/delight.png")
```

## Gradio interface 🤗

We also provide a Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> interface for a better experience, just run by:

```bash
# For Linux and Windows users (and macOS with Intel??)
python app.py
```

You can specify the `--server_port`, `--share`, `--server_name` arguments to satisfy your needs!

## Texture Delight for 3D Models 🎨

StableDelight not only works for single 2D image, but can also remove specular reflections from **3D model textures**.

### Input format
Your `source_dir/` should contain:
```bash
source_dir/
├── images/ 
│ ├── 000.png
│ ├── 000.cam
│ └── ...
└── 3DModel.ply 
```
### Step 1 — Delight the Texture Images
```bash
python stabledelight/script/image_delight.py -s source_dir
```
This will generate an images_delighted/ folder inside source_dir/ containing the reflection-free images.

### Step 2 — Binding Texture to the Mesh
The output data is organized in the same format as [mvs-texturing](https://github.com/nmoehrle/mvs-texturing/tree/master). Follow these steps to add texture to the mesh:

* Compile the mvs-texturing repository on your system.
* Add the build/bin directory to your PATH environment variable
* Navigate to the output directory containing the mesh.
* Run the following command:
```bash
texrecon ./images ./fused_mesh.ply ./textured_mesh --outlier_removal=gauss_clamping --data_term=area --no_intermediate_results
```
The final textured_mesh.obj will now contain the **specular-free texture**.
