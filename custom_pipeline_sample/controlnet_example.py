''' This script is used to show the model summary.

This script runs stable_diffusion_controlnet_clip.py under community pipelines.
UNet model used by runwayml/stable-diffusion-v1-5:

<class 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel'>

We now change the structure of Controlnet by processing embeddings straight,
shaving off the first 2 down blocks until the image size is 16 by 16.
This will be reconstructed from the CLIP embeddings (sep length of 257, throw away CLS embedding) and fed in.

Output of the down sample will be short by 2 blocks, so we will have to generate zeros to fill in.
This will then be added as `down_block_additional_residuals` back in the unet.

'''
# Enable import of StableDiffusionControlNetCLIPPipeline
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "examples" / "community"))

# Main imports
from torchinfo import summary
from PIL import Image
import cv2
import torch
import numpy as np
from diffusers.utils import load_image
from stable_diffusion_controlnet_clip import StableDiffusionControlNetCLIPPipeline
from diffusers import ControlNetCLIPModel, UniPCMultistepScheduler

from clip_preprocess import CLIPWrapper

# !pip install opencv-python transformers accelerate
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from ..examples.community.stable_diffusion_controlnet_clip import StableDiffusionControlNetCLIPPipeline


config = ControlNetCLIPModel.load_config("./controlnet_clip_config.json")
controlnet = ControlNetCLIPModel.from_config(config)


# print(controlnet)

# Input sizes for forward pass in ControlNetCLIPModel
# torch.Size([2, 4, 64, 64])
# torch.Size([])  # tensor(999)
# torch.Size([2, 77, 768])
# torch.Size([2, 3, 512, 512])


# summary(controlnet, input_size=[
#     (2, 4, 64, 64),
#     (1,),
#     (2, 77, 768),
#     (2, 3, 512, 512)
# ])


# download an image
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image = np.array(image)

# get canny image
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# load control net and stable diffusion v1-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    # Run with torch.float16
    # controlnet = ControlNetCLIPModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetCLIPPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )

    # # remove following line if xformers is not installed
    # pipe.enable_xformers_memory_efficient_attention()

    # pipe.enable_model_cpu_offload()
else:
    # Run with torch.float32 as CPU doesn't support half precision
    controlnet = ControlNetCLIPModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetCLIPPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
    )

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# generate image
generator = torch.manual_seed(0)
image = pipe(
    "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
).images[0]