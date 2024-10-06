from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch

pipe: StableDiffusionPipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    requires_safety_checker=False,
    feature_extractor=None
).to("cuda")

prompt = "A plug laid on the floor"
num_images = 100
gen = torch.Generator(device='cuda').manual_seed(42)
images = pipe(
    prompt,
    negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
    num_images_per_prompt=num_images,
    generator=gen
).images
for i in range(num_images):
    images[i].save(f"outputs/ego4d_data/v2/diffusion/{prompt.replace(' ', '-')}-neg--{i:02d}.png")
