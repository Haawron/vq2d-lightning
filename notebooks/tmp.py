from diffusers import DiffusionPipeline, StableDiffusionPipeline
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

pipe: StableDiffusionPipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    requires_safety_checker=False,
    feature_extractor=None
).to("cuda")
print(pipe)
pipe.__call__

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt = "A plug, realistic"
# prompt = "A plug laid on the floor"
prompt = "A bottle laid on the floor"

# img = Image.open('outputs/ego4d_data/v2/vq2d_crops/queries/train/plug/c46debfc-4260-4217-be0f-168930b903cb_237_4b6d213d-5ebc-4145-88fc-ed8e4d1c99ee_3.jpg')
# img = pil_to_tensor(img).unsqueeze(0).to("cuda")
num_images = 10
images = pipe(
    prompt,
    negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy",
    num_images_per_prompt=num_images
).images
for i in range(num_images):
    images[i].save(f"notebooks/diffusion/{prompt.replace(' ', '-')}-neg--{i:02d}.png")
