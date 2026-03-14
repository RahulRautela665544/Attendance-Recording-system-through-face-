import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

print("Loading model (CPU mode, first time is slow)...")

MODEL = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,   # CPU needs float32
    safety_checker=None
)

pipe = pipe.to("cpu")   # Force CPU

# Load input image
image = Image.open("input.jpg").convert("RGB")
image = image.resize((512,512))

prompts = [
    "photo of same person, clean shaved, realistic, studio lighting",
    "photo of same person with thick beard, ultra realistic",
    "photo of same person wearing glasses, professional portrait",
    "photo of same person, side view, natural light",
    "photo of same person, slightly older, DSLR photo",
    "passport style photo, same person, clear face"
]

os.makedirs("ai_generated", exist_ok=True)

for i, prompt in enumerate(prompts):

    print("Generating:", i+1, "/", len(prompts))

    result = pipe(
        prompt=prompt,
        image=image,
        strength=0.6,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]

    result.save(f"ai_generated/face_{i}.png")

print("DONE ✅ Check ai_generated folder")
