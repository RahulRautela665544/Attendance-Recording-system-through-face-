import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

print("Loading AI model... (This takes time)")

MODEL = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    safety_checker=None
)

pipe = pipe.to("cpu")

# Prompts
prompts = [
    "photo of same person, clean shaved, realistic",
    "photo of same person with thick beard",
    "photo of same person wearing glasses",
    "photo of same person, side view",
    "passport style photo, clear face"
]

dataset_dir = "dataset"
output_dir = "ai_generated_all"

os.makedirs(output_dir, exist_ok=True)

for student in os.listdir(dataset_dir):

    student_path = os.path.join(dataset_dir, student)

    if not os.path.isdir(student_path):
        continue

    print("\nProcessing:", student)

    images = os.listdir(student_path)

    if len(images) == 0:
        print("No images, skipping")
        continue

    first_img = os.path.join(student_path, images[0])

    try:
        image = Image.open(first_img).convert("RGB")
        image = image.resize((512,512))
    except:
        print("Cannot read image, skipping")
        continue

    save_path = os.path.join(output_dir, student)
    os.makedirs(save_path, exist_ok=True)

    for i, prompt in enumerate(prompts):

        print("  Generating", i+1, "/", len(prompts))

        result = pipe(
            prompt=prompt,
            image=image,
            strength=0.6,
            guidance_scale=7.5,
            num_inference_steps=25
        ).images[0]

        result.save(f"{save_path}/face_{i}.png")

print("\nALL STUDENTS DONE ✅")
