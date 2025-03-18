import torch
from diffusers import DiffusionPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Load the base model
pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
)

# Move to MPS (Metal Performance Shaders) for Mac M1
pipe.to("mps")

# Create a prompt
prompt = "A business man in a suit"

# Generate the image
print("Generating image... (this may take a few minutes)")
image = pipe(prompt=prompt).images[0]

# Save the image
image.save("business_man_in_suit.png")
print(f"Image saved as business_man_in_suit.png")