# app/main.py

from fastapi import FastAPI
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import io
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# from concurrent.futures import ThreadPoolExecutor


class InferenceParams(BaseModel):
    prompt: str 
    n_steps: int = 40
    high_noise_frac: float = 0.8
   

app = FastAPI()


base = None
refiner = None

@app.on_event("startup")
async def load_model():
    global base
    global refiner
    base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    
    )
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
    refiner.to("cuda")
# app.POOL: ThreadPoolExecutor = None

# @app.on_event("startup")
# def startup_event():
#     app.POOL = ThreadPoolExecutor(max_workers=1)
# @app.on_event("shutdown")
# def shutdown_event():
#     app.POOL.shutdown(wait=False)
# Load both base & refiner models




# Function to generate an image from a prompt
def generate_image(prompt, n_steps, high_noise_frac):
    # Run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    
    return image

@app.post("/generate_image/")
async def generate_image_endpoint(body:InferenceParams):
    # torch.cuda.empty_cache()
    image = generate_image(body.prompt, body.n_steps, body.high_noise_frac)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")
    