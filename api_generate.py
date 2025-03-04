from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch import load, no_grad
import torch
import os
from PIL import Image
import io
import base64
from safetensors import safe_open  # for loading .safetensor files

# Initialize FastAPI app
app = FastAPI()

# Define the model path (update this to your actual model path)
MODEL_PATH = r"a1111/stable-diffusion-webu/models/Lora/last.safetensors"
MODEL_PATH = r"C:\Users\rushi\Desktop\3rd_year_project\a1111\stable-diffusion-webui\models\Lora\last.safetensors"

# Define a class for incoming request data
class GenerateRequest(BaseModel):
    description: str

# Load the model
def load_model():
    model = torch.load(MODEL_PATH) if MODEL_PATH.endswith(".pt") else safe_open(MODEL_PATH, framework="pt")
    model.eval()
    return model

model = load_model()

# Define an endpoint to generate images
@app.post("/generate_image/")
async def generate_image(request: GenerateRequest):
    # Generate a sample input tensor based on description (modify as needed)
    description = request.description
    input_tensor = process_description(description)  # Define a function to process text descriptions if needed
    
    # Ensure no gradients are stored
    with no_grad():
        generated_image = model(input_tensor)

    # Convert generated tensor to an image (assuming output is [C, H, W] format)
    image = tensor_to_image(generated_image)
    
    # Convert image to base64 for API response
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": img_str}

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    image = Image.fromarray((tensor.numpy() * 255).astype("uint8").transpose(1, 2, 0))
    return image

def process_description(description):
    # Convert text description to tensor as required by model
    # Placeholder function; implement according to your model's requirements
    return torch.randn(1, 100)  # Replace with your embedding logic

# Run the API with:
# uvicorn api_filename:app --reload --host 0.0.0.0 --port 8000
