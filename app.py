from flask import Flask, request, jsonify
import torch
import base64
from io import BytesIO

from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the image generation model
import torch
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming your Generator class is already defined as 'ConditionalGenerator' (or similar)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, embed_dim=768, img_channels=3, img_size=256):
        super(Generator, self).__init__()
        self.init_size = img_size // 16  # Initial size before upsampling
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, 128 * self.init_size * self.init_size),
            nn.LeakyReLU(0.05, inplace=True)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, embed):
        gen_input = torch.cat((noise, embed), -1)
        out = self.fc(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Initialize the generator model
generator = Generator()  # Make sure to specify the device

# Path to the saved generator model
model_path = 'final_generator.pth'

# Load the saved state dictionary
generator.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
generator.eval()

# Function to get BERT embedding for a text description
def get_bert_embedding(description):
    inputs = tokenizer(description, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the [CLS] token embedding as the sentence embedding
    text_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return text_embedding.unsqueeze(0)  # Shape: (1, embed_dim)

# Function to generate an image from a text description
def generate_image_from_text(generator, description, noise_dim=100):
    # Get BERT embedding
    text_embedding = get_bert_embedding(description)

    # Generate noise vector
    noise = torch.randn(1, noise_dim)

    # Generate the image with the GAN
    generator.eval()
    with torch.no_grad():
        generated_image = generator(noise, text_embedding).cpu()

    # Convert tensor to PIL image
    generated_image = transforms.ToPILImage()(generated_image.squeeze())
    return generated_image

# Display the generated image
def display_generated_image(description):
    generated_image = generate_image_from_text(generator, description)
    return generated_image


# Endpoint for generating an image from a description
@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    description = data.get("description")
    
    # Example usage
    # description = "a floor plan with a living room and a kitchen"
    generated_image = display_generated_image(description)
    if not description:
        return jsonify({"error": "Description not provided"}), 400


    # Convert the generated tensor to an image
    image_tensor = generated_image
    pil_image = transforms.ToPILImage()
    # Encode the image to base64 for JSON transmission
    buffer = BytesIO()
    generated_image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({"image": encoded_image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
