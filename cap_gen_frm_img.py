import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch
from io import BytesIO
import re
import numpy as np
from torchvision import transforms
from transformers import AutoModelForVision2Seq, AutoTokenizer

# Step 1: Download and Preprocess the Image

# Define the URL of the image you want to process
image_url = "https://pbs.twimg.com/profile_images/428316729220276224/EdBZ2Kgp.jpeg"  # Replace with your image URL

# Download the image from the URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display the image (optional, for verification)
plt.imshow(image)
plt.axis('off')
plt.show(block=False)  # Non-blocking display
plt.pause(0.001) 

# Define the preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to a fixed size
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Apply the preprocessing steps to the image
image_tensor = preprocess(image)

# Add a batch dimension (since models expect a batch of images)
image_tensor = image_tensor.unsqueeze(0)

# Move the tensor to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)

# Verify the shape of the preprocessed image tensor
print(f"Preprocessed image tensor shape: {image_tensor.shape}")

# Step 2: Load the Pre-trained Model and Tokenizer

# Define the model and tokenizer names
model_name = "nlpconnect/vit-gpt2-image-captioning"  # You can also use other models like 'Salesforce/blip-image-captioning-large'

# Load the pre-trained model
model = AutoModelForVision2Seq.from_pretrained(model_name)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move the model to the appropriate device (CPU or GPU)
model.to(device)

# Set the model to evaluation mode
model.eval()

# Verify that the model and tokenizer are loaded correctly
# print(f"Model: {model}")
# print(f"Tokenizer: {tokenizer}")

# Step 3: Generate Text from the Image

# Create a dummy input_ids tensor (since the model expects it for generating the caption)
input_ids = torch.tensor([[101]]).to(device)  # Example: [BOS] token

# Create an attention mask based on the length of the input_ids
attention_mask = torch.ones_like(input_ids).to(device)

# Generate caption for the image
with torch.no_grad():
    output = model.generate(image_tensor, decoder_input_ids=input_ids, attention_mask=attention_mask)

# Decode the output to get the caption
caption = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated caption
print(f"Generated Caption: {caption}")
