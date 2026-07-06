from fastapi import FastAPI
import torch
import os
from app.model import Net
from fastapi import UploadFile, File
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

app = FastAPI()


model = Net()

MODEL_PATH = "models/fashion_mnist.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )
    model.eval()


CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("L")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = F.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return {
        "class_id": prediction.item(),
        "class_name": CLASSES[prediction.item()],
        "confidence": round(confidence.item() * 100, 2)
    }

@app.get("/model-info")
def model_info():

    return {
        "model": "Fashion-MNIST CNN",
        "classes": CLASSES,
        "model_loaded": True
    }