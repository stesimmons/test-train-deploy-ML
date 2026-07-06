from fastapi import FastAPI
import torch
import os
from app.model import Net

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

@app.get("/predict")
def predict():

    import torch

    sample = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        output = model(sample)
        prediction = output.argmax(dim=1).item()

    return {
        "class_id": prediction,
        "class_name": CLASSES[prediction]
    }

@app.get("/model-info")
def model_info():

    return {
        "model": "Fashion-MNIST CNN",
        "classes": CLASSES,
        "model_loaded": True
    }