import os

from fastapi import FastAPI, File, UploadFile
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.datasets import Food101

from app.model import Net


app = FastAPI()


CLASSES = Food101(
    root="./data",
    split="train",
    download=True
).classes


model = Net()

MODEL_PATH = "models/food101.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )
    model.eval()
    print("Food-101 model loaded successfully")
else:
    print(f"Model not found: {MODEL_PATH}")


@app.get("/")
def home():
    return {
        "message": "Food-101 Classifier API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return {
        "class_id": prediction.item(),
        "class_name": CLASSES[prediction.item()],
        "confidence": round(
            confidence.item() * 100,
            2
        )
    }