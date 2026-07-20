import os

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from app.model import Net
from app.classes import CLASSES


app = FastAPI()

MODEL_PATH = "models/food101.pth"

model = Net()

if os.path.exists(MODEL_PATH):

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    print("Food-101 model loaded")

else:

    print("food101.pth not found")


@app.get("/")
def home():

    return {
        "message": "Food-101 Classifier API"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():

    return {
        "model_name": "Food-101 CNN",
        "num_classes": len(CLASSES),
        "model_loaded": os.path.exists(MODEL_PATH)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    image = Image.open(
        file.file
    ).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image_tensor = (
        transform(image)
        .unsqueeze(0)
    )

    with torch.no_grad():

        output = model(
            image_tensor
        )

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