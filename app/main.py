import os

print("STEP 1: main.py imported")

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile

print("STEP 2: FastAPI imports loaded")

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

print("STEP 3: Torch imports loaded")

from app.model import Net
from app.classes import CLASSES

print("STEP 4: Classes loaded")

app = FastAPI()

MODEL_PATH = "models/food101.pth"

print(f"STEP 5: MODEL_PATH = {MODEL_PATH}")

model = Net()

print("STEP 6: Model created")

if os.path.exists(MODEL_PATH):

    model_size_mb = (
        os.path.getsize(MODEL_PATH)
        / 1024
        / 1024
    )

    print(
        f"STEP 7: Model file found "
        f"({model_size_mb:.2f} MB)"
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    print("STEP 8: Weights loaded")

    model.eval()

    print("STEP 9: Food-101 model loaded")

else:

    print("STEP X: food101.pth not found")


@app.get("/")
def home():

    return {
        "message": "Food-101 Classifier API",
        "version": "food101-debug-v1"
    }


@app.get("/food-test")
def food_test():

    return {
        "status": "food101 deployment active"
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
        "model_loaded": os.path.exists(MODEL_PATH),
        "version": "food101-debug-v1"
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