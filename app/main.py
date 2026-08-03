from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

import json
import os

import torch
import torch.nn.functional as F

from PIL import Image
from PIL import UnidentifiedImageError

import torchvision.transforms as transforms

from app.model import (
    BaselineCNN,
    WiderCNN,
    DeepCNN
)

app = FastAPI()

MODEL_PATH = "models/production-model.pth"
METRICS_PATH = "metrics.json"

MODEL_LOADED = False
BEST_ARCHITECTURE = "Unknown"


def create_model(architecture: str):

    if architecture == "BaselineCNN":
        return BaselineCNN()

    if architecture == "WiderCNN":
        return WiderCNN()

    if architecture == "DeepCNN":
        return DeepCNN()

    raise ValueError(
        f"Unknown architecture: {architecture}"
    )


if os.path.exists(METRICS_PATH):

    try:

        with open(
            METRICS_PATH,
            "r"
        ) as f:

            metrics = json.load(f)

        BEST_ARCHITECTURE = metrics.get(
            "best_architecture",
            "BaselineCNN"
        )

    except Exception as e:

        print(
            f"Failed loading metrics.json: {e}"
        )

        BEST_ARCHITECTURE = "BaselineCNN"

else:

    BEST_ARCHITECTURE = "BaselineCNN"


model = create_model(
    BEST_ARCHITECTURE
)

if os.path.exists(MODEL_PATH):

    try:

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location="cpu"
            )
        )

        model.eval()

        MODEL_LOADED = True

        print(
            f"Loaded model: {BEST_ARCHITECTURE}"
        )

    except Exception as e:

        print(
            f"Model load failed: {e}"
        )

        MODEL_LOADED = False


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
    "Ankle boot"
]


@app.get("/")
def home():

    return {
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "architecture": BEST_ARCHITECTURE
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        image = Image.open(
            file.file
        ).convert("L")

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    if not MODEL_LOADED:

        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image_tensor = transform(
        image
    ).unsqueeze(0)

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

    class_id = int(
        prediction.item()
    )

    confidence_pct = float(
        confidence.item() * 100
    )

    return {
        "class_id": class_id,
        "class_name": CLASSES[class_id],
        "confidence": round(
            confidence_pct,
            2
        ),
        "architecture": BEST_ARCHITECTURE
    }


@app.get("/model-info")
def model_info():

    return {
        "model": "Fashion-MNIST Production Model",
        "architecture": BEST_ARCHITECTURE,
        "model_loaded": MODEL_LOADED,
        "classes": CLASSES
    }