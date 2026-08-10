# FastAPI inference service for the Fashion-MNIST MLOps pipeline.
# Loads the current Production model from the Candidate → Staging →
# Production model registry, exposes deployment metadata, health checks,
# and prediction endpoints, and serves real-time image classification.

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

import json
import os

import torch
import torch.nn.functional as F

from PIL import Image

import torchvision.transforms as transforms

from app.model import (
    BaselineCNN,
    WiderCNN,
    DeepCNN
)

app = FastAPI()

MODEL_PATH = "models/production/production-model.pth"
METRICS_PATH = "metrics.json"

MODEL_LOADED = False
BEST_ARCHITECTURE = "Unknown"
MODEL_METADATA = {}


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

            MODEL_METADATA = json.load(f)

        BEST_ARCHITECTURE = MODEL_METADATA.get(
            "best_architecture",
            "BaselineCNN"
        )

    except Exception as e:

        print(
            f"Failed loading metrics.json: {e}"
        )

        BEST_ARCHITECTURE = "BaselineCNN"
        MODEL_METADATA = {}

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

    except Exception:

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

    return {
        "class_id": class_id,
        "class_name": CLASSES[class_id],
        "confidence": round(
            float(confidence.item() * 100),
            2
        ),
        "architecture": BEST_ARCHITECTURE,
        "version": MODEL_METADATA.get(
            "version",
            "Unknown"
        ),
        "stage": MODEL_METADATA.get(
            "stage",
            "Unknown"
        )
    }


@app.get("/model-info")
def model_info():

    return {
        "model": "Fashion-MNIST Production Model",
        "architecture": BEST_ARCHITECTURE,
        "accuracy": MODEL_METADATA.get(
            "best_accuracy"
        ),
        "stage": MODEL_METADATA.get(
            "stage"
        ),
        "version": MODEL_METADATA.get(
            "version"
        ),
        "registered_at": MODEL_METADATA.get(
            "registered_at"
        ),
        "run_id": MODEL_METADATA.get(
            "run_id"
        ),
        "model_loaded": MODEL_LOADED,
        "model_exists": os.path.exists(MODEL_PATH),
        "metrics_exists": os.path.exists(METRICS_PATH),
        "classes": CLASSES
    }