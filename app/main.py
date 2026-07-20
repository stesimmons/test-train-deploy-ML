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

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


@app.get("/")
def home():
    return {
        "message": "FOOD101_LIVE"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/food-test")
def food_test():
    return {
        "status": "food101 active"
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "Food-101 CNN",
        "num_classes": len(CLASSES),
        "model_loaded": True
    }