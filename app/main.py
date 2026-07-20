import os

from fastapi import FastAPI

print("STEP 1")

import torch

print("STEP 2")

from app.model import Net

print("STEP 3")

app = FastAPI()

MODEL_PATH = "models/food101.pth"

model_exists = os.path.exists(MODEL_PATH)

print(f"MODEL EXISTS: {model_exists}")

model = Net()

print("STEP 4: MODEL CREATED")


@app.get("/")
def home():
    return {
        "message": "FOOD101_MODEL_TEST"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/food-test")
def food_test():
    return {
        "status": "active"
    }


@app.get("/model-info")
def model_info():

    return {
        "deployment": "food101-model-test",
        "model_exists": model_exists
    }