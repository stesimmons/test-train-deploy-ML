import os

from fastapi import FastAPI

print("STEP 1")

import torch

print("STEP 2")

from app.model import Net

print("STEP 3")

app = FastAPI()

MODEL_PATH = "models/food101.pth"

print("STEP 4")

model = Net()

print("STEP 5")

loaded = False
error_message = None

try:

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    print("STEP 6")

    model.load_state_dict(
        state_dict
    )

    print("STEP 7")

    model.eval()

    loaded = True

except Exception as e:

    error_message = str(e)

    print(f"LOAD ERROR: {e}")


@app.get("/")
def home():

    return {
        "message": "FOOD101_LOAD_TEST"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():

    return {
        "model_loaded": loaded,
        "error": error_message
    }


@app.get("/food-test")
def food_test():

    return {
        "status": "active"
    }