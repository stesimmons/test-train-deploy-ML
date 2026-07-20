import os

from fastapi import FastAPI

app = FastAPI()

MODEL_PATH = "models/food101.pth"


@app.get("/")
def home():
    return {
        "message": "FOOD101_TEST_DEPLOYMENT"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/food-test")
def food_test():
    return {
        "status": "food101 deployment active"
    }


@app.get("/model-info")
def model_info():

    exists = os.path.exists(MODEL_PATH)

    size_mb = 0

    if exists:
        size_mb = round(
            os.path.getsize(MODEL_PATH) / 1024 / 1024,
            2
        )

    return {
        "deployment": "food101-test",
        "model_exists": exists,
        "model_size_mb": size_mb
    }