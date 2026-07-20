import os

from fastapi import FastAPI

print("STEP 1")

app = FastAPI()

print("STEP 2")

MODEL_PATH = "models/food101.pth"

print("STEP 3")

@app.get("/")
def home():
    return {
        "message": "FOOD101_MODEL_TEST"
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
        "model_exists": exists,
        "model_size_mb": size_mb
    }

@app.get("/food-test")
def food_test():
    return {
        "status": "active"
    }