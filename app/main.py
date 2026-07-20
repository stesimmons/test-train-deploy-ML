from fastapi import FastAPI

print("STARTUP STEP 1")

app = FastAPI()

print("STARTUP STEP 2")


@app.get("/")
def home():

    return {
        "message": "FOOD101_TEST_DEPLOYMENT"
    }


@app.get("/food-test")
def food_test():

    return {
        "status": "food101 deployment active"
    }


@app.get("/model-info")
def model_info():

    return {
        "deployment": "food101-test",
        "version": "1.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }