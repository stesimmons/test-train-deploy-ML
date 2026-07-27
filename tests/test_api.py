# This test script tests the api endpoints, invalid uploads, that the model file exists, 
# that the model loads, output shape, confidence range, and that the correct fashion classes are present.
#
# The purpose of these tests are to showcase automated testing through github actions.

from fastapi.testclient import TestClient
from app.main import app
from io import BytesIO
import os
import torch
from app.model import Net

client = TestClient(app)

def test_home():

    response = client.get("/")

    assert response.status_code == 200
    assert "status" in response.json()

def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model" in data
    assert "model_loaded" in data

def test_predict_route_exists():

    response = client.post("/predict")

    assert response.status_code != 404

def test_predict_rejects_invalid_file():

    response = client.post(
        "/predict",
        files={
            "file": (
                "fake.txt",
                BytesIO(b"hello"),
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_model_file_exists():

    assert os.path.exists(
        "models/fashion_mnist.pth"
    )


def test_model_loads():

    model = Net()

    model.load_state_dict(
        torch.load(
            "models/fashion_mnist.pth",
            map_location="cpu"
        )
    )

def test_model_output_shape():

    model = Net()

    sample = torch.randn(
        1,
        1,
        28,
        28
    )

    output = model(sample)

    assert output.shape == (1, 10)

def test_prediction_schema():

    with open(
        "tests/test_image.png",
        "rb"
    ) as f:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "test_image.png",
                    f,
                    "image/png"
                )
            }
        )

    data = response.json()

    assert "class_id" in data
    assert "class_name" in data
    assert "confidence" in data

def test_confidence_range():

    with open(
        "tests/test_image.png",
        "rb"
    ) as f:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "test_image.png",
                    f,
                    "image/png"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["confidence"] <= 100



EXPECTED_CLASSES = [
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


def test_fashion_classes():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["classes"] == EXPECTED_CLASSES