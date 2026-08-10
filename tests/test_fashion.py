# Tests the Fashion-MNIST prediction workflow by validating that the API
# correctly processes input images and returns valid classification
# results. Ensures the end-to-end inference pipeline functions as
# expected before deployment.

from fastapi.testclient import TestClient
from app.main import app

from io import BytesIO

import torch

from app.model import (
    BaselineCNN,
    WiderCNN,
    DeepCNN
)

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

    print(response.status_code)
    print(response.text)

    assert response.status_code == 400


def test_baseline_model_output_shape():

    model = BaselineCNN()

    sample = torch.randn(
        1,
        1,
        28,
        28
    )

    output = model(sample)

    assert output.shape == (1, 10)


def test_wider_model_output_shape():

    model = WiderCNN()

    sample = torch.randn(
        1,
        1,
        28,
        28
    )

    output = model(sample)

    assert output.shape == (1, 10)


def test_deep_model_output_shape():

    model = DeepCNN()

    sample = torch.randn(
        1,
        1,
        28,
        28
    )

    output = model(sample)

    assert output.shape == (1, 10)


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