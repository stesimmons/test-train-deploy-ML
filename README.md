# Fashion-MNIST Image Classification API

An end-to-end machine learning project that trains a Convolutional Neural Network (CNN) on the Fashion-MNIST dataset and serves predictions through a FastAPI REST API.

## Overview

This project demonstrates machine learning workflow:

- Training a CNN using PyTorch
- Saving and loading trained model weights
- Building REST API endpoints with FastAPI
- Writing automated tests with Pytest
- Implementing CI/CD with GitHub Actions
- Deploying the application to Render
- Serving real-time image classification predictions

The model classifies clothing items into 10 categories from the Fashion-MNIST dataset.

---

## Dataset

Fashion-MNIST consists of 70,000 grayscale images of clothing items.

### Classes

| ID | Class |
|----|---------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Model Architecture

The classifier is a Convolutional Neural Network (CNN) implemented with PyTorch.

```text
Input: 28 x 28 grayscale image

Conv2D (1 → 32)
ReLU
MaxPool

Conv2D (32 → 64)
ReLU
MaxPool

Flatten

Linear (1600 → 128)
ReLU

Linear (128 → 10)
```

---

## API Endpoints

### Home

```http
GET /
```

Response:

```json
{
  "message": "Fashion-MNIST Classifier API"
}
```

---

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

### Model Information

```http
GET /model-info
```

Response:

```json
{
  "model": "Fashion-MNIST CNN",
  "model_loaded": true
}
```

---

### Predict

```http
POST /predict
```

Upload a Fashion-MNIST image and receive a prediction.

Example Response:

```json
{
  "class_id": 9,
  "class_name": "Ankle boot",
  "confidence": 97.83
}
```

---

## Training

Train the model:

```bash
python -m src.train
```

The trained model will be saved as:

```text
models/fashion_mnist.pth
```

---

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

to access Swagger UI.

---

## Testing

Run unit tests:

```bash
pytest
```

Example:

```text
===================
4 passed
===================
```

---

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment.

Pipeline steps:

1. Checkout repository
2. Install dependencies
3. Run automated tests
4. Trigger deployment on Render

---

## Deployment

The API is deployed using Render.

Features:

- Automatic deployment from GitHub
- FastAPI application hosting
- Public prediction endpoint
- Continuous delivery through GitHub Actions

---

## Technologies Used

- Python
- PyTorch
- FastAPI
- Uvicorn
- Pytest
- GitHub Actions
- Render

---

## Skills Demonstrated

### Machine Learning

- CNN design and training
- Image classification
- Model evaluation
- Model serialization

### Software Engineering

- REST API development
- Automated testing
- CI/CD pipelines
- Version control with Git

### MLOps

- Model deployment
- Production inference
- Automated deployment workflows
- Cloud-hosted prediction services

---

## Example Prediction Workflow

```text
Image Upload
      │
      ▼
FastAPI Endpoint
      │
      ▼
Image Preprocessing
      │
      ▼
PyTorch CNN
      │
      ▼
Softmax Probabilities
      │
      ▼
Predicted Clothing Class
```

---

## Future Improvements

- Top-3 and Top-5 predictions
- Model performance monitoring
- Docker containerization
- Frontend web interface
- Additional CNN architectures
- Cloud model storage

---

## Author

Steven Simmons

Machine Learning | MLOps | Optimization | Software Engineering