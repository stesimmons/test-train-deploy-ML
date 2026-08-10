# Test-Train-Deploy ML: End-to-End MLOps Pipeline for Fashion-MNIST

## Author

**Steven Simmons**

Built to demonstrate practical Machine Learning Engineering and MLOps workflows including automated testing, experiment tracking, model promotion, versioning, deployment, and production inference.

## Overview

This project demonstrates a complete MLOps workflow for image classification using the Fashion-MNIST dataset.

The pipeline automatically:

- Runs unit tests
- Trains multiple CNN architectures
- Tracks experiments with MLflow
- Compares model performance
- Promotes the best model through a lightweight model registry
- Deploys the production model to a live FastAPI service on Render

The goal of the project is to showcase practical ML Engineering and MLOps concepts including experiment tracking, automated model promotion, CI/CD, model versioning, and deployment.

---

## Features

### Automated Testing

Every push triggers:

```text
Pytest
↓
Validation
↓
Training
```

Only code that passes tests can reach the training stage.

### Multi-Model Benchmarking

The pipeline trains and evaluates:

- BaselineCNN
- WiderCNN
- DeepCNN

The highest-performing model is promoted to Production.

### MLflow Experiment Tracking

Each run logs:

- Architecture
- Learning Rate
- Batch Size
- Epochs
- Version
- Accuracy
- Training Loss
- Model Artifacts

### Lightweight Model Registry

```text
Candidate
↓
Staging
↓
Production
```

Repository structure:

```text
models/
├── candidates/
├── staging/
└── production/
```

### Model Versioning

Each successful pipeline run produces a version such as:

```text
v1.0.61
v1.0.62
v1.0.63
```

### Automated Deployment

The production model is deployed automatically to Render through GitHub Actions.

---

## Technology Stack

### Machine Learning

- PyTorch
- TorchVision
- Fashion-MNIST

### API

- FastAPI
- Uvicorn

### MLOps

- MLflow
- GitHub Actions
- GitHub Artifacts

### Deployment

- Render

### Testing

- Pytest

---

## CI/CD Workflow

```text
Push to Main
        │
        ▼
 Run Pytest
        │
        ▼
 Restore MLflow History
        │
        ▼
 Train 3 Architectures
        │
        ▼
 MLflow Tracking
        │
        ▼
 Select Best Model
        │
        ▼
 Validate Accuracy
        │
        ▼
 Update Model Registry
        │
        ▼
 Upload Artifacts
        │
        ▼
 Deploy to Render
```

---

## API Endpoints

### Health Check

```http
GET /health
```

### Model Information

```http
GET /model-info
```

### Prediction

```http
POST /predict
```

Upload an image and receive the predicted Fashion-MNIST class and confidence score.

---

## Model Registry Metadata Example

```json
{
  "best_architecture": "DeepCNN",
  "best_accuracy": 91.82,
  "stage": "Production",
  "version": "v1.0.63"
}
```

---

## Skills Demonstrated

- Machine Learning
- CNN Architecture Design
- Experiment Tracking
- Model Versioning
- Model Promotion
- CI/CD Pipelines
- FastAPI Development
- GitHub Actions
- MLflow
- Render Deployment
- MLOps

---


