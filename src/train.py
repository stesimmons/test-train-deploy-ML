import json
import os

import mlflow

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from app.model import (
    BaselineCNN,
    WiderCNN,
    DeepCNN
)


def main():

    os.makedirs("models", exist_ok=True)

    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    architecture = os.getenv(
        "MODEL_ARCHITECTURE",
        "BaselineCNN"
    )

    print(
        f"Tracking URI: {mlflow.get_tracking_uri()}"
    )

    print(
        f"Architecture: {architecture}"
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    print("Loading Fashion-MNIST dataset...")

    trainset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    testset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=64,
        shuffle=True
    )

    testloader = DataLoader(
        testset,
        batch_size=64,
        shuffle=False
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if architecture == "BaselineCNN":

        model = BaselineCNN().to(device)

    elif architecture == "WiderCNN":

        model = WiderCNN().to(device)

    elif architecture == "DeepCNN":

        model = DeepCNN().to(device)

    else:

        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    num_epochs = 5

    mlflow.set_experiment(
        "fashion-mnist"
    )

    with mlflow.start_run(
        run_name=architecture
    ):

        mlflow.log_param(
            "architecture",
            architecture
        )

        mlflow.log_param(
            "epochs",
            num_epochs
        )

        mlflow.log_param(
            "batch_size",
            64
        )

        mlflow.log_param(
            "learning_rate",
            0.001
        )

        print("Starting training...")

        for epoch in range(num_epochs):

            model.train()

            running_loss = 0.0

            for images, labels in trainloader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            avg_loss = (
                running_loss /
                len(trainloader)
            )

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Loss: {avg_loss:.4f}"
            )

            mlflow.log_metric(
                f"epoch_{epoch + 1}_loss",
                avg_loss
            )

        print(
            "Evaluating model..."
        )

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in testloader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        accuracy = (
            100 * correct / total
        )

        print(
            f"Fashion-MNIST Accuracy: "
            f"{accuracy:.2f}%"
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        model_path = (
            f"models/{architecture}.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"Model saved to "
            f"{model_path}"
        )

        try:

            mlflow.log_artifact(
                model_path
            )

            print(
                "Model artifact logged to MLflow"
            )

        except Exception as e:

            print(
                f"MLflow artifact logging skipped: {e}"
            )

        metrics = {
            "architecture": architecture,
            "accuracy": round(
                accuracy,
                2
            ),
            "epochs": num_epochs,
            "batch_size": 64,
            "learning_rate": 0.001
        }

        with open(
            "metrics.json",
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

        print(
            "Metrics saved to metrics.json"
        )

    print(
        f"MLruns directory exists: "
        f"{os.path.exists('mlruns')}"
    )


if __name__ == "__main__":
    main()