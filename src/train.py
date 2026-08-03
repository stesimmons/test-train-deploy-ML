import json
import os
import shutil

from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

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


MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "v1.0.0"
)


def train_and_evaluate(
    model,
    architecture,
    trainloader,
    testloader,
    device
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    num_epochs = 1

    with mlflow.start_run(
        run_name=architecture
    ) as run:

        run_id = run.info.run_id

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

        mlflow.set_tags({
            "architecture": architecture,
            "stage": "Candidate",
            "registry_state": "Candidate",
            "version": MODEL_VERSION
        })

        print(
            f"\nTraining {architecture}"
        )

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
                running_loss
                / len(trainloader)
            )

            print(
                f"{architecture} | "
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Loss: {avg_loss:.4f}"
            )

            mlflow.log_metric(
                f"epoch_{epoch + 1}_loss",
                avg_loss
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

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        model_path = (
            f"models/candidates/{architecture}.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        try:

            mlflow.log_artifact(
                model_path
            )

        except Exception as e:

            print(
                f"MLflow artifact logging skipped: {e}"
            )

        print(
            f"{architecture} Accuracy: "
            f"{accuracy:.2f}%"
        )

        return (
            accuracy,
            architecture,
            model_path,
            run_id
        )


def main():

    os.makedirs(
        "models/candidates",
        exist_ok=True
    )

    os.makedirs(
        "models/staging",
        exist_ok=True
    )

    os.makedirs(
        "models/production",
        exist_ok=True
    )

    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    mlflow.set_experiment(
        "fashion-mnist"
    )

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

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

    experiments = [
        (
            "BaselineCNN",
            BaselineCNN()
        ),
        (
            "WiderCNN",
            WiderCNN()
        ),
        (
            "DeepCNN",
            DeepCNN()
        )
    ]

    results = []

    for architecture, model in experiments:

        result = train_and_evaluate(
            model=model.to(device),
            architecture=architecture,
            trainloader=trainloader,
            testloader=testloader,
            device=device
        )

        results.append(result)

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_accuracy = results[0][0]
    best_architecture = results[0][1]
    best_model_path = results[0][2]
    best_run_id = results[0][3]

    client = MlflowClient()

    client.set_tag(
        best_run_id,
        "stage",
        "Production"
    )

    client.set_tag(
        best_run_id,
        "registry_state",
        "Production"
    )

    client.set_tag(
        best_run_id,
        "version",
        MODEL_VERSION
    )

    shutil.copyfile(
        best_model_path,
        "models/staging/staging-model.pth"
    )

    shutil.copyfile(
        best_model_path,
        "models/production/production-model.pth"
    )

    metrics = {
        "best_architecture":
            best_architecture,

        "best_accuracy":
            round(
                best_accuracy,
                2
            ),

        "stage":
            "Production",

        "version":
            MODEL_VERSION,

        "run_id":
            best_run_id,

        "registered_at":
            datetime.utcnow().isoformat(),

        "all_results": [
            {
                "architecture":
                    result[1],

                "accuracy":
                    round(
                        result[0],
                        2
                    )
            }

            for result in results
        ]
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

    print("\nTraining Summary")

    for accuracy, architecture, _, _ in results:

        print(
            f"{architecture}: "
            f"{accuracy:.2f}%"
        )

    print(
        f"\nBest Model: "
        f"{best_architecture}"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Version: "
        f"{MODEL_VERSION}"
    )

    print(
        f"Production Run ID: "
        f"{best_run_id}"
    )


if __name__ == "__main__":
    main()