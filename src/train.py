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

    num_epochs = 5

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

        print(f"\nTraining {architecture}")

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

            mlflow.log_metric(
                f"epoch_{epoch+1}_loss",
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
            f"models/{architecture}.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        try:

            mlflow.log_artifact(
                model_path
            )

        except Exception:
            pass

        print(
            f"{architecture}: "
            f"{accuracy:.2f}%"
        )

        return accuracy, model_path


def main():

    os.makedirs(
        "models",
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

        accuracy, model_path = (
            train_and_evaluate(
                model.to(device),
                architecture,
                trainloader,
                testloader,
                device
            )
        )

        results.append(
            (
                accuracy,
                architecture,
                model_path
            )
        )

    results.sort(
        reverse=True
    )

    best_accuracy = results[0][0]
    best_architecture = results[0][1]
    best_model = results[0][2]

    import shutil

    shutil.copyfile(
        best_model,
        "models/production-model.pth"
    )

    metrics = {

        "best_architecture":
            best_architecture,

        "best_accuracy":
            round(
                best_accuracy,
                2
            ),

        "all_results": [

            {
                "architecture": r[1],
                "accuracy": round(
                    r[0],
                    2
                )
            }

            for r in results
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

    print(
        f"\nBest Model: "
        f"{best_architecture}"
    )

    print(
        f"Accuracy: "
        f"{best_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()