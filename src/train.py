import os

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.datasets import Food101

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from app.model import Net


def main():

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    print("Loading Food-101 training dataset...")

    trainset = Food101(
        root="./data",
        split="train",
        download=True,
        transform=transform
    )

    print("Loading Food-101 test dataset...")

    testset = Food101(
        root="./data",
        split="test",
        download=True,
        transform=transform
    )

    # Smaller subset for initial testing
    trainset = Subset(trainset, range(5000))
    testset = Subset(testset, range(1000))

    print(f"Training samples: {len(trainset)}")
    print(f"Testing samples: {len(testset)}")

    trainloader = DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    testloader = DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = Net()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    NUM_EPOCHS = 5

    print("Starting Food-101 training...")

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(trainloader):

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:

                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_idx}/{len(trainloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = (
            running_loss /
            len(trainloader)
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f}"
        )

    print("Evaluating model...")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in testloader:

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
        f"Food-101 Accuracy: "
        f"{accuracy:.2f}%"
    )

    torch.save(
        model.state_dict(),
        "models/food101.pth"
    )

    print(
        "Model saved to "
        "models/food101.pth"
    )


if __name__ == "__main__":
    main()