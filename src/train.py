import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import Food101

from app.model import Net


os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

trainset = Food101(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

testset = Food101(
    root="./data",
    split="test",
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=False
)

model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

NUM_EPOCHS = 1

print("Starting Food-101 training...")

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in trainloader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in testloader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

accuracy = 100 * correct / total

print(f"Food-101 Accuracy: {accuracy:.2f}%")

torch.save(
    model.state_dict(),
    "models/food101.pth"
)

print("Model saved to models/food101.pth")