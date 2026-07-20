import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from app.model import Net

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# Fashion-MNIST preprocessing
transform = transforms.ToTensor()

# Training dataset
trainset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Test dataset
testset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)

# Load CNN model
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


NUM_EPOCHS = 10

print("Starting Fashion-MNIST training...")

for epoch in range(NUM_EPOCHS):

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

# Evaluate model
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in testloader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Fashion-MNIST Accuracy: {accuracy:.2f}%")

# Save model
torch.save(
    model.state_dict(),
    "models/fashion_mnist.pth"
)

print("Model saved to models/fashion_mnist.pth")