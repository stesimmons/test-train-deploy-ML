# This script pulls an image from Fashion Mnist data set to test the model

from torchvision.datasets import FashionMNIST

classes = [
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

dataset = FashionMNIST(
    root="./data",
    train=False,
    download=True
)

image, label = dataset[3]

image.save("test_image.png")

print(f"Label: {label}")
print(f"Class: {classes[label]}")
print("Saved: test_image.png")