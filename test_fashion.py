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

image.save("fashion_test.png")

print(f"Label: {label}")
print(f"Class: {classes[label]}")
print("Saved: fashion_test.png")