from torchvision.datasets import FashionMNIST
from torchvision import transforms
from PIL import Image

dataset = FashionMNIST(
    root="./data",
    train=False,
    download=True
)

image, label = dataset[0]

image.save("fashion_test.png")

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

print("Label:", label)
print("Class:", classes[label])
print("Saved as fashion_test.png")