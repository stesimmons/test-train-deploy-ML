# Defines the Fashion-MNIST CNN architectures used throughout the
# project. Multiple models are benchmarked, tracked in MLflow, and
# promoted through the Candidate → Staging → Production lifecycle.

import torch.nn as nn


class BaselineCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


class WiderCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


class DeepCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)