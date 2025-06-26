import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, reid=True):
        super(Net, self).__init__()
        self.reid = reid

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # (B, 3, 128, 64) → (B, 32, 128, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # → (B, 32, 64, 32)

            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # → (B, 64, 64, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # → (B, 64, 32, 16)

            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # → (B, 128, 32, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # → (B, 128, 16, 8)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # → (B, 128, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),         # → (B, 128)
            nn.Linear(128, 128),  # → (B, 128)
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x
