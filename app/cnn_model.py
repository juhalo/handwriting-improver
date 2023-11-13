import torch
from torchvision import transforms
import torch.nn as nn


class CNN(nn.Module):
    """Class for the CNN."""

    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 20, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(20, 50, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )
        self.lin = nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(),
            nn.Linear(500, 26),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """"Forward pass of the CNN."""
        x = self.conv(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x
