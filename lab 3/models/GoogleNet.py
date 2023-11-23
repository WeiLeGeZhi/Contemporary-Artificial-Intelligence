import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(channels_in, 24, 1)
        )
        self.branch2 = nn.Conv2d(channels_in, 16, 1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels_in, 16, 1),
            nn.Conv2d(16, 24, 5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels_in, 16, 1),
            nn.Conv2d(16, 23, 3, padding=1),
            nn.Conv2d(23, 24, 3, padding=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(88, 20, 5)
        self.incep1 = InceptionModule(channels_in=10)
        self.incep2 = InceptionModule(channels_in=20)
        self.maxpool = nn.MaxPool2d(2)
        self.fully_connection = nn.Linear(1408, 10)

    def forward(self, x):
        size_fc = x.shape[0]
        x = F.relu(self.maxpool(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.maxpool(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(size_fc, -1)
        x = self.fully_connection(x)
        return x
