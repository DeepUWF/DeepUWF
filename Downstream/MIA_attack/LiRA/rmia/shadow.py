import torch
import torch.nn as nn
import torch.nn.functional as F


class ShadowModel(nn.Module):
    """Lightweight surrogate CNN for 448x448 SHDR RMIA.

    Adaptive pooling avoids a huge fully connected layer and makes the
    network independent of the exact spatial input size.
    """

    def __init__(self, channel=3, num_classes=2, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv2d(channel, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _extract_conv_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.pool(F.relu(self.bn2(self.conv2(x)), inplace=True))
        x = self.pool(F.relu(self.bn3(self.conv3(x)), inplace=True))
        x = self.pool(F.relu(self.bn4(self.conv4(x)), inplace=True))
        x = self.adaptive_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self._extract_conv_features(x)
        x = self.dropout(F.relu(self.fc1(x), inplace=True))
        return self.fc2(x)

    def feature(self, x):
        return self._extract_conv_features(x)
