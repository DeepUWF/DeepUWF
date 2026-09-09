import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc, accuracy_score
import math
import matplotlib.pyplot as plt 



# Assuming we have a simple neural network model
class ShadowModel(nn.Module):
    def __init__(self, channel=3, num_classes=2):
        super(ShadowModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # 修复：移除硬编码的 fc1_input_size，改为动态计算以支持不同输入尺寸
        # 原始代码：fc1_input_size = 1024（基于 32x32 图像计算：256 * 2 * 2 = 1024）
        # 对于 448x448 图像：256 * 28 * 28 = 200704
        self.fc1 = None  # 将在 forward 中首次调用时动态初始化
        self.fc2 = nn.Linear(1024, num_classes)
        self.num_classes = num_classes
        
    def _initialize_fc1(self, x):
        """动态计算 fc1 的输入维度并初始化（修复：支持不同输入图像尺寸）"""
        if self.fc1 is None:
            # 计算经过卷积和池化后的特征图尺寸
            with torch.no_grad():
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                x = self.pool(F.relu(self.bn4(self.conv4(x))))
                fc1_input_size = x.view(x.size(0), -1).size(1)
            # 在正确的设备上初始化 fc1
            self.fc1 = nn.Linear(fc1_input_size, 1024).to(x.device)
        
    def forward(self, x):
        # 修复：首次调用时动态初始化 fc1 层
        self._initialize_fc1(x)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flattening the output
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def feature(self,x):
        # 修复：首次调用时动态初始化 fc1 层
        self._initialize_fc1(x)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flattening the output
        x = x.view(x.size(0), -1)

        return x
