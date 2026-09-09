import os
import pathlib
import _pickle as pickle
import numpy as np
import random

import torchvision
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import sys
sys.path.append('..')

mean = IMAGENET_DEFAULT_MEAN
std = IMAGENET_DEFAULT_STD

def get_transform(size):
    t = []
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(t)
    return transform

def build_dataset(dir_path, input_size=224):
    transform = get_transform(input_size)
    dataset = torchvision.datasets.ImageFolder(dir_path, transform=transform)
    return dataset


if __name__ == '__main__':
    dataset = build_dataset('/path/to/workspace/uwf/dr_labeled_junior/train')
    print(dataset)
    print(dir(dataset))
    print(dataset.samples)