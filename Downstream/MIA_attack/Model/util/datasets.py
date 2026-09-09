# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import pathlib
import _pickle as pickle
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .circle_crop import circle_crop

IMG_FILE_SUFFIX = ['.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff']


def build_dataset(is_train, args):
    '''
    is_train: 'train' or 'test' or 'val'
    '''
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

        # # add circle crop to train transform
        # transform = transforms.Compose([
        #     transforms.Lambda(circle_crop),
        #     transform,
        # ])

        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))

    #t.append(transforms.Lambda(circle_crop))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def load_pkl(root, label_pkl_path):
    out = []
    with open(label_pkl_path, 'rb') as f:
        img_labels = pickle.load(f)
        f.close()
    for img_label in img_labels:
        if '.' in img_label[0]: # has suffix
            path = os.path.join(root, img_label[0])
            if os.path.exists(path):
                out.append(
                    [path, img_label[1]]
                )
        else:   # not have suffix
            for suffix in IMG_FILE_SUFFIX:
                path = os.path.join(root, img_label[0]+suffix)
                if os.path.exists(path):
                    out.append(
                        [path, img_label[1]]
                    )
                break
    return out


def split_dataset(data_dir, label_pkl_path, args, proportion=0.8):
    train_transform = build_transform('train', args)
    test_transform = build_transform('test', args)
    dataset = load_pkl(data_dir, label_pkl_path=label_pkl_path)
    
    train_size = int( proportion * len(dataset) )
    random.shuffle(dataset)
    train_samples = dataset[: train_size]
    test_samples = dataset[train_size :]
    
    train_dataset = LabeledDataset.fromarray(train_samples, train_transform)
    test_dataset = LabeledDataset.fromarray(test_samples, test_transform)
    print('Build training dataset with size {} and test dataset with size {}'.format(
        len(train_dataset), len(test_dataset)
    ))
    return train_dataset, test_dataset


class LabeledDataset(Dataset):
    def __init__(self, data_dir=None, label_pkl_path=None, is_train=None, args=None, **kwargs):
        '''
        used for dataset with .pkl labels in the following format:
        [
            [img_file_name: str, label: int],
            ...
        ]
        '''
        super(Dataset, self).__init__(**kwargs)
        if label_pkl_path is None:
            return
        self.root = data_dir
        with open(label_pkl_path, 'rb') as f:
            img_labels = pickle.load(f)
            f.close()
        # initialize labels (and check if file exists):
        self.img_labels = load_pkl(data_dir, label_pkl_path)
        self.multi_label = args.multi_label
        # initialize transform
        if is_train is None or args is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = build_transform(is_train, args)

    @classmethod
    def fromarray(cls, img_labels, transform=None, **kwargs):
        '''
        Build from array
        '''
        obj = cls(**kwargs)
        obj.img_labels = img_labels
        # initialize transform
        if transform is None:
            obj.transform = transforms.ToTensor()
        else:
            obj.transform = transform
        return obj

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path, label = self.img_labels[index]
        img = Image.open(img_path)
        img = self.transform(img)
        if self.multi_label == False:
            label = int(label)
        else:
            label = [int(x) for x in label]
        return img, torch.tensor(label)

    


class UnlabeledDataset(Dataset):
    # used for pretraining
    def __init__(self, dir_path, transform=None, **kwargs) -> None:
        '''
        If using domain adversial training, then it should return: img, (label,) domain_label.
        '''
        super(Dataset, self).__init__(**kwargs)
        self.root = dir_path
        self.transform = transform

        print('Loading image paths ...')
        self.img_paths = []
        for path in tqdm(pathlib.Path(self.root).glob('**/*')):
            # check the suffix
            if path.is_file() and (path.suffix.lower() in IMG_FILE_SUFFIX):
                self.img_paths.append(path.absolute())
        
        print('Done! {} images found.'.format(len(self.img_paths)))

        if self.transform is None:
            self.transform = transforms.ToTensor()
              
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        if len(img.getbands()) != 3:
            #print(img_path)
            img = img.resize((448,448))
            return transforms.ToTensor()(img.convert('RGB'))
        img = self.transform(img)


        return img



if __name__ == '__main__':
    path = '/path/to/workspace/UWFound/labeled_data/uwf_dr_labeled_senior'
    dataset = datasets.ImageFolder(path)
    print(dataset.imgs)