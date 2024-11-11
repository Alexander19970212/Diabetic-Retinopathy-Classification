import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


DEFAULT_TRANSFORMS = A.Compose([
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.08, p=0.8),
    A.ToGray(p=0.2),
    A.AdvancedBlur(p=0.5),
    A.Affine(rotate=(-180, 180), p=0.5),
    A.GaussNoise(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

])


def get_transform(mean, std, mode='train', crop_size=512):

    if mode == 'train':
        transform = A.Compose([
            A.Resize(crop_size, crop_size),
            A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.5, 1.0), p=0.8),
            DEFAULT_TRANSFORMS,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif mode == 'eval':
        transform = A.Compose([
            A.Resize(crop_size, crop_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f'Unknown transform mode: {mode}')

    return lambda x: transform(image=np.array(x))['image']
