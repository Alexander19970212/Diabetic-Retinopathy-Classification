import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


DEFAULT_TRANSFORMS = A.Compose([
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=4, min_width=4, p=0.3),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1, p=0.8),
    A.ToGray(p=0.1),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Affine(rotate=(-20, 20), scale=(0.9, 1.1), shear=(-10, 10), p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=50, p=0.2),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])



def get_transform(mean, std, mode='train', crop_size=512):

    if mode == 'train':
        transform = A.Compose([
            A.Resize(crop_size, crop_size),
            A.RandomResizedCrop(
                size=(crop_size, crop_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.8
            ),
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
