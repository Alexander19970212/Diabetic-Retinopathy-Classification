import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


DEFAULT_TRANSFORMS = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.8),
    A.ToGray(p=0.2),
    A.AdvancedBlur(p=0.75),
    A.Solarize(p=0.2),
    A.Affine(rotate=(-180, 180), p=0.8),
    A.GaussNoise(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

])


def collate_fn(batch):
    return {
        'pixel_values': torch.stack(batch['image']),
        'labels': torch.tensor(batch['label'])
    }


def get_transform(mean, std, mode='train', crop_size=512):

    if mode == 'train':
        transform = A.Compose([
            A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.5, 1.0), p=0.8),
            DEFAULT_TRANSFORMS,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    elif mode == 'eval':
        transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f'Unknown transform mode: {mode}')


    def batch_transform(example_batch):
        return {
            'image': [
                transform(image=np.array(image))['image']
                for image in example_batch['image']
            ],
            'label': example_batch['label']
        }
    return batch_transform
