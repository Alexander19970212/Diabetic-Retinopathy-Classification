#! /usr/bin/env python

from sklearn import metrics
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import os
import json
import argparse

from src.classifier import Classifier
from src.external import Config as ExtConfig
from src.SSiT import Config as SSiTConfig
from src.utils import load_config
from src.tools import LitClassifier, Metrics, SavePredictions
from data.preprocessing import get_transform


arg_parser = argparse.ArgumentParser(description='Train a classifier')
arg_parser.add_argument('--config_path', type=str, default='configs', help='Path to config file')
arg_parser.add_argument('--config_name', type=str, default='default.yaml', help='Path to config file')
arg_parser.add_argument('--logger', type=str, default='tensorboard', help='Logger to use: <tensorboard>, <wandb> or <None>')
arg_parser.add_argument('--project', type=str, default='DRGrading', help='Wandb project name')
arg_parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
args = arg_parser.parse_args()


def main():
    # Load config
    config = load_config(os.path.join(args.config_path, args.config_name))
    ext_config = load_config(config['ext_config_path'])
    seed_everything(config['random_seed'], workers=True)

    # Load logger
    if args.logger == 'tensorboard':
        logger = TensorBoardLogger(name=config['name'], save_dir=args.log_dir, default_hp_metric=False)
    elif args.logger == 'wandb':
        logger = WandbLogger(project=args.project, name=config['name'], save_dir=args.log_dir)
    else:
        logger = False

    # Load dataset
    print('Loading dataset...')
    with open(os.path.join(config['dataset']['path'], 'stats.json'), 'r') as f:
        stats = json.load(f)
        stats = {
            'mean': stats['mean'],
            'std': stats['std'],
            'crop_size': config['dataset']['size'],
        }

    train_transform = get_transform(mode='train', **stats)
    eval_transform = get_transform(mode='eval', **stats)

    train_ds = ImageFolder(os.path.join(config['dataset']['path'], 'train'), transform=train_transform)
    valid_ds = ImageFolder(os.path.join(config['dataset']['path'], 'valid'), transform=eval_transform)
    test_ds = ImageFolder(os.path.join(config['dataset']['path'], 'test'), transform=eval_transform)

    train_loader = DataLoader(
        train_ds, shuffle=True,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers']
    )

    valid_loader = DataLoader(
        valid_ds, shuffle=False,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
    )

    test_loader = DataLoader(
        test_ds, shuffle=False,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
    )


    # Build model
    SSiT_config = SSiTConfig(**config['SSiT'])
    external_arch = config['classifier']['external_arch']
    external_config = ExtConfig(**ext_config[external_arch])

    classifier = Classifier(
        num_classes=config['classifier']['num_classes'],
        num_heads=config['classifier']['num_heads'],
        dropout=config['classifier']['dropout'],
        mode=config['classifier']['mode'],
        freeze_SSiT=config['classifier']['freeze_SSiT'],
        freeze_external=config['classifier']['freeze_external'],
        external_config=external_config,
        SSiT_config=SSiT_config
    )

    # Train model
    model = LitClassifier(classifier, config['optimizer'])
    callbacks = [
        Metrics(num_classes=config['classifier']['num_classes'], cb_type='train'),
        Metrics(num_classes=config['classifier']['num_classes'], cb_type='validation'),
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(mode='max', save_top_k=2, monitor="validation_cohen_kappa")
    ]

    # Testing properties
    if '-1' in test_ds.classes: # if there are unknown samples
        codes = [
            os.path.splitext(os.path.basename(sample))[0]
            for sample, _ in test_ds.samples
        ]
        callbacks.append(SavePredictions(codes, os.path.join(logger.log_dir, 'test_results.csv')))
    else:
        callbacks.append(Metrics(num_classes=config['classifier']['num_classes'], cb_type='test'),)

    trainer = Trainer(
        logger=logger,
        **config['trainer'],
        callbacks=callbacks
    )
    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    main()
