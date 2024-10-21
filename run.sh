#!/bin/sh

wandb login 1fa58b4e42c64c2531b3abeb43c04f5991be307e
python train.py --backbone_name resnet50  --dataset_name DDR --run_name DDR05_2__resnet_freeze__SSiT384eye_freeze__sSave39k__1  --num_train_epochs 480  --feat_concat --external_embedings
