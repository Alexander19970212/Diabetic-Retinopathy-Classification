#!/bin/sh

wandb login 1fa58b4e42c64c2531b3abeb43c04f5991be307e
python train.py --backbone_name resnet50  --dataset_name DDR --run_name DDR05_2__resnet_pre_freeze__SSiT384eyeDdr_short_18 --num_train_epochs 5  --feat_concat --external_embedings
