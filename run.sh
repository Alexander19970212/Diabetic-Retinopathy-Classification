#!/bin/sh

wandb login 1fa58b4e42c64c2531b3abeb43c04f5991be307e
python train.py --backbone_name resnet50  --dataset_name DDR --run_name DDR__resnet224_pre__SSiT384 --num_train_epochs 30  --feat_concat --external_embedings 
