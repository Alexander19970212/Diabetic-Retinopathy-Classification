#!/bin/sh

wandb login 1fa58b4e42c64c2531b3abeb43c04f5991be307e
python train.py --backbone_name resnet50  --dataset_name DDR --input_size2 224  --run_name DDR_hachiko_resnet_attnHeadOneLayer_extrembds_withoutEncoder --num_train_epochs 30  --feat_concat --external_embedings 
