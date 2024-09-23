#!/bin/sh

wandb login 1fa58b4e42c64c2531b3abeb43c04f5991be307e
python train.py --backbone_name resnet50  --dataset_name DDR --run_name DDR6_080_1__SSiT384eye_sSave3k8__1_c  --num_train_epochs 30  --feat_concat --external_embedings --only_ssit_embds
python train.py --backbone_name resnet50  --dataset_name DDR --run_name DDR6_080_1__SSiT384eye_sSave3k8__2_c  --num_train_epochs 30  --feat_concat --external_embedings --only_ssit_embds
python train.py --backbone_name resnet50  --dataset_name DDR --run_name DDR6_080_1__SSiT384eye_sSave3k8__3_c  --num_train_epochs 30  --feat_concat --external_embedings --only_ssit_embds
