#!/bin/sh

wandb login 1fa58b4e42c64c2531b3abeb43c04f5991be307e
python train.py --dataset_name DDR --num_classes 5  --run_name DDR_MLP3_sfmax_hachiko_withSSITembs --num_train_epochs 30  --feat_concat --save_backbone --external_embedings  
