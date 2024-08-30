#!/bin/sh

wandb login 1fa58b4e42c64c2531b3abeb43c04f5991be307e
python train.py --backbone_name MedViT  --dataset_name DDR --num_classes 5 --input_size 224  --run_name test_medvit_remove_after --num_train_epochs 10  --feat_concat  
