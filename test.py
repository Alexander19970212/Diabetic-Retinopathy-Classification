# main for train
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # is need to train on 'hachiko'
import argparse
import sys
sys.path.append('model')
sys.path.append('model/SSIT')
sys.path.append('model/MedViT')

import torch
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

from model.classifier import ClfConfig, Classifier
from utils import train, test
from data.data_utils import build_datasets

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default="./Classifier")
    parser.add_argument('--evaluation_strategy', default="steps")
    parser.add_argument('--logging_steps', default=50)
    parser.add_argument('--save_steps', default=50)
    parser.add_argument('--eval_steps', default=50)
    parser.add_argument('--save_total_limit', default=3)
    parser.add_argument('--report_to', default="wandb")
    parser.add_argument('--run_name', default="clf_test")
    parser.add_argument('--dataloader_num_workers', default=16)
    parser.add_argument('--lr_scheduler_type', default="linear")
    parser.add_argument('--learning_rate', default=2e-5)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', default=4)
    parser.add_argument('--num_train_epochs', type=int, default=15)
    parser.add_argument('--warmup_ratio', default=0.02)
    parser.add_argument('--metric_for_best_model', default="kappa")
    parser.add_argument('--dataset_root_dir', default="../mnt/local/data/kalexu97")
    parser.add_argument('--plots_path', default="src")
    parser.add_argument('--save_hgf_model', default=True, action="store_true")
    parser.add_argument('--saved_model_dir', default="model/checkpoints")
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--input_size2', type=int, default=384)
    parser.add_argument('--save_backbone', default=False, action="store_true")
    parser.add_argument('--load_backbone', default=False, action="store_true")
    parser.add_argument('--backbone_checkpoint_path_load', default="model/checkpoints/resnet50_backbone.pt")
    parser.add_argument('--dataset_name', default="DDR")
    parser.add_argument('--backbone_checkpoint_path_save', default="model/checkpoints/resnet50_backbone.pt")
    args = parser.parse_args()
    
    model = Classifier.from_pretrained(f"{args.saved_model_dir}/{args.run_name}").to(device)
    test_dataset, train_dataset, valid_dataset = build_datasets(args.dataset_name, args.
            dataset_root_dir,
            input_size=args.input_size,
            input_size2=args.input_size2)

    model.embd_model.load_state_dict(torch.load("model/checkpoints/ssit_eyepack_pretrained.pt"))

    model.eval()
        
    if args.save_backbone:
        model.save_backbone_checkpoint(args.backbone_checkpoint_path_load)

    test(model, test_dataset, train_dataset, valid_dataset, args, device)
