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

from model.classifier import ClfConfig, Classifier
from utils import train, test
from data.data_utils import build_datasets

if __name__ == '__main__':
# def train_main():

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="1" # is need to train on 'hachiko'

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
    parser.add_argument('--learning_rate', default=5e-6)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--gradient_accumulation_steps', default=4)
    parser.add_argument('--num_train_epochs', type=int, default=15)
    parser.add_argument('--warmup_ratio', default=0.1) #0.02
    parser.add_argument('--metric_for_best_model', default="kappa")
    parser.add_argument('--dataset_root_dir', default="../mnt/local/data/kalexu97")
    # parser.add_argument('--with_emdedings', default=None)
    # model/checkpoints/pretrained_vits_imagenet_initialized.pt
    parser.add_argument('--emb_model_checkpoint', default="model/checkpoints/ssit_ddr_pretrained.pt")
    parser.add_argument('--plots_path', default="src")
    parser.add_argument('--save_hgf_model', default=True, action="store_true")
    parser.add_argument('--saved_model_dir', default="model/checkpoints")
    parser.add_argument('--backbone_name', default="resnet50")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--input_size2', type=int, default=384)
    parser.add_argument('--pretrained', default=True, action="store_true")
    parser.add_argument('--only_ssit_embds', default=False, action="store_true")
    parser.add_argument('--external_embedings', default=False, action="store_true")
    parser.add_argument('--external_embedings_len', default=384)
    parser.add_argument('--feat_concat', default=True, action="store_true")
    parser.add_argument('--apply_encoder', default=False, action="store_true")
    parser.add_argument('--save_backbone', default=False, action="store_true")
    parser.add_argument('--load_backbone', default=False, action="store_true")
    parser.add_argument('--backbone_checkpoint_path_load', default="model/checkpoints/resnet50_backbone.pt")
    parser.add_argument('--dataset_name', default="DDR")
    parser.add_argument('--backbone_checkpoint_path_save', default="model/checkpoints/resnet50_backbone.pt")
    parser.add_argument('--test_after_train', default=True, action="store_true")
    args = parser.parse_args()

    model_config = ClfConfig(backbone_name = args.backbone_name,
        num_classes = args.num_classes,
        input_size = args.input_size,
        input_size2 = args.input_size2,
        pretrained = args.pretrained,
        only_ssit_embds = args.only_ssit_embds,
        external_embedings = args.external_embedings,
        external_embedings_len = args.external_embedings_len,
        emb_model_checkpoint = args.emb_model_checkpoint,
        apply_encoder = args.apply_encoder,
        feat_concat = args.feat_concat)
    model = Classifier(model_config)

    if args.load_backbone:
       model.load_backbone_checkpoint(args.backbone_checkpoint_path_load)

    # model.embd_model.load_state_dict(torch.load(args.emb_model_checkpoint))
    # model.embd_model.load_state_dict(torch.load("model/checkpoints/ssit_eyepack_pretrained.pt"))
    # model.embd_model.load_state_dict(torch.load("model/checkpoints/ssit_ddr05_1_pretrained.pt"))
    model.embd_model.load_state_dict(torch.load("model/checkpoints/ssit_eye_ddr05_1_pretrained.pt"))

    test_dataset, train_dataset, valid_dataset = build_datasets(args.dataset_name,
            args.dataset_root_dir,
            input_size=args.input_size,
            input_size2=args.input_size2)

    model = train(model, train_dataset, valid_dataset, test_dataset, args)

    if args.save_backbone:
        model.save_backbone_checkpoint(args.backbone_checkpoint_path_load)

    # torch.save(model.embd_model.state_dict(), "model/checkpoints/ssit_eye_ddr05_1_pretrained.pt")
    # print("SSIT is saved...")

    # if args.test_after_train:
        # test(model, train_dataset, valid_dataset, test_dataset, args)
