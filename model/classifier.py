import torch
import torch.nn as nn
import torchvision
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from SSIT.ssit_models import SSitEncoder

backbone_options = {
    "resnet50": {"model": torchvision.models.resnet50, "feature_length":  2048}
}

class ClfConfig(PretrainedConfig):
    model_type = "clf"

    def __init__(
        self,
        backbone_name = "resnet50",
        num_classes = 5,
        input_size = 224,
        pretrained = True,
        external_embedings = False,
        external_embedings_len = 384,
        emb_model_checkpoint = '../checkpoints/pretrained_vits_imagenet_initialized.pt',
        feat_concat = True,
        **kwargs
    ):
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.input_size = input_size
        self.pretrained = pretrained

        self.external_embedings = external_embedings
        self.external_embedings_len = external_embedings_len
        self.emb_model_checkpoint = emb_model_checkpoint

        self.feat_concat = feat_concat
        self.global_pool = 'avg'
        self.use_fc_norm = True
            
        
class Classifier(PreTrainedModel):
    config_class = ClfConfig

    def __init__(self, config):
        super().__init__(config)
        self.external_embedings = config.external_embedings
        self.feat_concat = config.feat_concat
        self.global_pool = config.global_pool
    
        self.model = backbone_options[config.backbone_name]["model"](pretrained=True)
        self.model.fc = nn.Identity()

        if config.external_embedings:
            emd_chs = config.external_embedings_len * 2 if self.feat_concat else config.external_embedings_len
            input_head_size = backbone_options[config.backbone_name]["feature_length"]+emd_chs
            self.fc_norm = nn.LayerNorm(emd_chs) if config.use_fc_norm else nn.Identity()
            self.pre_logits = nn.Identity()

            self.embd_model = SSitEncoder('ViT-S-p16', config.emb_model_checkpoint)
            for param in self.embd_model.parameters():
                param.requires_grad = False
        else:
            input_head_size = backbone_options[config.backbone_name]["feature_length"]

        self.head = nn.Linear(input_head_size, config.num_classes)
    
    def save_backbone_checkpoint(self, checkpoint_path):
        torch.save(self.model.state_dict(), checkpoint_path)
        print("Checkpoint is saved")

    def load_backbone_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))  
        print("Checkpoint is loaded.")      

    def concat_embedings(self, x):
        if self.feat_concat:
            feats = x[:, 1:].mean(dim=1)
            x = torch.cat((x[:, 0], feats), dim=1)
        elif self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, pixel_values, labels=None):
        # define function in transformers library maner
        feateres = self.model(pixel_values)
        if self.external_embedings:
            embedings = self.embd_model(pixel_values)
            embedings = self.concat_embedings(embedings)
            feateres = torch.cat((feateres, embedings), dim=1)

        logits = self.head(feateres)

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            # loss = torch.nn.functional.mse_loss(logits, labels)
            return {"loss": loss, "logits": logits}
            
        return {"logits": logits}