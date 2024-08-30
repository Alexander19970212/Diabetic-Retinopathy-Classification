import torch
import torch.nn as nn
import torchvision
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from SSIT.ssit_models import SSitEncoder
from MedViT import MedViT, MedViT_large

backbone_options = {
    "resnet50": {"model": torchvision.models.resnet50, 
                 "feature_length":  2048, 
                 "output_type": "features",
                 "cut_layers": ["fc"],
                 "params": {"pretrained": True}},
    "MedViT": {"model": MedViT_large,
               "feature_length": 1024,
               "output_type": "features",
               "cut_layers": ["proj_head"],
               "pretrained_cfg": 'model/checkpoints/MedViT_large_im1k.pth',
               "params": {"use_checkpoint": False}}
}

class ClfConfig(PretrainedConfig):
    model_type = "clf"

    def __init__(
        self,
        backbone_name = "resnet50",
        num_classes = 5,
        input_size = 224,
        pretrained = True,
        only_ssit_embds = False,
        external_embedings = False,
        external_embedings_len = 384,
        emb_model_checkpoint = '../checkpoints/pretrained_vits_imagenet_initialized.pt',
        feat_concat = True,
        **kwargs
    ):
        assert (external_embedings==False and only_ssit_embds==True) == False, "Need to turn on external_embedings!"

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.input_size = input_size
        self.pretrained = pretrained

        self.only_ssit_embds = only_ssit_embds
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
        self.only_ssit_embds = config.only_ssit_embds
        self.external_embedings = config.external_embedings
        self.feat_concat = config.feat_concat
        self.global_pool = config.global_pool

        self.backbone_type = backbone_options[config.backbone_name]["output_type"]
        
        if self.only_ssit_embds != True:
            self.model = backbone_options[config.backbone_name]["model"](**backbone_options[config.backbone_name]["params"])
            
            if "pretrained_cfg" in backbone_options[config.backbone_name].keys():
                self.model.load_state_dict(torch.load(backbone_options[config.backbone_name]["pretrained_cfg"],
                                                 weights_only=True)['model'])

            self.remove_head(backbone_options[config.backbone_name]["cut_layers"])
            #coped_layers = (list(model.children())[:-backbone_options[config.backbone_name]["cut_id"]])
            #coped_layers.append(nn.Flatten())
            #self.model = torch.nn.Sequential(*coped_layers)
            # self.model.fc = torch.nn.Identity() 

        if self.external_embedings:
            print("External embedings are used")
            emd_chs = config.external_embedings_len * 2 if self.feat_concat else config.external_embedings_len
            if self.only_ssit_embds:
                input_head_size = emd_chs
            else:
                input_head_size = backbone_options[config.backbone_name]["feature_length"]+emd_chs
            self.fc_norm = nn.LayerNorm(emd_chs) if config.use_fc_norm else nn.Identity()
            self.pre_logits = nn.Identity()

            self.embd_model = SSitEncoder('ViT-S-p16', config.emb_model_checkpoint)
            for param in self.embd_model.parameters():
                param.requires_grad = False
        else:
            input_head_size = backbone_options[config.backbone_name]["feature_length"]

        self.head = nn.Sequential(
                nn.Linear(input_head_size, 512),
                # nn.BatchNorm1d(num_features = 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                # nn.BatchNorm1d(num_features = 128),
                nn.ReLU(),
                nn.Linear(128, config.num_classes)
                # nn.Softmax()
                )

        # nn.Linear(input_head_size, config.num_classes)

    def remove_head(self, cut_layers):
        for cut_layer_name in cut_layers:
            if cut_layer_name == "fc":
                self.model.fc = torch.nn.Identity()
            elif cut_layer_name == "proj_head":
                self.model.proj_head = torch.nn.Identity()
    
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
        # print("PPV: ", pixel_values)
        if self.only_ssit_embds == False:
            if self.backbone_type == "features":
                features = self.model(pixel_values)
            elif self.backbone_type == "embedings":
                bb_embedings = self.model(pixel_values)
                features = self.concat_embedings(bb_embedings)
                
        if self.external_embedings:
            embedings = self.embd_model(pixel_values)
            embedings = self.concat_embedings(embedings)
            if self.only_ssit_embds:
                features = embedings
            else:
                features = torch.cat((features, embedings), dim=1)

        logits = self.head(features)

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            # loss = torch.nn.functional.mse_loss(logits, labels)
            return {"loss": loss, "logits": logits}
            
        return {"logits": logits}
