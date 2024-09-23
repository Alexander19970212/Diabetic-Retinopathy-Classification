import torch
import torch.nn as nn
import torch.nn.functional as F
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

"""
class AttentionHead(nn.Module):
    # based on MultiHeadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

    class CrossAttention(nn.Module):
        def __init__(self, features_dim, ext_features_dim, num_heads=4, dropout=0.2):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                features_dim, num_heads, dropout=dropout
            )

            self.attention_ext = nn.MultiheadAttention(
                ext_features_dim, num_heads, dropout=dropout
            )

            self.norm1 = nn.LayerNorm(features_dim)
            self.norm2 = nn.LayerNorm(ext_features_dim)

            if ext_features_dim != features_dim:
                self.projector = nn.Linear(ext_features_dim, features_dim)
                self.projector_ext = nn.Linear(features_dim, ext_features_dim)

        def forward(self, features, ext_features):
            attn, _ = self.attention(
                F.relu(self.projector(ext_features)), features, features,           # query, key, value
                attn_mask=None, key_padding_mask=None
            )

            attn_ext, _ = self.attention_ext(
                F.relu(self.projector_ext(features)), ext_features, ext_features,   # query, key, value
                attn_mask=None, key_padding_mask=None
            )

            return torch.cat((self.norm1(features + attn), self.norm2(ext_features + attn_ext)), dim=1)


    def __init__(self, num_classes, features_dim, ext_features_dim=None, num_heads=4, dropout=0.2):
        '''
        num_classes: the number of classes in the dataset

        features_dim: the number of expected features in the input (from MedViT)
        ext_features_dim: the number of expected features in the external input (from SSiT) (if None, no external input)

        num_heads: the number of heads in the multiheadattention models (play with this, 4 or 8 would be a good start)
        '''
        super().__init__()
        self.cross_attention = self.CrossAttention(
            features_dim, ext_features_dim,
            num_heads, dropout
        ) if ext_features_dim is not None else None

        dim = (features_dim + ext_features_dim) if ext_features_dim is not None else features_dim
        self.norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        # replace it with your best classifier
        self.classifier = nn.Linear(dim, num_classes)

        # self.classifier = nn.Sequential(
        #         nn.Linear(features_dim, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, num_classes)
        #         )
"""
"""
class AttentionHead(nn.Module):
    # based on MultiHeadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

    def __init__(self, num_classes, features_dim, ext_features_dim=None, num_heads=4, dropout=0.1):
        '''
        num_classes: the number of classes in the dataset

        features_dim: the number of expected features in the input (from MedViT)
        ext_features_dim: the number of expected features in the external input (from SSiT) (if None, no external input)

        num_heads: the number of heads in the multiheadattention models (play with this, 4 or 8 would be a good start)
        '''
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            features_dim, num_heads, dropout=dropout
        )
        # project external (SSiT) features to the same dimension as the input (MedViT) features
        if ext_features_dim is not None and ext_features_dim != features_dim:
            self.projector = nn.Linear(ext_features_dim, features_dim)
        else:
            self.projector = nn.Identity()

        # replace it with your best classifier
        self.classifier = nn.Linear(features_dim, num_classes)
        self.norm = nn.LayerNorm(features_dim)

    def forward(self, features, ext_features=None):

        ext_features = F.relu(self.projector(ext_features)) if ext_features is not None else features

        attn, _ = self.self_attn(
            ext_features, features, features,         # query, key, value
            attn_mask=None, key_padding_mask=None
        )

        # classify the output
        return self.classifier(self.norm(features + attn))
"""
"""
class AttentionHead(nn.Module):
    # based on MultiHeadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

    def __init__(self, num_classes, features_dim, ext_features_dim=None, num_heads=4, dropout=0.2):
        '''
        num_classes: the number of classes in the dataset

        features_dim: the number of expected features in the input (from MedViT)
        ext_features_dim: the number of expected features in the external input (from SSiT) (if None, no external input)

        num_heads: the number of heads in the multiheadattention models (play with this, 4 or 8 would be a good start)
        '''
        super().__init__()

        dim = (features_dim + ext_features_dim) if ext_features_dim is not None else features_dim
        self.norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        # replace it with your best classifier
        self.classifier = nn.Linear(dim, num_classes)


    def forward(self, features, ext_features=None):

        # do cross_attention if needed
        if ext_features is not None:
            features = torch.cat((features, ext_features), dim=-1)

        # do self_attention
        attn, _ = self.self_attention(features, features, features)

        # classify the output
        return self.classifier(self.norm(features + attn))
"""
"""
class AttentionHead(nn.Module):
    # based on MultiHeadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

    class CrossAttention(nn.Module):
        def __init__(self, features_dim, ext_features_dim, num_heads=4, dropout=0.2):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                features_dim, num_heads, dropout=dropout
            )

            self.attention_ext = nn.MultiheadAttention(
                ext_features_dim, num_heads, dropout=dropout
            )

            self.norm1 = nn.LayerNorm(features_dim)
            self.norm2 = nn.LayerNorm(ext_features_dim)

            if ext_features_dim != features_dim:
                self.projector = nn.Linear(ext_features_dim, features_dim)
                self.projector_ext = nn.Linear(features_dim, ext_features_dim)

        def forward(self, features, ext_features):
            attn, _ = self.attention(
                # query, key, value
                F.relu(self.projector(ext_features)), features, features,
                attn_mask=None, key_padding_mask=None
            )

            attn_ext, _ = self.attention_ext(
                F.relu(self.projector_ext(features)
                       ), ext_features, ext_features,   # query, key, value
                attn_mask=None, key_padding_mask=None
            )

            return torch.concat((self.norm1(features + attn), self.norm2(ext_features + attn_ext)), dim=-1)

    def __init__(self, num_classes, features_dim, ext_features_dim=None, num_heads=4, dropout=0.2):
        '''
        num_classes: the number of classes in the dataset

        features_dim: the number of expected features in the input (from MedViT)
        ext_features_dim: the number of expected features in the external input (from SSiT) (if None, no external input)

        num_heads: the number of heads in the multiheadattention models (play with this, 4 or 8 would be a good start)
        '''
        super().__init__()
        self.cross_attention = self.CrossAttention(
            features_dim, ext_features_dim,
            num_heads, dropout
        ) if ext_features_dim is not None else None

        dim = (features_dim + ext_features_dim) if ext_features_dim is not None else features_dim

        # replace it with your best classifier
        self.classifier = nn.Linear(dim, num_classes)


    def forward(self, features, ext_features=None):

        # do cross_attention if needed
        if ext_features is not None:
            features = self.cross_attention(features, ext_features)

        # classify the output
        return self.classifier(features)
"""


class AttentionHead(nn.Module):
    # based on MultiHeadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

    class CrossAttention(nn.Module):
        def __init__(self, features_dim, ext_features_dim, num_heads=4, dropout=0.2):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                features_dim, num_heads, dropout=dropout
            )

            self.attention_ext = nn.MultiheadAttention(
                ext_features_dim, num_heads, dropout=dropout
            )

            self.norm1 = nn.LayerNorm(features_dim)
            self.norm2 = nn.LayerNorm(ext_features_dim)

            if ext_features_dim != features_dim:
                self.projector = nn.Linear(ext_features_dim, features_dim)
                self.projector_ext = nn.Linear(features_dim, ext_features_dim)

        def forward(self, features, ext_features):
            features_pr = F.relu(self.projector_ext(features))
            ext_features_pr = F.relu(self.projector(ext_features))

            attn, _ = self.attention(
                # query, key, value
                features, ext_features_pr, ext_features_pr,
                attn_mask=None, key_padding_mask=None
            )

            attn_ext, _ = self.attention_ext(
                ext_features, features_pr, features_pr,   # query, key, value
                attn_mask=None, key_padding_mask=None
            )

            return torch.concat((self.norm1(features + attn), self.norm2(ext_features + attn_ext)), dim=-1)

    def __init__(self, num_classes, features_dim, ext_features_dim=None, num_heads=4, dropout=0.2):
        '''
        num_classes: the number of classes in the dataset

        features_dim: the number of expected features in the input (from MedViT)
        ext_features_dim: the number of expected features in the external input (from SSiT) (if None, no external input)

        num_heads: the number of heads in the multiheadattention models (play with this, 4 or 8 would be a good start)
        '''
        super().__init__()
        self.cross_attention = self.CrossAttention(
            features_dim, ext_features_dim,
            num_heads, dropout
        ) if ext_features_dim is not None else None

        dim = (features_dim + ext_features_dim) if ext_features_dim is not None else features_dim

        # replace it with your best classifier
        self.classifier = nn.Linear(dim, num_classes)


    def forward(self, features, ext_features=None):

        # do cross_attention if needed
        if ext_features is not None:
            features = self.cross_attention(features, ext_features)

        # classify the output
        return self.classifier(features)

"""
class AttentionHead(nn.Module):
    # based on MultiHeadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

    class CrossAttention(nn.Module):
        def __init__(self, features_dim, ext_features_dim, num_heads=4, dropout=0.2):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                features_dim, num_heads, dropout=dropout
            )

            self.attention_ext = nn.MultiheadAttention(
                ext_features_dim, num_heads, dropout=dropout
            )

            self.norm1 = nn.LayerNorm(features_dim)
            self.norm2 = nn.LayerNorm(ext_features_dim)

            if ext_features_dim != features_dim:
                self.projector = nn.Linear(ext_features_dim, features_dim)
                self.projector_ext = nn.Linear(features_dim, ext_features_dim)

        def forward(self, features, ext_features):
            features_pr = F.relu(self.projector_ext(features))
            ext_features_pr = F.relu(self.projector(ext_features))

            attn, _ = self.attention(
                # query, key, value
                features, ext_features_pr, ext_features_pr,
                attn_mask=None, key_padding_mask=None
            )

            attn_ext, _ = self.attention_ext(
                ext_features, features_pr, features_pr,   # query, key, value
                attn_mask=None, key_padding_mask=None
            )

            return torch.concat((self.norm1(features + attn), self.norm2(ext_features + attn_ext)), dim=-1)

    def __init__(self, num_classes, features_dim, ext_features_dim=None, num_heads=4, dropout=0.2):
        '''
        num_classes: the number of classes in the dataset

        features_dim: the number of expected features in the input (from MedViT)
        ext_features_dim: the number of expected features in the external input (from SSiT) (if None, no external input)

        num_heads: the number of heads in the multiheadattention models (play with this, 4 or 8 would be a good start)
        '''
        super().__init__()
        self.cross_attention = self.CrossAttention(
            features_dim, ext_features_dim,
            num_heads, dropout
        ) if ext_features_dim is not None else None

        dim = (features_dim + ext_features_dim) if ext_features_dim is not None else features_dim
        self.norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        # replace it with your best classifier
        self.classifier = nn.Linear(dim, num_classes)


    def forward(self, features, ext_features=None):

        # do cross_attention if needed
        if ext_features is not None:
            features = self.cross_attention(features, ext_features)

        attn, _ = self.self_attention(features, features, features)

        # classify the output
        return self.classifier(self.norm(features + attn))
"""

class ClfConfig(PretrainedConfig):
    model_type = "clf"

    def __init__(
        self,
        backbone_name = "resnet50",
        num_classes = 5,
        input_size = 224,
        input_size2 = 384,
        pretrained = True,
        only_ssit_embds = False,
        external_embedings = False,
        external_embedings_len = 384,
        apply_encoder = False,
        emb_model_checkpoint = '../checkpoints/pretrained_vits_imagenet_initialized.pt',
        feat_concat = True,
        **kwargs
    ):
        assert (external_embedings==False and only_ssit_embds==True) == False, "Need to turn on external_embedings!"

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.input_size = input_size
        self.input_size2 = input_size2
        self.pretrained = pretrained

        self.only_ssit_embds = only_ssit_embds
        self.external_embedings = external_embedings
        self.external_embedings_len = external_embedings_len
        self.apply_encoder = apply_encoder
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

            self.embd_model = SSitEncoder('ViT-S-p16', config.emb_model_checkpoint, config.input_size2)
            # if self.only_ssit_embds == False:
            #     print("SSIT freezing ...")
            #     for param in self.embd_model.parameters():
            #         param.requires_grad = False
        else:
            emd_chs = None
            input_head_size = backbone_options[config.backbone_name]["feature_length"]

        # self.head = AttentionHead(num_classes=config.num_classes,
        #                         # features_dim=backbone_options[config.backbone_name]["feature_length"], #!!
        #                         features_dim=input_head_size, #!
        #                         # features_dim = 768/2),
        #                         ext_features_dim=emd_chs,
        #                         num_heads=4,
        #                         # apply_encoder=config.apply_encoder,
        #                         # inner_dim=512,
        #                         dropout=0.2
        #                         )

        # # TODO: create flag
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.head = nn.Sequential(
        #         nn.Linear(input_head_size, 512),
        #         # nn.BatchNorm1d(num_features = 512),
        #         nn.ReLU(),
        #         nn.Linear(512, 128),
        #         # nn.BatchNorm1d(num_features = 128),
        #         nn.ReLU(),
        #         nn.Linear(128, config.num_classes)
        #         # nn.Softmax()
        #         )

        self.head = nn.Linear(input_head_size, config.num_classes)

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

    def forward(self, pixel_values, pixel_values2, labels=None):
        # define function in transformers library maner
        # print("PPV: ", pixel_values)
        # print("PV", pixel_values.shape)
        # print("PV2", pixel_values2.shape)

        if self.only_ssit_embds == False:
            if self.backbone_type == "features":
                features = self.model(pixel_values2)
            elif self.backbone_type == "embedings":
                bb_embedings = self.model(pixel_values2)
                features = self.concat_embedings(bb_embedings)
                
        if self.external_embedings:
            # print("PV2", pixel_values2.shape)
            # print("PV", pixel_values.shape)
            embedings = self.embd_model(pixel_values)
            embedings = self.concat_embedings(embedings)
            if self.only_ssit_embds:
                # features = embedings #!
                
                logits = self.head(embedings) #!!
            else:
                # features = torch.cat((features, embedings), dim=1) #!
                # logits = self.head(features)
                logits = self.head(features, embedings) # !!
        else:  #!!
            logits = self.head(features) #!!
        
        # logits = self.head(features) #!

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            # loss = torch.nn.functional.mse_loss(logits, labels)
            return {"loss": loss, "logits": logits}
            
        return {"logits": logits}
