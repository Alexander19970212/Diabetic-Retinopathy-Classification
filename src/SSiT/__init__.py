from tabnanny import check
import torch
from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass
import sys

old_path = sys.path
sys.path.append('src/SSiT')
sys.path.append('SSiT')
from . import vits, funcs
sys.path = old_path


@dataclass
class Config:
    arch: str = 'ViT-S-p16'
    input_size: int = 384
    features_dim: int = 768
    checkpoint: str = 'checkpoints/pretrained_vits_imagenet_initialized.pt'
    checkpoint_key: str = 'base_encoder'
    linear_key: str = 'head'
    global_pool: str = 'token'
    feat_concat: bool = True
DEFAULT_CONFIG = Config()


class SSITEncoder(nn.Module):
    def __init__(self,
                 arch, input_size, features_dim,
                 checkpoint, checkpoint_key, linear_key,
                 global_pool='token', feat_concat=True):

        super().__init__()

        assert global_pool in ('', 'avg', 'token'), "Global pooling must be one of '', 'avg', 'token'"

        self.input_size = input_size
        self.features_dim = features_dim
        self.global_pool = global_pool
        self.feat_concat = feat_concat

        self.model = vits.archs[arch](
            pretrained=False,
            num_classes=1,          # for checkpoint compatibility
            img_size=input_size,
            feat_concat=feat_concat
        )

        if checkpoint:
            old_path = sys.path
            sys.path.append('src/SSiT')
            sys.path.append('SSiT')
            funcs.load_checkpoint(self.model, checkpoint, checkpoint_key, linear_key)
            sys.path = old_path
        else:
            print('No checkpoint provided. Training from scratch.')


    def _forward_head(self, x, pre_logits: bool = False):
        if self.feat_concat:
            feats = x[:, 1:].mean(dim=1)
            x = torch.cat((x[:, 0], feats), dim=1)
        elif self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.model.global_pool == 'avg' else x[:, 0]

        x = self.model.fc_norm(x)
        x = self.model.pre_logits(x)
        return x if pre_logits else self.model.head(x)

    def forward(self, X):
        X = F.interpolate(X, size=self.input_size)
        features = self.model.forward_features(X)
        return self._forward_head(features, pre_logits=True)
