from unittest.mock import DEFAULT
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    resnet50, resnet101,
    resnext50_32x4d, resnext101_32x8d,
    swin_v2_t, swin_v2_s, swin_v2_b,
    vit_b_16, vit_l_16,
    vgg16, vgg16_bn,
    efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l,
    inception_v3,
)

from typing import List
from dataclasses import dataclass, field

archs = {
    'resnet50': resnet50, 'resnet101': resnet101,
    'resnext50': resnext50_32x4d, 'resnext101': resnext101_32x8d,
    'swin_t': swin_v2_t, 'swin_s': swin_v2_s, 'swin_b': swin_v2_b,
    'vit_b': vit_b_16, 'vit_l': vit_l_16,
    'vgg16': vgg16, 'vgg16_bn': vgg16_bn,
    'efficientnet_s': efficientnet_v2_s, 'efficientnet_m': efficientnet_v2_m, 'efficientnet_l': efficientnet_v2_l,
    'inception': inception_v3
}

@dataclass
class Config:
    arch: str = 'resnet50'
    features_dim: int = 2048
    input_size: int = 224
    layer_keys: List[str] = field(default_factory=lambda: ['fc'])
    weights: str = 'IMAGENET1K_V2'

DEFAULT_CONFIG = Config()


class FeatureExtractor(nn.Module):
    def __init__(self, arch, features_dim, input_size, layer_keys, weights):
        super().__init__()

        assert arch in archs.keys(), 'Not implemented architecture.'

        self.input_size = input_size
        self.features_dim = features_dim
        self.layer_keys = layer_keys
        self.weights = weights

        # load model
        self.model = archs[arch](weights=self.weights)

        # cut layers
        if not self.layer_keys:
            return

        for layer_key in self.layer_keys:
            setattr(self.model, layer_key, nn.Identity())

    def forward(self, X):
        X = F.interpolate(X, size=self.input_size)
        model_output = self.model(X)
        if getattr(model_output, 'logits', None) is not None:
            return model_output.logits

        return model_output
