from torch import nn
from torchvision.models import (
    resnet50, resnet101,
    resnext50_32x4d, resnext101_32x8d,
    swin_v2_t, swin_v2_s, swin_v2_b,
    vit_b_16, vit_l_16,
    vgg16, vgg16_bn,
    efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l,
    inception_v3,
)
import yaml

archs = {
    'resnet50': resnet50, 'resnet101': resnet101,
    'resnext50': resnext50_32x4d, 'resnext101': resnext101_32x8d,
    'swin_t': swin_v2_t, 'swin_s': swin_v2_s, 'swin_b': swin_v2_b,
    'vit_b': vit_b_16, 'vit_l': vit_l_16,
    'vgg16': vgg16, 'vgg16_bn': vgg16_bn,
    'efficientnet_s': efficientnet_v2_s, 'efficientnet_m': efficientnet_v2_m, 'efficientnet_l': efficientnet_v2_l,
    'inception': inception_v3
}


class FeatureExtractor(nn.Module):
    def __init__(self, arch, config='configs/external.yaml'):
        super().__init__()

        assert arch in archs.keys(), 'Not implemented architecture.'
        with open(config) as file:
            self.config = yaml.safe_load(file)[arch]

        self.input_size = self.config['input_size']
        self.features_dim = self.config['features_dim']

        # load model
        self.model = archs[arch](weights=self.config['weights'])

        # cut layers
        if not self.config['layer_keys']:
            return

        for layer_key in self.config['layer_keys']:
            setattr(self.model, layer_key, nn.Identity())

    def forward(self, X):
        model_output = self.model(X)
        if getattr(model_output, 'logits', None) is not None:
            return model_output.logits

        return model_output
