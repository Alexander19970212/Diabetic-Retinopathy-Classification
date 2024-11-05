import torch
from torch import nn

import sys

old_path = sys.path
sys.path.append('src/SSiT')
sys.path.append('SSiT')
from . import vits, funcs
sys.path = old_path


class SSITEncoder(nn.Module):
    def __init__(self, arch, input_size, features_dim, checkpoint, checkpoint_key, linear_key):
        super().__init__()

        self.input_size = input_size
        self.features_dim = features_dim

        self.model = vits.archs[arch](
            pretrained=False,
            num_classes=1,          # for checkpoint compatibility
            img_size=input_size,
            # feat_concat=True # do we need this?
        )

        if checkpoint:
            old_path = sys.path
            sys.path.append('src/SSiT')
            sys.path.append('SSiT')
            funcs.load_checkpoint(self.model, checkpoint, checkpoint_key, linear_key)
            sys.path = old_path
        else:
            print('No checkpoint provided. Training from scratch.')

    def forward(self, X):
        return self.model.forward_features(X)[:, 0] # as in knn.py
