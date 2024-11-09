import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import asdict

try:
    from .SSiT import SSITEncoder, DEFAULT_CONFIG as SSIT_DEFAULT_CONFIG
    from .external import FeatureExtractor, DEFAULT_CONFIG as EXT_DEFAULT_CONFIG
except ImportError:
    from SSiT import SSITEncoder, DEFAULT_CONFIG as SSIT_DEFAULT_CONFIG
    from external import FeatureExtractor, DEFAULT_CONFIG as EXT_DEFAULT_CONFIG


class AttentionHead(nn.Module):

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

    def __init__(self, num_classes, features_dim, ext_features_dim, num_heads=4, dropout=0.2):
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
        )

        # replace it with your best classifier
        self.classifier = nn.Linear(features_dim + ext_features_dim, num_classes)


    def forward(self, features, ext_features):

        features = self.cross_attention(features, ext_features)

        # classify the output
        return self.classifier(features)


class LinearClassifier(nn.Module):
    def __init__(self, model, num_classes, features_dim):
        super().__init__()
        self.backbone = model
        self.classifier = nn.Linear(features_dim, num_classes)

    def forward(self, X):
        features = self.backbone(X)
        return self.classifier(features)


class AttentionClassifier(nn.Module):
    def __init__(self, num_classes, backbone, external, num_heads=4, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.external = external
        self.head = AttentionHead(
            num_classes=num_classes,
            features_dim=backbone.features_dim,
            ext_features_dim=external.features_dim,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, X):
        features = self.backbone(X)
        ext_features = self.external(X)
        return self.head(features, ext_features)


class Classifier(nn.Module):
    def __init__(self, num_classes=5, num_heads=4, dropout=0.2,
                 mode='attention', freeze_SSiT=False, freeze_external=False,
                 external_config=EXT_DEFAULT_CONFIG, SSiT_config=SSIT_DEFAULT_CONFIG):

        super().__init__()

        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.mode = mode
        self.freeze_SSiT = freeze_SSiT
        self.freeze_external = freeze_external


        # load external model if needed
        if self.mode in ['external_only', 'attention']:
            self.external_config = external_config
            self.feature_extractor = FeatureExtractor(**asdict(self.external_config))

            if self.freeze_external:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

        # load SSiT model if needed
        if self.mode in ['SSiT_only', 'attention']:
            self.SSiT_config = SSiT_config
            self.SSiT = SSITEncoder(**asdict(self.SSiT_config))

            if self.freeze_SSiT:
                for param in self.SSiT.parameters():
                    param.requires_grad = False

        # build classifier
        if self.mode == 'SSiT_only':
            self.classifier = LinearClassifier(
                model=self.SSiT,
                num_classes=self.num_classes,
                features_dim=self.SSiT_config.features_dim
            )
        elif self.mode == 'external_only':
            self.classifier = LinearClassifier(
                model=self.feature_extractor,
                num_classes=self.num_classes,
                features_dim=self.external_config.features_dim
            )
        elif self.mode == 'attention':
            self.classifier = AttentionClassifier(
                num_classes=self.num_classes,
                backbone=self.SSiT,
                external=self.feature_extractor,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Invalid mode: {self.config['mode']}")

    def forward(self, X):
        return self.classifier(X)
