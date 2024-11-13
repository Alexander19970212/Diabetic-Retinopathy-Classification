import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

import lightning as L
from lightning.pytorch.callbacks import Callback

from functools import partial

try:
    from .utils import get_metrics
except ImportError:
    from utils import get_metrics


class Metrics(Callback):
    def __init__(self, num_classes, cb_type):
        self.num_classes = num_classes
        self.reset()
        setattr(self, 'on_' + cb_type + '_batch_end', self.on_batch_end)
        setattr(self, 'on_' + cb_type + '_epoch_end', partial(self.on_epoch_end, prefix=cb_type))

    def reset(self):
        self.y_pred = []
        self.y_true = []
        self.y_score = []

    def on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _, labels = batch
        y_true = labels.cpu().numpy()
        y_score = F.softmax(outputs['pre-logits'], dim=-1).detach().cpu().numpy()
        y_pred = y_score.argmax(axis=1)

        self.y_true.extend(y_true)
        self.y_score.extend(y_score)
        self.y_pred.extend(y_pred)


    def on_epoch_end(self, trainer, pl_module, prefix=''):
        metrics = get_metrics(
            self.y_true, self.y_pred, self.y_score,
            num_classes=self.num_classes
        )

        for key, value in metrics.items():
            self.log(f'{prefix}_{key}', value)

        self.reset()



class LitClassifier(L.LightningModule):
    def __init__(self, backbone, optimizer_params):
        super().__init__()
        self.backbone = backbone
        self.num_classes = self.backbone.num_classes
        self.optimizer_params = optimizer_params

        self.loss = nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=['backbone'])

    def forward(self, images):
        return self.backbone(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss(logits, labels)
        self.log('train_loss', loss)
        return {
            'loss': loss,
            'pre-logits': logits,
        }

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss(logits, labels)
        self.log('val_loss', loss)
        return {
            'loss': loss,
            'pre-logits': logits,
        }


    def configure_optimizers(self):
        # Extracting parameters from the optimizer config
        lr = self.optimizer_params.get('lr', 5e-5)
        weight_decay = self.optimizer_params.get('weight_decay', 1e-4)
        warmup_epochs = self.optimizer_params.get('warmup_epochs', 20)
        total_epochs = self.optimizer_params.get('total_epochs', 100)


        def exclude_weight_decay(named_parameters):
            decay, no_decay = [], []
            for name, param in named_parameters:
                if param.requires_grad:
                    if "bn" in name or "ln" in name or "bias" in name or "norm" in name:
                        no_decay.append(param)  # Exclude BatchNorm and bias parameters
                    else:
                        decay.append(param)
            return [{'params': decay, 'weight_decay': weight_decay},
                    {'params': no_decay, 'weight_decay': 0.0}]

        # Use the function to split parameters into groups
        param_groups = exclude_weight_decay(self.named_parameters())

        # Setup optimizer with separated parameter groups
        optimizer = torch.optim.AdamW(param_groups, lr=lr)



        # Warmup scheduler for the first few epochs
        warmup_lr_lambda = lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

        # Cosine annealing scheduler for the rest of the epochs
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs), eta_min=0)

        # Combining schedulers
        scheduler = {
            'scheduler': SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
            ),
            'interval': 'epoch',
            'frequency': 1
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
