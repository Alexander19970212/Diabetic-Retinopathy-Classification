import torch
from torch import nn
from torch.nn import functional as F

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_params['lr'],
            weight_decay=self.optimizer_params['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.optimizer_params['start_factor'],
            end_factor=self.optimizer_params['end_factor'],
            total_iters=self.optimizer_params['total_iters']
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
