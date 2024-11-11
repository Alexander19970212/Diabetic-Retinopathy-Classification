import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)

import matplotlib.pyplot as plt


import os
import yaml



def get_metrics(y_true, y_pred, y_score=None, num_classes=None):
    metrics = {}
    labels = np.arange(num_classes) if num_classes is not None else np.unique(y_true)

    # metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=labels)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', labels=labels)

    return metrics


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config
