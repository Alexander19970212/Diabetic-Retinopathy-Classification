import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
)

import matplotlib.pyplot as plt

from collections import defaultdict
import os
import re
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

def get_best_checkpoint(log_dir, key='validation_cohen_kappa') -> str:
    checkpoints = defaultdict(dict)
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')

    pattern = r'(\w+)=([-+]?\d*\.\d+|\d+)'

    for path in [file for file in os.listdir(checkpoints_dir)
                 if file.endswith('.ckpt') and file != 'last.ckpt']:

        filename = os.path.splitext(os.path.basename(path))[0]
        matches = re.findall(pattern, filename)  # Extract key-value pairs
        checkpoints[path] = {k: float(v) if '.' in v or 'e' in v else int(v)
                             for k, v in matches}  # Create a dictionary for the checkpoint

    # Find the checkpoint with the maximum value for the given key
    cpkt = max(checkpoints.keys(), key=lambda cp: float(checkpoints[cp][key]))
    return os.path.join(checkpoints_dir, cpkt)
