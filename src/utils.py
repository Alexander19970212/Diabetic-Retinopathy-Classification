import torch

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    roc_auc_score
)

import matplotlib.pyplot as plt


from tqdm.auto import tqdm
import os
import yaml


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config
