import numpy as np
from tqdm import tqdm

from skimage.io import imread
from skimage.color import gray2rgb

import os

try:
    from utils import load_image
except ImportError:
    from .utils import load_image


def distributed_statistics(iterable):
    running_var = np.zeros(3, dtype=float)
    running_mean = np.array(3, dtype=float)


    n_processed = 0
    for chunk in tqdm(iterable, leave=False, desc='Calculating statistics'):

        n_chunk = np.prod(chunk.shape)

        var = np.var(chunk, axis=(0, 1))
        mean = np.mean(chunk, axis=(0, 1))

        n_total = n_processed + n_chunk

        # update var
        if running_var.sum() == 0:
            running_var = var
        else:
            delta = running_mean - mean
            running_var = running_var * (n_processed - 1) + var * (n_chunk - 1) + \
                          (delta ** 2) * (n_processed * n_chunk / n_total)
            running_var /= (n_total - 1)

        # update mean
        running_mean = running_mean * (n_processed / n_total) + \
                       mean * (n_chunk / n_total)
        n_processed = n_total

    return running_mean, running_var


def lazy_loader(root):
    for path, subdirs, files in os.walk(root):
        if not files:
            continue
        for file in files:
            if os.path.splitext(file)[1] in ['.jpg', '.png', '.jpeg']:
                yield load_image(os.path.join(path, file))
