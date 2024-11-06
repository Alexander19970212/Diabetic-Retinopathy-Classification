import numpy as np

from skimage.filters import gaussian, threshold_mean
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread
from skimage.morphology import convex_hull_image
from skimage.transform import resize, rescale


def load_image(path):
    img = imread(path) / 255.0
    return img if img.shape[-1] == 3 else gray2rgb(img)


def get_mask(img, downscale_factor=0.25):
    gray = rgb2gray(img)
    blurred = gaussian(rescale(gray, downscale_factor), sigma=1)
    thresh = threshold_mean(blurred)
    return resize(convex_hull_image(blurred > thresh), gray.shape)


def get_bbox(binary_mask):
    # find bbox
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def squarify(img, bbox, max_size=512, cut_mode=max):
    rmin, rmax, cmin, cmax = bbox
    size = cut_mode(rmax - rmin, cmax - cmin)
    max_size = min(size, max_size)

    # cut rectangular box
    img = img[rmin:rmax, cmin:cmax]
    height, width = img.shape[:2]

    if cut_mode == min:
        height_diff = (height - size) // 2
        width_diff = (width - size) // 2

        square = img[
            height_diff:height-height_diff,
            width_diff:width-width_diff,
        ]

    elif cut_mode == max:
        square = np.zeros((size, size, img.shape[-1]), dtype=img.dtype)
        height_diff = (size - height) // 2
        width_diff = (size - width) // 2

        square[
            height_diff:size-height_diff,
            width_diff:size-width_diff,
        ] = img
    else:
        raise ValueError("<cut_mode> must be either <min> or <max>; got <{}>".format(cut_mode))

    return resize(square, (max_size, max_size))
