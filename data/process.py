#! /usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map

import os
import json
import argparse
from sys import stderr
from functools import partial

try:
    from utils import load_image, get_mask, get_bbox, squarify, imsave
    from stats import distributed_statistics, lazy_loader
except ImportError:
    from .utils import load_image, get_mask, get_bbox, squarify, imsave
    from .stats import distributed_statistics, lazy_loader



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, help='path to raw dataset')
    parser.add_argument('--output-folder', type=str, help='path to output folder')
    parser.add_argument('--max-size', type=int, default=512, help='maximum size of image')
    parser.add_argument('--cut-mode', type=str, default='max', help='cut mode for squarification')
    parser.add_argument('--test-size', type=float, default=0.15, help='test set size')
    parser.add_argument('--val-size', type=float, default=0.15, help='validation set size')
    parser.add_argument('--random-state', type=int, default=0xC0FFEE, help='random state for data splitting')
    parser.add_argument('--num-processes', type=int, default=8, help='number of processes to use')
    args = parser.parse_args()


    root = args.image_folder
    print(f'Processing {os.path.basename(root).upper()} dataset...')

    X_train, X_test, X_val, y_train, y_test, y_val = scan_dataset(
        root,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )

    # create output directories
    root = args.output_folder
    train_dir, test_dir, val_dir = map(lambda x: os.path.join(root, x), ['train', 'test', 'valid'])
    unique_classes = np.unique(y_train)
    for path in [train_dir, test_dir, val_dir]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            for label in unique_classes:
                os.makedirs(os.path.join(path, str(label)), exist_ok=True)

    # process sets
    assert args.cut_mode in ['min', 'max'], "Invalid <cut-mode> argument; must be either <min> or <max>"
    args.cut_mode = min if args.cut_mode == 'min' else max

    for X, y, out_dir in [
        (X_train, y_train, train_dir),
        (X_test, y_test, test_dir),
        (X_val, y_val, val_dir)]:
        process_map(
            partial(process_single, max_size=args.max_size, cut_mode=args.cut_mode),
            X,
            [os.path.join(out_dir, str(y_), os.path.basename(x_)) for x_, y_ in zip(X, y)],
            total=len(y),
            desc=f'Processing <{os.path.basename(out_dir)}> subset',
            max_workers=args.num_processes,
            chunksize=1
        )

    # calculate and save train stats
    mean, var = distributed_statistics(lazy_loader(train_dir))
    print(f"MEAN:\t{mean}\nVAR:\t{var}\nSTD:\t{np.sqrt(var)}")
    with open(os.path.join(root, 'stats.json'), 'w') as f:
        json.dump({'mean': mean.tolist(), 'var': var.tolist(), 'std': np.sqrt(var).tolist()}, f)


def scan_dataset(root, test_size=0.15, val_size=0.15, random_state=0xC0FFEE):
    # load training set from the raw dataset
    train_dir = os.path.join(root, 'train')
    if not os.path.exists(train_dir) or not os.path.isdir(train_dir) or \
       not os.path.exists(os.path.join(root, 'train.csv')):
        raise ValueError("Train configuration not found!")

    train_info = pd.read_csv(os.path.join(root, 'train.csv'))
    X_train, y_train = train_info['filename'].apply(
        lambda x: os.path.join(train_dir, x)
    ).values, train_info['label'].values

    mask = ~np.isnan(y_train)
    X_train, y_train = X_train[mask], y_train[mask].astype(int)

    # check for missing directories
    test_dir = os.path.join(root, 'test')
    if os.path.exists(test_dir) and os.path.isdir(test_dir) and \
       os.path.exists(os.path.join(root, 'test.csv')):
        test_info = pd.read_csv(os.path.join(root, 'test.csv'))
        X_test, y_test = test_info['filename'].apply(
            lambda x: os.path.join(test_dir, x)
        ).values, test_info['label'].values

        mask = ~np.isnan(y_test)
        X_test, y_test = X_test[mask], y_test[mask].astype(int)
    else:
        test_dir = None
        print("\tTest set: not found; creating split...")

    val_dir = os.path.join(root, 'valid')
    if os.path.exists(val_dir) and os.path.isdir(val_dir) and \
       os.path.exists(os.path.join(root, 'val.csv')):
        val_info = pd.read_csv(os.path.join(root, 'val.csv'))
        X_val, y_val = val_info['filename'].apply(
            lambda x: os.path.join(val_dir, x)
        ).values, val_info['label'].values

        mask = ~np.isnan(y_val)
        X_val, y_val = X_val[mask], y_val[mask].astype(int)
    else:
        val_dir = None
        print("\tValidation set: not found; creating split...")


    # if both test and validation sets are missing, create them
    if test_dir is None and val_dir is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train,
            test_size=test_size+val_size,
            shuffle=True,
            random_state=random_state,
            stratify=y_train
        )

        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test,
            test_size=val_size / (test_size + val_size),
            shuffle=True,
            random_state=random_state,
            stratify=y_test
        )
    elif test_dir is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train,
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
            stratify=y_train
        )
    elif val_dir is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            shuffle=True,
            random_state=random_state,
            stratify=y_train
        )


    print(f"\tTraining set: {X_train.shape[0]} samples")
    print(f"\tTest set: {X_test.shape[0]} samples")
    print(f"\tValidation set: {X_val.shape[0]} samples")

    return X_train, X_test, X_val, y_train, y_test, y_val


def process_single(path, out_path, max_size=512, cut_mode=max):
    # load & process & save image
    try:
        img = load_image(path)
        mask = get_mask(img)
        bbox = get_bbox(mask)
        img = squarify(img, bbox, max_size, cut_mode)
        imsave(out_path, (img * 255.0).astype(np.uint8))
    except Exception as error:
        print(f'Error: file {path} was not processed.', file=stderr)
        print(f'\tSupposed output path: {out_path}', file=stderr)
        print(f'\tError message:\n{error}', end='\n\n', file=stderr)

if __name__ == "__main__":
    main()
