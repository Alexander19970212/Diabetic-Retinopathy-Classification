import cv2
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from PIL import Image, ImageFilter, ImageOps

ddr_mean = [0.4143788,  0.25651503, 0.12490026]
ddr_std = [0.29622576, 0.20603535, 0.14079799]

ttens_tr = transforms.ToTensor()
norm_tr = transforms.Normalize(ddr_mean, ddr_std)

def pil_loader(img_path):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def distributed_statistics(iterable, path):
    running_var = np.array([0, 0, 0])
    running_mean = np.array([0, 0, 0])
    
    # all_images = []
    
    n_processed = 0
    for chunk_name in tqdm(iterable):
        image_path = path+"/"+chunk_name
        # chunk = cv2.imread(image_path)/255
        chunk = pil_loader(image_path)
        chunk = ttens_tr(chunk)
        # chunk = norm_tr(chunk)
        chunk = np.array(chunk)
        n_chunk = np.prod(chunk.shape)
        
        var = np.var(chunk, axis=(1, 2))
        mean = np.mean(chunk, axis=(1, 2))

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
        

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default="../../mnt/local/data/kalexu97/DDR-dataset/DR_grading_processed/train", help='Name of folder')

def main():
    args = parser.parse_args()
    path = args.root_dir
    image_files = [f for f in listdir(path) if isfile(join(path, f))]
    mean, var = distributed_statistics(image_files, path)

    # print(f'Distributed mean:\t{mean:.4f}')
    # print(f'Distributed var:\t{var:.4}')
    print("Mean: ", mean)
    print("STD: ", np.sqrt(var))

if __name__ == '__main__':
    main()

