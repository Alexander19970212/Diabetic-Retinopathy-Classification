import cv2
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np

def distributed_statistics(iterable, path):
    running_var = 0
    running_mean = 0
    
    n_processed = 0
    for chunk_name in iterable:
        image_path = path+"/"+chunk_name
        chunk = cv2.imread(image_path)/255
        n_chunk = np.prod(chunk.shape)
        
        var = np.var(chunk)
        mean = np.mean(chunk)

        n_total = n_processed + n_chunk

        # update var
        if running_var == 0:
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

    print(f'Distributed mean:\t{mean:.4f}')
    print(f'Distributed var:\t{var:.4}')

if __name__ == '__main__':
    main()

