## !!! is taken from original SSIT repo:
# https://github.com/YijinHuang/SSiT

# ==========================================================================
# Base on https://github.com/sveitser/kaggle_diabetic/blob/master/convert.py
# ==========================================================================
import os
import random
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFilter
from multiprocessing import Process

from torchvision.transforms import transforms
import os
import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--image-folder', type=str, help='path to image folder')
parser.add_argument('--output-folder', type=str, help='path to output folder')
parser.add_argument('--crop-size', type=int, default=512, help='crop size of image')
parser.add_argument('-n', '--num-processes', type=int, default=8, help='number of processes to use')


###########################################################################################

def imread(file_path, c=None):
    # print(file_path)
    file_path = str(file_path)
    if c is None:
        im = cv2.imread(file_path)
    else:
        im = cv2.imread(file_path, c)

    if im is None:
        raise 'Can not read image'

    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def get_mask_BZ(img):
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    threhold = np.mean(gray_img)/3-5
    _, mask = cv2.threshold(gray_img, max(0, threhold), 1, cv2.THRESH_BINARY)
    nn_mask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), np.uint8)
    new_mask = (1-mask).astype(np.uint8)
    #  cv::floodFill(Temp, Point(0, 0), Scalar(255));
    # _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, [(0, 0),(0,new_mask.shape[0])], (0), cv2.FLOODFILL_MASK_ONLY)
    _, new_mask, _, _ = cv2.floodFill(new_mask, nn_mask, (0, 0), (0), cv2.FLOODFILL_MASK_ONLY)
    _, new_mask, _, _ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1, new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,  20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask

def _get_center_by_edge(mask):
    center = [0, 0]
    x = mask.sum(axis=1)
    center[0] = np.where(x > x.max()*0.95)[0].mean()
    x = mask.sum(axis=0)
    center[1] = np.where(x > x.max()*0.95)[0].mean()
    return center
    
def _get_radius_by_mask_center(mask, center):
    mask = mask.astype(np.uint8)
    ksize = max(mask.shape[1]//400*2+1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    # radius=
    #cv2.imshow('1',mask)
    #cv2.waitKey(0)
    index = np.where(mask > 0)
    d_int = np.sqrt((index[0]-center[0])**2+(index[1]-center[1])**2)
    b_count = np.bincount(np.ceil(d_int).astype(np.int_))
    radius = np.where(b_count > b_count.max()*0.995)[0].max()
    return radius

def _get_circle_by_center_bbox(shape, center, bbox, radius):
    center_mask = np.zeros(shape=shape).astype('uint8')
    tmp_mask = np.zeros(shape=bbox[2:4])
    center_tmp = (int(center[0]), int(center[1]))
    center_mask = cv2.circle(center_mask, center_tmp[::-1], int(radius), (1), -1)
    # center_mask[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3]]=tmp_mask
    # center_mask[bbox[0]:min(bbox[0]+bbox[2],center_mask.shape[0]),bbox[1]:min(bbox[1]+bbox[3],center_mask.shape[1])]=tmp_mask
    return center_mask

def get_mask(img):
    if img.ndim == 3:
        #raise 'image dim is not 3'
        g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        g_img = img.copy()
    else:
        raise 'image dim is not 1 or 3'
    h, w = g_img.shape
    shape = g_img.shape[0:2]
    g_img = cv2.resize(g_img, (0, 0), fx=0.5, fy=0.5)
    tg_img = cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask = get_mask_BZ(tg_img)
    center = _get_center_by_edge(tmp_mask)
    #bbox=_get_bbox_by_mask(tmp_mask)
    radius = _get_radius_by_mask_center(tmp_mask, center)
    #resize back
    center = [center[0]*2, center[1]*2]
    radius = int(radius*2)
    s_h = max(0, int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h-s_h, 2 * radius), min(w-s_w, 2 * radius))
    tmp_mask = _get_circle_by_center_bbox(shape, center, bbox, radius)
    return tmp_mask, bbox, center, radius

def remove_back_area(img, bbox=None, border=None):
    image = img
    if border is None:
        border = np.array((bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3], img.shape[0], img.shape[1]), dtype=np.int_)
    image = image[border[0]:border[1], border[2]:border[3], ...]
    return image, border

def supplemental_black_area(img, border=None):
    image = img
    if border is None:
        h, v = img.shape[0:2]
        max_l = max(h, v)
        if image.ndim > 2:
            image = np.zeros(shape=[max_l, max_l, img.shape[2]], dtype=img.dtype)
        else:
            image = np.zeros(shape=[max_l, max_l], dtype=img.dtype)
        border = (int(max_l/2-h/2), int(max_l/2-h/2)+h, int(max_l/2-v/2), int(max_l/2-v/2)+v, max_l)
    else:
        max_l = border[4]
        if image.ndim > 2:
            image = np.zeros(shape=[max_l, max_l, img.shape[2]], dtype=img.dtype)
        else:
            image = np.zeros(shape=[max_l, max_l], dtype=img.dtype)
    image[border[0]:border[1], border[2]:border[3], ...] = img
    return image, border

def mask_image(img, mask):
    img[mask <= 0, ...] = 0
    return img

def process_without_gb(img):
    # preprocess images
    #   img : origin image
    #   tar_height: height of tar image
    # return:
    #   result_img: preprocessed image
    #   borders: remove border, supplement mask
    #   mask: mask for preprocessed image
    borders = []
    mask, bbox, center, radius = get_mask(img)
    r_img = mask_image(img, mask)
    r_img, r_border = remove_back_area(r_img, bbox=bbox)
    mask, _ = remove_back_area(mask, border=r_border)
    borders.append(r_border)
    r_img, sup_border = supplemental_black_area(r_img)
    mask, _ = supplemental_black_area(mask, border=sup_border)
    borders.append(sup_border)
    return r_img, borders, (mask*255).astype(np.uint8)

def convert(img_path, img_save_path, crop_size, mask_save_path=None):
    tra=transforms.Compose([
                transforms.Resize([crop_size, crop_size])
                    ])
    try:
        img=imread(img_path)
        rimg,bd,rmask=process_without_gb(img)
        #rimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2BGR)    
        
        rimg=tra(Image.fromarray(rimg.astype(np.uint8)))
        
        #rimg.show()
        if img_save_path != None:
            rimg.save(img_save_path)
        if mask_save_path != None:
            rmask=tra(Image.fromarray(rmask.astype(np.uint8)))
    except:
        print("Error: ", img_path)
      

###########################################################################################


def main():
    args = parser.parse_args()
    image_folder = Path(args.image_folder)
    output_folder = Path(args.output_folder)

    jobs = []
    for root, _, imgs in os.walk(args.image_folder):
        root = Path(root)
        subfolders = root.relative_to(image_folder)
        output_root = output_folder.joinpath(subfolders)
        output_root.mkdir(parents=True, exist_ok=True)

        for img in tqdm(imgs):
            src_path = root.joinpath(img)
            tgt_path = output_root.joinpath(img)
            jobs.append((src_path, tgt_path, args.crop_size))
    random.shuffle(jobs)

    procs = []
    job_size = len(jobs) // args.num_processes
    for i in range(args.num_processes):
        if i < args.num_processes - 1:
            procs.append(Process(target=convert_list, args=(i, jobs[i * job_size:(i + 1) * job_size])))
        else:
            procs.append(Process(target=convert_list, args=(i, jobs[i * job_size:])))

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def convert_list(i, jobs):
    for j, job in enumerate(jobs):
        if j % 100 == 0:
            print('worker{} has finished {}.'.format(i, j))
        convert(*job)


def convert_ssit(fname, tgt_path, crop_size):
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    cropped = cropped.resize([crop_size, crop_size], Image.Resampling.LANCZOS)
    save(cropped, tgt_path)


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def save(img, fname):
    img.save(fname, quality=100, subsampling=0)


if __name__ == "__main__":
    main()