from __future__ import print_function, division
import os, sys
import pdb
import time
import pickle
import numpy as np
from PIL import Image

import torch
torch.manual_seed(0)
from torchvision import transforms


IMAGE_SIZE = 224


def add_base_folder(base_folder, img_path):
    return os.path.join(base_folder, img_path)    


def add_base_folder_list(base_folder, img_list):
    img_paths = [os.path.join(base_folder, img) for img in img_list]
    return img_paths


def load_resize_img(img_path, im_size=224):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    return img


def load_image_from_list(paths, im_size=224):
    imgs = [load_resize_img(path, im_size=im_size) for path in paths]
    return imgs


'''
# (TRAIN ONLY) convert np.array images to PIL.Image for builtin transforms
def img_array2PIL(img):
    img = Image.fromarray(img)
    return img
'''
# check input image_tensors have correct shape 
def check_input(images):
    assert images.size()[-1] == images.size()[-2] \
        == IMAGE_SIZE, 'Image size is not (224, 224)!'
    
def check_dataloader_efficiency(dataset):
    def _run_once(steps, params):
        for num_workers in range(5, 21, 1):    # less than 5 is too slow
            params['num_workers'] = num_workers
            dataloader = torch.utils.data.DataLoader(
                dataset, **params)
            dataloader = iter(dataloader)
            begin = time.time()
            for _step in range(steps):
                images = next(dataloader)
                images = images.cuda()
            end = time.time()
            total_time = end - begin
            if num_workers in worker_efficiency:
                worker_efficiency[num_workers].append(total_time)
            else:
                worker_efficiency[num_workers] = [total_time]

    steps = 600
    N = 1    # run n times and take average runtime
    params = {'batch_size': 128,
              'shuffle': True,
              'num_workers': 0}

    worker_efficiency = {}
    for n in range(N):
        _run_once(steps, params)
    print('batch_size {}'.format(params['batch_size']))
    for num_workers in worker_efficiency:
        t = np.mean(worker_efficiency[num_workers])
        print(f'{steps} steps - num_workers={num_workers}: {t} seconds')


        
import dataset        
if __name__ == '__main__':
    ImgntBuilder = dataset.ImgNtExposureBuilder(train=True)
    check_dataloader_efficiency(ImgntBuilder)
    
