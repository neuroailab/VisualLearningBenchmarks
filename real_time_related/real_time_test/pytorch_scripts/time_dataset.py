from __future__ import print_function, division
import os, sys
import pdb
import pickle
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from mmcv import Config
from openselfsup.datasets import build_dataset

import local_paths
import build_response
from utils import load_image_from_list, load_resize_img
import dataset


def color_denorm(images):
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
    std = std[np.newaxis, :, np.newaxis, np.newaxis]
    images = images * torch.from_numpy(std) + torch.from_numpy(mean)
    images = torch.clip(images, 0, 1)
    images = images * 255
    return images


class TimeExposureBuilder(dataset.ExposureBuilder):
    def __init__(
            self, window_size, 
            exposure_num_steps=None,
            *args, **kwargs):
        self.window_size = window_size # unit is minute
        self.pretrain_fns = None
        super().__init__(*args, **kwargs)
        if not self.is_legal():
            return
        self.exposure_num_steps = exposure_num_steps or self.num_steps
        self.build_time_sequence()
        self.build_batch_time()
        self.build_example_time()
        self.prepare_test_images()

    def build_time_sequence(self):
        self.phases = []
        for idx in range(9):
            curr_phase = 'test' if idx % 2 ==0 else 'exposure'
            self.phases.append((curr_phase, 10))
        self.num_exposure_phases = 4
        self.num_steps_per_phase = self.exposure_num_steps // self.num_exposure_phases
        self.time_per_step = 10 / self.num_steps_per_phase

    def build_batch_time(self):
        curr_time = 0
        batch_time = []
        for idx, (_phase, _time) in enumerate(self.phases):
            sta_time = curr_time
            curr_time += _time
            end_time = curr_time
            if _phase == 'test':
                continue
            batch_time.append(
                    np.arange(
                        sta_time, 
                        end_time, 
                        self.time_per_step))
        batch_time.append(
                np.arange(self.exposure_num_steps, self.num_steps)\
                * self.time_per_step)
        self.batch_time = np.concatenate(batch_time)

    def get_phase_for_t(self, t):
        if t <= 0:
            return 'pretrain'
        now_time = 0
        for _phase, _time in self.phases:
            now_time += _time
            if t <= now_time:
                return _phase
        return 'pretrain'

    def build_example_time(self):
        np.random.seed(self.which_pair)
        all_rel_pos = np.random.uniform(
                size=self.batch_size * self.num_steps)
        example_time = np.tile(self.batch_time[:, np.newaxis], [1, self.batch_size]).reshape(-1)
        self.example_time = example_time - all_rel_pos * self.window_size
        self.per_step_choices = np.random.uniform(size=self.batch_size * self.num_steps)

    def set_pretrain_fns(self, fns):
        self.pretrain_fns = fns

    def get_other_test_images(self):
        eval_folder = self.get_eval_image_folder()
        all_pkl_paths = os.listdir(eval_folder)
        other_sm_imgs = []
        other_big_imgs = []
        other_test_imgs = []
        for _path in all_pkl_paths:
            if _path == f'{self.which_pair}.pkl':
                continue
            curr_eval_info = dataset.load_eval_pkl(
                    os.path.join(eval_folder, _path))
            if self.face0 in curr_eval_info['test_faces']:
                continue
            if self.face1 in curr_eval_info['test_faces']:
                continue
            other_sm_imgs.append(curr_eval_info['small_imgs'])
            other_big_imgs.append(curr_eval_info['big_imgs'])
            other_test_imgs.append(curr_eval_info['test_imgs'])
        self.other_sm_imgs = color_denorm(torch.cat(other_sm_imgs))
        self.other_big_imgs = color_denorm(torch.cat(other_big_imgs))
        self.other_test_imgs = color_denorm(torch.cat(other_test_imgs))

    def get_curr_test_images(self):
        eval_info = self.get_eval_images()
        self.curr_sm_imgs = color_denorm(eval_info['small_imgs'])
        self.curr_big_imgs = color_denorm(eval_info['big_imgs'])
        self.curr_test_imgs = color_denorm(eval_info['test_imgs'])

    def prepare_test_images(self, im_size=224):
        self.im_size = im_size
        self.get_curr_test_images()
        self.get_other_test_images()
        sample_sm_idxs = np.random.choice(
                len(self.other_sm_imgs), len(self.curr_sm_imgs))
        sample_big_idxs = np.random.choice(
                len(self.other_big_imgs), len(self.curr_big_imgs))
        half_len = (len(self.curr_sm_imgs) + len(self.curr_big_imgs)) // 2
        sample_curr_test_idxs = np.random.choice(
                len(self.curr_test_imgs), half_len)
        sample_other_test_idxs = np.random.choice(
                len(self.other_test_imgs), half_len)
        self.all_test_imgs = torch.cat(
                [self.curr_sm_imgs, self.curr_big_imgs, 
                 self.curr_test_imgs[sample_curr_test_idxs],
                 self.other_sm_imgs[sample_sm_idxs], 
                 self.other_big_imgs[sample_big_idxs], 
                 self.other_test_imgs[sample_other_test_idxs],
                 ])

        _transform = transforms.ToPILImage()
        self.all_test_imgs = [
                _transform(_img)
                for _img in self.all_test_imgs]

    def load_pretrain_image(self, curr_choice):
        img_path = self.pretrain_fns[int(curr_choice * len(self.pretrain_fns))]
        img = Image.open(img_path)
        img = img.convert('RGB')
        return img, img

    def load_test_image(self, curr_choice):
        img = self.all_test_imgs[int(curr_choice * len(self.all_test_imgs))]
        return img, img

    def load_exposure_image(self, curr_choice):
        idx = int(len(self.imgs1) * curr_choice)
        img1 = self.imgs1[idx]
        img2 = self.imgs2[idx]
        return img1, img2

    def __getitem__(self, idx):
        phase = self.get_phase_for_t(self.example_time[idx])
        curr_choice = self.per_step_choices[idx]
        if phase == 'pretrain':
            img1, img2 = self.load_pretrain_image(curr_choice)
        elif phase == 'test':
            img1, img2 = self.load_test_image(curr_choice)
        elif phase == 'exposure':
            img1, img2 = self.load_exposure_image(curr_choice)
        else:
            raise NotImplementedError
        img1, img2 = self.apply_transforms(img1, img2)
        return torch.stack([img1, img2])
    
    def __len__(self):
        return len(self.example_time)


class NoSwapTimeExposureBuilder(TimeExposureBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset.NoSwapExposureBuilder.get_noswap_imgs(self)
        dataset.NoSwapExposureBuilder.noswap_expand_batch(self)

    def load_exposure_image(self, curr_choice):
        idx = int(len(self.noswap_imgs1) * curr_choice)
        img1 = self.noswap_imgs1[idx]
        img2 = self.noswap_imgs2[idx]
        return img1, img2

class NoswapSwapTimeExposureBuilder(NoSwapTimeExposureBuilder):
    def switch_training(self):
        pass

    def build_time_sequence(self):
        self.phases = []
        for idx in range(9):
            if idx % 2 == 0:
                curr_phase = 'test'
            elif idx < 5:
                curr_phase = 'noswap_exposure'
            else:
                curr_phase = 'swap_exposure'
            self.phases.append((curr_phase, 10))
        self.num_exposure_phases = 4
        self.num_steps_per_phase = self.exposure_num_steps // self.num_exposure_phases
        self.time_per_step = 10 / self.num_steps_per_phase

    def load_swap_exposure_image(self, curr_choice):
        idx = int(len(self.imgs1) * curr_choice)
        img1 = self.imgs1[idx]
        img2 = self.imgs2[idx]
        return img1, img2

    def load_noswap_exposure_image(self, curr_choice):
        idx = int(len(self.noswap_imgs1) * curr_choice)
        img1 = self.noswap_imgs1[idx]
        img2 = self.noswap_imgs2[idx]
        return img1, img2

    def __getitem__(self, idx):
        phase = self.get_phase_for_t(self.example_time[idx])
        curr_choice = self.per_step_choices[idx]
        if phase == 'pretrain':
            img1, img2 = self.load_pretrain_image(curr_choice)
        elif phase == 'test':
            img1, img2 = self.load_test_image(curr_choice)
        elif phase == 'noswap_exposure':
            img1, img2 = self.load_noswap_exposure_image(curr_choice)
        elif phase == 'swap_exposure':
            img1, img2 = self.load_swap_exposure_image(curr_choice)
        else:
            raise NotImplementedError
        img1, img2 = self.apply_transforms(img1, img2)
        return torch.stack([img1, img2])


def check_time_builder():
    which_pair = 7
    num_steps = 600
    window_size = 10
    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 10,}
    transform = dataset.compose_transforms(is_train=True)
    eval_transform = dataset.compose_transforms(is_train=False)
    exposure_builder = NoswapSwapTimeExposureBuilder(
            window_size=window_size,
            which_pair=which_pair,
            num_steps=num_steps,
            batch_size=params['batch_size'],
            which_stimuli='face',
            transform=transform,
            eval_transform=eval_transform)
    pdb.set_trace()
    pass


if __name__ == '__main__':
    check_time_builder()
