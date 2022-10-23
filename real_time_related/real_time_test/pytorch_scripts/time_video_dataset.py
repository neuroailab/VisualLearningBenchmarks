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
import time_dataset
from time_dataset import color_denorm
DEBUG = os.environ.get('DEBUG', '0')=='1'


class TimeVideoExpBuilder(dataset.ExposureBuilder):
    def __init__(
            self, window_size, aggre_time,
            exposure_num_steps=None,
            return_phases=False,
            min_aggre_time=None,
            include_test=False,
            test_more_trials=False,
            *args, **kwargs):
        self.window_size = window_size # unit is minute
        self.aggre_time = aggre_time # unit is second
        self.min_aggre_time = min_aggre_time # unit is second
        self.include_test = include_test
        self.test_more_trials = test_more_trials
        if self.min_aggre_time is not None:
            assert self.min_aggre_time < self.aggre_time
        self.pretrain_fns = None
        super().__init__(*args, **kwargs)
        if not self.is_legal():
            return
        self.topil_transform = transforms.ToPILImage()
        np.random.seed(self.which_pair)
        self.exposure_num_steps = exposure_num_steps or self.num_steps
        self.prepare_test_images()
        self.build_time_sequence()
        self.build_batch_time()
        self.build_example_time()
        self.gray_background = self.topil_transform(torch.ones(
                3, 224, 224, dtype=torch.float64) * 127)
        self.return_phases = return_phases

    def process_loaded_images(self, loaded_images):
        return [self.topil_transform(_img)
                for _img in color_denorm(loaded_images)]

    def get_other_test_images(self):
        eval_folder = self.get_eval_image_folder()
        all_pkl_paths = os.listdir(eval_folder)
        other_sm_imgs = []
        other_big_imgs = []
        other_test_imgs = []
        all_valid_other_pairs = []
        for _path in all_pkl_paths:
            if _path == f'{self.which_pair}.pkl':
                continue
            curr_eval_info = dataset.load_eval_pkl(
                    os.path.join(eval_folder, _path))
            if self.face0 in curr_eval_info['test_faces']:
                continue
            if self.face1 in curr_eval_info['test_faces']:
                continue
            all_valid_other_pairs.append(_path)
            other_sm_imgs.append(curr_eval_info['small_imgs'])
            other_big_imgs.append(curr_eval_info['big_imgs'])
            other_test_imgs.append(curr_eval_info['test_imgs'])

        chosen_other_pair = np.random.randint(low=0, high=len(all_valid_other_pairs))
        self.other_sm_imgs = self.process_loaded_images(
                other_sm_imgs[chosen_other_pair][::2])
        self.other_big_imgs = self.process_loaded_images(
                other_big_imgs[chosen_other_pair][::2])
        self.other_test_imgs = self.process_loaded_images(
                other_test_imgs[chosen_other_pair])
        self.chosen_other_pair_path = all_valid_other_pairs[chosen_other_pair]

    def get_curr_test_images(self):
        eval_info = self.get_eval_images()
        self.curr_sm_imgs = self.process_loaded_images(
                eval_info['small_imgs'][::2])
        self.curr_big_imgs = self.process_loaded_images(
                eval_info['big_imgs'][::2])
        self.curr_test_imgs = self.process_loaded_images(
                eval_info['test_imgs'])

    def prepare_test_images(self, im_size=224):
        self.im_size = im_size
        self.get_curr_test_images()
        self.get_other_test_images()

    def build_one_test_sub_seq(self, state, length):
        return [(state, _idx) for _idx in range(length)]

    def build_test_time_seq(self):
        all_test_imgs =\
                self.build_one_test_sub_seq('b', len(self.curr_big_imgs))\
                + self.build_one_test_sub_seq('s', len(self.curr_sm_imgs))\
                + self.build_one_test_sub_seq('ob', len(self.other_big_imgs))\
                + self.build_one_test_sub_seq('os', len(self.other_sm_imgs))
        all_test_imgs = np.random.permutation(all_test_imgs)
        curr_time_sequence = []
        # 75 = 1 + 6 * 10 + 14, 75 * 80 = 6000 (10 mins)
        num_saccades = 6
        saccade_interval = 10
        trial_interval = 14
        for _test_img in all_test_imgs:
            curr_time_sequence.append(_test_img)
            which_test = 't' if _test_img[0] in ['b', 's'] else 'ot'
            which_test_img = np.random.randint(low=0, high=2)
            for _ in range(num_saccades):
                curr_time_sequence.extend(
                        [(which_test, which_test_img)] * saccade_interval)
                which_test_img = 1 - which_test_img
            curr_time_sequence.extend([('g',)] * trial_interval)
        return curr_time_sequence

    def build_test_time_seq_more_trials(self):
        all_test_imgs =\
                self.build_one_test_sub_seq('b', len(self.curr_big_imgs))\
                + self.build_one_test_sub_seq('s', len(self.curr_sm_imgs))\
                + self.build_one_test_sub_seq('ob', len(self.other_big_imgs))\
                + self.build_one_test_sub_seq('os', len(self.other_sm_imgs))
        all_test_imgs = np.asarray(all_test_imgs)
        all_test_imgs = all_test_imgs[np.random.choice(len(all_test_imgs), 200)]
        curr_time_sequence = []
        # 30 = 1 + 4 * 6 + 5, 30 * 200 = 6000 (10 mins)
        num_saccades = 4
        saccade_interval = 6
        trial_interval = 5
        for _test_img in all_test_imgs:
            curr_time_sequence.append(_test_img)
            which_test = 't' if _test_img[0] in ['b', 's'] else 'ot'
            which_test_img = np.random.randint(low=0, high=2)
            for _ in range(num_saccades):
                curr_time_sequence.extend(
                        [(which_test, which_test_img)] * saccade_interval)
                which_test_img = 1 - which_test_img
            curr_time_sequence.extend([('g',)] * trial_interval)
        return curr_time_sequence

    def build_exposure_time_seq(self):
        num_exp_events = 400
        len_one_exp_event = 15
        curr_time_sequence = []
        for _ in range(num_exp_events):
            which_exp_img = np.random.randint(low=0, high=len(self.imgs1))
            curr_time_sequence.append(('e1', which_exp_img))
            curr_time_sequence.append(('e2', which_exp_img))
            curr_time_sequence.extend(
                    [('g',)] * (len_one_exp_event-2))
        return curr_time_sequence

    def build_time_sequence(self):
        self.phases = []
        for idx in range(9):
            curr_phase = 'test' if idx % 2 ==0 else 'exposure'
            self.phases.append((curr_phase, 10))

        all_time_sequence = []
        for _phase, _ in self.phases:
            if _phase == 'test':
                if not self.test_more_trials:
                    all_time_sequence.extend(self.build_test_time_seq())
                else:
                    all_time_sequence.extend(
                            self.build_test_time_seq_more_trials())
            else:
                all_time_sequence.extend(self.build_exposure_time_seq())
        self.all_time_sequence = all_time_sequence
        self.num_exposure_phases = 4 if not self.include_test else 9
        self.num_steps_per_phase = self.exposure_num_steps // self.num_exposure_phases
        self.time_per_step = 10 / self.num_steps_per_phase

    def build_batch_time(self):
        curr_time = 0
        batch_time = []
        for idx, (_phase, _time) in enumerate(self.phases):
            sta_time = curr_time
            curr_time += _time
            end_time = curr_time
            if (_phase == 'test') and (not self.include_test):
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

    def build_example_time(self):
        all_rel_pos = np.random.uniform(
                size=self.batch_size * self.num_steps)
        example_time = np.tile(
                self.batch_time[:, np.newaxis], 
                [1, self.batch_size]).reshape(-1)
        self.example_time = example_time - all_rel_pos * self.window_size
        self.per_step_choices = np.random.uniform(
                size=self.batch_size * self.num_steps)

    def load_pretrain_image(self, curr_choice):
        img_path = self.pretrain_fns[int(curr_choice * len(self.pretrain_fns))]
        img = Image.open(img_path)
        img = img.convert('RGB')
        return img, img

    def is_pretrain_phase(self, idx):
        curr_time = self.example_time[idx] * 60
        if (curr_time < 0) or (curr_time > 5400):
            return True
        return False

    def get_e1_img(self, idx):
        return self.imgs1[idx]

    def get_e2_img(self, idx):
        return self.imgs2[idx]

    def get_exp_imgs(self, curr_time, phase, idx):
        if phase == 'e1':
            return self.get_e1_img(idx)
        elif phase == 'e2':
            return self.get_e2_img(idx)

    def get_img_for_t(self, curr_time):
        curr_time = int(curr_time * 10)
        curr_time = max(0, curr_time)
        curr_time = min(len(self.all_time_sequence), curr_time)
        info = self.all_time_sequence[curr_time]
        phase = info[0]
        if DEBUG:
            print(phase)
        if phase == 'g':
            return self.gray_background
        idx = int(info[1])
        if phase == 'b':
            return self.curr_big_imgs[idx]
        elif phase == 's':
            return self.curr_sm_imgs[idx]
        elif phase == 'ob':
            return self.other_big_imgs[idx]
        elif phase == 'os':
            return self.other_sm_imgs[idx]
        elif phase == 't':
            return self.curr_test_imgs[idx]
        elif phase == 'ot':
            return self.other_test_imgs[idx]
        elif phase in ['e1', 'e2']:
            return self.get_exp_imgs(curr_time, phase, idx)
        else:
            print(phase)
            raise NotImplementedError

    def __getitem__(self, idx):
        curr_choice = self.per_step_choices[idx]
        if self.is_pretrain_phase(idx):
            img1, img2 = self.load_pretrain_image(curr_choice)
        else:
            curr_time = self.example_time[idx] * 60
            added_time = curr_choice * 2 * self.aggre_time - self.aggre_time
            if self.min_aggre_time is not None:
                if curr_choice <= 0.5:
                    added_time = -self.min_aggre_time\
                            - 2*curr_choice\
                              *(self.aggre_time - self.min_aggre_time)
                else:
                    added_time = self.min_aggre_time\
                            + 2*(curr_choice-0.5)\
                              *(self.aggre_time - self.min_aggre_time)
            another_time = curr_time + added_time
            img1 = self.get_img_for_t(curr_time)
            img2 = self.get_img_for_t(another_time)
        img1, img2 = self.apply_transforms(img1, img2)
        if not self.return_phases:
            return torch.stack([img1, img2])
        else:
            return torch.stack([img1, img2]), self.get_two_phases(idx)

    def get_phase_for_t(self, curr_time):
        curr_time = int(curr_time * 10)
        curr_time = max(0, curr_time)
        curr_time = min(len(self.all_time_sequence), curr_time)
        return self.all_time_sequence[curr_time][0]

    def get_two_phases(self, idx):
        curr_choice = self.per_step_choices[idx]
        if self.is_pretrain_phase(idx):
            return 'p', 'p'
        else:
            curr_time = self.example_time[idx] * 60
            another_time = curr_time\
                    + curr_choice * 2 * self.aggre_time - self.aggre_time
            phase1 = self.get_phase_for_t(curr_time)
            phase2 = self.get_phase_for_t(another_time)
        return phase1, phase2
    
    def __len__(self):
        return len(self.example_time)

    def set_pretrain_fns(self, fns):
        self.pretrain_fns = fns


class NoSwapTimeVideoExpBuilder(TimeVideoExpBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset.NoSwapExposureBuilder.get_noswap_imgs(self)
        dataset.NoSwapExposureBuilder.noswap_expand_batch(self)

    def get_e1_img(self, idx):
        return self.noswap_imgs1[idx]

    def get_e2_img(self, idx):
        return self.noswap_imgs2[idx]


class NoSwapSwapTimeVideoExpBuilder(NoSwapTimeVideoExpBuilder):
    def switch_training(self):
        pass

    def get_sw_e1_img(self, idx):
        return self.imgs1[idx]

    def get_sw_e2_img(self, idx):
        return self.imgs2[idx]

    def get_exp_imgs(self, curr_time, phase, idx):
        before_switch = curr_time <= 40 * 60 * 10
        if before_switch:
            if phase == 'e1':
                return self.get_e1_img(idx)
            elif phase == 'e2':
                return self.get_e2_img(idx)
        else:
            if phase == 'e1':
                return self.get_sw_e1_img(idx)
            elif phase == 'e2':
                return self.get_sw_e2_img(idx)


def check_time_video_builder():
    which_pair = 7
    num_steps = 600
    window_size = 0.5
    batch_size = 32 
    transform = dataset.compose_transforms(is_train=True)
    eval_transform = dataset.compose_transforms(is_train=False)
    exposure_builder = TimeVideoExpBuilder(
            window_size=window_size,
            aggre_time=0.2,
            which_pair=which_pair,
            num_steps=num_steps,
            batch_size=batch_size,
            which_stimuli='face',
            transform=transform,
            eval_transform=eval_transform)

    def check_freq(start_step):
        all_phases = []
        for idx in range(batch_size * 2):
            p1, p2 = exposure_builder.get_two_phases(idx + start_step * batch_size)
            all_phases.append(','.join(sorted((p1, p2))))
        print(np.unique(all_phases, return_counts=True))
    check_freq(100)
    pdb.set_trace()
    pass


if __name__ == '__main__':
    check_time_video_builder()
