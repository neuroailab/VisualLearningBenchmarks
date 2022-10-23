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
from local_paths import IMAGENET_FOLDER, BASE_FOLDER, STIM_PATH

EVAL_PKL_CACHE = {}
def load_eval_pkl(path):
    global EVAL_PKL_CACHE
    if path not in EVAL_PKL_CACHE:
        ret_values = pickle.load(open(path, 'rb'))
        ret_values['small_imgs'] = torch.stack(
            ret_values['small_imgs'])
        ret_values['big_imgs'] = torch.stack(
            ret_values['big_imgs'])
        ret_values['test_imgs'] = torch.stack(
            ret_values['test_imgs'])
        EVAL_PKL_CACHE[path] = ret_values
    return EVAL_PKL_CACHE[path]


def get_color_norm():
    norm_cfg = dict(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    return norm_cfg


""" MAE specific transforms """
#import torch.nn.functional as F
import torchvision.transforms.functional as F
class MAETransform(torch.nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.size = size        
        self.i, self.j, self.h, self.w = 0, 0, 0, 0
    
    def forward(self, img):        
        return F.resized_crop(
            img, self.i, self.j, self.h, self.w,
            self.size, self.interpolation)


# use 1 image to get cropping params
def compose_mae_transform(size=224):
    norm_cfg = get_color_norm()
    transform = transforms.Compose([
        MAETransform(size),
        transforms.ToTensor(),
        transforms.Normalize(**norm_cfg)
    ])
    return transform
    
    
def set_crop_params(mae_transform, size=224):
    img = torch.ones([3, size, size])
    torch_rrc = transforms.RandomResizedCrop(size, scale=(0.4, 1.0))
    i, j, h, w = torch_rrc.get_params(img, torch_rrc.scale, torch_rrc.ratio)    
    mae_transform.i = i
    mae_transform.j = j
    mae_transform.h = h
    mae_transform.w = w
    
    mae_transform.size = torch_rrc.size
    mae_transform.interpolation = torch_rrc.interpolation
    
    return mae_transform


def compose_transforms(is_train, size=224):
    norm_cfg = get_color_norm()
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.4, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg),
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg),
        ])
    return transform


def get_transforms_from_cfg(cfg):
    from openselfsup.utils import build_from_cfg
    from openselfsup.datasets.registry import PIPELINES

    def _get_transform_from_pipeline(pipeline):
        return transforms.Compose(
                [build_from_cfg(p, PIPELINES) for p in pipeline])
    # load default pipeline w/o any rdpd
    pipeline = cfg.train_pipeline
    transform = _get_transform_from_pipeline(pipeline)
    return transform


def is_color_jitter(pipeline):
    if pipeline['type'] != 'RandomAppliedTrans':
        return False
    if 'transforms' not in pipeline:
        return False
    if pipeline['transforms'][0]['type'] != 'ColorJitter':
        return False
    return True

def get_transforms_from_cfg_ncj(cfg):
    from openselfsup.utils import build_from_cfg
    from openselfsup.datasets.registry import PIPELINES

    def _get_transform_from_pipeline(pipeline):
        return transforms.Compose(
                [build_from_cfg(p, PIPELINES) for p in pipeline\
                 if not is_color_jitter(p)])
    # load default pipeline w/o any rdpd
    pipeline = cfg.train_pipeline
    transform = _get_transform_from_pipeline(pipeline)
    return transform

def get_transforms_from_cfg_ctrlCJ(cfg):
    from openselfsup.utils import build_from_cfg
    from openselfsup.datasets.registry import PIPELINES

    def _get_transform_from_pipeline(pipeline):
        return transforms.Compose(
                [build_from_cfg(p, PIPELINES) for p in pipeline])
    # load default pipeline w/o any rdpd
    pipeline = cfg.train_pipeline
    cj_idx = None
    for idx, p in enumerate(pipeline):
        if is_color_jitter(p):
            cj_idx = idx
            break
    assert cj_idx != None, "No color jittering found!"
    prev_unctrl = _get_transform_from_pipeline(pipeline[:cj_idx])
    ctrl = _get_transform_from_pipeline(pipeline[cj_idx:cj_idx+1])
    post_unctrl = _get_transform_from_pipeline(pipeline[cj_idx+1:])
    transform = dict(
            prev_unctrl=prev_unctrl,
            ctrl=ctrl,
            post_unctrl=post_unctrl,
            )
    return transform


def get_actual_transforms_from_cfg(cfg):
    from openselfsup.utils import build_from_cfg
    from openselfsup.datasets.registry import PIPELINES

    def _get_transform_from_pipeline(pipeline):
        return transforms.Compose(
                [build_from_cfg(p, PIPELINES) for p in pipeline])
    if 'pipeline1' not in cfg.data['train']:
        transform = _get_transform_from_pipeline(cfg.data['train'].pipeline)
        return transform
    else:
        transform1 = _get_transform_from_pipeline(cfg.data['train'].pipeline1)
        transform2 = _get_transform_from_pipeline(cfg.data['train'].pipeline2)
        return transform1, transform2


def replace_base_folder(metas):
    OLD_BASE_FOLDER = 'IMAGE_META_FOLDER'
    def _do_replace(old_str):
        if OLD_BASE_FOLDER in old_str:
            return old_str.replace(OLD_BASE_FOLDER, BASE_FOLDER)

    swapmb = metas['swapmb']
    for idx0 in range(len(swapmb)):
        for idx1 in range(4):
            for idx2 in range(2):
                old_path = swapmb[idx0][idx1][idx2]
                new_path = _do_replace(old_path)
                swapmb[idx0][idx1][idx2] = new_path
    metas['swapmb'] = swapmb
    
    all_test_stim_paths = []
    for idx, old_path in enumerate(metas['all_test_stim_paths']):
        new_path = _do_replace(old_path)
        all_test_stim_paths.append(new_path)
        
    metas['all_test_stim_paths'] = all_test_stim_paths
    
    all_stim_paths = []
    for idx, old_path in enumerate(metas['all_stim_paths']):
        new_path = _do_replace(old_path)
        all_stim_paths.append(new_path)
    metas['all_stim_paths'] = all_stim_paths
    return metas


class ExposureBuilder(Dataset):
    '''
    load stim_path (.pkl) wihch contains image_path, sizes and face name
    - swapmb (28, 4, 2): 28 sets of exposure experiments with 4 pairs of two
    different faces (not all of them are used). which_pair=0 is always valid
    - swapmb_objs (16,) {img_name: face_obj}: all exposure images names and
    their corresponding face identities (face1 to face8)
    - which_pair (int 0-27): index of current pair (4, 2) and return for
    current experiment (used to iterate all pairs and get valid ones)
    - valid_objs (6,): valid faces are face1 to face6
    - all_test_stim_paths (6,): list of test image paths (faces on clean bgd)
    - all_stim_paths (540,): all image paths (faces/objects on background)
    - all_objs (540,): all face id corresponds to images at each index
    - all_s (540,): all sizes corresponds to all_stim_paths
    - face0, face1: which faces are used in current pairs
    '''
    def __init__(self, which_pair,
                 num_steps,
                 batch_size,   # assert batch_size % 4=0 in experiment setup
                 im_size=224,
                 mae_transform=False,
                 which_stimuli='face',    # default face, option objectome
                 transform=None,
                 eval_transform=None):
        self.batch_size = batch_size
        self.repeat = max(batch_size // 4, 1)    # (2, 4) per pair
        self.num_steps = num_steps
        self.im_size = im_size        
        self.which_pair = which_pair
        self.transform = transform
        self.eval_transform = eval_transform
        self.which_stimuli = which_stimuli
        self.mae_transform = mae_transform
        
        if which_stimuli == 'face':
            metas = pickle.load(open(STIM_PATH, 'rb'))
        elif which_stimuli == 'objectome':
            metas = pickle.load(open(OBJECTOME_PATH, 'rb'))
        else:
            raise Exception(f'which_stimuli should be face or objectome ONLY!')

        metas = replace_base_folder(metas)
        
        self.swapmb = metas['swapmb']
        self.num_pairs = len(self.swapmb)
        self.swapmb_objs = metas['swapmb_objs']
        self.valid_objs = metas['all_test_objs']
        self.all_test_stim_paths = np.asarray(metas['all_test_stim_paths'])

        self.all_test_objs = np.asarray(metas['all_test_objs'])
        self.all_stim_paths = np.asarray(metas['all_stim_paths'])
        self.all_objs = np.asarray(metas['all_objs'])
        self.all_s = np.asarray(metas['all_s'])
        self.__get_faces()
        self.__create_face_swapmb()
        # get file names for pair used in current exp
        self.__expand_batch()

    # files that are in all_test_stim_paths are medium faces, and other files in swapmb_objs are clean big faces
    def __create_face_swapmb(self):
        def get_size(face_path):
            if face_path in self.all_test_stim_paths:
                return 'm'
            else:
                return 'b'
        face_size_swapmb = []
        for exp_num, exp_pairs in enumerate(self.swapmb):
            exp_face_size = []
            for n, pair in enumerate(exp_pairs):
                face1 = os.path.basename(pair[0])
                face1 = face1[ : face1.rfind('.')]
                if not isinstance(self.swapmb_objs[face1], str):
                    face1 = self.swapmb_objs[face1].decode("utf-8")
                else:
                    face1 = self.swapmb_objs[face1]                
                face1_size = get_size(pair[0])
                
                face2 = os.path.basename(pair[1])
                face2 = face2[ : face2.rfind('.')]
                if not isinstance(self.swapmb_objs[face2], str):
                    face2 = self.swapmb_objs[face2].decode("utf-8")
                else:
                    face2 = self.swapmb_objs[face2]
                    
                face2_size = get_size(pair[1])
                exp_face_size.append(
                    [f'{face1}_{face1_size}', f'{face2}_{face2_size}'])
            face_size_swapmb.append(exp_face_size)
        self.face_size_swapmb = np.array(face_size_swapmb)
        
        
    def __get_faces(self):
        base_name0 = os.path.basename(self.swapmb[self.which_pair][0][0])[:-4]
        base_name1 = os.path.basename(self.swapmb[self.which_pair][0][1])[:-4]
        self.face0 = self.swapmb_objs[base_name0]
        self.face1 = self.swapmb_objs[base_name1]

    # fill up current pair to a full batch
    def __expand_batch(self):
        img_pairs = self.swapmb[self.which_pair]
        face_sizes = self.face_size_swapmb[self.which_pair]
        paths1 = []
        paths2 = []
        face1_sizes = []
        face2_sizes = []        
        for i in range(len(img_pairs)):
            path1, path2 = img_pairs[i]
            face1_size, face2_size = face_sizes[i]
            paths1.append(path1)
            paths2.append(path2)
            face1_sizes.append(face1_size)
            face2_sizes.append(face2_size)
            
        imgs1 = load_image_from_list(paths1)
        imgs2 = load_image_from_list(paths2)
        self.imgs1 = np.tile(imgs1, self.repeat)
        self.imgs2 = np.tile(imgs2, self.repeat)
        self.face1_sizes = np.tile(face1_sizes, self.repeat)
        self.face2_sizes = np.tile(face2_sizes, self.repeat)
        
    # number of image pair for current batch
    def __len__(self):
        return len(self.imgs1) * self.num_steps

    def apply_transforms(self, img1, img2):
        if self.mae_transform:
            set_crop_params(self.transform.__dict__['transforms'][0])
            
        if self.transform is not None:
            if isinstance(self.transform, list) or isinstance(self.transform, tuple):
                img1 = self.transform[0](img1)
                img2 = self.transform[1](img2)
            elif isinstance(self.transform, dict):
                img1 = self.transform['prev_unctrl'](img1)
                img2 = self.transform['prev_unctrl'](img2)

                state = torch.get_rng_state()
                img1 = self.transform['ctrl'](img1)
                torch.set_rng_state(state)
                img2 = self.transform['ctrl'](img2)

                img1 = self.transform['post_unctrl'](img1)
                img2 = self.transform['post_unctrl'](img2)
            else:                                    
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        return img1, img2
        
    # return current pair as (img1, img2) used in training only
    # (B,2,C,H,W)
    def __getitem__(self, idx):        
        idx = idx % len(self.imgs1)
        img1 = self.imgs1[idx]
        img2 = self.imgs2[idx]
        img1, img2 = self.apply_transforms(img1, img2)
        return torch.stack([img1, img2])

    def is_legal(self):
        if self.face0 not in self.valid_objs:
            return False
        if self.face1 not in self.valid_objs:
            return False
        return True

    'Evaluation loader'
    def _load_transform_imgs_from_paths(self, img_paths):
        imgs = load_image_from_list(img_paths, self.im_size)
        eval_imgs = []
        for img in imgs:
            img = self.eval_transform(img) if \
                  self.eval_transform else img
            eval_imgs.append(img)
        return eval_imgs

    def premake_eval_images(self):
        print(f'saving testing images for pair {self.which_pair}')
        idx_for_small_faces \
            = np.logical_or(
                np.logical_and(self.all_objs==self.face0, self.all_s=='small'),
                np.logical_and(self.all_objs==self.face1, self.all_s=='small'))
        idx_for_big_faces \
            = np.logical_or(
                np.logical_and(self.all_objs==self.face0, self.all_s=='big'),
                np.logical_and(self.all_objs==self.face1, self.all_s=='big'))
        
        small_img_paths = self.all_stim_paths[idx_for_small_faces]
        small_faces = self.all_objs[idx_for_small_faces]
        small_imgs = self._load_transform_imgs_from_paths(small_img_paths)
        
        big_img_paths = self.all_stim_paths[idx_for_big_faces]
        big_faces = self.all_objs[idx_for_big_faces]
        big_imgs = self._load_transform_imgs_from_paths(big_img_paths)
        
        test_img_paths = [self.all_test_stim_paths[self.all_test_objs==self.face0][0],
                          self.all_test_stim_paths[self.all_test_objs==self.face1][0]]
        test_faces = [self.face0, self.face1]
        test_imgs = self._load_transform_imgs_from_paths(test_img_paths)
        
        ret_values = {
            'small_imgs': small_imgs,
            'small_faces': small_faces,
            'big_imgs': big_imgs,
            'big_faces': big_faces,
            'test_imgs': test_imgs,
            'test_faces': test_faces
        } 
        output_folder = os.path.join(
            BASE_FOLDER, 'data', 'premake_test_images',
            f'{self.which_stimuli}_{self.im_size}')
        try:
            os.makedirs(output_folder)
        except:
            pass
        output_path = os.path.join(
            output_folder, f'{self.which_pair}.pkl')        
        pickle.dump(ret_values, open(output_path, 'wb'))
        print(f'{output_path} saved')

    def get_eval_image_folder(self):
        test_image_path = os.path.join(
            BASE_FOLDER, 'data', 'premake_test_images',
            f'{self.which_stimuli}_{self.im_size}')
        return test_image_path

    # eval images sorted by big&small (test_imgs are the anchors)
    # imgs are stacked into one tensor to be fed into the model
    def get_eval_images(self):        
        test_image_path = os.path.join(
            self.get_eval_image_folder(),
            f'{self.which_pair}.pkl')
        ret_values = load_eval_pkl(test_image_path)
        return ret_values

    
class NoSwapExposureBuilder(ExposureBuilder):
    def __init__(self, which_pair,
                 num_steps,
                 batch_size,
                 im_size=224,
                 mae_transform=False,
                 which_stimuli='face',
                 transform=None,
                 eval_transform=None):
        super(NoSwapExposureBuilder, self).__init__(which_pair,
                                                    num_steps,
                                                    batch_size,
                                                    im_size=im_size,
                                                    mae_transform=mae_transform,
                                                    which_stimuli=which_stimuli,
                                                    transform=transform,
                                                    eval_transform=eval_transform)        
        self.get_noswap_imgs()
        self.noswap_expand_batch()
                
    def get_noswap_imgs(self):
        swaped_images = np.concatenate(self.swapmb[self.which_pair])
        no_swapped_images = []
        face0_images = []
        face1_images = []
        # pair paths are for the same faces but different imgs
        self.img1_paths = []
        self.img2_paths = []
        for _image in swaped_images:
            base_name = os.path.basename(_image)[:-4]
            which_face = self.swapmb_objs[base_name]
            if which_face == self.face0 and _image not in face0_images:
                face0_images.append(_image)
            if which_face == self.face1 and _image not in face1_images:
                face1_images.append(_image)

        self.img1_paths.extend(face0_images)
        self.img2_paths.extend(reversed(face0_images))
        self.img1_paths.extend(face1_images)
        self.img2_paths.extend(reversed(face1_images))
        
    def noswap_expand_batch(self):
        imgs1 = load_image_from_list(self.img1_paths, self.im_size)
        imgs2 = load_image_from_list(self.img2_paths, self.im_size)
        self.noswap_imgs1 = np.tile(imgs1, self.repeat)
        self.noswap_imgs2 = np.tile(imgs2, self.repeat)
        
    def __len__(self):
        return len(self.noswap_imgs1) * self.num_steps

    def __getitem__(self, idx):        
        idx = idx % len(self.noswap_imgs1)
        img1 = self.noswap_imgs1[idx]
        img2 = self.noswap_imgs2[idx]
        img1, img2 = self.apply_transforms(img1, img2)
        return torch.stack([img1, img2])


class NoswapSwapExposureBuilder(NoSwapExposureBuilder):
    def __init__(self, which_pair,                 
                 num_steps,
                 batch_size,
                 im_size=224,
                 mae_transform=False,
                 which_stimuli='face',
                 transform=None,
                 eval_transform=None):
        self.batch_size = batch_size
        super(NoswapSwapExposureBuilder, self).__init__(
            which_pair, num_steps, batch_size,
            im_size=im_size,
            mae_transform=mae_transform,
            which_stimuli=which_stimuli,
            transform=transform, eval_transform=eval_transform)

    # go from noswap to swapped
    def switch_training(self):
        self.noswap_imgs1 = self.imgs1
        self.noswap_imgs2 = self.imgs2

    def __getitem__(self, idx):
        idx = idx % len(self.noswap_imgs1)
        img1 = self.noswap_imgs1[idx]
        img2 = self.noswap_imgs2[idx]
        img1, img2 = self.apply_transforms(img1, img2)
        return torch.stack([img1, img2])
    

def premake_test_imgs(which_stimuli, im_size=224):
    num_pairs = 28
    eval_transform = compose_transforms(
        is_train=False, size=im_size)
    for which_pair in range(num_pairs):        
        exposure_builder = ExposureBuilder(
            which_pair, 1, 4, im_size, which_stimuli,
            eval_transform=eval_transform)
        if exposure_builder.is_legal():
            exposure_builder.premake_eval_images()
        
    
# co-training dataset (imgnt or imgnt&VGG)    
def build_imgnt_dataset(cfg=None):
    if cfg is None:
        config = local_paths.CONFIG_MOCO_V2
        cfg = Config.fromfile(config)
    if ('root' in cfg.data.train.data_source):
        if (not os.path.exists(cfg.data.train.data_source['root'])):
            cfg.data.train.data_source['root'] = IMAGENET_FOLDER
    if 'memcached' in cfg.data.train.data_source:
        cfg.data.train.data_source['memcached'] = False
    
    return build_dataset(cfg.data.train)
    
    
if __name__ == '__main__':
    pass
