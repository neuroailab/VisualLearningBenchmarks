import os
from PIL import Image
import copy
import numpy as np

from ..registry import DATASOURCES
from .utils import McLoader
from .image_list import ImageList


@DATASOURCES.register_module
class MSTImageList(ImageList):
    def __init__(self, root, which_set='all', oversample_len=None, **kwargs):
        self.memcached = False
        self.mclient_path = None
        self.has_labels = False

        all_sets = [
                'Set {}'.format(_idx) for _idx in range(1, 7)]
        used_sets = []
        if which_set == 'all':
            used_sets = copy.copy(all_sets)
        elif which_set in all_sets:
            used_sets = [which_set]
        else:
            raise NotImplementedError

        fns = []
        for _set in used_sets:
            imgs = os.listdir(os.path.join(root, _set))
            imgs.sort()
            fns.extend(
                    [os.path.join(root, _set, _img) 
                     for _img in imgs])
        self.raw_fns = fns
        self.oversample_len = oversample_len
        self.set_epoch(0)

    def set_epoch(self, epoch):
        if self.oversample_len is not None:
            np.random.seed(epoch)
            self.fns = np.random.choice(
                    self.raw_fns, self.oversample_len, replace=True)
        else:
            self.fns = self.raw_fns


@DATASOURCES.register_module
class MSTPairImageList(ImageList):
    def __init__(self, root, which_set='all', oversample_len=None, **kwargs):
        self.memcached = False
        self.mclient_path = None
        self.has_labels = False

        all_sets = [
                'Set {}'.format(_idx) for _idx in range(1, 7)]
        used_sets = []
        if which_set == 'all':
            used_sets = copy.copy(all_sets)
        elif which_set in all_sets:
            used_sets = [which_set]
        else:
            raise NotImplementedError

        fns = []
        for _set in used_sets:
            imgs = os.listdir(os.path.join(root, _set))
            imgs.sort()
            fns.extend(
                    [os.path.join(root, _set, _img) 
                     for _img in imgs])
        fns = np.asarray(fns)
        fns = np.reshape(fns, [-1, 2])
        self.raw_fns = fns
        self.oversample_len = oversample_len
        self.set_epoch(0)

    def set_epoch(self, epoch):
        if self.oversample_len is not None:
            np.random.seed(epoch)
            fns_idx = np.random.choice(
                    len(self.raw_fns), 
                    self.oversample_len, replace=True)
            self.fns = self.raw_fns[fns_idx]
        else:
            self.fns = self.raw_fns

    def get_sample(self, idx):
        img1 = Image.open(self.fns[idx][0])
        img2 = Image.open(self.fns[idx][1])
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')
        return img1, img2
