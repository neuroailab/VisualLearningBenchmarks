import torch
from PIL import Image
from .base import BaseDataset
from .registry import DATASETS, PIPELINES
from openselfsup.utils import print_log, build_from_cfg
import numpy as np
import torchvision.transforms as transforms


@DATASETS.register_module
class NaiveCuriousDataset(BaseDataset):
    def __init__(
            self, data_source, pipeline, 
            paramed_pipeline, 
            attempt_num=20,
            use_max=True):
        self.attempt_num = attempt_num
        self.use_max = use_max
        self.paramed_pipeline = [
                build_from_cfg(p, PIPELINES) \
                for p in paramed_pipeline]
        super(NaiveCuriousDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1, img2 = self.get_paramed_pair(img)
        img1 = self.pipeline(img1)
        img2 = self.pipeline(img2)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def get_paramed_pair(self, img):
        _aug = self.paramed_pipeline[0]
        def _get_one_param():
            return _aug.get_params(img, _aug.scale, _aug.ratio)
        all_pairs = [
                (_get_one_param(), _get_one_param()) \
                for _ in range(self.attempt_num)]
        all_dist = [
                np.sum(np.abs(np.asarray(_pair[0]) - np.asarray(_pair[1]))) \
                for _pair in all_pairs]
        if self.use_max:
            want_pair = all_pairs[np.argmax(all_dist)]
        else:
            want_pair = all_pairs[np.argmin(all_dist)]

        def _apply_aug(_crop_params):
            _img = transforms.functional.resized_crop(
                    img, 
                    _crop_params[0], _crop_params[1], 
                    _crop_params[2], _crop_params[3], 
                    _aug.size, 
                    _aug.interpolation)
            return _img
        img1 = _apply_aug(want_pair[0])
        img2 = _apply_aug(want_pair[1])
        return img1, img2

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
