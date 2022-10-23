import torch
from PIL import Image
from torchvision import transforms as _transforms
from .registry import FFCVLOADER, PIPELINES

from typing import List
from pathlib import Path
import torchvision
import copy

import numpy as np
import pdb
from ..framework.dist_utils import get_dist_info
from openselfsup.utils import build_from_cfg

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


class ToContiguous(torch.nn.Module):
    def forward(self, img):
        return img.to(memory_format=torch.channels_last)


class MultiTransform(torch.nn.Module):
    def __init__(
            self, num_imgs_one_sp, transform):
        super().__init__()
        self.num_imgs_one_sp = num_imgs_one_sp
        self.transform = transform

    def forward(self, img):
        for sta_idx in range(0, img.size(0), self.num_imgs_one_sp):
            end_idx = min(sta_idx + self.num_imgs_one_sp, img.size(0))
            img[sta_idx:end_idx] = self.transform(img[sta_idx:end_idx])
        return img


@FFCVLOADER.register_module
class ContrastiveFFCVLoader:
    def __init__(
            self, data_path, pipeline, shuffle, 
            in_memory, batch_size, num_workers,
            indices=None, multi_trans=None,
            use_torch_gauss=True, color_jitter_nohue=False):
        from ffcv.pipeline.operation import Operation
        from ffcv.loader import Loader, OrderOption

        self.multi_trans = multi_trans
        self.use_torch_gauss = use_torch_gauss
        self.color_jitter_nohue = color_jitter_nohue
        order = OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL
        loader_kwargs = dict(
                fname=data_path,
                batch_size=batch_size,
                num_workers=num_workers,
                order=order,
                os_cache=in_memory,
                drop_last=False,
                distributed=True,
                indices=indices)
        loader1 = Loader(
                pipelines={
                    'image': self.get_one_image_pipeline(pipeline)},
                **loader_kwargs)
        loader2 = Loader(
                pipelines={
                    'image': self.get_one_image_pipeline(pipeline)},
                **loader_kwargs)
        self.loaders = [loader1, loader2]

    def map_to_ffcv_trans(self, one_pipeline):
        import openselfsup.datasets.pipelines.ffcv_color as ffcv_color
        import openselfsup.datasets.pipelines.ffcv_flip as ffcv_flip
        import openselfsup.datasets.pipelines.ffcv_gaussian as ffcv_gaussian

        new_pipeline = None
        if one_pipeline['type'] == 'RandomGrayscale':
            add_gray_prob = one_pipeline.get('p', 0.5)
            new_pipeline = ffcv_color.RandomGrayScale(prob=add_gray_prob)
        if one_pipeline['type'] == 'RandomVerticalFlip':
            add_flip_prob = one_pipeline.get('p', 0.5)
            new_pipeline = ffcv_flip.RandomVerticalFlip(flip_prob=add_flip_prob)
        if one_pipeline['type'] == 'RandomAppliedTrans'\
                and one_pipeline['transforms'][0]['type'] == 'ColorJitter':
            add_jitter_prob = one_pipeline.get('p', 0.5)
            color_jitter_kwargs = copy.copy(one_pipeline['transforms'][0])
            color_jitter_kwargs.pop('type')
            color_jitter_builder = ffcv_color.RandomColorJitter
            if self.color_jitter_nohue:
                color_jitter_builder = ffcv_color.RandomColorJitterNoHue
            new_pipeline = color_jitter_builder(
                        prob=add_jitter_prob,
                        **color_jitter_kwargs)
        if one_pipeline['type'] == 'RandomAppliedTrans'\
                and one_pipeline['transforms'][0]['type'] == 'GaussianBlur'\
                and (not self.use_torch_gauss):
            prob = one_pipeline.get('p', 0.5)
            gaussian_kwargs = copy.copy(one_pipeline['transforms'][0])
            gaussian_kwargs.pop('type')
            gaussian_kwargs['prob'] = prob
            new_pipeline = ffcv_gaussian.RandomGaussianBlur(
                        **gaussian_kwargs)
        return new_pipeline

    def get_one_image_pipeline(self, pipeline):
        from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
            RandomHorizontalFlip, ToTorchImage
        from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
        pipeline = copy.deepcopy(pipeline)
        rank, _ = get_dist_info()
        this_device = f'cuda:{rank}'

        res = pipeline[0]['size']
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        before_tensor_pipeline = [
                self.decoder,
                RandomHorizontalFlip(),
                ]
        pipeline_from_torch = []
        need_multi_trans = []
        # First 2 (crop, flip) and last 2 (to_tensor, normalize) are skipped
        for _pipeline in pipeline[2:-2]:
            new_pipeline = self.map_to_ffcv_trans(_pipeline)
            if new_pipeline is not None:
                before_tensor_pipeline.append(new_pipeline)
                continue

            _need_multi_tran = False
            if _pipeline['type'] == 'RandomAppliedTrans':
                _pipeline['type'] = 'RandomApply'
                _need_multi_tran = True
            if 'transforms' in _pipeline:
                for _transform in _pipeline['transforms']:
                    if _transform['type'] == 'GaussianBlur':
                        _transform['type'] = 'GaussianBlurTorch'
                        if 'kernel_size' not in _transform:
                            _transform['kernel_size'] = 23
                        sigma_min = _transform.pop('sigma_min')
                        sigma_max = _transform.pop('sigma_max')
                        _transform['sigma'] = (sigma_min, sigma_max)
                        _need_multi_tran = True
                _pipeline['transforms'] = [
                        build_from_cfg(p, PIPELINES) 
                        for p in _pipeline['transforms']]
            need_multi_trans.append(_need_multi_tran)
            pipeline_from_torch.append(_pipeline)
        pipeline_from_torch = [
                build_from_cfg(p, PIPELINES) for p in pipeline_from_torch]
        if self.multi_trans is not None:
            for idx, _need in enumerate(need_multi_trans):
                if _need:
                    pipeline_from_torch[idx] = MultiTransform(
                            self.multi_trans, pipeline_from_torch[idx])
        image_pipeline: List[Operation] = before_tensor_pipeline \
            + [
                ToTensor(),
                ToDevice(torch.device(this_device), non_blocking=True),
                ToTorchImage(),
                ]\
            + pipeline_from_torch\
            + [ToContiguous(), 
               NormalizeImage(
                   IMAGENET_MEAN, IMAGENET_STD, 
                   np.float32)]
        return image_pipeline

    def __len__(self):
        return len(self.loaders[0])

    def __iter__(self):
        sentinel = object()
        iterators = [iter(it) for it in self.loaders]
        while iterators:
            result = []
            for it in iterators:
                elem = next(it, sentinel)
                if elem is sentinel:
                    return
                result.append(elem)
            ret_dict = {}
            ret_dict['img'] = torch.stack(
                    [_result[0] for _result in result], 
                    dim=1)
            yield ret_dict
