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


@FFCVLOADER.register_module
class ExtractFFCVLoader:
    def __init__(
            self, data_path, input_size, meta_path,
            in_memory, batch_size, num_workers,
            ):
        from ffcv.pipeline.operation import Operation
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
            RandomHorizontalFlip, ToTorchImage
        from ffcv.fields.rgb_image import CenterCropRGBImageDecoder

        with open(meta_path, 'r') as f:
            lines = f.readlines()
        _, self.labels = zip(*[l.strip().split() for l in lines])
        self.labels = [int(l) for l in self.labels]
        self.data_source = self

        order = OrderOption.SEQUENTIAL
        loader_kwargs = dict(
                fname=data_path,
                batch_size=batch_size,
                num_workers=num_workers,
                order=order,
                os_cache=in_memory,
                drop_last=False,
                distributed=False)

        res_tuple = (input_size, input_size)
        DEFAULT_CROP_RATIO = 224/256
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        rank, world_size = get_dist_info()
        this_device = f'cuda:{rank}'
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
        ]
        self.loader = Loader(
                pipelines={
                    'image': image_pipeline},
                **loader_kwargs)
        from openselfsup.datasets.loader.sampler import DistributedSampler
        self.dataset = list(range(len(self.labels)))
        #sampler = DistributedSampler(
        #        self.dataset, world_size, rank, 
        #        shuffle=False)
        #self.loader.traversal_order.sampler = sampler
        #self.sampler = sampler
        self.sampler = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        sentinel = object()
        iterator = iter(self.loader)
        while iterator:
            elem = next(iterator, sentinel)
            if elem is sentinel:
                return
            ret_dict = {'img': elem[0]}
            yield ret_dict
