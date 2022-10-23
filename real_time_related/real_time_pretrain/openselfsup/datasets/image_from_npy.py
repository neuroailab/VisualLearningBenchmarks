import torch
from torch.utils.data import Dataset
import numpy as np
import os

from openselfsup.utils import print_log, build_from_cfg
from torchvision.transforms import Compose

from .registry import DATASETS, PIPELINES
from .builder import build_datasource


@DATASETS.register_module
class ImageFromNpy(Dataset):
    """Base dataset.

    Args:
        data_source (dict): Data source defined in
            `openselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `oenselfsup.datasets.pipelines`.
    """

    def __init__(self, npy_path, pipeline, meta_path):
        self.images = np.load(npy_path)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)

        with open(meta_path, 'r') as f:
            lines = f.readlines()
        _, self.labels = zip(*[l.strip().split() for l in lines])
        self.labels = np.asarray([int(l) for l in self.labels])
        assert len(self.images) == len(self.labels)
        self.data_source = self

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.pipeline(img)
        return {'img': img}


def get_typical_dataset_cfg():
    import openselfsup
    fwk_path = openselfsup.__path__[0] 
    data_path = os.path.join(
            fwk_path, '..', 'data/imagenet/in_val_processed.npy')
    meta_path = os.path.join(
            fwk_path, '..', 'data/imagenet/meta/part_train_val_labeled.txt')
    img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pipeline = [
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
    ]
    val_svm_cfg=dict(
            type='ImageFromNpy',
            npy_path=data_path,
            pipeline=pipeline,
            meta_path=meta_path)
    return val_svm_cfg
