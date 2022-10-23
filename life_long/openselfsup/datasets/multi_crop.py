import torch
from PIL import Image
from .registry import DATASETS, PIPELINES
from .base import BaseDataset
from .builder import build_datasource
import copy

from openselfsup.utils import print_log, build_from_cfg
from torchvision.transforms import Compose


@DATASETS.register_module
class MultiCropDataset(BaseDataset):
    def __init__(
            self, data_source, pipeline,
            size_crops, nmb_crops, 
            min_scale_crops, max_scale_crops,
            ):
        self.data_source = build_datasource(data_source)
        self.base_pipeline = pipeline

        trans = []
        for i in range(len(size_crops)):
            pipeline = copy.deepcopy(self.base_pipeline)
            pipeline[0]['size'] = size_crops[i]
            pipeline[0]['scale'] = (min_scale_crops[i], max_scale_crops[i])
            pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
            pipeline = Compose(pipeline)
            trans.extend([pipeline] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, idx):
        image = self.data_source.get_sample(idx)
        if isinstance(image, Image.Image):
            multi_crops = list(map(lambda trans: trans(image), self.trans))
        else:
            assert isinstance(image, tuple) and len(image)==len(self.trans)
            multi_crops = [_tran(_image) for _tran, _image in zip(self.trans, image)]
        return dict(img=multi_crops)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented


@DATASETS.register_module
class DINOMultiCropDataset(BaseDataset):
    def __init__(
            self, data_source, 
            pipeline1, pipeline2, local_pipeline,
            size_crops, nmb_crops, 
            min_scale_crops, max_scale_crops,
            ):
        self.data_source = build_datasource(data_source)

        pipeline1[0]['size'] = size_crops[0]
        pipeline1[0]['scale'] = (min_scale_crops[0], max_scale_crops[0])
        pipeline1 = [build_from_cfg(p, PIPELINES) for p in pipeline1]
        pipeline1 = Compose(pipeline1)

        pipeline2[0]['size'] = size_crops[0]
        pipeline2[0]['scale'] = (min_scale_crops[1], max_scale_crops[1])
        pipeline2 = [build_from_cfg(p, PIPELINES) for p in pipeline2]
        pipeline2 = Compose(pipeline2)

        local_pipeline[0]['size'] = size_crops[1]
        local_pipeline = [build_from_cfg(p, PIPELINES) for p in local_pipeline]
        local_pipeline = Compose(local_pipeline)

        self.trans = [pipeline1, pipeline2] + [local_pipeline] * nmb_crops[1]

    def __getitem__(self, idx):
        image = self.data_source.get_sample(idx)
        if isinstance(image, Image.Image):
            multi_crops = list(map(lambda trans: trans(image), self.trans))
        else:
            assert isinstance(image, tuple) and len(image)==len(self.trans)
            multi_crops = [_tran(_image) for _tran, _image in zip(self.trans, image)]
        return dict(img=multi_crops)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
