from .registry import DATASETS, PIPELINES
from .base import BaseDataset
from PIL import Image
from torch.utils.data import Dataset
import torch

from openselfsup.utils import print_log, build_from_cfg

from torchvision.transforms import Compose



@DATASETS.register_module
class ExtractDataset(BaseDataset):
    """Dataset for feature extraction.
    """

    def __init__(self, data_source, pipeline):
        super(ExtractDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        if not isinstance(img, Image.Image):
            img, _ = img
        img = self.pipeline(img)
        return dict(img=img)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented


@DATASETS.register_module
class ExtractDatasetWidx(Dataset):
    """Dataset for feature extraction.
    """
    def __init__(self, data_source, pipeline, num_crops=1):
        self.data_source = data_source
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.num_crops = num_crops

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        if self.num_crops == 1:
            img = self.pipeline(img)
        else:
            imgs = [self.pipeline(img).unsqueeze(0)
                    for _ in range(self.num_crops)]
            img = torch.cat(imgs, dim=0)
        return dict(img=img, idx=idx)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
