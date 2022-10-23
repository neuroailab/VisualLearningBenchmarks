import torch
from PIL import Image
from .registry import DATASETS
from .base import BaseDataset


@DATASETS.register_module
class ContrastiveDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline, with_idx=False):
        self.with_idx = with_idx
        super(ContrastiveDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        if not self.with_idx:
            return dict(img=img_cat)
        else:
            return dict(img=img_cat, idx=idx)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented


@DATASETS.register_module
class TriContrastiveDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline):
        super().__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img3 = self.pipeline(img)
        img_cat = torch.cat(
                (img1.unsqueeze(0), img2.unsqueeze(0), img3.unsqueeze(0)), 
                dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented


@DATASETS.register_module
class ContrastiveTwoImageDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline):
        super().__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        if isinstance(img, Image.Image):
            img1 = img
            img2 = img
        else:
            assert isinstance(img, tuple) and len(img)==2
            img1, img2 = img
        img1 = self.pipeline(img1)
        img2 = self.pipeline(img2)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented


@DATASETS.register_module
class SamAugTwoImageDataset(BaseDataset):
    """Dataset for contrastive learning methods that forward
        two views of the image at a time (MoCo, SimCLR).
    """

    def __init__(self, data_source, pipeline):
        super().__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        if isinstance(img, Image.Image):
            img1 = img
            img2 = img
        else:
            assert isinstance(img, tuple) and len(img)==2
            img1, img2 = img
        state = torch.get_rng_state()
        img1 = self.pipeline(img1)
        torch.set_rng_state(state)
        img2 = self.pipeline(img2)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
