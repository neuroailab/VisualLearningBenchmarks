import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import builder
from .registry import MODELS
from openselfsup.utils import print_log


@MODELS.register_module
class ReconTrain(nn.Module):
    def __init__(
            self, model_cfg, vae_cfg,
            vae_pretrained=None,
            ):
        super().__init__()
        self.model = builder.build_model(model_cfg)
        self.vae = builder.build_model(vae_cfg)
        self.vae_pretrained = vae_pretrained

        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.init_weights(vae_pretrained)

    def forward_train(self, img):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        with torch.no_grad():
            num_views = img.size(1)
            img = img.reshape(
                img.size(0) * num_views, img.size(2), img.size(3), img.size(4))
            recon = self.vae(img, mode='recon')
            recon = recon.reshape(
                    recon.size(0) // num_views, num_views,
                    recon.size(1), recon.size(2), recon.size(3))
        return self.model(recon)

    def forward_test(self, img):
        assert img.dim() == 4, \
            f"Input must have 4 dims for forward_test, got {img.dim()}"
        with torch.no_grad():
            recon = self.vae(img, mode='recon')
            return self.model(recon, mode='test')

    def forward_extract(self, img):
        assert img.dim() == 4, \
            f"Input must have 4 dims for forward_test, got {img.dim()}"
        with torch.no_grad():
            recon = self.vae(img, mode='recon')
            return self.model(recon, mode='extract')

    def forward(self, img, mode='train', *args, **kwargs):
        if mode == 'train':
            return self.forward_train(img)
        elif mode == 'test':
            return self.forward_test(img)
        elif mode == 'extract':
            return self.forward_extract(img)
        else:
            raise Exception("No such mode: {}".format(mode))
