import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import builder
from .registry import MODELS
from openselfsup.utils import print_log


@MODELS.register_module
class InterAutoEncoder(nn.Module):
    def __init__(
            self, backbone, vae_cfg, padding_size,
            pretrained=None, image_region=None,
            ):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.vae = builder.build_model(vae_cfg)
        self.vae_cfg = vae_cfg
        self.padding = nn.ZeroPad2d(padding_size)
        self.image_region = image_region
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)

    def get_inter_x(self, img):
        with torch.no_grad():
            if self.image_region is not None:
                _l, _r = self.image_region
                img = img[:, :, _l:_r, _l:_r].detach().clone() 
                img = img.reshape(
                        img.size(0), img.size(1), _r-_l, _r-_l)
            x = self.backbone(img)[0]
            x = self.padding(x)
            x = x[:, :self.vae_cfg['image_channels']]
        return x

    def forward_train(self, img):
        x = self.get_inter_x(img)
        return self.vae.forward(x, mode='train')

    def forward_recon(self, img):
        x = self.get_inter_x(img)
        x_recon = self.vae.forward(x, mode='recon')
        return {'x': x, 'x_recon': x_recon}

    def forward(self, img, mode='train', *args, **kwargs):
        if mode == 'train':
            return self.forward_train(img)
        elif mode == 'recon':
            return self.forward_recon(img)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class InterBNAutoEncoder(nn.Module):
    def __init__(
            self, backbone, vae_cfg, inter_channels,
            pretrained=None, image_region=None,
            vae_pretrained=None,
            ):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.vae = builder.build_model(vae_cfg)
        self.vae_cfg = vae_cfg
        self.inter_channels = inter_channels
        self.image_region = image_region
        self.init_weights(pretrained=pretrained, vae_pretrained=vae_pretrained)

    def init_weights(self, pretrained=None, vae_pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        if vae_pretrained is not None:
            print_log(
                    'load VAE model from: {}'.format(vae_pretrained), 
                    logger='root')
        self.vae.init_weights(pretrained=vae_pretrained)

    def get_img_inter_x(self, img):
        with torch.no_grad():
            if self.image_region is not None:
                _l, _r = self.image_region
                img = img[:, :, _l:_r, _l:_r].detach().clone() 
                img = img.reshape(
                        img.size(0), img.size(1), _r-_l, _r-_l)
            x = self.backbone(img)[0]
            x = x[:, :self.inter_channels]
        return img, x

    def forward_train(self, img):
        img, x = self.get_img_inter_x(img)
        vae_returns = self.vae.forward(img, mode='with_recon')
        recon_batch = vae_returns.pop('recon_batch')
        x_recon = self.backbone(recon_batch)[0]
        x_recon = x_recon[:, :self.inter_channels]
        batch_size = x.size(0)
        new_reconL = self.vae.calculate_recon_loss(
                x=x.view(batch_size, -1), average=True,
                x_recon=x_recon.view(batch_size, -1))
        vae_returns['loss'] = vae_returns['variat'] + new_reconL
        vae_returns['recon'] = new_reconL
        return vae_returns

    def forward_recon(self, img):
        img, x = self.get_img_inter_x(img)
        vae_returns = self.vae.forward(img, mode='with_recon')
        recon_batch = vae_returns.pop('recon_batch')
        x_recon = self.backbone(recon_batch)[0]
        x_recon = x_recon[:, :self.inter_channels]
        ret_dict = {
                'x': x, 'x_recon': x_recon,
                'input': img, 'recon': recon_batch,
                }
        return ret_dict

    def forward(self, img, mode='train', *args, **kwargs):
        if mode == 'train':
            return self.forward_train(img)
        elif mode == 'recon':
            return self.forward_recon(img)
        else:
            raise Exception("No such mode: {}".format(mode))
