import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class BarlowTwins(nn.Module):
    def __init__(
            self, backbone, neck=None, head=None,
            pretrained=None, neg_subsamples=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
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
        self.neck.init_weights(init_linear='kaiming')

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def get_embeddings(self, img):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        num_views = img.size(1)
        img = img.reshape(
            img.size(0) * num_views, img.size(2), img.size(3), img.size(4))
        x = self.forward_backbone(img)  # 2n
        z = self.neck(x)[0]  # (2n)xd
        z = z.reshape(img.size(0) // num_views, num_views, -1)
        return z

    def forward_train(
            self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.
            mode (str): normal is used in pretraining for large dataset, exposure_cotrain 
                is used in exposure co-training to avoid same faces are treated as 
                negative pairs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        z = self.get_embeddings(img)
        z = torch.transpose(z, 0, 1)
        z = z.reshape(-1, z.size(2))
        loss = self.head(z)
        losses = dict(loss=loss)
        return losses
    
    def forward_test(self, img, to_cpu=True, **kwargs):
        img = img.reshape(
            img.size(0), img.size(1), img.size(2), img.size(3))
        x = self.forward_backbone(img)
        z = self.neck(x)[0]        
        z = z.reshape(img.size(0), -1)
        if to_cpu:
            return {'embd': z.cpu()}
        else:
            return {'embd': z}

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
