import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .utils import add_noise_to_target


@MODELS.register_module
class Siamese(nn.Module):
    '''Siamese unofficial implementation.
    '''

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 **kwargs):
        super(Siamese, self).__init__()
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.online_net[0]
        self.head = builder.build_head(head)

        sync_bn_convert = torch.nn.SyncBatchNorm.convert_sync_batchnorm
        self.online_net = sync_bn_convert(self.online_net)
        self.head = sync_bn_convert(self.head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.online_net[0].init_weights(pretrained=pretrained) # backbone
        self.online_net[1].init_weights(init_linear='kaiming') # projection
        # init the predictor in the head
        self.head.init_weights()

    def forward_train(self, img, add_noise=None, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        img_cat1 = torch.cat([img_v1, img_v2], dim=0)
        img_cat2 = torch.cat([img_v2, img_v1], dim=0)
        # compute query features
        proj_online = self.online_net(img_cat1)[0]
        with torch.no_grad():
            proj_target = self.online_net(img_cat2)[0].clone().detach()

        if add_noise is not None:
            proj_target = add_noise_to_target(proj_target, add_noise)

        losses = self.head(proj_online, proj_target)
        return losses

    def forward_test(self, img, **kwargs):
        assert img.dim() == 4, \
            "Input must have 4 dims, got: {}".format(img.dim())
        feature = self.online_net(img)[0]
        feature = nn.functional.normalize(feature)  # BxC
        return {'embd': feature.cpu()}

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class SiameseOneView(Siamese):
    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()

        # compute query features
        proj_online = self.online_net(img_v1)[0]
        with torch.no_grad():
            proj_target = self.online_net(img_v2)[0].clone().detach()
        losses = self.head(proj_online, proj_target)
        return losses


@MODELS.register_module
class SiameseSepCrop(Siamese):
    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()

        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]

        loss = self.head(proj_online_v1, proj_online_v2.detach())['loss'] \
               + self.head(proj_online_v2, proj_online_v1.detach())['loss']
        loss = loss / 2
        losses = {'loss': loss}
        return losses
