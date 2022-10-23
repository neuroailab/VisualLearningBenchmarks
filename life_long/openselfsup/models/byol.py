import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .utils import add_noise_to_target, GatherLayer


@MODELS.register_module
class BYOL(nn.Module):
    """BYOL.

    Implementation of "Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning (https://arxiv.org/abs/2006.07733)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        base_momentum (float): The base momentum coefficient for the target network.
            Default: 0.996.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 **kwargs):
        super(BYOL, self).__init__()
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.online_net[0]
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        sync_bn_convert = torch.nn.SyncBatchNorm.convert_sync_batchnorm
        self.online_net = sync_bn_convert(self.online_net)
        self.target_net = sync_bn_convert(self.target_net)
        self.head = sync_bn_convert(self.head)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_onl_tgt(self, pretrained):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.online_net[0].init_weights(pretrained=pretrained) # backbone
        self.online_net[1].init_weights(init_linear='kaiming') # projection
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        self.init_onl_tgt(pretrained)
        # init the predictor in the head
        self.head.init_weights()

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    def forward_train(self, img, add_noise=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        with torch.no_grad():
            proj_target_v1 = self.target_net(img_v1)[0].clone().detach()
            proj_target_v2 = self.target_net(img_v2)[0].clone().detach()

        if add_noise is not None:
            proj_target_v1 = add_noise_to_target(proj_target_v1, add_noise)
            proj_target_v2 = add_noise_to_target(proj_target_v2, add_noise)

        loss = self.head(proj_online_v1, proj_target_v2)['loss'] + \
               self.head(proj_online_v2, proj_target_v1)['loss']
        return dict(loss=loss)

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
class BYOLNeg(BYOL):
    def __init__(self, predictor, **kwargs):
        super().__init__(**kwargs)
        self.predictor = builder.build_neck(predictor)
        sync_bn_convert = torch.nn.SyncBatchNorm.convert_sync_batchnorm
        self.predictor = sync_bn_convert(self.predictor)
        self.predictor.init_weights(init_linear='normal')

    def init_weights(self, pretrained=None):
        self.init_onl_tgt(pretrained)

    @staticmethod
    def _create_buffer(N):
        neg_mask = 1 - torch.eye(N, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N).cuda(), torch.arange(N).cuda())
        return pos_ind, neg_mask

    def get_ce_loss(self, proj_onl, proj_tgt):
        proj_onl = nn.functional.normalize(proj_onl, dim=1)
        proj_tgt = nn.functional.normalize(proj_tgt, dim=1)

        proj_onl = torch.cat(GatherLayer.apply(proj_onl), dim=0)
        proj_tgt = torch.cat(GatherLayer.apply(proj_tgt), dim=0)
        s = torch.matmul(proj_onl, proj_tgt.permute(1, 0))
        pos_ind, neg_mask = self._create_buffer(proj_onl.size(0))
        positive = s[pos_ind].unsqueeze(1)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        loss = self.head(positive, negative)['loss']
        return loss

    def forward_train(self, img, add_noise=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        with torch.no_grad():
            proj_target_v1 = self.predictor(self.target_net(img_v1))[0]
            proj_target_v1 = proj_target_v1.clone().detach()
            proj_target_v2 = self.predictor(self.target_net(img_v2))[0]
            proj_target_v2 = proj_target_v2.clone().detach()

        if add_noise is not None:
            proj_target_v1 = add_noise_to_target(proj_target_v1, add_noise)
            proj_target_v2 = add_noise_to_target(proj_target_v2, add_noise)

        loss = self.get_ce_loss(proj_online_v1, proj_target_v2) + \
               self.get_ce_loss(proj_online_v2, proj_target_v1)
        return dict(loss=loss)


@MODELS.register_module
class SepBatchBYOLPosNeg(BYOL):
    def __init__(self, neg_head, scale_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.neg_head = builder.build_head(neg_head)
        sync_bn_convert = torch.nn.SyncBatchNorm.convert_sync_batchnorm
        self.neg_head = sync_bn_convert(self.neg_head)
        self.scale_ratio = scale_ratio

    def init_weights(self, pretrained=None):
        self.init_onl_tgt(pretrained)

    def get_ce_loss(self, proj_onl, proj_tgt):
        proj_onl = nn.functional.normalize(proj_onl, dim=1)
        proj_tgt = nn.functional.normalize(proj_tgt, dim=1)

        proj_onl = torch.cat(GatherLayer.apply(proj_onl), dim=0)
        proj_tgt = torch.cat(GatherLayer.apply(proj_tgt), dim=0)
        s = torch.matmul(proj_onl, proj_tgt.permute(1, 0))
        pos_ind, neg_mask = BYOLNeg._create_buffer(proj_onl.size(0))
        positive = s[pos_ind].unsqueeze(1)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        loss = self.neg_head(positive, negative)['loss']
        return loss

    def get_curr_part(self, embds):
        batch_size = embds.size(0) // (self.scale_ratio+1)
        return embds[:batch_size]

    def get_storage_part(self, embds):
        batch_size = embds.size(0) // (self.scale_ratio+1)
        return embds[batch_size:]

    def forward_train(self, img, add_noise=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        with torch.no_grad():
            proj_target_v1 = self.target_net(img_v1)[0]
            proj_target_v1 = proj_target_v1.clone().detach()
            proj_target_v2 = self.target_net(img_v2)[0]
            proj_target_v2 = proj_target_v2.clone().detach()

        if add_noise is not None:
            proj_target_v1 = add_noise_to_target(proj_target_v1, add_noise)
            proj_target_v2 = add_noise_to_target(proj_target_v2, add_noise)

        loss = self.get_ce_loss(
                    self.get_curr_part(proj_online_v1), 
                    self.get_curr_part(proj_target_v2)) + \
               self.get_ce_loss(
                    self.get_curr_part(proj_online_v2), 
                    self.get_curr_part(proj_target_v1)) + \
               self.head(
                    self.get_storage_part(proj_online_v1), 
                    self.get_storage_part(proj_target_v2))['loss'] + \
               self.head(
                    self.get_storage_part(proj_online_v2), 
                    self.get_storage_part(proj_target_v1))['loss']
        return dict(loss=loss)
