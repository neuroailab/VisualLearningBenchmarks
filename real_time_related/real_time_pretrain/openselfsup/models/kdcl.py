import torch
import torch.nn as nn

from openselfsup.utils import print_log
from mmcv.runner import load_checkpoint
from openselfsup.utils import get_root_logger

from . import builder
from .registry import MODELS
from .utils import add_noise_to_target, GatherLayer


@MODELS.register_module
class KDCL(nn.Module):
    def __init__(self,
                 backbone,
                 target_net,
                 target_pretrained,
                 neck=None,
                 head=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = builder.build_model(target_net)
        self.backbone = self.online_net[0]
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.target_pretrained = target_pretrained
        self.init_weights(pretrained=pretrained)

        sync_bn_convert = torch.nn.SyncBatchNorm.convert_sync_batchnorm
        self.online_net = sync_bn_convert(self.online_net)
        self.target_net = sync_bn_convert(self.target_net)
        self.head = sync_bn_convert(self.head)

    def init_onl_tgt(self, pretrained):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.online_net[0].init_weights(pretrained=pretrained) # backbone
        self.online_net[1].init_weights(init_linear='kaiming') # projection

        assert isinstance(self.target_pretrained, str)
        load_checkpoint(
                self.target_net, self.target_pretrained, 
                strict=True, logger=None)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        self.init_onl_tgt(pretrained)
        # init the predictor in the head
        self.head.init_weights()

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        proj_online_v1, proj_target_v1, proj_online_v2, proj_target_v2 =\
                self.get_projs(img, **kwargs)

        loss = self.head(proj_online_v1, proj_target_v2)['loss'] + \
               self.head(proj_online_v2, proj_target_v1)['loss']
        return dict(loss=loss)

    def get_projs(self, img, add_noise=None, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        with torch.no_grad():
            proj_target_v1 = self.target_net(
                    img_v1, mode='test', to_cpu=False)['embd'].clone().detach()
            proj_target_v2 = self.target_net(
                    img_v2, mode='test', to_cpu=False)['embd'].clone().detach()

        if add_noise is not None:
            proj_target_v1 = add_noise_to_target(proj_target_v1, add_noise)
            proj_target_v2 = add_noise_to_target(proj_target_v2, add_noise)
        return proj_online_v1, proj_target_v1, proj_online_v2, proj_target_v2

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
class KDCLNeg(KDCL):
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
        proj_online_v1, proj_target_v1, proj_online_v2, proj_target_v2 =\
                self.get_projs(img, **kwargs)

        loss = self.get_ce_loss(proj_online_v1, proj_target_v2) + \
               self.get_ce_loss(proj_online_v2, proj_target_v1)
        return dict(loss=loss)
