import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models
import pdb

from . import builder
from .registry import MODELS
from .utils import GatherLayer


@MODELS.register_module
class PTMAE(nn.Module):
    def __init__(self, model_name, mask_ratio, norm_pix_loss=True, extract_last_n=4):
        from .pt_mae_modules import models_mae
        super().__init__()
        self.model = models_mae.__dict__[model_name](norm_pix_loss=norm_pix_loss)
        self.extract_last_n = extract_last_n
        self.mask_ratio = mask_ratio

    def forward_train(self, img):
        loss, _, _ = self.model(img, mask_ratio=self.mask_ratio)
        return {'loss': loss}

    def forward_extract(self, img):
        feat = self.model.get_encoder_intermediate_layers(
                img, self.extract_last_n)
        feat = torch.cat(feat, dim=-1)
        return [feat]

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'extract':
            return self.forward_extract(img)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class PTMAENeg(PTMAE):
    def __init__(
            self, neck, head, 
            neck_last_n=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.neck_last_n = neck_last_n or self.extract_last_n

    def l2_normalize(self, z):
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        return z

    @staticmethod
    def _create_buffer(N):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def get_ce_loss(self, z):
        z = z.reshape(-1, z.size(-1), 1, 1)
        z = self.neck([z])[0]
        z = self.l2_normalize(z)

        if torch.distributed.is_initialized():
            z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        permu_z = z.permute(1, 0)
        s = torch.matmul(z, permu_z)  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)
        return losses['loss']

    def forward_train(self, img):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img = img.reshape(
                img.size(0) * 2, img.size(2), 
                img.size(3), img.size(4))
        mse_loss, _, _, feat = self.model(
                img, mask_ratio=self.mask_ratio,
                with_last_n=self.neck_last_n)
        ce_loss = self.get_ce_loss(torch.cat(feat, dim=-1))
        return {'loss': mse_loss * 2 + ce_loss}
