import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models
import pdb
try:
    from einops import rearrange
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from .mae_modules import modeling_pretrain
except:
    pass

from . import builder
from .registry import MODELS
from .utils import GatherLayer


@MODELS.register_module
class MAE(nn.Module):
    def __init__(
            self, model_name, mask_ratio, window_size,
            extract_last_n=4, model_kwargs={}, normlize_target=True,
            patch_size=16, two_image_input=False,
            ):
        super().__init__()
        self.model_name = model_name
        self.extract_last_n = extract_last_n
        self.model_kwargs = model_kwargs
        self.normlize_target = normlize_target
        self.patch_size = patch_size
        self.two_image_input = two_image_input

        from .mae_modules import masking_generator
        self.masked_pos_generator = masking_generator.RandomMaskingGenerator(
            window_size, mask_ratio)
        self.build_networks()

    def build_networks(self):
        model_func = getattr(modeling_pretrain, self.model_name)
        self.backbone = model_func(**self.model_kwargs)
        self.loss_func = nn.MSELoss()

    def forward_extract(self, img):
        feat = self.backbone.encoder.get_intermediate_layers(
                img, self.extract_last_n)
        feat = torch.cat(feat, dim=-1)
        return [feat]

    def get_masked_pos(self, img):
        num_imgs = img.size(0)
        bool_masked_pos = []
        for _ in range(num_imgs):
            bool_masked_pos.append(self.masked_pos_generator())
        bool_masked_pos = np.stack(bool_masked_pos)
        bool_masked_pos = torch.from_numpy(bool_masked_pos)
        bool_masked_pos = bool_masked_pos.cuda(non_blocking=True).flatten(1).to(torch.bool)
        return bool_masked_pos

    def get_mask_labels(self, img):
        bool_masked_pos = self.get_masked_pos(img)

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).cuda()[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).cuda()[None, :, None, None]
            unnorm_images = img * std + mean  # in [0, 1]

            if self.normlize_target:
                images_squeeze = rearrange(
                        unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', 
                        p1=self.patch_size, p2=self.patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(
                        unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                        p1=self.patch_size, p2=self.patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)
        return bool_masked_pos, labels

    def forward_train(self, img, **kwargs):
        bool_masked_pos, labels = self.get_mask_labels(img)
        outputs = self.backbone(img, bool_masked_pos)
        loss = self.loss_func(input=outputs, target=labels)
        return {'loss': loss}

    # training for human experiment
    def forward_train_exposure(self, img, **kwargs):
        img_mask = img[:, 0, ...].contiguous()
        img_full = img[:, 1, ...].contiguous()
        bool_masked_pos = self.get_masked_pos(img_mask)
        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).cuda()[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).cuda()[None, :, None, None]
            unnorm_images = img_full * std + mean  # in [0, 1]

            if self.normlize_target:
                images_squeeze = rearrange(
                        unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', 
                        p1=self.patch_size, p2=self.patch_size)
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(
                        unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                        p1=self.patch_size, p2=self.patch_size)

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)

        outputs = self.backbone(img_mask, bool_masked_pos)
        loss = self.loss_func(input=outputs, target=labels)
        return {'loss': loss}

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            if not self.two_image_input:
                return self.forward_train(img, **kwargs)
            else:
                return self.forward_train_exposure(
                        img, **kwargs)
        elif mode == 'extract':
            return self.forward_extract(img)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class MAENeg(MAE):
    def __init__(self, neck, head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)

    def get_mse_loss_feat(self, img):
        bool_masked_pos, labels = self.get_mask_labels(img)
        outputs, feat = self.backbone(
                img, bool_masked_pos, 
                with_last_n=self.extract_last_n)
        mse_loss = self.loss_func(input=outputs, target=labels)
        feat = torch.cat(feat, dim=-1)
        return mse_loss, feat

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

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        mse_loss, feat = self.get_mse_loss_feat(img.reshape(
                img.size(0) * 2, img.size(2), 
                img.size(3), img.size(4)))
        ce_loss = self.get_ce_loss(feat)
        return {'loss': mse_loss * 2 + ce_loss}
