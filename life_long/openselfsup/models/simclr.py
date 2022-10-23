import torch
import torch.nn as nn
import pdb

from openselfsup.utils import print_log


from torchvision.transforms import Compose
from openselfsup.utils import build_from_cfg
from ..datasets.registry import PIPELINES

from . import builder
from .registry import MODELS
from .utils import GatherLayer


@MODELS.register_module
class SimCLR(nn.Module):
    """SimCLR.

    Implementation of "A Simple Framework for Contrastive Learning
    of Visual Representations (https://arxiv.org/abs/2002.05709)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(
            self, backbone, neck=None, head=None, hipp_head=None,
            pretrained=None, neg_subsamples=None, add_transform=None):
        super(SimCLR, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.hipp_head = None
        self.neg_subsamples = neg_subsamples
        if hipp_head is not None:
            self.hipp_head = builder.build_head(hipp_head)
        if add_transform is not None:
            self.add_transform = Compose([build_from_cfg(p, PIPELINES) for p in add_transform])
        else:
            self.add_transform = None
        self.init_weights(pretrained=pretrained)

    @staticmethod
    def _create_buffer(N, mode='normal'):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        if mode == 'normal':
            neg_mask[pos_ind] = 0
        elif mode == 'exposure_cotrain':
            neg_mask[: N, : N-1] = 0    # make sure face pairs are never treated as negatives
            neg_mask[pos_ind] = 0
            
        return mask, pos_ind, neg_mask

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
        if self.hipp_head is not None:
            self.hipp_head.init_weights(init_linear='kaiming')

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

    def l2_normalize(self, z):
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        return z

    def get_embeddings(self, img):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        num_views = img.size(1)
        if self.add_transform is not None:
            for idx in range(num_views):
                img[:, idx] = self.add_transform(img[:, idx])
        img = img.reshape(
            img.size(0) * num_views, img.size(2), img.size(3), img.size(4))
        x = self.forward_backbone(img)  # 2n
        z = self.neck(x)[0]  # (2n)xd
        z = self.l2_normalize(z)
        if num_views > 2:
            z = z.reshape(z.size(0) // num_views, num_views, z.size(1))
            z_0 = z[:, 0, :]
            z_others = z[:, 1:, :]
            z_others = torch.mean(z_others, dim=1)
            z_others = self.l2_normalize(z_others)
            z = torch.stack([z_0, z_others], dim=1)
            z = z.reshape(z.size(0)*2, z.size(2))
        return z

    def forward_train(
            self, img, mode='normal', within_batch_ctr=None, 
            add_noise=None,
            **kwargs):
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
        if self.hipp_head is not None:
            hipp_loss = self.hipp_head(z)
        if torch.distributed.is_initialized():
            z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        permu_z = z.permute(1, 0)
        if add_noise is not None:
            noise = torch.randn(permu_z.shape[0], permu_z.shape[1]).cuda() * add_noise
            permu_z += noise
            permu_z = nn.functional.normalize(permu_z, dim=0)
        s = torch.matmul(z, permu_z)  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N, mode=mode)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        if self.neg_subsamples:
            negative = negative[:, ::(negative.size(1)//self.neg_subsamples)]
        losses = self.head(positive, negative)
        if within_batch_ctr is not None:
            _t = self.head.temperature
            l_batch_ctr = torch.exp(negative / _t)
            l_batch_ctr = torch.sum(l_batch_ctr, dim=1)
            l_batch_ctr = torch.mean(torch.log(1.0 / l_batch_ctr))
            losses['loss'] -= within_batch_ctr * l_batch_ctr
        if self.hipp_head is not None:
            losses['loss'] += hipp_loss.pop('loss')
            for key in hipp_loss:
                assert key not in losses
            losses.update(hipp_loss)
        return losses

    def forward_train_asy(
            self, img, mode='normal', 
            **kwargs):
        z = self.get_embeddings(img)
        z = z.reshape(z.size(0)//2, 2, z.size(1))
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        z_0 = z[:, 0, :]
        z_1 = z[:, 1, :]
        positive = torch.sum(z_0 * z_1, dim=1, keepdim=True)
        negative = torch.matmul(z_1, z_1.permute(1, 0))
        nega_mask = 1 - torch.eye(z_1.size(0), dtype=torch.uint8).cuda()
        negative = torch.masked_select(
                negative, nega_mask==1).reshape(negative.size(0), -1)
        losses = self.head(positive, negative)
        return losses

    def forward_test(self, img, to_cpu=True, **kwargs):
        assert img.dim() == 4, \
            f"Input must have 4 dims for forward_test, got {img.dim()}"
        x = self.forward_backbone(img)
        z = self.neck(x)[0]
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True))
        if to_cpu:
            return {'embd': z.cpu()}
        else:
            return {'embd': z}

    def forward_sparse_mask(self, img, **kwargs):
        assert img.dim() == 4, \
            f"Input must have 4 dims for forward_test, got {img.dim()}"
        all_x = self.forward_backbone(img)
        ret_x = all_x[-1]
        if ret_x.dim() == 4:
            ret_x = torch.mean(ret_x, dim=(-2, -1))
        ret_x = ret_x > 0
        return {'embd': ret_x.cpu()}
        
    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, mode='normal', **kwargs)
        elif mode == 'exposure_cotrain':
            return self.forward_train(img, mode='exposure_cotrain', **kwargs)
        elif mode == 'asy_train':
            return self.forward_train_asy(img, mode='normal', **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        elif mode == 'sparse_mask':
            return self.forward_sparse_mask(img)
        elif mode == 'momentum_test':
            return self.forward_test(img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class LessNegSimCLR(SimCLR):
    def __init__(
            self, less_method='cont-16', *args, **kwargs):
        self.less_method = less_method
        super().__init__(*args, **kwargs)

    @staticmethod
    def _create_buffer(N, less_method):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.zeros((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        if less_method.startswith('cont-'):
            cont_K = int(less_method.split('-')[1])
            assert N % cont_K == 0, (N, cont_K)
            y_start_row = torch.arange(N-1, N-1+cont_K, dtype=torch.long).cuda()
            x_start_row = torch.zeros(cont_K, dtype=torch.long).cuda()
            x_rows = [x_start_row]
            y_rows = [y_start_row]
            for idx in range(cont_K):
                x_rows.append(x_start_row.clone() + idx)
                y_rows.append(y_start_row.clone())
            x_rows = torch.cat(x_rows)
            y_rows = torch.cat(y_rows)
            all_x_rows = []
            all_y_rows = []
            for idx in range(N // cont_K):
                all_x_rows.append(x_rows.clone() + idx * cont_K)
                all_y_rows.append(y_rows.clone() + idx * cont_K)
            all_x_rows = torch.cat(all_x_rows)
            all_y_rows = torch.cat(all_y_rows)
            all_x_rows = torch.cat([all_x_rows, all_x_rows + N])
            all_y_rows = torch.cat([all_y_rows, all_y_rows - (N-1)])
            final_neg_idx = (all_x_rows, all_y_rows)
            neg_mask[final_neg_idx] = 1
        else:
            raise NotImplementedError
        return mask, pos_ind, neg_mask

    def forward_train(self, img, **kwargs):
        z = self.get_embeddings(img)
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        permu_z = z.permute(1, 0)
        s = torch.matmul(z, permu_z)  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(
                N, less_method=self.less_method)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)
        return losses


@MODELS.register_module
class GroupSimCLR(SimCLR):
    def __init__(
            self, group_no=32, *args, **kwargs):
        self.group_no = group_no
        super().__init__(*args, **kwargs)

    def get_loss_for_one_group(self, z):
        N = z.size(0) // 2
        permu_z = z.permute(1, 0)
        s = torch.matmul(z, permu_z)  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)
        return losses['loss']

    def forward_train(self, img, **kwargs):
        z = self.get_embeddings(img)
        assert z.size(0) % (2 * self.group_no) == 0
        loss = []
        N = z.size(0) // 2
        for idx in range(len(z) // (2*self.group_no)):
            sta_idx = idx * self.group_no
            end_idx = (idx+1) * self.group_no
            z_sub_0 = z[sta_idx:end_idx]
            z_sub_1 = z[(sta_idx + N):(end_idx + N)]
            loss.append(self.get_loss_for_one_group(
                torch.cat([z_sub_0, z_sub_1], dim=0)))
        losses = {'loss': sum(loss) / len(loss)}
        return losses


@MODELS.register_module
class SepBatchSimCLR(SimCLR):
    def __init__(
            self, scale_ratio=1, stop_neg_grad=False, *args, **kwargs):
        self.scale_ratio = scale_ratio
        self.stop_neg_grad = stop_neg_grad
        super().__init__(*args, **kwargs)

    def get_loss_with_add_negs(self, z, z_negs=None):
        N = z.size(0) // 2
        permu_z = z.permute(1, 0)
        s = torch.matmul(z, permu_z)  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        if z_negs is not None:
            add_negs = torch.sum(z.unsqueeze(1) * z_negs, dim=-1)
            negative = torch.cat([negative, add_negs], dim=1)
        losses = self.head(positive, negative)
        return losses['loss']

    def forward_train(self, img, **kwargs):
        z = self.get_embeddings(img)
        assert z.size(0) % (2 * (self.scale_ratio+1)) == 0
        embd_d = z.size(-1)
        z = z.reshape(-1, 2, embd_d)
        batch_size = z.size(0) // (self.scale_ratio+1)
        z_curr = z[:batch_size].reshape(-1, embd_d)
        z_storage = z[batch_size:]
        loss_storage = self.get_loss_with_add_negs(z_storage.reshape(-1, embd_d))
        z_storage = z_storage.reshape(self.scale_ratio, -1, 2, embd_d)
        z_storage = torch.transpose(z_storage, 0, 1) # [BS, SR, 2, D]
        z_storage = z_storage.reshape(batch_size, -1, embd_d)
        z_storage = z_storage.unsqueeze(1).repeat(1, 2, 1, 1)
        z_storage = z_storage.reshape(batch_size*2, -1, embd_d)
        if self.stop_neg_grad:
            z_storage = z_storage.detach()
        loss_curr = self.get_loss_with_add_negs(z_curr, z_storage)
        losses = {'loss': loss_curr+loss_storage}
        return losses


@MODELS.register_module
class InterOutSimCLR(SimCLR):
    def __init__(
            self, backbone, all_necks=[None], head=None,
            pretrained=None, inter_mix_weights=[1., 1., 1.],
            head_mode=None):
        nn.Module.__init__(self)
        self.inter_mix_weights = inter_mix_weights
        self.backbone = builder.build_backbone(backbone)

        self.neck_names = []
        for idx, curr_neck in enumerate(all_necks):
            neck_name = f'neck_{idx}'
            curr_neck = builder.build_neck(curr_neck)
            setattr(self, neck_name, curr_neck)
            self.neck_names.append(neck_name)

        if head_mode is None:
            head_mode = [None] * len(all_necks)
        self.head_mode = head_mode
        assert len(self.head_mode) == len(all_necks)

        if isinstance(head, dict):
            head = [head] * len(all_necks)
        self.head_names = []
        for idx, curr_head in enumerate(head):
            head_name = f'head_{idx}'
            curr_head = builder.build_head(curr_head)
            setattr(self, head_name, curr_head)
            self.head_names.append(head_name)
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
        for neck_name in self.neck_names:
            getattr(self, neck_name).init_weights(init_linear='kaiming')

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, mode='normal', **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)[-1:]
        elif mode == 'test':
            return self.forward_test(img)
        elif mode == 'sparse_mask':
            return self.forward_sparse_mask(img)
        else:
            raise Exception("No such mode: {}".format(mode))

    def get_embeddings(self, img):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        num_views = img.size(1)
        img = img.reshape(
            img.size(0) * num_views, img.size(2), img.size(3), img.size(4))
        all_x = self.forward_backbone(img)  # 2n

        all_x = list(all_x)
        while len(all_x) < len(self.neck_names):
            all_x.append(all_x[-1])
        all_z = []
        for x, neck_name in zip(all_x, self.neck_names):
            z = getattr(self, neck_name)([x])[0]  # (2n)xd
            z = self.l2_normalize(z)
            if num_views > 2:
                raise NotImplementedError
            all_z.append(z)
        return all_z

    def get_loss_from_embd(self, z, mode, head):
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        permu_z = z.permute(1, 0)
        s = torch.matmul(z, permu_z)  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N, mode=mode)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = head(positive, negative)
        return losses['loss']

    def forward_train(
            self, img, mode='normal', 
            **kwargs):
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
        embds = self.get_embeddings(img)
        loss = 0.
        for weight, z, head_name, head_mode in \
                zip(self.inter_mix_weights, embds, 
                    self.head_names, self.head_mode):
            head = getattr(self, head_name)
            if (head_mode == 'contrastive') or (head_mode is None):
                _inter_loss = self.get_loss_from_embd(z, mode, head) * weight
            elif (head_mode == 'embd_head'):
                _inter_loss = head(z)['loss'] * weight
            else:
                raise NotImplementedError
            loss = loss + _inter_loss
        return {'loss': loss}

    def forward_test(self, img, to_cpu=True, **kwargs):
        assert img.dim() == 4, \
            f"Input must have 4 dims for forward_test, got {img.dim()}"
        all_x = self.forward_backbone(img)
        last_neck = getattr(self, self.neck_names[-1])
        z = last_neck(all_x[-1:])[0]
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True))
        if to_cpu:
            return {'embd': z.cpu()}
        else:
            return {'embd': z}

    def forward_sparse_mask(self, img, **kwargs):
        assert img.dim() == 4, \
            f"Input must have 4 dims for forward_test, got {img.dim()}"
        all_x = self.forward_backbone(img)
        ret_x = all_x[-1]
        ret_x = torch.mean(ret_x, dim=(-2, -1))
        ret_x = ret_x > 0
        return {'embd': ret_x.cpu()}


@MODELS.register_module
class SimCLRMomentum(SimCLR):
    def __init__(
            self, 
            backbone, neck=None,
            base_momentum=0.99,
            *args, **kwargs):
        super().__init__(
                backbone=backbone, neck=neck, *args, **kwargs)
        self.base_momentum = base_momentum
        self.momentum = base_momentum

        self.momentum_backbone = builder.build_backbone(backbone)
        self.momentum_neck = builder.build_neck(neck)
        for param in self.momentum_backbone.parameters():
            param.requires_grad = False
        for param in self.momentum_neck.parameters():
            param.requires_grad = False
        self.init_momentum_weights()
        
    def init_momentum_weights(self):
        for param_ol, param_tgt in zip(self.backbone.parameters(),
                                       self.momentum_backbone.parameters()):
            param_tgt.data.copy_(param_ol.data)
        for param_ol, param_tgt in zip(self.neck.parameters(),
                                       self.momentum_neck.parameters()):
            param_tgt.data.copy_(param_ol.data)

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.backbone.parameters(),
                                       self.momentum_backbone.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)
        for param_ol, param_tgt in zip(self.neck.parameters(),
                                       self.momentum_neck.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    def forward_momentum_test(self, img, to_cpu=True, **kwargs):
        assert img.dim() == 4, \
            f"Input must have 4 dims for forward_test, got {img.dim()}"
        x = self.momentum_backbone(img)
        z = self.momentum_neck(x)[0]
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True))
        if to_cpu:
            return {'embd': z.cpu()}
        else:
            return {'embd': z}

    def forward(self, mode='train', *args, **kwargs):
        if mode == 'momentum_test':
            return self.forward_momentum_test(*args, **kwargs)
        else:
            return super().forward(mode=mode, *args, **kwargs)
