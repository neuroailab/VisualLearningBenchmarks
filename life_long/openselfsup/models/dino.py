import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models
import pdb

from torch.nn.utils import clip_grad
from .vit_modules import utils
from .vit_modules import vision_transformer as vits
from .vit_modules.vision_transformer import DINOHead

from . import builder
from .registry import MODELS
from .utils import GatherLayer
from ..framework.hooks.hook import Hook
from ..framework.hooks import optimizer
from ..framework.checkpoint import load_checkpoint


class WDScheduleHook(Hook):
    def before_run(self, runner):
        args = runner.model.module.args
        self.wd_schedule = utils.cosine_scheduler(
            args.weight_decay,
            args.weight_decay_end,
            runner._max_epochs, len(runner.data_loader),
        )

    def before_train_iter(self, runner):
        it = runner.iter
        for i, param_group in enumerate(runner.optimizer.param_groups):
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[it]


class TeacherTempHook(Hook):
    def before_epoch(self, runner):
        epoch = runner.epoch
        runner.model.module.dino_loss.teacher_temp = \
                runner.model.module.dino_loss.teacher_temp_schedule[epoch]


try:
    import apex
except:
    pass

class DistOptimizerHook(optimizer.DistOptimizerHook):
    """Optimizer hook for distributed training."""

    def __init__(self, freeze_last_layer=1, last_layer_dim=65536, *args, **kwargs):
        self.freeze_last_layer = freeze_last_layer
        self.last_layer_dim = last_layer_dim
        super().__init__(*args, **kwargs)

    def cancel_gradients_last_layer(self, runner):
        if runner.epoch >= self.freeze_last_layer:
            return
        all_params = apex.amp.master_params(runner.optimizer)
        # Tricky solution to find the last layer params
        last_layer_params = 0
        for _param in all_params:
            if self.last_layer_dim in _param.shape:
                last_layer_params += 1
                _param.grad = None
        assert last_layer_params == 4, "There should only be 4 last_layer params"

    def after_train_iter(self, runner):
        runner.iter_outputs['loss'] /= self.update_interval
        if self.use_fp16:
            with apex.amp.scale_loss(runner.iter_outputs['loss'], runner.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            runner.iter_outputs['loss'].backward()
        if self.every_n_iters(runner, self.update_interval):
            if self.grad_clip is not None:
                if not self.use_fp16:
                    self.clip_grads(runner.model.parameters())
                else:
                    clip_grad.clip_grad_norm_(
                            apex.amp.master_params(runner.optimizer), **self.grad_clip)
            self.cancel_gradients_last_layer(runner)
            runner.optimizer.step()
            runner.optimizer.zero_grad(set_to_none=True)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, neg_head=None):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp = self.teacher_temp_schedule[0]
        self.neg_head = neg_head
        if self.neg_head is not None:
            self.neg_head = builder.build_head(self.neg_head)

    @staticmethod
    def _create_buffer(N):
        neg_mask = 1 - torch.eye(N, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N).cuda(), torch.arange(N).cuda())
        return pos_ind, neg_mask

    def get_ce_loss(self, proj_onl, proj_tgt):
        proj_onl = torch.cat(GatherLayer.apply(proj_onl), dim=0)
        proj_tgt = torch.cat(GatherLayer.apply(proj_tgt), dim=0)
        s = torch.matmul(proj_onl, proj_tgt.permute(1, 0))
        pos_ind, neg_mask = self._create_buffer(proj_onl.size(0))
        positive = s[pos_ind].unsqueeze(1)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        loss = self.neg_head(positive, negative)['loss']
        return loss

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        #temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                if self.neg_head is None:
                    loss = torch.sum(
                            -q * F.log_softmax(student_out[v], dim=-1), 
                            dim=-1)
                else:
                    loss = self.get_ce_loss(
                            F.log_softmax(student_out[v], dim=-1), q)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINONegLoss(nn.Module):
    def __init__(self, ncrops, neg_head, nepochs):
        super().__init__()
        self.ncrops = ncrops
        self.neg_head = builder.build_head(neg_head)
        self.teacher_temp_schedule = [None] * nepochs
        self.teacher_temp = None

    @staticmethod
    def _create_buffer(N):
        neg_mask = 1 - torch.eye(N, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N).cuda(), torch.arange(N).cuda())
        return pos_ind, neg_mask

    def get_ce_loss(self, proj_onl, proj_tgt):
        proj_onl = torch.cat(GatherLayer.apply(proj_onl), dim=0)
        proj_tgt = torch.cat(GatherLayer.apply(proj_tgt), dim=0)
        s = torch.matmul(proj_onl, proj_tgt.permute(1, 0))
        pos_ind, neg_mask = self._create_buffer(proj_onl.size(0))
        positive = s[pos_ind].unsqueeze(1)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        loss = self.neg_head(positive, negative)['loss']
        return loss

    def forward(self, student_output, teacher_output):
        student_out = student_output.chunk(self.ncrops)
        teacher_out = teacher_output.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = self.get_ce_loss(student_out[v], q)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


@MODELS.register_module
class DINO(nn.Module):
    def __init__(
            self, dino_args, extract_last_n=4,
            dino_head_kwargs={}, neg_head=None, use_neg_loss=False):
        super().__init__()
        self.args = dino_args
        self.extract_last_n = extract_last_n
        self.dino_head_kwargs = dino_head_kwargs
        self.neg_head = neg_head
        self.use_neg_loss = use_neg_loss

        self.build_networks()

    def build_networks(self):
        args = self.args
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if args.arch in vits.__dict__.keys():
            self.student = vits.__dict__[args.arch](
                patch_size=args.patch_size,
                drop_path_rate=args.drop_path_rate,  # stochastic depth
            )
            self.teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
            embed_dim = self.student.embed_dim

        # multi-crop wrapper handles forward with inputs of different resolutions
        self.backbone = self.student
        self.student = utils.MultiCropWrapper(self.student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
            **self.dino_head_kwargs))
        self.teacher = utils.MultiCropWrapper(
            self.teacher,
            DINOHead(
                embed_dim, args.out_dim, args.use_bn_in_head,
                **self.dino_head_kwargs),
            )

        if utils.has_batchnorms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)
        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

        if not self.use_neg_loss:
            self.dino_loss = DINOLoss(
                args.out_dim,
                args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
                args.warmup_teacher_temp,
                args.teacher_temp,
                args.warmup_teacher_temp_epochs,
                args.epochs,
                neg_head=self.neg_head,
            )
        else:
            assert self.neg_head is not None
            self.dino_loss = DINONegLoss(
                args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
                self.neg_head,
                args.epochs,
            )

        self.momentum = args.momentum_teacher
        self.base_momentum = args.momentum_teacher

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the teacher network."""
        for param_ol, param_tgt in zip(self.student.parameters(),
                                       self.teacher.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    def forward_train(self, img, **kwargs):
        teacher_output = self.teacher(img[:2])  # only the 2 global views pass through the teacher
        student_output = self.student(img)
        loss = self.dino_loss(student_output, teacher_output)
        return {'loss': loss}

    def forward_extract(self, img):
        feat = self.backbone.get_intermediate_layers(
                img, self.extract_last_n)
        feat = torch.cat([x[:, 0] for x in feat], dim=-1)
        return [feat]

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'extract':
            return self.forward_extract(img)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class DINOLinearEval(nn.Module):
    def __init__(
            self, arch, patch_size, drop_path_rate,
            extract_last_n=4,
            head=None,
            pretrained=None):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.drop_path_rate = drop_path_rate
        self.extract_last_n = extract_last_n
        self.pretrained = pretrained

        if head is not None:
            self.head = builder.build_head(head)
        self.build_networks()

    def build_networks(self):
        if self.arch in vits.__dict__.keys():
            self.backbone = vits.__dict__[self.arch](
                patch_size=self.patch_size,
                drop_path_rate=self.drop_path_rate,  # stochastic depth
            )
        else:
            raise NotImplementedError

        if self.pretrained is not None:
            assert isinstance(self.pretrained, str)
            load_checkpoint(
                    self.backbone, self.pretrained, 
                    map_location='cpu')
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        feat = self.backbone.get_intermediate_layers(
                img, self.extract_last_n)
        feat = torch.cat([x[:, 0] for x in feat], dim=-1)
        feat = torch.unsqueeze(torch.unsqueeze(feat, -1), -1)
        return [feat]

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.forward_backbone(img)
        outs = self.head(x)
        loss_inputs = (outs, gt_label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, **kwargs):
        x = self.forward_backbone(img)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
