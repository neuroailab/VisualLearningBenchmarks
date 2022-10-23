import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import os
import os.path as osp
import pdb

from . import builder
from .registry import MODELS
from .utils import GatherLayer
from ..framework.hooks.hook import Hook


class QueueHook(Hook):
    def __init__(self, args, save_dir):
        self.args = args
        self.queue_path = os.path.join(
                save_dir, "queue" + str(args.rank) + ".pth")

    def before_run(self, runner):
        args = self.args
        queue = None
        if osp.isfile(self.queue_path):
            queue = torch.load(self.queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        args.queue_length -= args.queue_length % (args.batch_size * args.world_size)
        self.queue = queue
        runner.model.module.queue_hook = self

    def before_epoch(self, runner):
        queue = self.queue
        args = self.args

        # optionally starts a queue
        if args.queue_length > 0 \
                and runner.epoch >= args.epoch_queue_starts \
                and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()
        self.queue = queue
        self.use_the_queue = False

    def after_epoch(self, runner):
        queue = self.queue
        if queue is not None:
            torch.save(
                    {"queue": queue}, 
                    self.queue_path)


class OptimizerHook(Hook):
    def __init__(self, use_fp16, freeze_prototypes_niters):
        self.use_fp16 = use_fp16
        self.freeze_prototypes_niters = freeze_prototypes_niters

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        loss = runner.iter_outputs['loss']
        if self.use_fp16:
            with apex.amp.scale_loss(loss, runner.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # cancel some gradients
        if runner.iter < self.freeze_prototypes_niters:
            for name, p in runner.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        runner.optimizer.step()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


def distributed_sinkhorn(Q, nmb_iters, world_size):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (world_size * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            dist.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


@MODELS.register_module
class SwAV(nn.Module):
    def __init__(
            self, args, backbone, neck=None,
            nmb_prototypes=0, pretrained=None,
            ):
        super().__init__()
        self.args = args
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.loss_func = nn.Softmax(dim=1)

        output_dim = neck['out_channels']
        # prototype layer
        self.prototypes = None
        if nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
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

    def forward_head(self, x):
        x = self.neck([x])[0]
        x = nn.functional.normalize(x, dim=1, p=2)
        return x, self.prototypes(x)

    def get_embd_proto(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))[0]

            if start_idx == 0:
                output = _out
            else:
                output = torch.mean(
                        output, dim=(2, 3), keepdim=True)
                _out = torch.mean(
                        _out, dim=(2, 3), keepdim=True)
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)

    def reshape_input_if_needed(self, img):
        if isinstance(img, list):
            return img
        if img.dim() == 5:
            inputs = [img[:, _idx] for _idx in range(img.shape[1])]
            return inputs
        else:
            return img

    def forward_train(self, img, **kwargs):
        #inputs = img
        inputs = self.reshape_input_if_needed(img)
        args = self.args
        queue = self.queue_hook.queue

        # normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        embedding, output = self.get_embd_proto(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)]

                # time to use the queue
                if queue is not None:
                    if self.queue_hook.use_the_queue \
                            or not torch.all(queue[i, -1, :] == 0):
                        self.queue_hook.use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            self.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                # get assignments
                q = out / args.epsilon
                if args.improve_numerical_stability:
                    M = torch.max(q)
                    dist.all_reduce(M, op=dist.ReduceOp.MAX)
                    q -= M
                q = torch.exp(q).t()
                q = distributed_sinkhorn(
                        q, args.sinkhorn_iterations,
                        world_size=args.world_size)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                p = self.loss_func(output[bs * v: bs * (v + 1)] / args.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)
        self.queue_hook.queue = queue
        return {'loss': loss}

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
