# Copyright (c) Open-MMLab. All rights reserved.
import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad
from .hook import Hook


class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad(set_to_none=True)
        runner.iter_outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                pass
                # Add grad norm to the logger
                #runner.log_buffer.update({'grad_norm': float(grad_norm)},
                #                         runner.outputs['num_samples'])
        runner.optimizer.step()


try:
    import apex
except:
    pass

class DistOptimizerHook(OptimizerHook):
    """Optimizer hook for distributed training."""

    def __init__(self, update_interval=1, grad_clip=None, coalesce=True, bucket_size_mb=-1, use_fp16=False):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16

    def before_run(self, runner):
        runner.optimizer.zero_grad(set_to_none=True)

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
            runner.optimizer.step()
            runner.optimizer.zero_grad(set_to_none=True)
