import torch
import torch.nn as nn
from packaging import version
from mmcv.cnn import kaiming_init, normal_init
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from openselfsup.utils import get_root_logger
import os
import pdb

from .registry import NECKS
from .utils import build_norm_layer
from .necks import _init_weights


@NECKS.register_module
class NaiveMLP(nn.Module):
    def __init__(
            self, in_channels, hid_channels, out_channels,
            init_linear=None, act_func='tanh', pretrained=None):
        super().__init__()
        self.act_func = act_func
        if isinstance(hid_channels, int):
            hid_channels = [hid_channels]
        module_list = []
        _in_c = in_channels
        for _hid_c in hid_channels:
            module_list.append(nn.Linear(_in_c, _hid_c))
            module_list.append(self.get_act_func())
            _in_c = _hid_c
        module_list.append(nn.Linear(_in_c, out_channels))
        self.mlp =  nn.Sequential(*module_list)
        self.init_linear = init_linear
        self.pretrained = pretrained

    def get_act_func(self):
        if self.act_func=='relu':
            return nn.ReLU(inplace=True)
        elif self.act_func=='tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError

    def init_weights(self, init_linear='normal'):
        if self.pretrained is None:
            init_linear = self.init_linear or init_linear
            _init_weights(self, init_linear)
        else:
            assert isinstance(self.pretrained, str)
            logger = get_root_logger()
            load_checkpoint(
                    self, self.pretrained, 
                    strict=True, logger=logger)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class SparseMLP(NaiveMLP):
    def __init__(self, sparsity, *args, **kwargs):
        self.sparsity = sparsity
        super().__init__(*args, **kwargs)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        out = self.mlp(x.view(x.size(0), -1))
        mask = torch.zeros_like(out)
        _, indices = torch.topk(
                out, k=self.sparsity, dim=1, sorted=False)
        mask = mask.scatter(1, indices, 1)
        out = out.masked_fill(mask==0, 0)
        return [out]


@NECKS.register_module
class DynamicMLP2L(nn.Module):
    def __init__(
            self, in_channels, hid_channels, out_channels,
            init_linear=None, act_func='tanh', pretrained=None,
            learning_rate=1e-3):
        super().__init__()
        self.act_func = act_func
        self.nonlnrty = self.get_act_func()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels

        self.weights_0 = torch.nn.Parameter(
                torch.zeros(in_channels, hid_channels))
        self.bias_0 = torch.nn.Parameter(torch.zeros(hid_channels))
        self.weights_1 = torch.nn.Parameter(
                torch.zeros(hid_channels, out_channels))
        self.bias_1 = torch.nn.Parameter(torch.zeros(out_channels))

        self.param_names = [
                'weights_0', 'bias_0',
                'weights_1', 'bias_1',
                ]
        for _param in self.param_names:
            getattr(self, _param).requires_grad = True

        self.init_linear = init_linear
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.loss_logs = None

    def get_act_func(self):
        if self.act_func=='relu':
            return nn.ReLU(inplace=True)
        elif self.act_func=='tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError

    def init_weights(self, init_linear='normal'):
        if self.pretrained is None:
            init_linear = self.init_linear or init_linear
            _init_weights(self, init_linear)
        else:
            assert isinstance(self.pretrained, str)
            logger = get_root_logger()
            load_checkpoint(
                    self, self.pretrained, 
                    strict=True, logger=logger)

    def start_seq(self, batch_size):
        if self.loss_logs is not None:
            DEBUG = os.environ.get('DEBUG', '0')=='1'
            if DEBUG:
                #print(self.loss_logs[-1], self.loss_logs[0])
                pass

        self.init_d_params(batch_size)
        self.loss_logs = []
        self.backup_grad()

    def init_d_params(self, batch_size):
        for _name in self.param_names:
            curr_param = getattr(self, _name)
            new_name = 'd_' + _name
            setattr(self, new_name, torch.zeros_like(curr_param))

    def end_seq(self):
        self.recover_grad()

    def backup_grad(self):
        for _name in self.param_names:
            curr_param = getattr(self, _name)
            curr_param.grad_back = curr_param.grad
            curr_param.grad = None

    def recover_grad(self):
        for _name in self.param_names:
            curr_param = getattr(self, _name)
            curr_param.grad = curr_param.grad_back

    def compute_sim_loss(self, target_pat, curr_pat):
        target_pat = target_pat.detach().clone()
        target_pat = nn.functional.normalize(target_pat, dim=-1)
        curr_pat = nn.functional.normalize(curr_pat, dim=-1)
        loss = 1 - torch.mean(torch.sum(curr_pat * target_pat, dim=-1))
        return loss

    def update_wrt_loss(self, loss):
        self.zero_grad()
        self.loss_logs.append(loss)
        loss.backward()
        for _name in self.param_names:
            curr_param = getattr(self, _name)
            d_param = getattr(self, 'd_'+_name)
            d_param = d_param - curr_param.grad * self.learning_rate
            setattr(self, 'd_'+_name, d_param)
        self.zero_grad()

    def update(self, target_pat, curr_pat):
        loss = self.compute_sim_loss(target_pat, curr_pat)
        self.update_wrt_loss(loss)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        x = x.view(x.size(0), -1)
        x = torch.matmul(
                x, (self.weights_0+self.d_weights_0)) \
            + (self.bias_0+self.d_bias_0).unsqueeze(0)
        x = self.nonlnrty(x)
        x = torch.matmul(
                x, (self.weights_1+self.d_weights_1)) \
            + (self.bias_1+self.d_bias_1).unsqueeze(0)
        return [x]


@NECKS.register_module
class NStepDynamicMLP2L(DynamicMLP2L):
    def __init__(self, step_num, *args, **kwargs):
        self.step_num = step_num
        super().__init__(*args, **kwargs)

    def reset_step_losses(self):
        self.curr_losses = []

    def start_seq(self, batch_size):
        super().start_seq(batch_size)
        self.reset_step_losses()

    def end_seq(self):
        super().recover_grad()
        self.reset_step_losses()

    def update(self, target_pat, curr_pat):
        loss = self.compute_sim_loss(target_pat, curr_pat)
        self.curr_losses.append(loss)
        if len(self.curr_losses) == self.step_num:
            loss = sum(self.curr_losses) / self.step_num
            self.update_wrt_loss(loss)
            self.reset_step_losses()


@NECKS.register_module
class RepDynamicMLP2L(DynamicMLP2L):
    def __init__(self, rep_num, *args, **kwargs):
        self.rep_num = rep_num
        super().__init__(*args, **kwargs)

    def init_d_params(self, batch_size):
        assert batch_size // self.rep_num
        for _name in self.param_names:
            curr_param = getattr(self, _name)
            new_name = 'd_' + _name
            setattr(
                    self, new_name, 
                    torch.zeros(
                        [batch_size // self.rep_num,] + list(curr_param.size()),
                        dtype=curr_param.dtype, device=curr_param.device))

    def repeat_d_params(self, bs, d_param):
        d_param = d_param.unsqueeze(1)
        d_param = d_param.repeat(1, self.rep_num, *([1] * (len(d_param.size())-2)))
        d_param = d_param.reshape(-1, *(list(d_param.size())[2:]))
        return d_param

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        x = x.view(x.size(0), -1)
        bs = x.size(0)
        assert bs % self.rep_num == 0
        x = torch.einsum(
                'bpq, bqo->bpo',
                x.unsqueeze(1), 
                self.weights_0.unsqueeze(0) \
                    + self.repeat_d_params(bs, self.d_weights_0),
                ).squeeze(1) \
            + (self.bias_0.unsqueeze(0) + self.repeat_d_params(bs, self.d_bias_0))
        x = self.nonlnrty(x)
        x = torch.einsum(
                'bpq, bqo->bpo',
                x.unsqueeze(1), 
                self.weights_1.unsqueeze(0) \
                    + self.repeat_d_params(bs, self.d_weights_1),
                ).squeeze(1) \
            + (self.bias_1.unsqueeze(0) + self.repeat_d_params(bs, self.d_bias_1))
        return [x]

    def compute_sim_loss(self, target_pat, curr_pat):
        target_pat = target_pat.detach().clone()
        target_pat = nn.functional.normalize(target_pat, dim=-1)
        curr_pat = nn.functional.normalize(curr_pat, dim=-1)
        loss = 1 - torch.sum(curr_pat * target_pat, dim=-1)
        return loss

    def update_wrt_loss(self, loss):
        loss = loss.reshape(-1, self.rep_num)
        loss = torch.mean(loss, dim=-1)
        self.loss_logs.append(torch.mean(loss))
        for idx in range(loss.size(0)):
            self.zero_grad(set_to_none=True)
            loss[idx].backward(retain_graph=True)
            for _name in self.param_names:
                curr_param = getattr(self, _name)
                d_param = getattr(self, 'd_'+_name)
                d_param[idx] = d_param[idx] - curr_param.grad * self.learning_rate
                setattr(self, 'd_'+_name, d_param)


@NECKS.register_module
class NStepRepDynamicMLP2L(RepDynamicMLP2L):
    def __init__(self, step_num, *args, **kwargs):
        self.step_num = step_num
        super().__init__(*args, **kwargs)

    def reset_step_losses(self):
        self.curr_losses = []

    def start_seq(self, batch_size):
        super().start_seq(batch_size)
        self.reset_step_losses()

    def end_seq(self):
        super().recover_grad()
        self.reset_step_losses()

    def update(self, target_pat, curr_pat):
        loss = self.compute_sim_loss(target_pat, curr_pat)
        self.curr_losses.append(loss)
        if len(self.curr_losses) == self.step_num:
            loss = sum(self.curr_losses) / self.step_num
            self.update_wrt_loss(loss)
            self.reset_step_losses()


@NECKS.register_module
class SelfGradDynamicMLP2L(DynamicMLP2L):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.act_func == 'tanh'

    def init_d_params(self, batch_size):
        for _name in self.param_names:
            curr_param = getattr(self, _name)
            new_name = 'd_' + _name
            setattr(
                    self, new_name, 
                    torch.zeros(
                        [batch_size,] + list(curr_param.size()),
                        dtype=curr_param.dtype, device=curr_param.device))

    def get_forward_output_w_updates(self, x, update_target=None):
        assert len(x) == 1
        x = x[0]
        x = x.view(x.size(0), -1)
        bs = x.size(0)

        i = x
        w = self.weights_0.unsqueeze(0) + self.d_weights_0
        b = self.bias_0.unsqueeze(0) + self.d_bias_0
        w2 = self.weights_1.unsqueeze(0) + self.d_weights_1
        b2 = self.bias_1.unsqueeze(0) + self.d_bias_1

        tanh_out = self.nonlnrty(
                torch.einsum(
                    'bpq, bqo->bpo',
                    i.unsqueeze(1), w).squeeze(1) \
                + b)
        out2 = \
                torch.einsum(
                    'bpq, bqo->bpo',
                    tanh_out.unsqueeze(1), w2).squeeze(1) \
               + b2
        l2_norm = torch.sum(out2 ** 2, dim=-1, keepdim=True)
        l2_normed_out2 = out2 / l2_norm

        if update_target is None:
            return [l2_normed_out2], {}

        t = update_target.detach()
        t = nn.functional.normalize(t, dim=-1)

        self_grad_b2 = \
                -t / l2_norm \
                + torch.sum(t * out2, dim=-1, keepdim=True) / (l2_norm**2) \
                  * 2 * out2
        self_grad_w2 = torch.einsum(
                'bpq, bqo->bpo',
                tanh_out.unsqueeze(-1), self_grad_b2.unsqueeze(1))
        self_grad_b = \
                (1 - tanh_out**2) \
                * torch.einsum(
                    'bpq, bqo->bpo',
                    self_grad_b2.unsqueeze(1), w2.transpose(2, 1)
                    ).squeeze(1)
        self_grad_w = torch.einsum(
                'bpq, bqo->bpo',
                i.unsqueeze(-1), self_grad_b.unsqueeze(1))
        self_grads = dict(
                d_bias_1=self_grad_b2,
                d_weights_1=self_grad_w2,
                d_bias_0=self_grad_b,
                d_weights_0=self_grad_w,
                )
        return [l2_normed_out2], self_grads

    def do_update_weights_from_grads(self, self_grads):
        for _name in self.param_names:
            d_param = getattr(self, 'd_'+_name)
            d_param = d_param - self_grads.get('d_'+_name) * self.learning_rate
            setattr(self, 'd_'+_name, d_param)

    def forward(self, x, update_target=None):
        output, self_grads = self.get_forward_output_w_updates(x, update_target)
        if update_target is not None:
            self.do_update_weights_from_grads(self_grads)
        return output

    def compute_sim_loss(self, target_pat, curr_pat):
        raise NotImplementedError('Update within forward!')

    def update_wrt_loss(self, loss):
        raise NotImplementedError('Update within forward!')

    def update(self, target_pat, curr_pat):
        raise NotImplementedError('Update within forward!')


@NECKS.register_module
class MomentumSGDynamicMLP2L(SelfGradDynamicMLP2L):
    def __init__(self, momentum=0.9, *args, **kwargs):
        self.momentum = momentum
        super().__init__(*args, **kwargs)

    def init_d_params(self, batch_size):
        super().init_d_params(batch_size)
        for _name in self.param_names:
            curr_param = getattr(self, _name)
            new_name = 'mtm_' + _name
            setattr(
                    self, new_name, 
                    torch.zeros(
                        [batch_size,] + list(curr_param.size()),
                        dtype=curr_param.dtype, device=curr_param.device))

    def do_update_weights_from_grads(self, self_grads):
        for _name in self.param_names:
            mtm_param = getattr(self, 'mtm_'+_name)
            mtm_param = self.momentum * mtm_param + self_grads.get('d_'+_name)
            d_param = getattr(self, 'd_'+_name)
            d_param = d_param - mtm_param * self.learning_rate
            setattr(self, 'd_'+_name, d_param)
            setattr(self, 'mtm_'+_name, mtm_param)
