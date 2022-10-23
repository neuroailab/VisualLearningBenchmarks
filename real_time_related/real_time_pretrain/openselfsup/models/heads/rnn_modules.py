import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
from openselfsup.framework.dist_utils import get_dist_info

from ..moco import concat_all_gather
from .. import builder
from .early_rnn_modules import *


class NaivePatSoftmaxRNN(nn.Module):
    def __init__(
            self, input_size, hidden_size, pattern_size,
            pat_init_norm=0.01):
        """"Constructor of the class"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pattern_size = pattern_size
        self.pat_init_norm = pat_init_norm
        self.setup_modules()
        self.reset_parameters()
        self.setup_other_params()

    def setup_modules(self):
        self.softmax = nn.Softmax(dim=1)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def setup_other_params(self):
        self.decay = torch.nn.Parameter(torch.tensor(0.999))
        self.decay.requires_grad = False
        self.update_rate = torch.nn.Parameter(torch.tensor(1.0))
        self.update_rate.requires_grad = False

    def l2_normalize(self, z):
        z = z / (torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-10)
        return z

    def get_init_states(self, input):
        init_pat = self.pat_init_norm * torch.randn(
                input.size(1),
                self.pattern_size, self.hidden_size,
                dtype=input.dtype, device=input.device)
        init_pat = self.l2_normalize(init_pat)
        return init_pat

    def before_seq_setup(self, input, pat):
        if pat is None:
            pat = self.get_init_states(input)
        return pat

    def forward(self, input, pat=None):
        pat = self.before_seq_setup(input, pat)
        num_seq = input.size(0)
        outputs = []
        all_pats = []
        for idx in range(num_seq):
            hx, pat = self.forward_one_time(
                    input[idx], pat)
            outputs.append(hx)
            all_pats.append(pat)
        outputs = torch.stack(outputs, 0)
        return outputs, all_pats

    def get_softmax_resp(self, pat_resp):
        pat_resp_max, _ = torch.max(pat_resp, dim=1, keepdim=True)
        mask = pat_resp.ge(pat_resp_max*torch.tensor(0.9))
        pat_resp = pat_resp * mask
        pat_resp = pat_resp / pat_resp_max * torch.tensor(10.0)
        pat_resp = self.softmax(pat_resp)
        return pat_resp

    def get_up_pat(self, h, new_h, pat, pat_resp):
        up_pat = torch.matmul(
                pat_resp.unsqueeze(-1),
                h.unsqueeze(1))
        new_pat = self.decay * pat + self.update_rate * up_pat
        new_pat = self.l2_normalize(new_pat)
        return new_pat

    def get_hidden(self, input):
        return input

    def get_new_h(self, pat, pat_resp, raw_pat_resp):
        new_h = torch.matmul(
                pat.transpose(1,2), 
                pat_resp.unsqueeze(-1)).squeeze(-1)
        return new_h

    def forward_one_time(self, input, pat):
        self.curr_pat_mat = pat
        h = self.get_hidden(input)
        raw_pat_resp = torch.matmul(pat, h.unsqueeze(-1)).squeeze(-1)
        pat_resp = self.get_softmax_resp(raw_pat_resp)
        new_h = self.get_new_h(pat, pat_resp, raw_pat_resp)
        new_pat = self.get_up_pat(h, new_h, pat, pat_resp)
        return new_h, new_pat


class MaxNorm(nn.Module):
    def forward(self, pat_resp):
        pat_resp_max, _ = torch.max(pat_resp, dim=1, keepdim=True)
        pat_resp_normed = pat_resp / pat_resp_max
        return pat_resp_normed


class ConcatMaxNorm(nn.Module):
    def forward(self, pat_resp):
        pat_resp_max, _ = torch.max(pat_resp, dim=1, keepdim=True)
        pat_resp_normed = pat_resp / pat_resp_max
        return torch.cat([pat_resp, pat_resp_normed], 1)


class ConcatReLU(nn.Module):
    def __init__(self, pattern_size, gate_out_size):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.threshold = torch.nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(pattern_size*2, gate_out_size)

    def forward(self, raw_pat_resp):
        relu_th_pr = self.relu(raw_pat_resp - self.threshold)
        relu_th_pr = relu_th_pr + self.threshold * torch.greater(relu_th_pr, 0)
        new_pr = torch.cat([raw_pat_resp, relu_th_pr], 1)
        return self.linear(new_pr)


class ConcatMax(nn.Module):
    def __init__(self, pattern_size, gate_out_size, max_num):
        super().__init__()
        self.max_num = max_num
        self.linear = nn.Linear(pattern_size + max_num, gate_out_size)

    def forward(self, raw_pat_resp):
        max_pr = torch.topk(raw_pat_resp, self.max_num, dim=1, sorted=True)
        new_pr = torch.cat([raw_pat_resp, max_pr[0]], 1)
        return self.linear(new_pr)


class ConcatReLUMax(nn.Module):
    def __init__(self, pattern_size, gate_out_size, max_num):
        super().__init__()
        self.max_num = max_num
        self.relu = nn.ReLU(inplace=True)
        self.threshold = torch.nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(
                (pattern_size+max_num)*2, gate_out_size)

    def forward(self, raw_pat_resp):
        max_pr = torch.topk(raw_pat_resp, self.max_num, dim=1, sorted=True)
        new_pr = torch.cat([raw_pat_resp, max_pr[0]], 1)
        relu_th_pr = self.relu(new_pr - self.threshold)
        relu_th_pr = relu_th_pr + self.threshold * torch.greater(relu_th_pr, 0)
        new_pr = torch.cat([new_pr, relu_th_pr], 1)
        return self.linear(new_pr)


class ResMLP(nn.Module):
    def __init__(self, pattern_size, mlp_layers, act='tanh'):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise NotImplementedError
        modules = []
        for _ in range(mlp_layers):
            modules.append(
                    nn.Linear(pattern_size, pattern_size))
        self.lin_modules = nn.ModuleList(modules)

    def forward(self, raw_pat_resp):
        new_pr = raw_pat_resp
        for idx in range(len(self.lin_modules)):
            new_pr = self.lin_modules[idx](new_pr)
            if idx < len(self.lin_modules) - 1:
                new_pr = self.act(new_pr)
        new_pr = self.relu(new_pr + raw_pat_resp)
        return new_pr


class GnrlPatSoftmaxRNN(NaivePatSoftmaxRNN):
    '''
    To release hand coded procedures and parameters
    '''
    GATE_SETTINGS = ['gate_two_mlps', 'gate_mlp_stpmlp', 'sim_gate_two_mlps']
    def __init__(
            self, hand_coded_softmax=True, 
            naive_hidden=True,
            parallel_mmax=None,
            parallel_use_newh=False,
            newh_from_parallel_mmax=False,
            use_stpmlp=None,
            gate_update=False,
            stpmlp_kwargs={},
            new_h_cat_input=False,
            with_layernorm=False,
            act_func='relu',
            max_norm=None,
            state_pass_method='mean_tile',
            pr_gate_method=None,
            max_mlp_layers=2,
            noisy_decay=None,
            sync_update=True,
            pt_queue_len=10240,
            **kwargs):
        assert not with_layernorm, "Layernorm deprecated"
        self.hand_coded_softmax = hand_coded_softmax
        self.naive_hidden = naive_hidden
        self.parallel_mmax = parallel_mmax
        self.parallel_use_newh = parallel_use_newh
        self.use_stpmlp = use_stpmlp
        self.gate_update = gate_update
        self.stpmlp_kwargs = stpmlp_kwargs
        self.new_h_cat_input = new_h_cat_input
        self.newh_from_parallel_mmax = newh_from_parallel_mmax
        self.act_func = act_func
        self.max_norm = max_norm
        self.state_pass_method = state_pass_method
        self.pr_gate_method = pr_gate_method
        self.max_mlp_layers = max_mlp_layers
        self.noisy_decay = noisy_decay
        self.sync_update = sync_update
        self.pt_queue_len = pt_queue_len

        super().__init__(**kwargs)
        if self.state_pass_method == 'sample_from_q':
            self.setup_pt_queue()

    def setup_pt_queue(self):
        self.register_buffer(
                "pt_queue",
                torch.randn(
                    self.pt_queue_len, self.pattern_size, self.hidden_size))
        self.pt_queue = self.l2_normalize(self.pt_queue)
        self.register_buffer(
                "pt_queue_ptr", torch.zeros(1, dtype=torch.long))
        from openselfsup.utils import AliasMethod
        self.multinomial = AliasMethod(torch.ones(self.pt_queue_len))
        self.multinomial.cuda()

    def get_activation_func(self):
        if self.act_func=='relu':
            return nn.ReLU(inplace=True)
        elif self.act_func=='tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError

    def get_one_mlp_for_max(self):
        if isinstance(self.max_norm, dict):
            return builder.build_neck(self.max_norm)
        if self.max_norm is 'res_mlp':
            return ResMLP(self.pattern_size, self.max_mlp_layers)
        modules = []
        for idx in range(self.max_mlp_layers):
            modules.append(nn.Linear(self.pattern_size, self.pattern_size))
            if idx < self.max_mlp_layers-1:
                modules.append(self.get_activation_func())
        modules.append(nn.ReLU(inplace=True))
        if self.max_norm is 'max':
            modules.insert(0, MaxNorm())
        elif self.max_norm is 'concat_max':
            del modules[0]
            modules.insert(0, ConcatMaxNorm())
            modules.insert(
                    1, nn.Linear(2*self.pattern_size, self.pattern_size))
        else:
            assert self.max_norm is None
        mlp = nn.Sequential(*modules)
        return mlp

    def get_one_PRgate(self):
        if self.pr_gate_method is None:
            return nn.Linear(self.pattern_size, self.gate_out_size)
        elif self.pr_gate_method=='relu_th':
            return ConcatReLU(self.pattern_size, self.gate_out_size)
        elif self.pr_gate_method=='cat_max10':
            return ConcatMax(self.pattern_size, self.gate_out_size, 10)
        elif self.pr_gate_method=='relu_th_cat_max10':
            return ConcatReLUMax(self.pattern_size, self.gate_out_size, 10)
        elif self.pr_gate_method=='mlp2':
            modules = [
                    nn.Linear(self.pattern_size, self.pattern_size),
                    self.get_activation_func(),
                    nn.Linear(self.pattern_size, self.gate_out_size),
                    ]
            return nn.Sequential(*modules)
        else:
            raise NotImplementedError

    def get_one_mlp_for_hidden(self, input_size):
        mlp_hidden =  nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                self.get_activation_func(),
                nn.Linear(self.hidden_size, self.hidden_size),
                )
        return mlp_hidden

    def setup_max_modules(self):
        self.softmax = nn.Softmax(dim=1)
        if not self.hand_coded_softmax:
            self.mlp_softmax = self.get_one_mlp_for_max()
            if self.parallel_mmax is not None:
                mlps = []
                for _ in range(self.parallel_mmax):
                    mlps.append(self.get_one_mlp_for_max())
                self.other_softmax_mlps = nn.ModuleList(mlps)
            if self.newh_from_parallel_mmax:
                self.newh_mlp_softmax = self.get_one_mlp_for_max()

    def setup_update_modules(self):
        self.pat_mat = torch.nn.Parameter(
                torch.zeros(self.pattern_size, self.pattern_size))
        self.pat_mat.requires_grad = True
        self.hid_mat = torch.nn.Parameter(
                torch.zeros(self.hidden_size, self.hidden_size))
        self.hid_mat.requires_grad = True

        if self.gate_update:
            self.pat_resp_to_gate = nn.Linear(
                    self.pattern_size, self.pattern_size)
            self.hidden_to_gate = nn.Linear(
                    self.hidden_size, self.pattern_size)

    def setup_PR_gates(self):
        self.gate_out_size = self.hidden_size
        if self.use_stpmlp == 'sim_gate_two_mlps':
            self.gate_out_size = 1
        self.gate_pat_resp_to_hid = self.get_one_PRgate()
        self.gate_pat_resp_to_hid_2 = self.get_one_PRgate()
        self.gate_pat_resp_to_hid_re = self.get_one_PRgate()
        self.gate_pat_resp_to_hid_re_2 = self.get_one_PRgate()

    def setup_hidden_modules(self):
        if self.new_h_cat_input:
            self.input_size += self.hidden_size

        if not self.naive_hidden:
            if self.use_stpmlp is None:
                self.mlp_hidden = self.get_one_mlp_for_hidden(self.input_size)
            elif self.use_stpmlp == 'one_layer_io':
                assert self.parallel_mmax is None
                self.inp_stpmlp = STPMLP(
                        self.input_size, self.hidden_size, 
                        **self.stpmlp_kwargs)
                self.out_stpmlp = STPMLP(
                        self.hidden_size, self.hidden_size,
                        **self.stpmlp_kwargs)
            elif self.use_stpmlp == 'one_sim_layer_io':
                assert self.parallel_mmax is None
                self.inp_stpmlp = SimSTPMLP(
                        self.input_size, self.hidden_size,
                        **self.stpmlp_kwargs)
                self.out_stpmlp = SimSTPMLP(
                        self.hidden_size, self.hidden_size,
                        **self.stpmlp_kwargs)
            elif self.use_stpmlp == 'one_lstm_layer_io':
                assert self.parallel_mmax is None
                self.inp_stpmlp = SelfLSTM(
                        self.input_size, self.hidden_size, 
                        num_layers=1,
                        **self.stpmlp_kwargs)
                self.out_stpmlp = SelfLSTM(
                        self.hidden_size, self.hidden_size,
                        num_layers=1,
                        **self.stpmlp_kwargs)
            elif self.use_stpmlp in self.GATE_SETTINGS:
                self.mlp_hidden = self.get_one_mlp_for_hidden(self.input_size)
                self.mlp_hidden_recover = self.get_one_mlp_for_hidden(
                        self.hidden_size)

                if self.use_stpmlp == 'gate_mlp_stpmlp':
                    self.inp_stpmlp = STPMLP(
                            self.input_size, self.hidden_size, 
                            **self.stpmlp_kwargs)
                    self.out_stpmlp = STPMLP(
                            self.hidden_size, self.hidden_size,
                            **self.stpmlp_kwargs)
                else:
                    self.mlp_hidden_2 = self.get_one_mlp_for_hidden(
                            self.input_size)
                    self.mlp_hidden_recover_2 = self.get_one_mlp_for_hidden(
                            self.hidden_size)
                self.setup_PR_gates()
            else:
                raise NotImplementedError

    def setup_modules(self):
        self.setup_max_modules()
        self.setup_update_modules()
        self.setup_hidden_modules()

    def get_passed_states(self, input, pat):
        if self.state_pass_method == 'mean_tile':
            pat = torch.mean(pat, dim=0, keepdim=True)
            pat = pat.repeat(input.size(1), 1, 1)
            pat = self.l2_normalize(pat)
        elif self.state_pass_method == 'raw_pass':
            assert pat.size(0) == input.size(1)
        elif self.state_pass_method == 'sample_from_q':
            _idx = self.multinomial.draw(input.size(1))
            selected_queue = torch.index_select(self.pt_queue, 0, _idx)
            self._dequeue_and_enqueue(pat)
            pat = selected_queue
        else:
            raise NotImplementedError
        pat = pat.clone().detach()
        return pat

    @torch.no_grad()
    def _dequeue_and_enqueue(self, pat):
        """Update queue."""
        # gather keys before updating queue
        _, world_size = get_dist_info()
        if self.sync_update and world_size > 1:
            pat = concat_all_gather(pat)
        batch_size = pat.shape[0]
        #assert self.pt_queue_len % batch_size == 0, \
        #        "%i, %i" % (self.queue_len, batch_size)  # for simplicity
        
        ptr = int(self.pt_queue_ptr)
        end_up = min(ptr + batch_size, self.pt_queue_len)
        self.pt_queue[ptr:end_up] = pat[:(end_up-ptr)]
        ptr = end_up % self.pt_queue_len
        self.pt_queue_ptr[0] = ptr

    def before_seq_setup(self, input, pat):
        if pat is None:
            pat = self.get_init_states(input)
        else:
            pat = self.get_passed_states(input, pat)

        if self.use_stpmlp in [
                'one_layer_io', 'one_sim_layer_io', 'gate_mlp_stpmlp']:
            self.inp_stpmlp.init_stp(input.size(1))
            self.out_stpmlp.init_stp(input.size(1))
        elif self.use_stpmlp == 'one_lstm_layer_io':
            self.inp_hid = self.inp_stpmlp.get_zero_states(input)
            self.out_hid = self.out_stpmlp.get_zero_states(input)

        if self.new_h_cat_input:
            self.last_new_h = torch.zeros(
                    input.size(1), self.hidden_size).cuda()
        return pat

    def get_hidden(self, input):
        if self.new_h_cat_input:
            input = torch.cat([input, self.last_new_h], -1)
        if self.naive_hidden:
            return input
        else:
            if self.use_stpmlp is None:
                h = self.mlp_hidden(input)
                h = self.l2_normalize(h)
            elif self.use_stpmlp in ['one_layer_io', 'one_sim_layer_io']:
                h = self.inp_stpmlp(input)
                h = self.l2_normalize(h)
            elif self.use_stpmlp == 'one_lstm_layer_io':
                self.inp_hid = self.inp_stpmlp.forward_one_time(
                        input, self.inp_hid)
                h = self.inp_hid[0][-1]
                h = self.l2_normalize(h)
            elif self.use_stpmlp in self.GATE_SETTINGS:
                h = self.mlp_hidden(input)
                h = self.l2_normalize(h)
                self.raw_pat_resp_for_gate = torch.matmul(
                        self.curr_pat_mat, h.unsqueeze(-1)).squeeze(-1)
                if self.use_stpmlp == 'gate_mlp_stpmlp':
                    h_2 = self.inp_stpmlp(input)
                else:
                    h_2 = self.mlp_hidden_2(input)
                h_2 = self.l2_normalize(h_2)
                _gate = torch.sigmoid(
                        self.gate_pat_resp_to_hid(
                            self.raw_pat_resp_for_gate))
                _gate_2 = torch.sigmoid(
                        self.gate_pat_resp_to_hid_2(
                            self.raw_pat_resp_for_gate))
                h = _gate * h + _gate_2 * h_2
                h = self.l2_normalize(h)
            return h

    def get_softmax_resp(self, pat_resp):
        if self.hand_coded_softmax:
            pat_resp = super().get_softmax_resp(pat_resp)
        else:
            pat_resp = self.mlp_softmax(pat_resp)
        return pat_resp

    def get_new_h(self, pat, pat_resp, raw_pat_resp):
        new_h = super().get_new_h(pat, pat_resp, raw_pat_resp)
        if self.parallel_mmax is not None:
            all_new_h = [new_h]
            if self.parallel_use_newh:
                raw_pat_resp = torch.matmul(
                        pat, new_h.unsqueeze(-1)).squeeze(-1)
            for idx in range(self.parallel_mmax):
                curr_pat_resp = self.other_softmax_mlps[idx](raw_pat_resp)
                all_new_h.append(
                        super().get_new_h(
                            pat, curr_pat_resp, raw_pat_resp))
            new_h = torch.cat(all_new_h, -1)
        if self.newh_from_parallel_mmax:
            curr_pat_resp = self.newh_mlp_softmax(raw_pat_resp)
            new_h = super().get_new_h(
                    pat, curr_pat_resp, raw_pat_resp)
        if self.use_stpmlp in ['one_layer_io', 'one_sim_layer_io']:
            new_h = self.out_stpmlp(new_h)
        elif self.use_stpmlp == 'one_lstm_layer_io':
            self.out_hid = self.out_stpmlp.forward_one_time(
                    new_h, self.out_hid)
            new_h = self.out_hid[0][-1]
        elif self.use_stpmlp in self.GATE_SETTINGS:
            _gate = torch.sigmoid(
                    self.gate_pat_resp_to_hid_re(
                        self.raw_pat_resp_for_gate))
            _gate_2 = torch.sigmoid(
                    self.gate_pat_resp_to_hid_re_2(
                        self.raw_pat_resp_for_gate))
            if self.use_stpmlp == 'gate_mlp_stpmlp':
                out2 = self.out_stpmlp(new_h)
            else:
                out2 = self.mlp_hidden_recover_2(new_h)
            new_h = self.mlp_hidden_recover(new_h) * _gate + out2 * _gate_2
        if self.new_h_cat_input:
            self.last_new_h = new_h
        return new_h

    def get_up_pat(self, h, new_h, pat, pat_resp):
        if self.gate_update:
            pat_gate = torch.sigmoid(
                    self.pat_resp_to_gate(pat_resp)\
                    + self.hidden_to_gate(h))
            pat_resp = pat_resp * pat_gate
        up_pat = torch.matmul(
                pat_resp.unsqueeze(-1),
                h.unsqueeze(1))
        up_pat = torch.matmul(
                self.pat_mat.unsqueeze(0),
                torch.matmul(up_pat, self.hid_mat.unsqueeze(0)))
        if self.noisy_decay is None:
            new_pat = self.decay * pat + self.update_rate * up_pat
        else:
            new_pat = pat \
                    + self.noisy_decay * torch.randn(
                        pat.size(0), pat.size(1), pat.size(2),
                        dtype=pat.dtype, device=pat.device) \
                    + self.update_rate * up_pat
        new_pat = self.l2_normalize(new_pat)
        return new_pat


class MultiSoftmaxMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

        self.mix_softmax = torch.nn.Parameter(
                torch.ones(1, 40))
        self.mix_softmax.requires_grad = True
        self.softmax_taus = np.concatenate(
                [np.arange(0.1, 1, 0.1), np.arange(1, 22, 2)])

    def forward(self, pat_resp):
        all_resps = []
        pat_resp_max, _ = torch.max(pat_resp, dim=1, keepdim=True)
        pat_resp_normed = pat_resp / pat_resp_max
        for tau in self.softmax_taus:
            each_resp = pat_resp * tau
            all_resps.append(each_resp)

            each_resp = pat_resp_normed * tau
            all_resps.append(each_resp)
             
        all_resps = torch.stack(all_resps, 1)
        pat_resp = torch.matmul(
                self.mix_softmax.unsqueeze(0),
                all_resps).squeeze(1)
        pat_resp = self.relu(pat_resp)
        return pat_resp


class MultiSoftmaxGnrlPatRNN(GnrlPatSoftmaxRNN):
    def setup_max_modules(self):
        assert not self.hand_coded_softmax
        self.softmax = nn.Softmax(dim=1)
        self.mix_softmax = MultiSoftmaxMix()

        if self.parallel_mmax is not None:
            raise NotImplementedError
        if self.newh_from_parallel_mmax:
            self.newh_mlp_softmax = MultiSoftmaxMix()

    def get_softmax_resp(self, pat_resp):
        pat_resp = self.mix_softmax(pat_resp)
        return pat_resp


class SimGnrlPatSoftmaxRNN(GnrlPatSoftmaxRNN):
    def setup_update_modules(self):
        if self.gate_update:
            self.pat_resp_to_gate = nn.Linear(
                    self.pattern_size, self.pattern_size)
            self.hidden_to_gate = nn.Linear(
                    self.hidden_size, self.pattern_size)

    def setup_other_params(self):
        super().setup_other_params()
        self.pat_mat = torch.nn.Parameter(torch.eye(self.pattern_size))
        self.pat_mat.requires_grad = True
        self.hid_mat = torch.nn.Parameter(torch.eye(self.hidden_size))
        self.hid_mat.requires_grad = True

    def get_up_pat(self, h, new_h, pat, pat_resp):
        if self.gate_update:
            pat_gate = torch.sigmoid(
                    self.pat_resp_to_gate(pat_resp)\
                    + self.hidden_to_gate(h))
            pat_resp = pat_resp * pat_gate
        up_pat = torch.matmul(
                pat_resp.unsqueeze(-1),
                h.unsqueeze(1))
        up_pat = torch.matmul(
                self.pat_mat.unsqueeze(0),
                torch.matmul(
                    up_pat, self.hid_mat.unsqueeze(0)))
        new_pat = self.decay * pat + self.update_rate * up_pat
        new_pat = self.l2_normalize(new_pat)
        return new_pat


class GateSepHPathRNN(GnrlPatSoftmaxRNN):
    def __init__(self, use_stpmlp, **kwargs):
        assert use_stpmlp in ['gate_two_mlps', 'sim_gate_two_mlps']
        super().__init__(use_stpmlp=use_stpmlp, **kwargs)
        assert not self.new_h_cat_input
        assert self.newh_from_parallel_mmax

    def setup_max_modules(self):
        assert not self.hand_coded_softmax
        self.softmax = nn.Softmax(dim=1)
        self.mlp_softmax = self.get_one_mlp_for_max()
        self.newh_mlp_softmax = self.get_one_mlp_for_max()
        self.newh_mlp_softmax_2 = self.get_one_mlp_for_max()
        if self.parallel_mmax is not None:
            raise NotImplementedError

    def forward(self, input, pat=None):
        self.all_pat_resps = []
        pat = self.before_seq_setup(input, pat)
        num_seq = input.size(0)
        all_pats = []
        for idx in range(num_seq-1):
            pat = self.forward_one_time(input[idx], pat)
            all_pats.append(pat)
        hx = self.forward_for_sep_h(input[-1], pat)
        return [hx], all_pats

    def forward_pred_all(self, input, pat=None):
        self.all_pat_resps = []
        pat = self.before_seq_setup(input, pat)
        num_seq = input.size(0)
        all_pats = []
        hxs = []
        for idx in range(num_seq):
            hx = self.forward_for_sep_h(input[idx], pat)
            pat = self.forward_one_time(input[idx], pat)
            hxs.append(hx)
            all_pats.append(pat)
        return hxs, all_pats

    def forward_one_time(self, input, pat):
        self.curr_pat_mat = pat
        h = self.get_hidden(input)
        raw_pat_resp = torch.matmul(pat, h.unsqueeze(-1)).squeeze(-1)
        pat_resp = self.get_softmax_resp(raw_pat_resp)
        self.all_pat_resps.append(pat_resp)
        new_pat = self.get_up_pat(h, None, pat, pat_resp)
        return new_pat

    def forward_for_sep_h(self, input, pat):
        h = self.mlp_hidden(input)
        h = self.l2_normalize(h)
        raw_pat_resp = torch.matmul(pat, h.unsqueeze(-1)).squeeze(-1)
        curr_pat_resp = self.newh_mlp_softmax(raw_pat_resp)
        new_h = NaivePatSoftmaxRNN.get_new_h(
                self, pat, curr_pat_resp, raw_pat_resp)

        h_2 = self.mlp_hidden_2(input)
        h_2 = self.l2_normalize(h_2)
        raw_pat_resp_2 = torch.matmul(pat, h_2.unsqueeze(-1)).squeeze(-1)
        curr_pat_resp_2 = self.newh_mlp_softmax_2(raw_pat_resp_2)
        new_h_2 = NaivePatSoftmaxRNN.get_new_h(
                self, pat, curr_pat_resp_2, raw_pat_resp_2)

        _gate = torch.sigmoid(self.gate_pat_resp_to_hid_re(raw_pat_resp))
        _gate_2 = torch.sigmoid(self.gate_pat_resp_to_hid_re_2(raw_pat_resp_2))
        new_h = _gate * new_h + _gate_2 * new_h_2
        return new_h


class FWGateSepHPathRNN(GateSepHPathRNN):
    def __init__(self, keep_fw=0.996, *args, **kwargs):
        self.keep_fw = keep_fw
        super().__init__(*args, **kwargs)

    def get_hidden(self, input):
        hidden = super().get_hidden(input)
        self.update_fast_weight(input, hidden)
        return hidden

    def update_fast_weight(self, input, hidden):
        if isinstance(self.keep_fw, float):
            self.fast_weight = self.fast_weight * self.keep_fw
        else:
            raise NotImplementedError
        _update = torch.einsum('bp, bq->bpq', input, hidden)
        self.fast_weight = self.fast_weight + _update

    def get_init_fast_weight(self, input, fast_weight):
        if fast_weight is not None:
            self.fast_weight = fast_weight
        else:
            self.fast_weight = torch.zeros(
                    input.size(1), input.size(2), self.hidden_size,
                    ).cuda()

    def before_seq_setup_wfw(self, input, pat, fast_weight):
        pat = self.before_seq_setup(input, pat)
        self.get_init_fast_weight(input, fast_weight)
        return pat

    def forward(self, input, pat=None, fast_weight=None):
        self.all_pat_resps = []
        pat = self.before_seq_setup_wfw(input, pat, fast_weight)
        num_seq = input.size(0)
        all_pats = []
        for idx in range(num_seq-1):
            pat = self.forward_one_time(input[idx], pat)
            all_pats.append(pat)
        hx = self.forward_for_sep_h(input[-1], pat)
        return ([hx], self.fast_weight), all_pats


class FxEcGateSepHPathRNN(GateSepHPathRNN):
    def __init__(
            self, ca3_cfg=None, dg_cfg=None, fix_ca3_dg_weights=True,
            *args, **kwargs):
        self.ca3_cfg = ca3_cfg
        self.dg_cfg = dg_cfg
        self.fix_ca3_dg_weights = fix_ca3_dg_weights
        super().__init__(*args, **kwargs)
        self.init_mlp_fix_weights()

    def init_mlp_fix_weights(self):
        if self.ca3_cfg is None and self.dg_cfg is None:
            return
        if self.fix_ca3_dg_weights:
            for param in self.mlp_hidden.parameters():
                param.requires_grad = False
            for param in self.mlp_hidden_2.parameters():
                param.requires_grad = False
        self.mlp_hidden.init_weights()
        self.mlp_hidden_2.init_weights()

    def setup_hidden_modules(self):
        if self.ca3_cfg is None and self.dg_cfg is None:
            super().setup_hidden_modules()
            return
        if self.new_h_cat_input:
            self.input_size += self.hidden_size

        self.mlp_hidden = builder.build_model(self.ca3_cfg)
        self.mlp_hidden_2 = builder.build_model(self.dg_cfg)

        self.mlp_hidden_recover = self.get_one_mlp_for_hidden(
                self.hidden_size)
        self.mlp_hidden_recover_2 = self.get_one_mlp_for_hidden(
                self.hidden_size)
        self.setup_PR_gates()


class DGCA3GateSepHPathRNN(GateSepHPathRNN):
    def __init__(
            self, ca3_cfg=None, dg_cfg=None, fix_ca3_dg_weights=False,
            *args, **kwargs):
        self.ca3_cfg = ca3_cfg
        self.dg_cfg = dg_cfg
        self.fix_ca3_dg_weights = fix_ca3_dg_weights
        super().__init__(*args, **kwargs)
        self.init_dg_ca3()

    def init_dg_ca3(self):
        if self.ca3_cfg is None and self.dg_cfg is None:
            return
        if self.fix_ca3_dg_weights:
            for param in self.mlp_hidden.parameters():
                param.requires_grad = False
            for param in self.mlp_hidden_2.parameters():
                param.requires_grad = False
        self.dg_mlp.init_weights()
        self.ca3_mlp.init_weights()

    def setup_hidden_modules(self):
        if self.ca3_cfg is None and self.dg_cfg is None:
            super().setup_hidden_modules()
            return
        self.dg_mlp = builder.build_model(self.dg_cfg)
        self.ca3_mlp = builder.build_model(self.ca3_cfg)

    def setup_max_modules(self):
        assert not self.hand_coded_softmax
        self.softmax = nn.Softmax(dim=1)
        self.mlp_softmax = self.get_one_mlp_for_max()
        self.newh_mlp_softmax = self.get_one_mlp_for_max()
        if self.parallel_mmax is not None:
            raise NotImplementedError

    def forward(self, input, pat=None):
        self.ca3_mimic_dg_losses = []
        return super().forward(input, pat)

    def get_hidden(self, input):
        h = self.dg_mlp(input)
        h = self.l2_normalize(h)
        h_ca3 = self.ca3_mlp(input)
        h_ca3 = self.l2_normalize(h_ca3)
        _loss = 1 - torch.mean(torch.sum(h_ca3 * h, dim=-1))
        self.ca3_mimic_dg_losses.append(_loss)
        return h

    def get_additional_loss(self):
        return sum(self.ca3_mimic_dg_losses)/len(self.ca3_mimic_dg_losses)

    def forward_for_sep_h(self, input, pat):
        h = self.ca3_mlp(input)
        h = self.l2_normalize(h)
        raw_pat_resp = torch.matmul(pat, h.unsqueeze(-1)).squeeze(-1)
        curr_pat_resp = self.newh_mlp_softmax(raw_pat_resp)
        new_h = NaivePatSoftmaxRNN.get_new_h(
                self, pat, curr_pat_resp, raw_pat_resp)
        return new_h


class DGDynCA3GSepHPRNN(DGCA3GateSepHPathRNN):
    def forward(self, input, pat=None):
        with torch.enable_grad():
            self.ca3_mlp.start_seq(input.size(1))
            ret_dict = super().forward(input, pat)
            self.ca3_mlp.end_seq()
            return ret_dict

    def get_additional_loss(self):
        raise NotImplementedError

    def setup_hidden_modules(self):
        if self.ca3_cfg is None and self.dg_cfg is None:
            super().setup_hidden_modules()
            return
        self.dg_mlp = builder.build_neck(self.dg_cfg)
        self.ca3_mlp = builder.build_neck(self.ca3_cfg)

    def get_hidden(self, input):
        h = self.dg_mlp([input])[0]
        h = self.l2_normalize(h)
        h_ca3 = self.ca3_mlp([input])[0]
        h_ca3 = self.l2_normalize(h_ca3)
        self.ca3_mlp.update(h, h_ca3)
        return h

    def forward_for_sep_h(self, input, pat):
        h = self.ca3_mlp([input])[0]
        h = self.l2_normalize(h)
        raw_pat_resp = torch.matmul(pat, h.unsqueeze(-1)).squeeze(-1)
        curr_pat_resp = self.newh_mlp_softmax(raw_pat_resp)
        new_h = NaivePatSoftmaxRNN.get_new_h(
                self, pat, curr_pat_resp, raw_pat_resp)
        return new_h


class DGDynCA3CA1GSepHPRNN(DGDynCA3GSepHPRNN):
    def __init__(
            self, ca3_cfg, dg_cfg, 
            ca1_cfg, ca3to1_cfg,
            readout_mlp_cfg,
            precomp_dg_ca1=False,
            update_in_fwd=False,
            *args, **kwargs):
        self.ca1_cfg = ca1_cfg
        self.ca3_cfg = ca3_cfg
        self.dg_cfg = dg_cfg
        self.ca3to1_cfg = ca3to1_cfg
        self.readout_mlp_cfg = readout_mlp_cfg
        self.precomp_dg_ca1 = precomp_dg_ca1
        self.update_in_fwd = update_in_fwd
        GateSepHPathRNN.__init__(self, *args, **kwargs)
        self.init_other_modules()

    def setup_hidden_modules(self):
        self.dg_mlp = builder.build_neck(self.dg_cfg)
        self.ca3_mlp = builder.build_neck(self.ca3_cfg)
        self.ca1_mlp = builder.build_neck(self.ca1_cfg)
        self.ca3to1_mlp = builder.build_neck(self.ca3to1_cfg)
        self.readout_mlp = builder.build_neck(self.readout_mlp_cfg)

    def init_other_modules(self):
        self.dg_mlp.init_weights()
        self.ca3_mlp.init_weights()
        self.ca1_mlp.init_weights()
        self.ca3to1_mlp.init_weights()
        self.readout_mlp.init_weights(init_linear='kaiming')

    def get_additional_loss(self):
        return sum(self.ca1_rec_losses)/len(self.ca1_rec_losses)

    def get_hidden(self, input):
        if not self.precomp_dg_ca1:
            h = self.dg_mlp([input])[0]
        else:
            h = self.precomp_dg_outs[self.now_seq_idx]
        h = self.l2_normalize(h)
        if not self.update_in_fwd:
            h_ca3 = self.ca3_mlp([input])[0]
            h_ca3 = self.l2_normalize(h_ca3)
            self.ca3_mlp.update(h, h_ca3)
        else:
            h_ca3 = self.ca3_mlp([input], update_target=h)[0]
        return h

    def get_precomp_dg_ca1(self, input):
        seq_len, batch_size = input.size(0), input.size(1)
        input_2d = input.reshape(seq_len * batch_size, -1)
        precomp_dg_outs = self.dg_mlp([input_2d])[0]
        precomp_ca1_outs = self.ca1_mlp([input_2d])[0]
        self.precomp_dg_outs = precomp_dg_outs.reshape(seq_len, batch_size, -1)
        self.precomp_ca1_outs = precomp_ca1_outs.reshape(seq_len, batch_size, -1)

    def forward(self, input, pat=None):
        with torch.enable_grad():
            self.ca1_rec_losses = []
            self.ca3_mlp.start_seq(input.size(1))
            self.ca3to1_mlp.start_seq(input.size(1))
            if self.precomp_dg_ca1:
                self.get_precomp_dg_ca1(input)
            self.now_seq_idx = 0
            ret_dict = GateSepHPathRNN.forward(self, input, pat)
            self.ca3_mlp.end_seq()
            self.ca3to1_mlp.end_seq()
            return ret_dict

    def forward_for_sep_h(self, input, pat):
        new_h = DGDynCA3GSepHPRNN.forward_for_sep_h(
                self, input, pat)
        new_h = new_h.type(self.ca3to1_mlp.weights_0.dtype)
        new_i = self.ca3to1_mlp([new_h])[0]
        rec_input = self.readout_mlp([new_i])[0]
        return rec_input

    def forward_one_time(self, input, pat):
        pat = DGDynCA3GSepHPRNN.forward_one_time(
                self, input, pat)

        with torch.no_grad():
            sim_new_h = DGDynCA3GSepHPRNN.forward_for_sep_h(
                    self, input, pat)
            sim_new_h = sim_new_h.detach().clone()
            # to solve some f16 training weird issues here
            sim_new_h = sim_new_h.type(self.ca3to1_mlp.weights_0.dtype)
        if not self.precomp_dg_ca1:
            ca1_output = self.ca1_mlp([input])[0]
        else:
            ca1_output = self.precomp_ca1_outs[self.now_seq_idx]
        ca1_output = self.l2_normalize(ca1_output)
        if not self.update_in_fwd:
            ca3to1_output = self.ca3to1_mlp([sim_new_h])[0]
            self.ca3to1_mlp.update(ca1_output, ca3to1_output)
        else:
            ca3to1_output = self.ca3to1_mlp(
                    [sim_new_h], update_target=ca1_output)[0]
        _curr_rec_input = self.readout_mlp([ca1_output])[0]
        _curr_rec_input = self.l2_normalize(_curr_rec_input)
        _loss = 1 - torch.mean(
                torch.sum(
                    _curr_rec_input * input, 
                    dim=-1))
        self.ca1_rec_losses.append(_loss)
        self.now_seq_idx += 1
        return pat


class LegMaskFxEcGSepHPRNN(FxEcGateSepHPathRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.state_pass_method == 'sample_from_q'
        self.setup_pt_leg_queue()

    def setup_pt_leg_queue(self):
        self.leg_seq_keep_len = 200
        self.register_buffer(
                "leg_seq_queue",
                torch.randn(
                    self.pt_queue_len, self.leg_seq_keep_len, self.input_size))
        self.leg_seq_queue = self.l2_normalize(self.leg_seq_queue)

    def get_init_states(self, input):
        pat = super().get_init_states(input)
        leg_seq = torch.randn(
                input.size(1), self.leg_seq_keep_len, self.input_size,
                device=input.device, dtype=input.dtype)
        leg_seq = self.l2_normalize(leg_seq)
        self.leg_seq = leg_seq
        self.curr_input = input
        return pat

    def get_passed_states(self, input, pat):
        _idx = self.multinomial.draw(input.size(1))
        selected_queue = torch.index_select(self.pt_queue, 0, _idx)
        selected_leg_seq = torch.index_select(self.leg_seq_queue, 0, _idx)
        self._dequeue_and_enqueue(pat)
        pat = selected_queue.clone().detach()
        self.leg_seq = selected_leg_seq.clone().detach()
        self.curr_input = input
        return pat

    def get_leg_mask(self, target):
        curr_cue = self.curr_input[-1]
        target_sim = torch.sum(curr_cue * target, dim=-1)
        curr_cue = curr_cue.unsqueeze(1)
        curr_leg_sim = torch.sum(curr_cue * self.leg_seq, dim=-1)
        curr_leg_sim, _ = torch.max(curr_leg_sim, dim=-1)
        leg_mask = target_sim.ge(curr_leg_sim)
        input_to_update = self.curr_input[:-1].permute(1, 0, 2)
        seq_len = input_to_update.size(1)
        self.leg_seq = torch.cat(
                [self.leg_seq[:, seq_len:], input_to_update], 
                dim=1)
        return leg_mask

    @torch.no_grad()
    def _dequeue_and_enqueue(self, pat):
        """Update queue."""
        # gather keys before updating queue
        prev_leg_seq = self.leg_seq.clone().detach()
        prev_leg_seq = concat_all_gather(prev_leg_seq)
        pat = concat_all_gather(pat)
        batch_size = pat.shape[0]
        assert self.pt_queue_len % batch_size == 0, \
                "%i, %i" % (self.queue_len, batch_size)  # for simplicity
        
        ptr = int(self.pt_queue_ptr)
        self.pt_queue[ptr:ptr + batch_size] = pat
        self.leg_seq_queue[ptr:ptr + batch_size] = prev_leg_seq
        ptr = (ptr + batch_size) % self.pt_queue_len
        self.pt_queue_ptr[0] = ptr
