import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

from ..moco import concat_all_gather
from .. import builder


class SelfRNN(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers,
            activation='tanh'):
        """"Constructor of the class"""
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.setup_modules()
        self.reset_parameters()

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise NotImplementedError

    def setup_modules(self):
        ih, hh = [], []
        for i in range(self.num_layers):
            if i==0:
                ih.append(nn.Linear(self.input_size, self.hidden_size))
            else:
                ih.append(nn.Linear(self.hidden_size, self.hidden_size))
            hh.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def get_zero_states(self, input):
        h_zeros = torch.zeros(
                self.num_layers, input.size(1),
                self.hidden_size,
                dtype=input.dtype, device=input.device)
        hx = (h_zeros,)
        return hx

    def forward(self, input, hx=None):
        if hx is None:
            hx = self.get_zero_states(input)
        num_seq = input.size(0)
        outputs = []
        all_states = [[] for _ in range(len(hx))]
        for idx in range(num_seq):
            hx = self.forward_one_time(
                    input[idx], hx)
            outputs.append(hx[0][-1])
            for idx in range(len(hx)):
                all_states[idx].append(hx[idx])
        outputs = torch.stack(outputs, 0)
        all_states = (torch.cat(_state, 0) for _state in all_states)
        return outputs, all_states

    def forward_one_time(self, input, hidden):
        hy = []
        for i in range(self.num_layers):
            hx = hidden[0][i]
            nhx = self.activation(self.w_ih[i](input) + self.w_hh[i](hx))
            hy.append(nhx)
            input = nhx

        hy = torch.stack(hy, 0)
        return (hy,)


class STPMLP(nn.Module):
    def __init__(
            self, input_size, hidden_size,
            gate_update=False, act_func='relu'):
        """"Constructor of the class"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_update = gate_update
        self.act_func = act_func
        self.setup_modules()
        self.reset_parameters()
        self.setup_other_params()

    def get_activation_func(self):
        if self.act_func=='relu':
            return nn.ReLU(inplace=True)
        elif self.act_func=='tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def setup_other_params(self):
        self.decay = torch.nn.Parameter(torch.tensor(0.999))
        self.decay.requires_grad = False

    def setup_modules(self):
        self.nonlnrty = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                self.get_activation_func(),
                nn.Linear(self.hidden_size, self.hidden_size),
                )

        self.inpt_mat = torch.nn.Parameter(
                torch.zeros(self.input_size, self.input_size))
        self.inpt_mat.requires_grad = True
        self.hid_mat = torch.nn.Parameter(
                torch.zeros(self.hidden_size, self.hidden_size))
        self.hid_mat.requires_grad = True

        self.main_mat = torch.nn.Parameter(
                torch.zeros(self.hidden_size, self.input_size))
        self.main_mat.requires_grad = True
        self.main_bias = torch.nn.Parameter(torch.zeros(self.hidden_size))
        self.main_bias.requires_grad = True

        if self.gate_update:
            self.y_t_prime_to_gate = nn.Linear(self.hidden_size, self.hidden_size)
            self.x_t_to_gate = nn.Linear(self.input_size, self.hidden_size)

    def init_stp(self, batch_size):
        self.stp_mat = torch.zeros(
                batch_size, self.hidden_size, self.input_size).cuda()

    def update_stpmat(self, x_t, y_t, y_t_prime):
        if self.gate_update:
            _gate = torch.sigmoid(
                    self.y_t_prime_to_gate(y_t_prime)\
                    + self.x_t_to_gate(x_t))
            y_t_prime = y_t_prime * _gate
        raw_update = torch.matmul(
                y_t_prime.unsqueeze(-1), x_t.unsqueeze(1))
        raw_update = torch.matmul(
                self.hid_mat.unsqueeze(0),
                torch.matmul(raw_update, self.inpt_mat.unsqueeze(0)))
        self.stp_mat = self.stp_mat * self.decay + raw_update

    def forward(self, x_t):
        _tmp_mat = self.main_mat.unsqueeze(0) + self.stp_mat
        y_t = torch.matmul(
                _tmp_mat, x_t.unsqueeze(-1)).squeeze(-1)
        y_t = y_t + self.main_bias.unsqueeze(0)
        y_t_prime = self.nonlnrty(y_t)
        self.update_stpmat(x_t, y_t, y_t_prime)
        return y_t_prime


class SimSTPMLP(STPMLP):
    def setup_modules(self):
        self.nonlnrty = torch.tanh

        self.inpt_mat_raw = torch.nn.Parameter(torch.zeros(self.input_size))
        self.inpt_mat_raw.requires_grad = True
        self.hid_mat_raw = torch.nn.Parameter(torch.zeros(self.hidden_size))
        self.hid_mat_raw.requires_grad = True

        self.main_mat = torch.nn.Parameter(
                torch.zeros(self.hidden_size, self.input_size))
        self.main_mat.requires_grad = True
        self.main_bias = torch.nn.Parameter(torch.zeros(self.hidden_size))
        self.main_bias.requires_grad = True

    def update_stpmat(self, x_t, y_t, y_t_prime):
        inpt_mat = torch.diag(self.inpt_mat_raw)
        hid_mat = torch.diag(self.hid_mat_raw)
        raw_update = torch.matmul(
                y_t_prime.unsqueeze(-1), x_t.unsqueeze(1))
        raw_update = torch.matmul(
                hid_mat.unsqueeze(0),
                torch.matmul(raw_update, inpt_mat.unsqueeze(0)))
        self.stp_mat = self.stp_mat * self.decay + raw_update


class DWSelfRNN(SelfRNN):
    def setup_modules(self):
        super().setup_modules()

        whi, whh = [], []
        for i in range(self.num_layers):
            if i==0:
                whi.append(nn.Linear(1, self.input_size))
            else:
                whi.append(nn.Linear(1, self.hidden_size))
            whh.append(nn.Linear(1, self.hidden_size))
        self.dw_whi = nn.ModuleList(whi)
        self.dw_whh = nn.ModuleList(whh)
        self.decay = torch.nn.Parameter(torch.tensor(0.9))
        self.decay.requires_grad = True

    def get_zero_states(self, input):
        h_zeros = super().get_zero_states(input)[0]
        dwi_zeros = []
        for i in range(self.num_layers):
            if i==0:
                dwi_zeros.append(torch.zeros(
                        input.size(1),
                        self.hidden_size, self.input_size,
                        dtype=input.dtype, device=input.device))
            else:
                dwi_zeros.append(torch.zeros(
                        input.size(1),
                        self.hidden_size, self.hidden_size,
                        dtype=input.dtype, device=input.device))
        dwh_zeros = torch.zeros(
                self.num_layers, input.size(1),
                self.hidden_size, self.hidden_size,
                dtype=input.dtype, device=input.device)
        return (h_zeros, dwi_zeros, dwh_zeros)

    def update_dws(self, i, dwi_x, dwh_x, nhx, input):
        dwi_x = self.decay * dwi_x \
                + self.dw_whi[i](nhx.unsqueeze(-1))
        dwh_x = self.decay * dwh_x \
                + self.dw_whh[i](nhx.unsqueeze(-1))
        return dwi_x, dwh_x

    def forward_one_time(self, input, hidden):
        hy, dwi_y, dwh_y = [], [], []
        for i in range(self.num_layers):
            hx = hidden[0][i]
            dwi_x, dwh_x = hidden[1][i], hidden[2][i]
            nhx = self.w_ih[i](input) + self.w_hh[i](hx)
            nhx = nhx \
                    + torch.matmul(dwi_x, input.unsqueeze(-1)).squeeze(-1)\
                    + torch.matmul(dwh_x, hx.unsqueeze(-1)).squeeze(-1)
            nhx = self.activation(nhx)
            hy.append(nhx)
            dwi_x, dwh_x = self.update_dws(
                    i, dwi_x, dwh_x, nhx, input)
            dwi_y.append(dwi_x)
            dwh_y.append(dwh_x)
            input = nhx

        hy = torch.stack(hy, 0)
        dwh_y = torch.stack(dwh_y, 0)
        return hy, dwi_y, dwh_y


class DWTransSelfRNN(SelfRNN):
    def reset_parameters(self):
        stdv = .1 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def setup_modules(self):
        super().setup_modules()

        wi, wh = [], []
        for i in range(self.num_layers):
            wi.append(nn.Linear(1, self.hidden_size))
            wh.append(nn.Linear(1, self.hidden_size))
        self.dw_wi = nn.ModuleList(wi)
        self.dw_wh = nn.ModuleList(wh)
        self.register_buffer("decay", torch.tensor(0.9))

    def get_zero_states(self, input):
        h_zeros = super().get_zero_states(input)[0]
        dwi_zeros = []
        for i in range(self.num_layers):
            if i==0:
                dwi_zeros.append(torch.zeros(
                        input.size(1),
                        self.hidden_size, self.input_size,
                        dtype=input.dtype, device=input.device))
            else:
                dwi_zeros.append(torch.zeros(
                        input.size(1),
                        self.hidden_size, self.hidden_size,
                        dtype=input.dtype, device=input.device))
        dwh_zeros = torch.zeros(
                self.num_layers, input.size(1),
                self.hidden_size, self.hidden_size,
                dtype=input.dtype, device=input.device)
        return (h_zeros, dwi_zeros, dwh_zeros)

    def update_dws(self, i, dwi_x, dwh_x, nhx, input):
        dwi_x = self.decay * dwi_x \
                + self.dw_wi[i](input.unsqueeze(-1)).transpose(1, 2)
        dwh_x = self.decay * dwh_x \
                + self.dw_wh[i](nhx.unsqueeze(-1)).transpose(1, 2)
        return dwi_x, dwh_x

    def forward_one_time(self, input, hidden):
        hy, dwi_y, dwh_y = [], [], []
        for i in range(self.num_layers):
            hx = hidden[0][i]
            dwi_x, dwh_x = hidden[1][i], hidden[2][i]
            nhx = self.w_ih[i](input) + self.w_hh[i](hx)
            nhx = nhx \
                    + torch.matmul(dwi_x, input.unsqueeze(-1)).squeeze(-1)\
                    + torch.matmul(dwh_x, hx.unsqueeze(-1)).squeeze(-1)
            nhx = self.activation(nhx)
            hy.append(nhx)
            dwi_x, dwh_x = self.update_dws(
                    i, dwi_x, dwh_x, nhx, input)
            dwi_y.append(dwi_x)
            dwh_y.append(dwh_x)
            input = nhx

        hy = torch.stack(hy, 0)
        dwh_y = torch.stack(dwh_y, 0)
        return hy, dwi_y, dwh_y


class SelfLSTM(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers,
            activation='tanh'):
        """"Constructor of the class"""
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.setup_modules()
        self.reset_parameters()

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise NotImplementedError

    def setup_modules(self):
        ih, hh = [], []
        for i in range(self.num_layers):
            if i==0:
                ih.append(nn.Linear(self.input_size, 4 * self.hidden_size))
            else:
                ih.append(nn.Linear(self.hidden_size, 4 * self.hidden_size))
            hh.append(nn.Linear(self.hidden_size, 4 * self.hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def get_zero_states(self, input):
        h_zeros = torch.zeros(
                self.num_layers, input.size(1),
                self.hidden_size,
                dtype=input.dtype, device=input.device)
        c_zeros = torch.zeros(
                self.num_layers, input.size(1),
                self.hidden_size,
                dtype=input.dtype, device=input.device)
        hx = (h_zeros, c_zeros)
        return hx

    def forward(self, input, hx=None):
        if hx is None:
            hx = self.get_zero_states(input)
        num_seq = input.size(0)
        outputs = []
        all_states = [[] for _ in range(len(hx))]
        for idx in range(num_seq):
            hx = self.forward_one_time(
                    input[idx], hx)
            outputs.append(hx[0][-1])
            for idx in range(len(hx)):
                all_states[idx].append(hx[idx])
        outputs = torch.stack(outputs, 0)
        all_states = (torch.cat(_state, 0) for _state in all_states)
        return outputs, all_states

    def get_ncx_nhx_from_gates(self, gates, cx):
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        c_gate = self.activation(c_gate)
        o_gate = torch.sigmoid(o_gate)

        ncx = (f_gate * cx) + (i_gate * c_gate)
        nhx = o_gate * self.activation(ncx)
        return ncx, nhx

    def forward_one_time(self, input, hidden):
        hy, cy = [], []
        for i in range(self.num_layers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            ncx, nhx = self.get_ncx_nhx_from_gates(gates, cx)
            cy.append(ncx)
            hy.append(nhx)
            input = nhx

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        return hy, cy


class DWSelfLSTM(SelfLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("decay", torch.tensor(0.9))

    def setup_modules(self):
        super().setup_modules()

        whi, wci, whh, wch = [], [], [], []
        for i in range(self.num_layers):
            if i==0:
                whi.append(nn.Linear(1, self.input_size))
                wci.append(nn.Linear(1, self.input_size))
            else:
                whi.append(nn.Linear(1, self.hidden_size))
                wci.append(nn.Linear(1, self.hidden_size))
            whh.append(nn.Linear(1, self.hidden_size))
            wch.append(nn.Linear(1, self.hidden_size))
        self.dw_whi = nn.ModuleList(whi)
        self.dw_wci = nn.ModuleList(wci)
        self.dw_whh = nn.ModuleList(whh)
        self.dw_wch = nn.ModuleList(wch)

    def get_zero_states(self, input):
        h_zeros, c_zeros = super().get_zero_states(input)
        dwi_zeros = []
        for i in range(self.num_layers):
            if i==0:
                dwi_zeros.append(torch.zeros(
                        input.size(1),
                        4*self.hidden_size, self.input_size,
                        dtype=input.dtype, device=input.device))
            else:
                dwi_zeros.append(torch.zeros(
                        input.size(1),
                        4*self.hidden_size, self.hidden_size,
                        dtype=input.dtype, device=input.device))
        dwh_zeros = torch.zeros(
                self.num_layers, input.size(1),
                4*self.hidden_size, self.hidden_size,
                dtype=input.dtype, device=input.device)
        return (h_zeros, c_zeros, dwi_zeros, dwh_zeros)

    def forward_one_time(self, input, hidden):
        hy, cy, dwi_y, dwh_y = [], [], [], []
        for i in range(self.num_layers):
            hx, cx = hidden[0][i], hidden[1][i]
            dwi_x, dwh_x = hidden[2][i], hidden[3][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            gates = gates \
                    + torch.matmul(dwi_x, input.unsqueeze(-1)).squeeze(-1)\
                    + torch.matmul(dwh_x, hx.unsqueeze(-1)).squeeze(-1)
            ncx, nhx = self.get_ncx_nhx_from_gates(gates, cx)
            cy.append(ncx)
            hy.append(nhx)
            dwi_x = self.decay * dwi_x \
                    + self.dw_whi[i](nhx.repeat(1,4).unsqueeze(-1)) \
                    + self.dw_wci[i](ncx.repeat(1,4).unsqueeze(-1))
            dwh_x = self.decay * dwh_x \
                    + self.dw_whh[i](nhx.repeat(1,4).unsqueeze(-1)) \
                    + self.dw_wch[i](ncx.repeat(1,4).unsqueeze(-1))
            dwi_y.append(dwi_x)
            dwh_y.append(dwh_x)
            input = nhx

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        dwh_y = torch.stack(dwh_y, 0)
        return hy, cy, dwi_y, dwh_y


class DWTransSelfLSTM(SelfLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("decay", torch.tensor(0.9))

    def setup_modules(self):
        super().setup_modules()

        wi, wh, wc = [], [], []
        for i in range(self.num_layers):
            wi.append(nn.Linear(1, 4*self.hidden_size, bias=False))
            wh.append(nn.Linear(1, 4*self.hidden_size, bias=False))
            wc.append(nn.Linear(1, 4*self.hidden_size, bias=False))
        self.dw_wi = nn.ModuleList(wi)
        self.dw_wh = nn.ModuleList(wh)
        self.dw_wc = nn.ModuleList(wc)

    def get_zero_states(self, input):
        h_zeros, c_zeros = super().get_zero_states(input)
        dwi_zeros = []
        for i in range(self.num_layers):
            if i==0:
                dwi_zeros.append(torch.zeros(
                        input.size(1),
                        4*self.hidden_size, self.input_size,
                        dtype=input.dtype, device=input.device))
            else:
                dwi_zeros.append(torch.zeros(
                        input.size(1),
                        4*self.hidden_size, self.hidden_size,
                        dtype=input.dtype, device=input.device))
        dwh_zeros = torch.zeros(
                self.num_layers, input.size(1),
                4*self.hidden_size, self.hidden_size,
                dtype=input.dtype, device=input.device)
        dwc_zeros = torch.zeros(
                self.num_layers, input.size(1),
                4*self.hidden_size, self.hidden_size,
                dtype=input.dtype, device=input.device)
        return (h_zeros, c_zeros, dwi_zeros, dwh_zeros, dwc_zeros)

    def forward_one_time(self, input, hidden):
        hy, cy, dwi_y, dwh_y, dwc_y = [], [], [], [], []
        for i in range(self.num_layers):
            hx, cx = hidden[0][i], hidden[1][i]
            dwi_x, dwh_x, dwc_x = hidden[2][i], hidden[3][i], hidden[4][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            gates = gates \
                    + torch.matmul(dwi_x, input.unsqueeze(-1)).squeeze(-1)\
                    + torch.matmul(dwh_x, hx.unsqueeze(-1)).squeeze(-1)\
                    + torch.matmul(dwc_x, cx.unsqueeze(-1)).squeeze(-1)
            ncx, nhx = self.get_ncx_nhx_from_gates(gates, cx)
            cy.append(ncx)
            hy.append(nhx)
            dwi_x = self.decay * dwi_x \
                    + self.dw_wi[i](input.unsqueeze(-1)).transpose(1,2)
            dwh_x = self.decay * dwh_x \
                    + self.dw_wh[i](nhx.unsqueeze(-1)).transpose(1,2)
            dwc_x = self.decay * dwc_x \
                    + self.dw_wc[i](ncx.unsqueeze(-1)).transpose(1,2)
            dwi_y.append(dwi_x)
            dwh_y.append(dwh_x)
            dwc_y.append(dwc_x)
            input = nhx

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        dwh_y, dwc_y = torch.stack(dwh_y, 0), torch.stack(dwc_y, 0)
        return hy, cy, dwi_y, dwh_y, dwc_y


class FastWeights(SelfLSTM):
    def __init__(self, update_fast_weight = "outer_product", **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("decay", torch.tensor(0.95))
        self.register_buffer("fast_lr", torch.tensor(0.5))
        self.layernorm_cx = nn.LayerNorm(self.hidden_size)
        self.layernorm_hx = nn.LayerNorm(self.hidden_size)
        if update_fast_weight == "outer_product":
            self.update_fast_weight = self.outerproduct_fast_weight
        elif update_fast_weight == "resnet_1_layer_cx_hx_separate":
            self.write_cx = nn.Linear(self.hidden_size, self.hidden_size)
            self.write_hx = nn.Linear(self.hidden_size, self.hidden_size)
            self.update_fast_weight = self.resnet_fast_weight_separate
        elif update_fast_weight == "resnet_1_layer_cx_hx_together":
            self.write_hx_cx = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.update_fast_weight = self.resnet_fast_weight_together
        else:
            raise NotImplementedError(f"Update fast weight method {update_fast_weight} not implemented")
            
        torch.autograd.set_detect_anomaly(True)
        
    def get_zero_fastweight(self, input):
        fastweight_zeros = torch.zeros(
            self.num_layers, input.size(1),
            self.hidden_size, self.hidden_size, 
            dtype=input.dtype, device=input.device)
    
        return fastweight_zeros
    
    def outerproduct_fast_weight(self, fastweight_layer, cx, hx):
        batch_outer_prod = torch.einsum('bp, bq->bpq', hx, hx)
        tmp = self.decay * fastweight_layer + self.fast_lr * batch_outer_prod
        return tmp
    
    def resnet_fast_weight_separate(self, fastweight_layer, cx, hx):
        outer_prod_cx = torch.einsum('bp, bq->bpq', cx, F.relu(self.write_cx(cx)))
        outer_prod_hx = torch.einsum('bp, bq->bpq', hx, F.relu(self.write_cx(hx)))
        tmp = fastweight_layer + outer_prod_cx + outer_prod_hx
        return tmp
    
    def resnet_fast_weight_together(self, fastweight_layer, cx, hx):
        hx_cx = torch.cat([hx, cx], dim=1)
        outer_prod_hx = torch.einsum('bp, bq->bpq', hx, F.relu(self.write_hx_cx(hx_cx)))
        tmp = fastweight_layer + outer_prod_hx
        return tmp
    
    def resnet_fast_weight_together(self, fastweight_layer, cx, hx):
        hx_cx = torch.cat([hx, cx], dim=1)
        outer_prod_hx = torch.einsum('bp, bq->bpq', hx, F.relu(self.write_hx_cx(hx_cx)))
        tmp = fastweight_layer + outer_prod_hx
        return tmp

    def forward(self, input, hx=None, fastweight = None):
        if hx is None:
            hx = self.get_zero_states(input)
        if fastweight is None:
            fastweight = self.get_zero_fastweight(input)
        num_seq = input.size(0)
        outputs = []
        #all_states = [[] for _ in range(len(hx) )]
        for idx in range(num_seq):
            
            hy, cy, fastweight = self.forward_one_time(
                    input[idx], hx, fastweight)
            outputs.append(hy[-1])
            #for idx in range(len(hx)):
            #    all_states[idx].append(hx[idx])
        outputs = torch.stack(outputs, 0)
        #all_states = (torch.cat(_state, 0) for _state in all_states)
        #return outputs, all_states
        return outputs, fastweight
    
    def forward_one_time(self, input, hidden, fastweight):
        hy, cy = [], []
        updated_fastweight = torch.zeros_like(fastweight)
        for i in range(self.num_layers):
            hx, cx = hidden[0][i], hidden[1][i]
            #print("hx.shape, cx.shape", hx.shape, cx.shape)
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            ncx, nhx = self.get_ncx_nhx_from_gates(gates, cx)
            nhx = nhx + torch.bmm(fastweight[i], nhx.unsqueeze(2)).squeeze(2)
            updated_fastweight[i] = self.update_fast_weight(fastweight[i], ncx, nhx)
            cy.append(self.layernorm_cx(ncx))
            hy.append(self.layernorm_hx(nhx))
            input = nhx

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        return hy, cy, updated_fastweight
