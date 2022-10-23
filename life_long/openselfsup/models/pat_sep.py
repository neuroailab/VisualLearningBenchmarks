import torch
import torch.nn as nn
import os
import pdb

from . import builder
from .registry import MODELS

#from openselfsup.framework.checkpoint import load_checkpoint
#from openselfsup.framework.utils import get_root_logger
from openselfsup.utils import get_root_logger
from mmcv.runner import load_checkpoint


@MODELS.register_module
class PatSep(nn.Module):
    def __init__(self, ps_neck):
        super().__init__()
        self.ps_neck = builder.build_neck(ps_neck)
        self.init_weights()

    def init_weights(self):
        self.ps_neck.init_weights(init_linear='kaiming')

    def get_off_diag_mean(self, mat):
        N = mat.size(1)
        mask = 1 - torch.eye(N, dtype=torch.uint8).cuda()
        mask = mask.unsqueeze(0)
        return torch.mean(torch.masked_select(mat, mask==1))

    def get_pat_sep_loss(self, hid_pat):
        hid_pat = nn.functional.normalize(hid_pat, dim=-1)
        sim_mat = torch.einsum('bpq, bqo->bpo', hid_pat, hid_pat.permute(0, 2, 1))
        sim_mat = torch.abs(sim_mat)
        return self.get_off_diag_mean(sim_mat)

    def forward_train(self, embd):
        embd = embd[:, :-1] # remove the last cue
        batch_size, seq_len = embd.size(0), embd.size(1)
        embd = embd.reshape(batch_size*seq_len, -1)
        hid_pat = self.ps_neck([embd])[0]
        hid_pat = hid_pat.reshape(batch_size, seq_len, -1)
        losses = {'loss': self.get_pat_sep_loss(hid_pat)}
        return losses

    def forward_extract(self, embd):
        batch_size, seq_len = embd.size(0), embd.size(1)
        embd = embd.reshape(batch_size*seq_len, -1)
        hid_pat = self.ps_neck([embd])[0]
        hid_pat = hid_pat.reshape(batch_size, seq_len, -1)
        hid_pat = nn.functional.normalize(hid_pat, dim=-1)
        return hid_pat
        
    def forward(self, img, target=None, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(embd=img, **kwargs)
        elif mode == 'extract':
            return self.forward_extract(embd=img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class PatSepRec(PatSep):
    def __init__(self, ps_neck, pr_neck):
        nn.Module.__init__(self)
        self.ps_neck = builder.build_neck(ps_neck)
        self.pr_neck = builder.build_neck(pr_neck)
        self.init_weights()

    def init_weights(self):
        self.ps_neck.init_weights(init_linear='kaiming')
        self.pr_neck.init_weights(init_linear='kaiming')

    def get_pat_rec_loss(self, rec_pat, embd):
        rec_pat = nn.functional.normalize(rec_pat, dim=-1)
        rec_loss = 1 - torch.mean(torch.sum(rec_pat * embd, dim=-1))
        return rec_loss

    def forward_train(self, embd):
        embd = embd[:, :-1] # remove the last cue
        batch_size, seq_len = embd.size(0), embd.size(1)
        embd = embd.reshape(batch_size*seq_len, -1)
        hid_pat = self.ps_neck([embd])[0]
        rec_pat = self.pr_neck([hid_pat])[0]
        hid_pat = hid_pat.reshape(batch_size, seq_len, -1)
        sep_loss = self.get_pat_sep_loss(hid_pat)
        rec_loss = self.get_pat_rec_loss(rec_pat, embd)
        total_loss = sep_loss + rec_loss
        losses = {
                'loss': total_loss,
                'sep_error': sep_loss,
                'rec_error': rec_loss,
                }
        return losses
        
    def forward(self, img, target=None, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(embd=img, **kwargs)
        elif mode == 'extract':
            return self.forward_extract(embd=img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class PatSepNoGrad(nn.Module):
    def __init__(self, ps_neck, pretrained=None):
        super().__init__()
        self.ps_neck = builder.build_neck(ps_neck)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                    self, self.pretrained, strict=True, logger=logger,
                    map_location='cpu')

    def forward(self, x):
        with torch.no_grad():
            output = self.ps_neck([x])[0]
        return output


@MODELS.register_module
class PatSepRecNoGrad(nn.Module):
    def __init__(self, ps_neck, pr_neck, pretrained=None, with_grad=False):
        super().__init__()
        self.ps_neck = builder.build_neck(ps_neck)
        self.pr_neck = builder.build_neck(pr_neck)
        self.pretrained = pretrained
        self.with_grad = with_grad

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                    self, self.pretrained, strict=True, logger=logger,
                    map_location='cpu')

    def forward(self, x):
        if not self.with_grad:
            with torch.no_grad():
                output = self.ps_neck([x])[0]
        else:
            output = self.ps_neck([x])[0]
        return output


@MODELS.register_module
class DgCA3PatSepRec(PatSep):
    def __init__(self, dg_neck, ca3_neck):
        nn.Module.__init__(self)
        self.dg_neck = builder.build_neck(dg_neck)
        self.ca3_neck = builder.build_neck(ca3_neck)
        self.init_weights()

    def init_weights(self):
        self.dg_neck.init_weights(init_linear='kaiming')
        self.ca3_neck.init_weights(init_linear='kaiming')

    def get_pat_rec_loss(self, rec_pat, embd):
        embd = nn.functional.normalize(embd, dim=-1)
        rec_pat = nn.functional.normalize(rec_pat, dim=-1)
        rec_loss = 1 - torch.mean(torch.sum(rec_pat * embd, dim=-1))
        return rec_loss

    def forward_train(self, embd, target):
        ps_embd = embd[:, :-1] # remove the last cue
        batch_size, seq_len = ps_embd.size(0), ps_embd.size(1)
        all_hid_pats = []
        self.ca3_neck.start_seq()
        for seq_idx in range(seq_len):
            _dg_hid_pat = self.dg_neck([ps_embd[:, seq_idx]])[0]
            _ca3_hid_pat = self.ca3_neck([ps_embd[:, seq_idx]])[0]
            self.ca3_neck.update(_dg_hid_pat, _ca3_hid_pat)
            all_hid_pats.append(_dg_hid_pat)

        all_hid_pats = torch.stack(all_hid_pats, dim=1)
        sep_loss = self.get_pat_sep_loss(all_hid_pats)
        rec_loss = self.get_pat_rec_loss(
                self.dg_neck([target])[0], 
                self.ca3_neck([embd[:, -1]])[0], 
                )
        total_loss = sep_loss + rec_loss
        losses = {
                'loss': total_loss,
                'sep_error': sep_loss,
                'rec_error': rec_loss,
                }
        return losses
        
    def forward(self, img, target=None, mode='train', **kwargs):
        if mode == 'train':
            with torch.enable_grad():
                return self.forward_train(embd=img, target=target, **kwargs)
        elif mode == 'extract':
            return self.forward_extract(embd=img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))
