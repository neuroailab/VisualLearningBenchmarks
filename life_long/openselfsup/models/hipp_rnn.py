import torch
import torch.nn as nn
import os
import pdb

from . import builder
from .registry import MODELS

DEBUG = os.environ.get('DEBUG', '0')=='1'


@MODELS.register_module
class HippRNN(nn.Module):
    def __init__(
            self, hipp_head, seq_len, noise_norm, 
            num_negs=3, loss_type='default', add_target_to_others=False,
            add_target_range=None, add_random_target_to_others=False,
            randomize_noise_norm=False,
            context_remove_test=False,
            mask_use_kNN=False,
            use_leg_mask=False,
            only_process_valid_input=False,
            include_additional_loss=False,
            ):
        super().__init__()
        self.hipp_head = builder.build_head(hipp_head)
        self.seq_len = seq_len
        self.noise_norm = noise_norm
        self.num_negs = num_negs
        self.loss_type = loss_type
        self.add_target_to_others = add_target_to_others
        self.add_target_range = add_target_range
        self.add_random_target_to_others = add_random_target_to_others
        self.randomize_noise_norm = randomize_noise_norm
        self.context_remove_test = context_remove_test
        self.mask_use_kNN = mask_use_kNN
        self.use_leg_mask = use_leg_mask
        self.only_process_valid_input = only_process_valid_input
        self.include_additional_loss = include_additional_loss
        self.init_weights()

    def init_weights(self):
        self.hipp_head.init_weights(init_linear='kaiming')

    def get_seq_nn(
            self, target_vec, seq_vecs, 
            nn_num=None):
        if nn_num is None:
            nn_num = self.num_negs + 1
        dp_seq_target = torch.sum(
                seq_vecs * target_vec.unsqueeze(0).repeat(seq_vecs.size(0), 1, 1),
                dim=-1)
        _, _idx = torch.topk(
                dp_seq_target, nn_num, dim=0, sorted=True)
        _idx = _idx * seq_vecs.size(1)
        batch_idx = torch.arange(seq_vecs.size(1)).cuda()
        _idx += batch_idx.unsqueeze(0).repeat(nn_num, 1)
        _idx = _idx.view(-1)
        target_neg_vecs = torch.index_select(
                seq_vecs.reshape(-1, seq_vecs.size(2)), 
                0, _idx).view(nn_num, seq_vecs.size(1), seq_vecs.size(2))
        return target_neg_vecs

    def add_target(self, seq_vecs, target_vec):
        if self.add_target_range is None:
            add_min = 0.2
            add_max = 0.6
        else:
            try:
                add_min, add_max = self.add_target_range.split(',')
                add_min, add_max = float(add_min), float(add_max)
            except:
                raise NotImplementedError('Wrong format for add_target_range')

        all_add_norm = torch.rand(
                seq_vecs.size(0), 
                seq_vecs.size(1), 1)* (add_max - add_min) + add_min
        seq_vecs = all_add_norm.cuda() * target_vec.unsqueeze(0) + seq_vecs
        seq_vecs = nn.functional.normalize(seq_vecs, dim=-1)
        return seq_vecs

    def get_target_vec(self, seq_vecs):
        batch_idx = torch.arange(seq_vecs.size(1)).cuda()
        target_vec_idx = torch.randint(
                            0, self.seq_len, 
                            [seq_vecs.size(1)]).cuda() * seq_vecs.size(1) \
                         + batch_idx
        target_vec = torch.index_select(
                seq_vecs.reshape(-1, seq_vecs.size(2)), 
                0, target_vec_idx)
        return target_vec

    def build_seq_targets(self, embd, target):
        if len(embd.shape) == 2:
            seq_vecs = []
            assert self.seq_len < embd.size(0)
            for idx in range(1, self.seq_len+1):
                new_vec = torch.cat([embd[idx:], embd[:idx]], dim=0)
                seq_vecs.append(new_vec.unsqueeze(0))
            seq_vecs = torch.cat(seq_vecs, dim=0)
        elif len(embd.shape) == 3:
            seq_vecs = embd.permute(1, 0, 2)
        else:
            raise NotImplementedError

        if target is None:
            target_vec = self.get_target_vec(seq_vecs)
            if self.add_target_to_others:
                if not self.add_random_target_to_others:
                    seq_vecs = self.add_target(seq_vecs, target_vec)
                else:
                    _random_target = self.get_target_vec(seq_vecs)
                    seq_vecs = self.add_target(seq_vecs, _random_target)
                    target_vec = self.get_target_vec(seq_vecs)
            target_neg_vecs = self.get_seq_nn(target_vec, seq_vecs)
        else:
            target_vec = target
            target_neg_vecs = self.get_seq_nn(target_vec, seq_vecs[:-1])

        target_neg_vecs = target_neg_vecs[1:]
        if target is None:
            noise = torch.randn(
                        target_vec.shape[0], target_vec.shape[1]).cuda()
            if not self.randomize_noise_norm:
                noise = noise * self.noise_norm
            else:
                noise = noise * torch.rand(
                        target_vec.shape[0], 1).cuda() * self.noise_norm
            noisy_target_vec = nn.functional.normalize(
                    target_vec + noise, dim=1)
            seq_vecs = torch.cat(
                    [seq_vecs, noisy_target_vec.unsqueeze(0)], dim=0)
        return seq_vecs, target_vec, target_neg_vecs

    def get_sim_loss(self, pred_target_vec, target_vec, mask=None):
        if mask is None:
            sim_loss = 2 - 2 * (pred_target_vec * target_vec).sum() \
                             / (torch.numel(pred_target_vec) \
                                / pred_target_vec.size(-1))
        else:
            sim_loss = 2 - 2 * (pred_target_vec \
                                * target_vec \
                                * mask.unsqueeze(-1)).sum()\
                             / mask.sum()
        return sim_loss

    def get_pat_resp_corr(self):
        pat_resp_corr = None
        corr_count = 0
        all_pat_resps = self.hipp_head.rnn.all_pat_resps
        for idx in range(len(all_pat_resps)):
            all_pat_resps[idx] = nn.functional.normalize(
                    all_pat_resps[idx], dim=-1)
        for idx_0 in range(1, len(all_pat_resps)):
            for idx_1 in range(idx_0):
                corr_count += 1
                curr_sum = (all_pat_resps[idx_0] * all_pat_resps[idx_1]).sum()
                if pat_resp_corr is None:
                    pat_resp_corr = curr_sum
                else:
                    pat_resp_corr += curr_sum
        return pat_resp_corr / (corr_count * all_pat_resps[0].size(0))

    def get_pred_outputs(self, seq_vecs):
        if self.context_remove_test:
            context_vec = seq_vecs[:1].clone()
            seq_vecs -= context_vec
        pred_embd = self.hipp_head(seq_vecs)
        assert pred_embd.size(1) == (1+self.num_negs) * seq_vecs.size(2)
        pred_embd = pred_embd.view(-1, 1+self.num_negs, seq_vecs.size(2))
        if self.context_remove_test:
            pred_embd += context_vec[0].unsqueeze(1)
        pred_embd = nn.functional.normalize(pred_embd, dim=-1)
        pred_target_vec = pred_embd[:, 0, :]
        if self.num_negs > 0:
            pred_target_neg_vecs = pred_embd[:, 1:, :].transpose(0, 1)
        else:
            pred_target_neg_vecs = None
        return pred_target_vec, pred_target_neg_vecs

    def get_input_mask(self, kNN_target_vecs, target_vec):
        mask = (kNN_target_vecs[0] * target_vec).sum(dim=-1)
        mask = mask.ge(0.99)
        if self.use_leg_mask:
            leg_mask = self.hipp_head.get_leg_mask(target_vec)
            mask = torch.logical_and(mask, leg_mask)
        return mask

    def forward_train(self, embd, target):
        seq_vecs, target_vec, target_neg_vecs = self.build_seq_targets(
                embd, target)
        losses = {}
        kNN_target_vecs = self.get_seq_nn(seq_vecs[-1], seq_vecs[:-1])
        losses['kNN_pred_error'] = self.get_sim_loss(
                kNN_target_vecs[0], target_vec)
        losses['kNN_pred_neg_error'] = self.get_sim_loss(
                kNN_target_vecs[1:], target_neg_vecs)
        if not self.only_process_valid_input:
            pred_target_vec, pred_target_neg_vecs = self.get_pred_outputs(seq_vecs)
            if not self.mask_use_kNN:
                losses['pred_error'] = self.get_sim_loss(
                        pred_target_vec, target_vec)
            else:
                mask = self.get_input_mask(kNN_target_vecs, target_vec)
                if int(torch.sum(mask)) == 0:
                    mask[0] = True
                losses['pred_error'] = self.get_sim_loss(
                        pred_target_vec, target_vec, mask=mask)
        else:
            assert self.mask_use_kNN
            mask = self.get_input_mask(kNN_target_vecs, target_vec)
            if int(torch.sum(mask)) == 0:
                mask[0] = True
            seq_vecs = torch.masked_select(
                    seq_vecs, mask.unsqueeze(0).unsqueeze(2),
                    ).reshape(seq_vecs.size(0), -1, seq_vecs.size(2))
            target_vec = torch.masked_select(
                    target_vec, mask.unsqueeze(1),
                    ).reshape(-1, target_vec.size(1))
            target_neg_vecs = torch.masked_select(
                    target_neg_vecs, mask.unsqueeze(0).unsqueeze(2),
                    ).reshape(target_neg_vecs.size(0), -1, target_neg_vecs.size(2))
            pred_target_vec, pred_target_neg_vecs = self.get_pred_outputs(seq_vecs)
            losses['pred_error'] = self.get_sim_loss(
                    pred_target_vec, target_vec)
        if self.num_negs > 0:
            losses['pred_neg_error'] = self.get_sim_loss(
                    pred_target_neg_vecs, target_neg_vecs)
        if DEBUG:
            #pdb.set_trace()
            pass
        if self.loss_type == 'default':
            losses['loss'] = losses['pred_error'] + losses['pred_neg_error']
        elif self.loss_type == 'just_pos':
            losses['loss'] = losses['pred_error']
        elif self.loss_type == 'pos_with_corr':
            losses['corr'] = self.get_pat_resp_corr()
            losses['loss'] = losses['pred_error'] + losses['corr']
        else:
            raise NotImplementedError
        if self.include_additional_loss:
            losses['add_error'] = self.hipp_head.get_additional_loss()
            losses['loss'] = losses['loss'] + losses['add_error']
        return losses

    def forward_mst(self, embd):
        assert len(embd.shape) == 3
        seq_vecs = embd.permute(1, 0, 2)
        pred_embd = self.hipp_head.forward_pred_all(seq_vecs)
        return pred_embd
        
    def forward(self, img, target=None, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(embd=img, target=target, **kwargs)
        elif mode == 'mst':
            return self.forward_mst(embd=img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))
