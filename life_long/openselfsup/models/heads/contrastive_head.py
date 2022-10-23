import torch
import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module
class ContrastiveHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(
            self, temperature=0.1, pos_nn_num=None, 
            neg_fn_num=None, neg_th_value=None, margin=None,
            loss_reduced=True,
            ):
        super(ContrastiveHead, self).__init__()
        if loss_reduced:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.pos_nn_num = pos_nn_num
        self.neg_fn_num = neg_fn_num
        self.neg_th_value = neg_th_value
        self.margin = margin

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        if self.neg_fn_num is not None:
            neg, _ = torch.topk(-neg, self.neg_fn_num, dim=-1)
            neg = -neg
        if self.neg_th_value is not None:
            neg = neg.masked_fill(neg > self.neg_th_value, -float('inf'))
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        if self.margin is not None:
            logits = logits - self.margin 
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        if self.pos_nn_num is None:
            losses['loss'] = self.criterion(logits, labels)
        else:
            logits = torch.exp(logits)
            pos_prob = torch.sum(logits[:, :(self.pos_nn_num+1)], dim=1)
            all_prob = torch.sum(logits, dim=1)
            losses['loss'] = -torch.mean(torch.log(pos_prob / all_prob))
        return losses


@HEADS.register_module
class CorrOptHead(nn.Module):
    """Head for contrastive learning.
    """
    def __init__(
            self, mix_weight=1.0,
            adaptive_low_th=None, 
            adaptive_high_th=None):
        super().__init__()
        self.mix_weight = mix_weight
        self.adaptive_low_th = adaptive_low_th
        self.adaptive_high_th = adaptive_high_th

    def adaptive_filter_neg(
            self, neg, adaptive_th, low_or_high):
        neg_flat = neg.reshape(-1)
        filter_num = int(adaptive_th * neg_flat.size(0))
        if low_or_high == 'low':
            neg_flat, _ = torch.topk(-neg_flat, filter_num)
            neg_flat = -neg_flat
            neg = neg.masked_fill(neg < torch.max(neg_flat), 0)
        else:
            neg_flat, _ = torch.topk(neg_flat, filter_num)
            neg = neg.masked_fill(neg > torch.min(neg_flat), 0)
        return neg

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = dict()
        if self.adaptive_low_th is not None:
            neg = self.adaptive_filter_neg(
                    neg, self.adaptive_low_th, 'low')
        if self.adaptive_high_th is not None:
            neg = self.adaptive_filter_neg(
                    neg, self.adaptive_high_th, 'high')
        losses['loss'] = \
                (1 - torch.mean(pos)) \
                + self.mix_weight * torch.mean(torch.abs(neg))
        return losses
