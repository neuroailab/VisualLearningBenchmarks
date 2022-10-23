import torch
import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module
class L1SparseHead(nn.Module):
    def __init__(
            self, weight=1.0,
            ):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        loss = torch.mean(torch.abs(x)) * self.weight
        return {'loss': loss}
