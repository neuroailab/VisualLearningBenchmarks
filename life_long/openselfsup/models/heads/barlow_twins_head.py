import torch
import torch.nn as nn

from ..registry import HEADS
from ...utils import all_reduce_mean, SyncNormalizeFunction


@HEADS.register_module
class BarlowTwinsHead(nn.Module):
    def __init__(self, lambda_: float, scale_loss: float, embedding_dim: int):
        super().__init__()

        self.lambda_ = lambda_
        self.scale_loss = scale_loss
        self.embedding_dim = embedding_dim
        self.num_copies = 2
        self.eps = 1e-5

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """
        return a flattened view of the off-diagonal elements of a square matrix
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, embedding: torch.Tensor) -> torch.tensor:
        """
        Calculate the loss. Operates on embeddings tensor.
        Args:
            embedding (torch.Tensor):   NxEMBEDDING_DIM
                                        Must contain the concatenated embeddings
                                        of the two image copies:
                                        [emb_img1_0, emb_img2_0, ....., emb_img1_1, emb_img2_1,...]
        """
        assert embedding.ndim == 2 and embedding.shape[1] == int(
            self.embedding_dim
        ), f"Incorrect embedding shape: {embedding.shape} but expected Nx{self.embedding_dim}"

        batch_size = embedding.shape[0]
        assert (
            batch_size % self.num_copies == 0
        ), f"Batch size {batch_size} should be divisible by num_copies ({self.num_copies})."

        # normalize embeddings along the batch dimension
        embedding_normed = SyncNormalizeFunction.apply(embedding, self.eps)

        # split embedding between copies
        embedding_normed_a, embedding_normed_b = torch.split(
            embedding_normed,
            split_size_or_sections=batch_size // self.num_copies,
            dim=0,
        )

        # cross-correlation matrix
        correlation_matrix = torch.mm(embedding_normed_a.T, embedding_normed_b) / (
            batch_size / self.num_copies
        )

        # Reduce cross-correlation matrices from all processes
        correlation_matrix = all_reduce_mean(correlation_matrix)

        # loss
        on_diag = (
            torch.diagonal(correlation_matrix).add(-1).pow(2).sum().mul(self.scale_loss)
        )
        off_diag = (
            self._off_diagonal(correlation_matrix).pow(2).sum().mul(self.scale_loss)
        )
        loss = on_diag + self.lambda_ * off_diag

        return loss
