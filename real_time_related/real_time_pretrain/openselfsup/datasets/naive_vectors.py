import torch
import torch.nn as nn
from .registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module
class NaiveVectorDataset(Dataset):
    def __init__(self, vector_dim=128):
        self.vector_dim = vector_dim

    def __len__(self):
        return 256 * 5000

    def __getitem__(self, idx):
        vector = torch.randn(self.vector_dim)
        vector = nn.functional.normalize(vector, dim=0)
        return dict(img=vector)


@DATASETS.register_module
class SeqVectorDataset(Dataset):
    def __init__(self, vector_dim=128, seq_len=32):
        self.vector_dim = vector_dim
        self.seq_len = seq_len

    def __len__(self):
        return 256 * 5000

    def __getitem__(self, idx):
        vector = torch.randn(self.seq_len, self.vector_dim)
        vector = nn.functional.normalize(vector, dim=1)
        return dict(img=vector)
