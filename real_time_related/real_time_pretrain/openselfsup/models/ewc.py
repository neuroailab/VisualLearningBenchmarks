import torch
import torch.nn as nn
from tqdm import tqdm
from openselfsup.framework.dist_utils import get_dist_info

from . import builder
from .registry import MODELS


@MODELS.register_module
class OnlineEWC(nn.Module):
    def __init__(self, model, ewc_lambda, gamma, num_samples=500):
        super().__init__()
        self.model = builder.build_model(model)
        self.ewc_lambda = ewc_lambda
        self.num_samples = num_samples
        self.gamma = gamma
        self.init_fisher()

    def init_fisher(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer(
                        '{}_ewc_prev_task'.format(n), p.detach().clone())
                self.register_buffer(
                        '{}_ewc_fisher'.format(n), p.detach().clone().zero_())

    def estimate_fisher(self, data_loader):
        mode = self.training
        self.eval()

        rank, _ = get_dist_info()
        if rank == 0:
            to_enum = tqdm(
                    data_loader, desc='Update Fisher Info', 
                    total=self.num_samples)
        else:
            to_enum = data_loader
        new_fisher_info = {}
        for i, data_batch in enumerate(to_enum):
            self.zero_grad()
            if i >= self.num_samples:
                break
            for key, value in data_batch.items():
                data_batch[key] = value.cuda()
            model_losses = self.model(mode='train', **data_batch)
            loss_value = model_losses['loss']
            loss_value.backward()
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        if n not in new_fisher_info:
                            new_fisher_info[n] = p.detach().clone().zero_()
                        new_fisher_info[n] += p.grad.detach() ** 2
        new_fisher_info = {
                n: p/self.num_samples 
                for n, p in new_fisher_info.items()}
        self.train(mode=mode)
        return new_fisher_info

    def update_buffer(self, data_loader):
        new_fisher_info = self.estimate_fisher(data_loader)
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                curr_fisher_value = getattr(self, '{}_ewc_fisher'.format(n))
                curr_fisher_value = \
                        curr_fisher_value * self.gamma \
                        + new_fisher_info[n] * (1 - self.gamma)
                setattr(self, '{}_ewc_fisher'.format(n), curr_fisher_value)
                setattr(self, '{}_ewc_prev_task'.format(n), p.detach().clone())

    def get_ewc_loss(self):
        losses = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                n = n.replace('.', '__')
                mean = getattr(self, '{}_ewc_prev_task'.format(n))
                fisher = getattr(self, '{}_ewc_fisher'.format(n))
                # Calculate EWC-loss
                losses.append((fisher * (p-mean)**2).sum())
	# Sum EWC-loss from all parameters
        return (1./2)*sum(losses)*self.ewc_lambda

    def forward_train(self, img, **kwargs):
        model_losses = self.model(img, mode='train', *kwargs)
        ewc_loss = self.get_ewc_loss()
        model_losses['ewc_penalty'] = ewc_loss
        model_losses['loss'] += ewc_loss
        return model_losses

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        else:
            return self.model(img, mode=mode, **kwargs)
