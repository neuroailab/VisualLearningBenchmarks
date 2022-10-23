import torch
import torch.nn as nn


def add_noise_to_target(proj_target, add_noise):
    noise = torch.randn(
            proj_target.shape[0], proj_target.shape[1]).cuda() \
            * add_noise
    proj_target += noise
    proj_target = nn.functional.normalize(proj_target, dim=1)
    return proj_target
