import numpy as np


def basic_r18_cfg_func(cfg):
    cfg.model['neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=2048,
            out_channels=2048,
            num_layers=4,
            with_avg_pool=True,
            with_last_bn=False,
            )
    cfg.model['head'] = dict(
            type='BarlowTwinsHead',
            lambda_=0.0051, 
            scale_loss=0.024, 
            embedding_dim=2048,
            )
    cfg.model['type'] = 'BarlowTwins'
    return cfg


def set_head_params(**kwargs):
    def _func(cfg):
        cfg.model['head'].update(kwargs)
        return cfg
    return _func


def change_embd_dim(new_dim):
    def _func(cfg):
        cfg.model['neck']['hid_channels'] = new_dim
        cfg.model['neck']['out_channels'] = new_dim
        cfg.model['head']['embedding_dim'] = new_dim
        return cfg
    return _func
