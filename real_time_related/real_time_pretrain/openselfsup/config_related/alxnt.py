from .saycam_funcs import random_saycam_two_img_cfg_func, random_saycam_two_img_rd_cfg_func


def ctl64(cfg):
    cfg.model['backbone'] = dict(
            type='AlexNet',
            pool_size=1,
            )
    cfg.data['train']['pipeline'][0]['size'] = 64
    cfg.model['neck']['in_channels'] = 4096
    cfg.model['neck']['hid_channels'] = 4096
    cfg.model['neck']['with_avg_pool'] = False
    cfg.data['imgs_per_gpu'] = 1024
    cfg.data['workers_per_gpu'] = 48
    return cfg


def sy_two_img_ctl64(cfg):
    cfg = random_saycam_two_img_cfg_func(cfg)
    cfg = ctl64(cfg)
    cfg.data['workers_per_gpu'] = 24
    return cfg


def sy_two_img_rd_ctl64(cfg):
    cfg = random_saycam_two_img_rd_cfg_func(cfg)
    cfg = ctl64(cfg)
    cfg.data['workers_per_gpu'] = 24
    return cfg
