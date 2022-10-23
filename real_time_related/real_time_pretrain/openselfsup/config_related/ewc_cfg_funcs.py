def ewc_wrap(cfg):
    cfg.model = {
            'type': 'OnlineEWC',
            'ewc_lambda': 100,
            'gamma': 0.9,
            'model': cfg.model,
            }
    return cfg


def ewc_s_wrap(cfg):
    cfg.model = {
            'type': 'OnlineEWC',
            'ewc_lambda': 30,
            'gamma': 0.9,
            'model': cfg.model,
            }
    return cfg


def ewc_l_wrap(cfg):
    cfg.model = {
            'type': 'OnlineEWC',
            'ewc_lambda': 300,
            'gamma': 0.9,
            'model': cfg.model,
            }
    return cfg


def ewc_ll_wrap(cfg):
    cfg.model = {
            'type': 'OnlineEWC',
            'ewc_lambda': 1000,
            'gamma': 0.9,
            'model': cfg.model,
            }
    return cfg


def ewc_l10_wrap(cfg):
    cfg.model = {
            'type': 'OnlineEWC',
            'ewc_lambda': 10,
            'gamma': 0.9,
            'model': cfg.model,
            }
    return cfg


def ewc_gd5_wrap(cfg):
    cfg.model = {
            'type': 'OnlineEWC',
            'ewc_lambda': 100,
            'gamma': 0.5,
            'model': cfg.model,
            }
    return cfg


def ewc_l300gd5_wrap(cfg):
    cfg.model = {
            'type': 'OnlineEWC',
            'ewc_lambda': 300,
            'gamma': 0.5,
            'model': cfg.model,
            }
    return cfg


def ewc_l30gd5_wrap(cfg):
    cfg.model = {
            'type': 'OnlineEWC',
            'ewc_lambda': 30,
            'gamma': 0.5,
            'model': cfg.model,
            }
    return cfg
