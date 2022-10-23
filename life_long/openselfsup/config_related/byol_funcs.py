def more_mlp_layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 3
    cfg.model['head']['predictor']['num_layers'] = 3
    return cfg


def mlp_4layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    cfg.model['head']['predictor']['num_layers'] = 4
    return cfg


def mlp_4L1bn_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    cfg.model['neck']['bn_settings'] = [True, False, False]
    cfg.model['head']['predictor']['num_layers'] = 4
    cfg.model['head']['predictor']['bn_settings'] = [True, False, False]
    return cfg


def mlp_4L1bnl_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    cfg.model['neck']['bn_settings'] = [False, False, True]
    cfg.model['head']['predictor']['num_layers'] = 4
    cfg.model['head']['predictor']['bn_settings'] = [True, False, False]
    return cfg


def mlp_5layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 5
    cfg.model['head']['predictor']['num_layers'] = 5
    return cfg


def mlp_6layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 6
    cfg.model['head']['predictor']['num_layers'] = 6
    return cfg


def mlp_layers_x_cfg_func(x):
    def _func(cfg):
        cfg.model['neck']['num_layers'] = x
        cfg.model['head']['predictor']['num_layers'] = x
        return cfg
    return _func

def mlp_x_L1bnl_cfg_func(x):
    def _func(cfg):
        cfg.model['neck']['num_layers'] = x
        cfg.model['neck']['bn_settings'] = [False] * (x-2) + [True]
        cfg.model['head']['predictor']['num_layers'] = x
        cfg.model['head']['predictor']['bn_settings'] = [True] + [False] * (x-2)
        return cfg
    return _func

def mlp_res_inter_cfg_func(cfg):
    cfg.model['neck']['res_inter'] = True
    cfg.model['head']['predictor']['res_inter'] = True
    return cfg


def base_mtm_x_cfg_func(x):
    def _func(cfg):
        cfg.model['base_momentum'] = x
        return cfg
    return _func
