from openselfsup.config_related.moco.r18_funcs import IMG_FACE_LIST_FILE, IMG_FACE_ROOT

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


def img_rdpd_face_cfg_func(cfg):
    rdpd_sm_pipeline = dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='ResizeCenterPad')],
        p=0.6)
    cfg.data['train']['pipeline1'].insert(1, rdpd_sm_pipeline)
    cfg.data['train']['pipeline2'].insert(1, rdpd_sm_pipeline)
    cfg.data['train']['data_source']['list_file'] = IMG_FACE_LIST_FILE
    cfg.data['train']['data_source']['root'] = IMG_FACE_ROOT
    return cfg


def mlp_3layers_rdpd_face_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 3
    cfg.model['head']['predictor']['num_layers'] = 3
    cfg = img_rdpd_face_cfg_func(cfg)
    return cfg


def mlp_4layers_rdpd_face_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    cfg.model['head']['predictor']['num_layers'] = 4
    cfg = img_rdpd_face_cfg_func(cfg)
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
