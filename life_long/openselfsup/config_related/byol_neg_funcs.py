def byol_neg_cfg_func(cfg):
    #cfg.model['predictor'] = cfg.model['head']['predictor']
    cfg.model['predictor'] = dict(type='Identity')
    cfg.model['head'] = dict(type='ContrastiveHead', temperature=0.1)
    cfg.model['type'] = 'BYOLNeg'
    return cfg


def sepb_byol_posneg_cfg_func_w_sr(scale_ratio=1):
    def _func(cfg):
        cfg.model['neg_head'] = dict(type='ContrastiveHead', temperature=0.1)
        cfg.model['type'] = 'SepBatchBYOLPosNeg'
        cfg.model['scale_ratio'] = scale_ratio
        return cfg
    return _func
