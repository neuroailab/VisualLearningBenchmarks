def moco_cld(cfg):
    cfg.model = dict(type='CLDMoCo')
    cfg.data['train']['type'] = 'TriContrastiveDataset'
    return cfg
