def more_hid_bn(cfg):
    cfg.model['neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=1024,
            out_channels=512,
            num_layers=3,
            sync_bn=True,
            with_bias=True,
            with_last_bn=True,
            with_avg_pool=True)
    cfg.model['head']['predictor']['hid_channels'] = 512
    return cfg


def change_to_l4_sim_neck(cfg):
    cfg.model['neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=1024,
            out_channels=512,
            num_layers=4,
            sync_bn=True,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=True)
    cfg.model['head']['predictor'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=1024,
            out_channels=512,
            num_layers=4,
            sync_bn=True,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False)
    return cfg


def add_sim_neck_2048(
        cfg, neck_layers=2, head_pred_layers=2,
        neck_with_last_bn=False,
        ):
    cfg.model['neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=2048,
            out_channels=2048,
            num_layers=neck_layers,
            sync_bn=True,
            with_bias=True,
            with_last_bn=neck_with_last_bn,
            with_avg_pool=True)
    cfg.model['head']['predictor'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=2048,
            hid_channels=512,
            out_channels=2048,
            num_layers=head_pred_layers,
            sync_bn=True,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False)
    return cfg


def sep_crop_sim_neck_2048(cfg):
    cfg.model['type'] = 'SiameseSepCrop'
    cfg = add_sim_neck_2048(cfg)
    return cfg


def sep_crop_sim_neck_2048_w_params(**kwargs):
    def _func(cfg):
        cfg.model['type'] = 'SiameseSepCrop'
        cfg = add_sim_neck_2048(cfg, **kwargs)
        return cfg
    return _func
