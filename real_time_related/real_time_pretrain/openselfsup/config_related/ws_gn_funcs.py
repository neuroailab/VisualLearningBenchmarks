def moco_ws_gn_cfg_func(cfg):
    cfg.model['backbone']['norm_cfg'] = dict(type='GN', num_groups=4)
    cfg.model['backbone']['conv_cfg'] = dict(type='ConvWS')
    cfg.model['neck'] = dict(
            type='NonLinearNeckGNV2',
            in_channels=512,
            hid_channels=2048,
            out_channels=128,
            with_avg_pool=True,
            num_groups=4)
    return cfg


def simclr_ws_gn_cfg_func(cfg):
    cfg.model['backbone']['norm_cfg'] = dict(type='GN', num_groups=4)
    cfg.model['backbone']['conv_cfg'] = dict(type='ConvWS')
    cfg.model['neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=2048,
            out_channels=128,
            num_layers=2,
            with_avg_pool=True,
            use_group_norm=4)
    return cfg


def byol_ws_gn_cfg_func(cfg):
    cfg.model['backbone']['norm_cfg'] = dict(type='GN', num_groups=4)
    cfg.model['backbone']['conv_cfg'] = dict(type='ConvWS')
    cfg.model['neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=1024,
            out_channels=256,
            num_layers=2,
            sync_bn=False,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=True,
            use_group_norm=4)
    cfg.model['head']['predictor'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=256, hid_channels=1024,
            out_channels=256, num_layers=2, sync_bn=False,
            with_bias=True, with_last_bn=False, with_avg_pool=False,
            use_group_norm=4)
    cfg.model['base_momentum'] = 0.999
    return cfg
