def ep300_cfg_func(cfg):
    cfg.total_epochs = 300
    cfg.optimizer = dict(
            type='LARS', lr=4.8, weight_decay=0.000001, momentum=0.9,
            paramwise_options={
                '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                'bias': dict(weight_decay=0., lars_exclude=True)})
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=0.,
            warmup='linear',
            warmup_iters=10,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg


def ep300_fxPrd_cfg_func(cfg):
    cfg.total_epochs = 300
    cfg.optimizer = dict(
            type='LARS', lr=4.8, weight_decay=0.000001, momentum=0.9,
            paramwise_options={
                '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                'bias': dict(weight_decay=0., lars_exclude=True),
                'predictor': dict(use_fix_lr=0.1),
                })
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=0.,
            warmup='linear',
            warmup_iters=10,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg


def ep300_SGD_cfg_func(cfg):
    cfg.total_epochs = 300
    cfg.optimizer = dict(type='SGD', lr=0.8, weight_decay=0.0001, momentum=0.9)
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=0.,
            warmup='linear',
            warmup_iters=10,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg


def ep300_SGD_BNex_cfg_func(cfg):
    cfg.total_epochs = 300
    cfg.optimizer = dict(
            type='SGD', lr=0.8, weight_decay=0.0001, momentum=0.9,
            paramwise_options={
                '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
                'bias': dict(weight_decay=0.)})
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=0.,
            warmup='linear',
            warmup_iters=10,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg


def ep300_SGD_lrfx_cfg_func(cfg):
    cfg.total_epochs = 300
    cfg.optimizer = dict(type='SGD', lr=0.8, weight_decay=0.0001, momentum=0.9)
    cfg.lr_config = dict(
            policy='Fixed',
            warmup='linear',
            warmup_iters=10,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg


def ep300_SGD_fxPrd_cfg_func(cfg):
    cfg.total_epochs = 300
    cfg.optimizer = dict(
            type='SGD', lr=0.1, weight_decay=0.0001, momentum=0.9,
            paramwise_options={
                'predictor': dict(use_fix_lr=0.1)},
            )
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=0.,
            warmup='linear',
            warmup_iters=10,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg
