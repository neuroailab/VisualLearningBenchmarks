def add_model_setting(cfg):
    input_size = 224
    patch_size = 16
    cfg.model = dict(
            type='MAE',
            model_name='pretrain_mae_small_patch16_224', 
            mask_ratio=0.75, 
            window_size=(
                input_size // patch_size, 
                input_size // patch_size),
            )
    return cfg

def add_AdamW(cfg):
    cfg.total_epochs = 1600
    cfg.optimizer = dict(
            type='AdamW', lr=6e-4,
            weight_decay=0.05,
            betas=(0.9, 0.95),
            )
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=1e-5,
            warmup='linear',
            warmup_iters=40,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg

def change_data_transform(cfg):
    p = cfg.data['train']['pipeline']
    p = [p[0], p[-2], p[-1]]
    cfg.data['train']['pipeline'] = p
    cfg.data['train']['type'] = 'ExtractDataset'
    return cfg

def basic_settings(cfg):
    cfg = add_model_setting(cfg)
    cfg = add_AdamW(cfg)
    cfg = change_data_transform(cfg)
    return cfg

def small_vit(cfg):
    cfg = basic_settings(cfg)
    return cfg

def small_vit_ep300(cfg):
    cfg = basic_settings(cfg)
    cfg.total_epochs = 300
    cfg.lr_config['warmup_iters'] = 20
    return cfg

def small_vit_ep800_corr_wd(cfg):
    cfg = basic_settings(cfg)
    cfg.total_epochs = 800
    cfg.optimizer['paramwise_options']= {
            '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.)}
    return cfg

def small_vit_ep400_corr_wd(cfg):
    cfg = basic_settings(cfg)
    cfg.total_epochs = 400
    cfg.optimizer['paramwise_options']= {
            '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.)}
    return cfg

def small_vit_ep400_corr_wd_bs4096(cfg):
    cfg = basic_settings(cfg)
    cfg.total_epochs = 400
    cfg.optimizer['paramwise_options']= {
            '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'norm(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.)}
    cfg.optimizer['lr'] = 1.5e-4  * (4096/256)
    return cfg

def small_vit_ep100_corr_wd(cfg):
    cfg = basic_settings(cfg)
    cfg.total_epochs = 100
    cfg.optimizer['paramwise_options']= {
            '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.)}
    cfg.lr_config['warmup_iters'] = 10
    return cfg

def use_two_image(cfg):
    cfg.model['two_image_input'] = True
    return cfg


def change_to_res112(cfg):
    input_size = 112
    patch_size = 16
    cfg.model.update(dict(
            model_name='pretrain_mae_small_patch16_112',
            window_size=(
                input_size // patch_size, 
                input_size // patch_size),
            ))
    return cfg

def neg_basic_setting(cfg):
    cfg.model['type'] = 'MAENeg'
    cfg.model['head'] = dict(type='ContrastiveHead', temperature=0.1)
    cfg.model['neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=384 * 4,
            hid_channels=2048,
            out_channels=256,
            num_layers=2,
            with_avg_pool=True)
    cfg.data['train']['type'] = 'ContrastiveDataset'
    return cfg

def neg_small_vit_ep200_corr_wd_bs4096(cfg):
    cfg = small_vit_ep400_corr_wd_bs4096(cfg)
    cfg.total_epochs = 200
    cfg = neg_basic_setting(cfg)
    return cfg

def neg_small_vit_ep200_corr_wd(cfg):
    cfg = basic_settings(cfg)
    cfg.optimizer['paramwise_options']= {
            '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'norm(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.)}
    cfg.total_epochs = 200
    cfg = neg_basic_setting(cfg)
    return cfg


def use_pt_vit_s_model(cfg):
    cfg.model = dict(
            type='PTMAE',
            model_name='mae_vit_small_patch16_dec512d8b', 
            mask_ratio=0.75, 
            )
    return cfg

def pt_vit_s_ep400_corr_wd_bs4096(cfg):
    cfg = basic_settings(cfg)
    cfg.total_epochs = 400
    cfg.optimizer['paramwise_options']= {
            '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'norm(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.)}
    cfg.optimizer['lr'] = 1.5e-4  * (4096/256)
    cfg = use_pt_vit_s_model(cfg)
    return cfg

def neg_pt_vit_s_ep200_corr_wd_bs4k(cfg):
    cfg = basic_settings(cfg)
    cfg.total_epochs = 200
    cfg.optimizer['paramwise_options']= {
            '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'norm(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.)}
    cfg.optimizer['lr'] = 1.5e-4  * (4096/256)
    cfg = use_pt_vit_s_model(cfg)
    cfg.model['type'] = 'PTMAENeg'
    cfg.model['head'] = dict(type='ContrastiveHead', temperature=0.1)
    cfg.model['neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=384,
            hid_channels=2048,
            out_channels=256,
            num_layers=3,
            with_avg_pool=True)
    cfg.model['neck_last_n'] = 1
    cfg.data['train']['type'] = 'ContrastiveDataset'
    return cfg

def base_vit(cfg):
    cfg = basic_settings(cfg)
    cfg.model['model_name'] = 'pretrain_mae_base_patch16_224'
    return cfg

def large_vit(cfg):
    cfg = basic_settings(cfg)
    cfg.model['model_name'] = 'pretrain_mae_large_patch16_224'
    return cfg

def large_vit_ep300(cfg):
    cfg = basic_settings(cfg)
    cfg.model['model_name'] = 'pretrain_mae_large_patch16_224'
    cfg.total_epochs = 300
    cfg.lr_config['warmup_iters'] = 20
    return cfg
