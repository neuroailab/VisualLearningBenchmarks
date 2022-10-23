import os

def sequential_func(*args):
    def ret_func(cfg):
        for _func in args:
            cfg = _func(cfg)
        return cfg
    return ret_func


def resX(X):
    def _func(cfg):
        def _change_one(data_cfg):
            if 'pipeline' in data_cfg:
                data_cfg['pipeline'][0]['size'] = X
            elif 'pipeline1' in data_cfg:
                data_cfg['pipeline1'][0]['size'] = X 
                data_cfg['pipeline2'][0]['size'] = X
            else:
                raise NotImplementedError
            if 'size_crops' in data_cfg:
                # For multi-crop dataset
                data_cfg['size_crops'] = [X, 96 // (224 / X)]
            return data_cfg
        cfg.data['train'] = _change_one(cfg.data['train'])
        if 'train1' in cfg.data:
            cfg.data['train1'] = _change_one(cfg.data['train1'])
        if 'train2' in cfg.data:
            cfg.data['train2'] = _change_one(cfg.data['train2'])
        return cfg
    return _func


def res112(cfg):
    return resX(112)(cfg)

def change_to_same_aug_type(cfg):
    if 'train1' in cfg.data:
        cfg.data['train1']['type'] = 'SamAugTwoImageDataset'
        cfg.data['train2']['type'] = 'SamAugTwoImageDataset'
    else:
        cfg.data['train']['type'] = 'SamAugTwoImageDataset'
    return cfg


USER_NAME = os.getlogin()
meta_dir = os.environ.get(
        'META_DIR',
        f'/mnt/fs4/{USER_NAME}/openselfsup_models/metas/')
def use_IN_gx_meta(x, num_classes=100):
    def _func(cfg):
        cfg.data['train']['data_source']['list_file'] = os.path.join(
                meta_dir, 'train_cls_g{}_labeled.txt'.format(x))
        cfg.model['head']['num_classes'] = num_classes
        return cfg
    return _func


def set_total_epochs(total_epochs):
    def _func(cfg):
        cfg.total_epochs = total_epochs
        return cfg
    return _func


def wider_r18_w_param(base_n_channels):
    def _func(cfg):
        cfg.model['backbone']['base_n_channels'] = base_n_channels
        cfg.model['neck']['in_channels'] = 512 // 64 * base_n_channels
        return cfg
    return _func

def per_stage_wider_wp(per_stage_wide, **kwargs):
    def _func(cfg):
        cfg.model['backbone']['per_stage_wide'] = per_stage_wide
        cfg.model['backbone'].update(kwargs)
        cfg.model['neck']['in_channels'] = 512 * per_stage_wide[-1]
        return cfg
    return _func

def per_stage_block_wider_wp(
        per_stage_wide, per_stage_block, **kwargs):
    def _func(cfg):
        cfg.model['backbone']['per_stage_wide'] = per_stage_wide
        cfg.model['backbone']['per_stage_block'] = per_stage_block
        cfg.model['backbone'].update(kwargs)
        cfg.model['neck']['in_channels'] = 512 * per_stage_wide[-1]
        if per_stage_block[-1] == 'Bottleneck':
            cfg.model['neck']['in_channels'] *= 4
        return cfg
    return _func


def change_to_r18(cfg):
    cfg.model['backbone']['depth'] = 18
    cfg.model['head']['in_channels'] = 512
    return cfg

def change_to_r10(cfg):
    cfg.model['backbone']['depth'] = 10
    cfg.model['backbone']['base_n_channels'] = 48
    cfg.model['neck']['in_channels'] = 384
    cfg.model['neck']['hid_channels'] = 1024
    return cfg

def add_backbone_neck_for_sup(cfg):
    cfg.model['backbone']['add_neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=2048,
            out_channels=2048,
            num_layers=2,
            with_avg_pool=True)
    cfg.model['head']['in_channels'] = 2048
    cfg.model['head']['with_avg_pool'] = False
    return cfg

def ep100_sup_SGD_cfg_func(cfg):
    cfg.total_epochs = 100
    cfg.optimizer = dict(
            type='SGD', lr=0.1, weight_decay=0.0001, momentum=0.9,
            )
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=0.,
            warmup='linear',
            warmup_iters=10,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg

def get_update_lr_config_func(**kwargs):
    def _func(cfg):
        cfg.lr_config.update(kwargs)
        return cfg
    return _func

def get_update_opt_func(**kwargs):
    def _func(cfg):
        cfg.optimizer.update(kwargs)
        return cfg
    return _func

def get_use_adam_func(lr):
    def _func(cfg):
        cfg.optimizer = dict(
                type='Adam', lr=lr, weight_decay=0.0001)
        cfg.lr_config = dict(policy='Fixed')
        return cfg
    return _func

def get_ffcv_mtrans_func(mtrans):
    def _func(cfg):
        cfg.data['train']['multi_trans'] = mtrans
        if 'train1' in cfg.data:
            cfg.data['train1']['multi_trans'] = mtrans
            cfg.data['train2']['multi_trans'] = mtrans
        return cfg
    return _func

def get_update_both_source_func(**kwargs):
    def _func(cfg):
        cfg.data['train1']['data_source'].update(kwargs)
        cfg.data['train2']['data_source'].update(kwargs)
        return cfg
    return _func


rdpd_sm_pipeline = dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='ResizeCenterPad')],
        p=0.6)
def rdpd_sm_cfg_func(cfg):
    if 'pipeline' in cfg.data['train']:
        cfg.data['train']['pipeline'].insert(
                1, rdpd_sm_pipeline)
    if 'pipeline1' in cfg.data['train']:
        cfg.data['train']['pipeline1'].insert(
                1, rdpd_sm_pipeline)
        cfg.data['train']['pipeline2'].insert(
                1, rdpd_sm_pipeline)
    return cfg


rdpd_ssm_pipeline = dict(
        type='RandomAppliedTrans',
        transforms=[dict(
            type='ResizeCenterPad',
            min_edge=25)],
        p=0.6)
def rdpd_ssm_cfg_func(cfg):
    if 'pipeline' in cfg.data['train']:
        cfg.data['train']['pipeline'].insert(
                1, rdpd_ssm_pipeline)
    if 'pipeline1' in cfg.data['train']:
        cfg.data['train']['pipeline1'].insert(
                1, rdpd_ssm_pipeline)
        cfg.data['train']['pipeline2'].insert(
                1, rdpd_ssm_pipeline)
    return cfg


IMG_FACE_LIST_FILE = os.path.join(meta_dir, 'img_face_train.txt')
SAYCam_root = os.environ.get(
        'SAYCAM_ROOT',
        f'/data5/{USER_NAME}/Dataset/infant_headcam/jpgs_extracted')
from pathlib import Path
IMG_FACE_ROOT = Path(SAYCam_root).parent.parent.absolute()
def img_face_cfg_func(cfg):
    cfg.data['train']['data_source']['list_file'] = IMG_FACE_LIST_FILE
    cfg.data['train']['data_source']['root'] = IMG_FACE_ROOT
    return cfg

def img_face_rdpd_sm_eqlen_cfg_func(cfg):
    cfg = rdpd_sm_cfg_func(cfg)
    cfg = img_face_cfg_func(cfg)
    cfg.data['train']['data_source']['data_len'] = 1281167
    return cfg
