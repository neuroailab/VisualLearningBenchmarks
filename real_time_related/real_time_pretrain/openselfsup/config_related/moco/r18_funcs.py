import os
import openselfsup


rdpd_sm_pipeline = dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='ResizeCenterPad')],
        p=0.6)
def rdpd_sm_cfg_func(cfg):
    cfg.data['train']['pipeline'].insert(
            1, rdpd_sm_pipeline)
    return cfg

# different paths on multiple servers
USER_NAME = os.getlogin()
FS_BASE = f'/data1/{USER_NAME}/pub_clean_related/real_time_related'
FS_BASE = os.environ.get('FS_BASE', FS_BASE)
IMG_FACE_LIST_FILE = os.path.join(FS_BASE, 'img_face_train.txt')
IMG_FACE_ROOT = f'/data5/{USER_NAME}/Dataset'
IMG_FACE_ROOT = os.environ.get('IMG_FACE_ROOT', IMG_FACE_ROOT)

def img_face_cfg_func(cfg):
    cfg.data['train']['data_source']['list_file'] = IMG_FACE_LIST_FILE
    cfg.data['train']['data_source']['root'] = IMG_FACE_ROOT
    return cfg


def img_face_rdpd_sm_cfg_func(cfg):
    cfg = rdpd_sm_cfg_func(cfg)
    cfg = img_face_cfg_func(cfg)
    return cfg


def img_face_rdpd_sm_pos5prob8_cfg_func(cfg):
    prob8_rdpd_sm_pipeline = dict(
            type='RandomAppliedTrans',
            transforms=[dict(type='ResizeCenterPad')],
            p=0.8)
    cfg.data['train']['pipeline'].insert(
            5, prob8_rdpd_sm_pipeline)
    cfg = img_face_cfg_func(cfg)
    return cfg


def avoid_update_in_forward(cfg):
    cfg.model['update_in_forward'] = False
    return cfg


def use_sync_bn(cfg):
    cfg.model['sync_bn'] = True
    return cfg


def small_queue(cfg):
    cfg.model['queue_len'] = 1024
    return cfg


def moco_simclr_concat_neg(cfg):
    cfg.model['type'] = 'MOCOSimCLR'
    cfg.model['neck']['type'] = 'NonLinearNeckV2'
    cfg.model['queue_len'] = 512
    cfg.model['concat_neg'] = True
    return cfg


def moco_simclr_concat_more_neg(cfg):
    cfg.model['type'] = 'MOCOSimCLR'
    cfg.model['neck']['type'] = 'NonLinearNeckV2'
    cfg.model['queue_len'] = 1536
    cfg.model['concat_neg'] = True
    return cfg


def simclr_neck(cfg):
    cfg.model['neck'] = dict(
        type='NonLinearNeckSimCLR',
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True)
    return cfg


def simclr_neck_nosybn_noltbn(cfg):
    cfg.model['neck'] = dict(
        type='NonLinearNeckSimCLR',
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True,
        with_last_bn=False,
        sync_bn=False,
        )
    return cfg


def simclr_neck_tau(cfg):
    cfg = simclr_neck_tau(cfg)
    cfg.model['head']['temperature'] = 0.1
    return cfg


def ms_concat_more_simneck(cfg):
    cfg = moco_simclr_concat_more_neg(cfg)
    cfg = simclr_neck_tau(cfg)
    return cfg


def ms_concat_more_simneck_fn(cfg):
    cfg = moco_simclr_concat_more_neg(cfg)
    cfg = simclr_neck_tau(cfg)
    cfg.model['head']['neg_fn_num'] = 1600
    return cfg


def two_queues(cfg):
    cfg.model['two_queues'] = True
    return cfg
