import os
import copy

USER_NAME = os.getlogin()
ego4d_root = os.environ.get(
        'EGO4D_ROOT',
        f'/data5/{USER_NAME}/Dataset/ego4d/jpgs')
SAYCam_meta_root = os.environ.get(
        'SAYCAM_META_ROOT',
        '/mnt/fs1/Dataset/infant_headcam/')
ego4d_meta_root = os.path.join(
        SAYCam_meta_root, '..', 'ego4d')


ego4d_num_frames_meta_file = os.path.join(ego4d_meta_root, 'num_frames_meta.txt')
ep300_meta_file = os.path.join(ego4d_meta_root, 'meta_for_ep300.txt')
ep100_meta_file = os.path.join(ego4d_meta_root, 'meta_for_ep100.txt')
LIST_DATA_SOURCE_DICT = {
        'type': 'SAYCamFromList',
        'list_file': ep300_meta_file,
        'num_frames_meta_file': ego4d_num_frames_meta_file,
        'root': ego4d_root,
        }
def ctl_short_ep300_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(LIST_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['one_epoch_img_num'] = 1281167 // 3
    return cfg


CONT_DATA_SOURCE_DICT = {
        'type': 'SAYCamCont',
        'list_file': ep300_meta_file,
        'num_frames_meta_file': ego4d_num_frames_meta_file,
        'root': ego4d_root,
        }
def cotrain_short_ep300_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['one_epoch_img_num'] = 1281167 // 6
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source']['type'] = 'SAYCamContAccu'
    return cfg

def cotrain_ep100_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['one_epoch_img_num'] = 1281167 // 2
    cfg.data['train']['data_source']['list_file'] = ep100_meta_file
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source']['type'] = 'SAYCamContAccu'
    return cfg
