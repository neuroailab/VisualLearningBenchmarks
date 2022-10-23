import copy
import os
from ..datasets import svm_eval as svm_eval

USER_NAME = os.getlogin()
SAYCam_root = os.environ.get(
        'SAYCAM_ROOT',
        f'/data5/{USER_NAME}/Dataset/infant_headcam/jpgs_extracted')
SAYCam_meta_root = os.environ.get(
        'SAYCAM_META_ROOT',
        './metas')


def add_vertical_flip(cfg):
    def _add_flip_to_one_pipeline(pipeline):
        has_flip = False
        for _item in pipeline:
            if _item['type'] == 'RandomVerticalFlip':
                if _item.get('p', 0) == 1:
                    has_flip = True
        if has_flip:
            print('Already has flip, Skip!')
        else:
            pipeline.insert(
                -2, dict(type='RandomVerticalFlip', p=1))
    def _add_flip_to_one_data(data):
        if 'pipeline1' in data:
            _add_flip_to_one_pipeline(data['pipeline1'])
            _add_flip_to_one_pipeline(data['pipeline2'])
        elif 'pipeline' in data:
            _add_flip_to_one_pipeline(data['pipeline'])
        else:
            raise NotImplementedError
    if 'train1' in cfg.data:
        _add_flip_to_one_data(cfg.data['train1'])
        _add_flip_to_one_data(cfg.data['train2'])
    elif 'train' in cfg.data:
        _add_flip_to_one_data(cfg.data['train'])
    else:
        raise NotImplementedError
    return cfg


def scale_total_epochs(total_epochs):
    def _func(cfg):
        cfg.data['train']['data_source']['scale_total_epochs'] = total_epochs
        if 'train1' in cfg.data:
            cfg.data['train1']['data_source']['scale_total_epochs'] = total_epochs
        if 'train2' in cfg.data:
            cfg.data['train2']['data_source']['scale_total_epochs'] = total_epochs
        return cfg
    return _func


SAM_ONLY_CONT_EP100_META = os.path.join(SAYCam_meta_root, 'meta_for_sam_cont_ep100_fx.txt')
SAYCam_num_frames_meta_file = os.path.join(SAYCam_meta_root, 'num_frames_meta.txt')
CONT_DATA_SOURCE_DICT = {
        'type': 'SAYCamCont',
        'list_file': SAM_ONLY_CONT_EP100_META,
        'num_frames_meta_file': SAYCam_num_frames_meta_file,
        'root': SAYCam_root,
        }
def cont_sam_only_ep100_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg = add_vertical_flip(cfg)
    return cfg

def cotrain_sam_only_half_ep100_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['one_epoch_img_num'] = 1281167 // 2
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source']['type'] = 'SAYCamContAccu'
    cfg = add_vertical_flip(cfg)
    return cfg

def cotr_cont_set_window_size(learn_window_size):
    def _func(cfg):
        if 'train1' in cfg.data:
            cfg.data['train1']['data_source']['learn_window_size'] = learn_window_size
        else:
            cfg.data['train']['data_source']['learn_window_size'] = learn_window_size
        return cfg
    return _func

def cotr_switch_datasource(cfg):
    train1_num = cfg.data['train1']['data_source']['one_epoch_img_num']
    train2_num = cfg.data['train2']['data_source']['one_epoch_img_num']
    _source = cfg.data['train1']['data_source']
    cfg.data['train1']['data_source'] = cfg.data['train2']['data_source']
    cfg.data['train2']['data_source'] = _source
    cfg.data['train1']['data_source']['one_epoch_img_num'] = train1_num
    cfg.data['train2']['data_source']['one_epoch_img_num'] = train2_num
    return cfg

def set_aggre_window_size(aggre_window):
    def _func(cfg):
        if 'train1' in cfg.data:
            cfg.data['train1']['data_source']['aggre_window'] = aggre_window
            cfg.data['train1']['type'] = 'ContrastiveTwoImageDataset'
            cfg.data['train2']['data_source']['aggre_window'] = aggre_window
            cfg.data['train2']['type'] = 'ContrastiveTwoImageDataset'
        else:
            cfg.data['train']['data_source']['aggre_window'] = aggre_window
            cfg.data['train']['type'] = 'ContrastiveTwoImageDataset'
        return cfg
    return _func

def set_aggre_window_size_keep_type(aggre_window):
    def _func(cfg):
        if 'train1' in cfg.data:
            cfg.data['train1']['data_source']['aggre_window'] = aggre_window
            cfg.data['train2']['data_source']['aggre_window'] = aggre_window
        else:
            cfg.data['train']['data_source']['aggre_window'] = aggre_window
        return cfg
    return _func

def storage_sam_only_half_ep100_cfg_func(cfg):
    cfg.data['train']['data_source'] = copy.deepcopy(CONT_DATA_SOURCE_DICT)
    cfg.data['train']['data_source']['list_file'] = SAM_ONLY_CONT_EP100_META
    cfg.data['train']['data_source']['type'] = 'SAYCamContAccu'
    cfg = add_vertical_flip(cfg)
    return cfg


def eq_scale_datasets(scale_ratio):
    def _func(cfg):
        train1_num = cfg.data['train1']['data_source']['one_epoch_img_num']
        train2_num = cfg.data['train2']['data_source']['one_epoch_img_num']
        new_train2_num = scale_ratio * train2_num
        new_train2_num = \
                ((train1_num + train2_num) \
                 / (new_train2_num + train1_num)) \
                * new_train2_num
        new_train2_num = int(new_train2_num)
        new_train2_num -= new_train2_num % scale_ratio
        new_train1_num = new_train2_num // scale_ratio
        cfg.data['train1']['data_source']['one_epoch_img_num'] = new_train1_num
        cfg.data['train2']['data_source']['one_epoch_img_num'] = new_train2_num
        return cfg
    return _func

def scale_both_datasets(scale_ratio):
    def _func(cfg):
        cfg.data['train1']['data_source']['one_epoch_img_num'] *= scale_ratio
        cfg.data['train2']['data_source']['one_epoch_img_num'] *= scale_ratio
        return cfg
    return _func
