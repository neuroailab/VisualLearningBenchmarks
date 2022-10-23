import copy
from . import saycam_funcs


def imgnt_cont_cfg_func(cfg):
    cfg.data['train']['data_source']['type'] = 'ImageNetCont'
    cfg.data['train']['data_source']['num_epochs'] = 300
    cfg.data['train']['data_source']['keep_prev_epoch_num'] = 2
    return cfg


def imgnt_cont_ep100_cfg_func(cfg):
    cfg.data['train']['data_source']['type'] = 'ImageNetCont'
    cfg.data['train']['data_source']['num_epochs'] = 100
    cfg.data['train']['data_source']['keep_prev_epoch_num'] = 0
    cfg.data['train']['data_source']['fix_end_idx'] = True
    return cfg


def imgnt_cont_cfg_func_w_ep(epoch):
    def _func(cfg):
        cfg.data['train']['data_source']['type'] = 'ImageNetCont'
        cfg.data['train']['data_source']['num_epochs'] = epoch
        cfg.data['train']['data_source']['keep_prev_epoch_num'] = 0
        cfg.data['train']['data_source']['fix_end_idx'] = True
        return cfg
    return _func


def cotr_imgnt_cont_cfg_func(cfg):
    cfg = imgnt_cont_cfg_func(cfg)
    cfg.data['train']['data_source']['data_len'] = 1281167 // 2
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source']['accu'] = True
    return cfg


def cotr_imgnt_cont_ep100_cfg_func(cfg):
    cfg = imgnt_cont_ep100_cfg_func(cfg)
    cfg.data['train']['data_source']['data_len'] = 1281167 // 2
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source']['accu'] = True
    return cfg


def cotr_imgnt_cont_cfg_func_w_ep(epoch):
    def _func(cfg):
        cfg = imgnt_cont_cfg_func_w_ep(epoch)(cfg)
        cfg.data['train']['data_source']['data_len'] = 1281167 // 2
        cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
        cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
        cfg.data['train2']['data_source']['accu'] = True
        return cfg
    return _func


def cnd_storage_cfg_func(cfg):
    cfg = saycam_funcs.cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['type'] = 'ImageNetCndCont'
    return cfg


def max40k_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_40000'
    return cfg


def max20k_cnd_storage_cfg_func(cfg):
    cfg = cnd_storage_cfg_func(cfg)
    cfg.data['train2']['data_source']['cond_method'] = 'max_20000'
    return cfg


def max_cnd_storage_cfg_func_w_param(k):
    def _func(cfg):
        cfg = cnd_storage_cfg_func(cfg)
        cfg.data['train2']['data_source']['cond_method'] = 'max_{}'.format(k)
        return cfg
    return _func


def get_cnd_loss_func(**kwargs):
    def _func(cfg):
        cfg = cnd_storage_cfg_func(cfg)
        cfg.data['train2']['data_source']['type'] = 'ImageNetCndLoss'
        cfg.data['train2']['data_source']['pipeline'] = cfg.data['train2']['pipeline']
        cfg.data['train2']['data_source']['loss_head'] = dict(
                type='ContrastiveHead', temperature=0.1,
                loss_reduced=False)
        cfg.data['train2']['data_source'].update(kwargs)
        return cfg
    return _func

def get_cnd_mean_sim_func(**kwargs):
    def _func(cfg):
        cfg = cnd_storage_cfg_func(cfg)
        cfg.data['train2']['data_source']['type'] = 'ImageNetCndMeanSim'
        cfg.data['train2']['data_source'].update(kwargs)
        return cfg
    return _func


def imgnt_batchLD_cfg_func(cfg):
    cfg.data['train']['data_source']['type'] = 'ImageNetBatchLD'
    cfg.data['train']['data_source']['batch_size'] = 256
    cfg.data['train']['data_source']['no_cate_per_batch'] = 10
    return cfg


def imgnt_batchLD512_cfg_func(cfg):
    cfg.data['train']['data_source']['type'] = 'ImageNetBatchLD'
    cfg.data['train']['data_source']['batch_size'] = 512
    cfg.data['train']['data_source']['no_cate_per_batch'] = 10
    return cfg


def imgnt_batchLD_w_param(**kwargs):
    def _func(cfg):
        cfg.data['train']['data_source']['type'] = 'ImageNetBatchLD'
        cfg.data['train']['data_source']['batch_size'] = 512
        cfg.data['train']['data_source']['no_cate_per_batch'] = 10
        cfg.data['train']['data_source'].update(kwargs)
        return cfg
    return _func


def cotr_imgnt_cont_acsy_cfg_func(cfg, end_epoch=100):
    cfg = imgnt_cont_cfg_func(cfg)
    cfg.data['train']['data_source']['data_len'] = 1281167 // 2
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source'].update(
            dict(
                type='ImageNetCntAccuSY',
                sy_root=saycam_funcs.SAYCam_root,
                sy_epoch_meta_path=saycam_funcs.SAYCam_list_file_cont_ep300,
                sy_file_num_meta_path=saycam_funcs.SAYCam_num_frames_meta_file,
                sy_end_epoch=end_epoch,
                ))
    return cfg
