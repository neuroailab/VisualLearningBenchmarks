from ..basic_param_setter import ParamsBuilder
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.mae_cfg_funcs as mae_cfg_funcs
from ..simclr.r18 import BASIC_SIMCLR_CFG
import copy
import os


def get_typical_params(
        args, exp_id, cfg_func=None, 
        cfg_path=BASIC_SIMCLR_CFG,
        batch_size=512, 
        builder_class=ParamsBuilder,
        model_find_unused=None,
        **kwargs):
    def _apply_func(cfg):
        if cfg_func is not None:
            cfg = cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = builder_class(
            args=args, exp_id=exp_id, cfg_path=cfg_path, 
            add_svm_val=True, col_name='mae_in',
            cfg_change_func=_apply_func,
            col_name_in_work_dir=True,
            model_find_unused=model_find_unused,
            opt_use_fp16=True,
            valid_interval=10,
            **kwargs)
    params = param_builder.build_params()
    return params

def small_vit(args):
    params = get_typical_params(
            args, 'small_vit_2',
            cfg_func=mae_cfg_funcs.small_vit,
            seed=0)
    return params

def small_vit_ep300(args):
    params = get_typical_params(
            args, 'small_vit_ep300',
            cfg_func=mae_cfg_funcs.small_vit_ep300,
            seed=0)
    return params

def small_vit_ep800_corr_wd_face(args):
    params = get_typical_params(
            args, 'small_vit_ep800_corr_wd_face',
            cfg_func=gnrl_funcs.sequential_func(
                mae_cfg_funcs.small_vit_ep400_corr_wd,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ),
            seed=0)
    return params

def small_vit_ep400_corr_wd_bs4096(args):
    params = get_typical_params(
            args, 'small_vit_ep400_corr_wd_bs4096',
            cfg_func=mae_cfg_funcs.small_vit_ep400_corr_wd_bs4096,
            seed=0, opt_update_interval=4)
    return params

def neg_vit_ssctr_ep200_corr_wd_face(args):
    params = get_typical_params(
            args, 'neg_vit_ssctr_ep200_corr_wd_face',
            gnrl_funcs.sequential_func(
                mae_cfg_funcs.neg_small_vit_ep200_corr_wd,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ),
            seed=0)
    return params

def pt_vit_s_ep400_corr_wd_bs4096(args):
    params = get_typical_params(
            args, 'pt_vit_s_ep400_corr_wd_bs4096',
            cfg_func=mae_cfg_funcs.pt_vit_s_ep400_corr_wd_bs4096,
            seed=0, opt_update_interval=4,
            model_find_unused=True)
    return params

def large_vit_ep300(args):
    params = get_typical_params(
            args, 'large_vit_ep300',
            cfg_func=mae_cfg_funcs.large_vit_ep300,
            seed=0,
            batch_size=128, opt_update_interval=2)
    return params

def large_vit_face_ep300(args):
    params = get_typical_params(
            args, 'large_vit_face_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                mae_cfg_funcs.large_vit_ep300,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ),
            seed=0,
            batch_size=128, opt_update_interval=2)
    return params
