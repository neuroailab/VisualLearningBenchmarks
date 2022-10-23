from ..basic_param_setter import ParamsBuilder
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.dino_cfg_funcs as dino_cfg_funcs
from openselfsup.models.dino import TeacherTempHook, WDScheduleHook, DistOptimizerHook
from ..byol.r18 import add_byol_hook_to_params, BASIC_BYOL_EP300_CFG
import copy
import os


def add_all_hooks(params):
    params = add_byol_hook_to_params(params, 1)
    if isinstance(params['extra_hook_params'], dict):
        params['extra_hook_params'] = [params['extra_hook_params']]
    params['extra_hook_params'].append({'builder': WDScheduleHook})
    params['extra_hook_params'].append({'builder': TeacherTempHook})
    return params


def add_freeze(params):
    params['optimizer_hook_params']['builder'] = DistOptimizerHook
    params['optimizer_hook_params']['builder_kwargs'].update(
            {'freeze_last_layer': dino_cfg_funcs.default_args.freeze_last_layer,
             'last_layer_dim': dino_cfg_funcs.default_args.out_dim})
    return params


def get_typical_ep100_params(
        args, exp_id, cfg_func=None, 
        cfg_path=BASIC_BYOL_EP300_CFG,
        batch_size=128, 
        builder_class=ParamsBuilder,
        **kwargs):
    def _apply_func(cfg):
        if cfg_func is not None:
            cfg = cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = builder_class(
            args=args, exp_id=exp_id, cfg_path=cfg_path, 
            add_svm_val=True, col_name='dino_in',
            cfg_change_func=_apply_func,
            col_name_in_work_dir=True,
            model_find_unused=None,
            opt_use_fp16=True,
            **kwargs)
    params = param_builder.build_params()
    params = add_all_hooks(params)
    return params

def vit_s_ncrp(args):
    params = get_typical_ep100_params(
            args, 'vit_s_ncrp_s0',
            cfg_func=dino_cfg_funcs.vit_s_ncrp,
            seed=0)
    return params

def vit_s_ncrp_corr_s0(args):
    params = get_typical_ep100_params(
            args, 'vit_s_ncrp_corr_s0_2',
            cfg_func=dino_cfg_funcs.vit_s_ncrp_corr,
            opt_grad_clip={'max_norm': 3.0},
            seed=0)
    params = add_freeze(params)
    return params

def vit_s_ncrp_corr_face_s0(args):
    params = get_typical_ep100_params(
            args, 'vit_s_ncrp_corr_face_s0',
            cfg_func=gnrl_funcs.sequential_func(
                dino_cfg_funcs.vit_s_ncrp_corr,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ),
            opt_grad_clip={'max_norm': 3.0},
            seed=0)
    params = add_freeze(params)
    return params

def vit_s_ncrp_corr_face_mlp5_s0(args):
    params = get_typical_ep100_params(
            args, 'vit_s_ncrp_corr_face_mlp5_s0',
            cfg_func=gnrl_funcs.sequential_func(
                dino_cfg_funcs.vit_s_ncrp_corr,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                dino_cfg_funcs.mlp_w_layer(5),
                ),
            opt_grad_clip={'max_norm': 3.0},
            seed=0)
    params = add_freeze(params)
    return params

def vit_s_ncrp_corrwd_face300_s0(args):
    params = get_typical_ep100_params(
            args, 'vit_s_ncrp_corrwd_face300_s0',
            cfg_func=gnrl_funcs.sequential_func(
                dino_cfg_funcs.vit_s_ncrp_corr_wd,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                gnrl_funcs.resX(300),
                ),
            opt_grad_clip={'max_norm': 3.0},
            seed=0)
    params = add_freeze(params)
    return params

def vit_s_ncrp_corrwd_p8_face_s0(args):
    params = get_typical_ep100_params(
            args, 'vit_s_ncrp_corrwd_p8_face_s0',
            cfg_func=gnrl_funcs.sequential_func(
                dino_cfg_funcs.vit_s_ncrp_corr_wd,
                dino_cfg_funcs.vit_change_to_p8,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ),
            opt_grad_clip={'max_norm': 3.0},
            seed=0)
    params = add_freeze(params)
    return params

def vit_s_ncrp_corr_wd_s0(args):
    params = get_typical_ep100_params(
            args, 'vit_s_ncrp_corr_wd_s0',
            cfg_func=dino_cfg_funcs.vit_s_ncrp_corr_wd,
            opt_grad_clip={'max_norm': 3.0},
            seed=0)
    params = add_freeze(params)
    return params

def neg_vit_s_ncrp(args):
    params = get_typical_ep100_params(
            args, 'neg_vit_s_ncrp',
            cfg_func=dino_cfg_funcs.neg_vit_s_ncrp,
            seed=0)
    return params

def neg_nsf_vit_s_ncrp(args):
    params = get_typical_ep100_params(
            args, 'neg_nsf_vit_s_ncrp',
            cfg_func=dino_cfg_funcs.neg_nsf_vit_s_ncrp,
            seed=0)
    return params

def neg_nsf_vit_s_ncrp_corr_wd_face(args):
    params = get_typical_ep100_params(
            args, 'neg_nsf_vit_s_ncrp_corr_wd_face',
            cfg_func=gnrl_funcs.sequential_func(
                dino_cfg_funcs.neg_nsf_vit_s_ncrp_corr_wd,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ),
            seed=0)
    return params

def neg_nsf_vit_s_ncrp_corr_wd(args):
    params = get_typical_ep100_params(
            args, 'neg_nsf_vit_s_ncrp_corr_wd',
            cfg_func=dino_cfg_funcs.neg_nsf_vit_s_ncrp_corr_wd,
            seed=0)
    return params

def vit_s_ncrp_s112(args):
    params = get_typical_ep100_params(
            args, 'vit_s_ncrp_s112_s0',
            cfg_func=gnrl_funcs.sequential_func(
                dino_cfg_funcs.vit_s_ncrp,
                gnrl_funcs.res112,
                ),
            seed=0)
    return params

def vit_s_ncrp_p8_s112_s0(args):
    params = get_typical_ep100_params(
            args, 'vit_s_ncrp_p8_s112_s0',
            cfg_func=gnrl_funcs.sequential_func(
                dino_cfg_funcs.vit_s_ncrp_p8,
                gnrl_funcs.res112,
                ),
            seed=0)
    return params
