from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.dino_cfg_funcs as dino_cfg_funcs
from openselfsup.models.dino import TeacherTempHook, WDScheduleHook, DistOptimizerHook
from ..byol.sam_s112ep100 import add_byol_hook_to_params, BASIC_BYOL_EP300_CFG
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
        builder_class=SAYCamParamBuilder,
        opt_use_fp16=True,
        **kwargs):
    def _apply_func(cfg):
        if cfg_func is not None:
            cfg = cfg_func(cfg)
        cfg = gnrl_funcs.res112(cfg)
        cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = builder_class(
            args=args, exp_id=exp_id, cfg_path=cfg_path, 
            add_svm_val=True, col_name='dino_sy_sep',
            cfg_change_func=_apply_func,
            col_name_in_work_dir=True,
            model_find_unused=None,
            opt_use_fp16=opt_use_fp16,
            **kwargs)
    params = param_builder.build_params()
    params = add_all_hooks(params)
    return params


COTRAIN_PARAMS = dict(
        builder_class=CotrainSAYCamParamBuilder,
        mix_weight=1.0, batch_size=64, concat_batches=True,
        )

def add_cotr_ct_wd_x_scale_y_func(
        func_name, x, y, 
        seed=0, exp_name=None,
        post_func=lambda cfg: cfg,
        dino_func=dino_cfg_funcs.vit_s_ncrp_corr_wd,
        do_freeze=True,
        which_cotr=saycam_funcs.cotrain_sam_only_half_ep100_cfg_func,
        **kwargs):
    def _func(args):
        local_kwargs = copy.copy(COTRAIN_PARAMS)
        curr_bs = local_kwargs.pop('batch_size')
        local_kwargs.update(kwargs)
        params = get_typical_ep100_params(
                args, exp_name or func_name,
                gnrl_funcs.sequential_func(
                    dino_func,
                    which_cotr,
                    saycam_funcs.cotr_cont_set_window_size(x),
                    saycam_funcs.eq_scale_datasets(y),
                    post_func),
                opt_grad_clip={'max_norm': 3.0}, seed=seed, 
                scale_ratio=y, batch_size=2 * curr_bs // (1+y),
                **local_kwargs)
        if do_freeze:
            params = add_freeze(params)
        return params
    all_things = globals()
    all_things[func_name] = _func

add_cotr_ct_wd_x_scale_y_func(
        'vsn_sam_cotr_wd20m_eq3_aw5',
        20 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_cotr_ct_wd_x_scale_y_func(
        'vsn_sam_cotr_wd30s_eq3_aw5',
        0.5 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_cotr_ct_wd_x_scale_y_func(
        'vsn_sam_cotr_wd30s_eq1_aw5',
        0.5 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_cotr_ct_wd_x_scale_y_func(
        'vsn_sam_cotr_wd20m_eq1_aw5',
        20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5))

add_cotr_ct_wd_x_scale_y_func(
        'vsn_sam_cotr_wd30s_eqd3_aw5',
        0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ),
        opt_use_fp16=False)
add_cotr_ct_wd_x_scale_y_func(
        'vsn_sam_cotr_wd20m_eqd3_aw5',
        20 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ))

add_cotr_ct_wd_x_scale_y_func(
        'neg_vsn_sam_cotr_wd20m_eq3_aw5',
        20 * 60 * 25, 3,
        dino_func=dino_cfg_funcs.neg_nsf_vit_s_ncrp_corr_wd,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5),
        do_freeze=False)
add_cotr_ct_wd_x_scale_y_func(
        'neg_vsn_sam_cotr_wd30s_eq3_aw5',
        0.5 * 60 * 25, 3,
        dino_func=dino_cfg_funcs.neg_nsf_vit_s_ncrp_corr_wd,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5),
        do_freeze=False)

add_cotr_ct_wd_x_scale_y_func(
        'neg_vsn_sam_cotr_wd30s_eq1_aw5',
        0.5 * 60 * 25, 1,
        dino_func=dino_cfg_funcs.neg_nsf_vit_s_ncrp_corr_wd,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5),
        do_freeze=False)
add_cotr_ct_wd_x_scale_y_func(
        'neg_vsn_sam_cotr_wd20m_eq1_aw5',
        20 * 60 * 25, 1,
        dino_func=dino_cfg_funcs.neg_nsf_vit_s_ncrp_corr_wd,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5),
        do_freeze=False)

add_cotr_ct_wd_x_scale_y_func(
        'neg_vsn_sam_cotr_wd30s_eqd3_aw5',
        0.5 * 60 * 25, 3,
        dino_func=dino_cfg_funcs.neg_nsf_vit_s_ncrp_corr_wd,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ),
        do_freeze=False)
add_cotr_ct_wd_x_scale_y_func(
        'neg_vsn_sam_cotr_wd20m_eqd3_aw5',
        20 * 60 * 25, 3,
        dino_func=dino_cfg_funcs.neg_nsf_vit_s_ncrp_corr_wd,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ),
        do_freeze=False)
