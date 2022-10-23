from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
from .byol_hook import BYOLHook, EWCBYOLHook
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.byol_neg_funcs as byol_neg_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
import openselfsup.config_related.byol_funcs as byol_funcs
import copy
import numpy as np


def add_byol_hook_to_params(
        params, update_interval=1, 
        use_ewc_hook=False, fx_up=False):
    byol_hook_params = {
            'builder': BYOLHook,
            'update_interval': update_interval,
            }
    if fx_up:
        byol_hook_params['builder_kwargs'] = {
                'update_interval': update_interval,
                }
    if use_ewc_hook:
        byol_hook_params['builder'] = EWCBYOLHook
    if 'extra_hook_params' not in params:
        params['extra_hook_params'] = byol_hook_params
    else:
        if isinstance(params['extra_hook_params'], dict):
            params['extra_hook_params'] = [params['extra_hook_params']]
        params['extra_hook_params'].append(byol_hook_params)
    return params

BASIC_BYOL_EP300_CFG = './configs/selfsup/byol/r18_ep300.py'
R50_BASIC_BYOL_CFG = './configs/selfsup/byol/r50_bs256_accumulate16_ep300.py'
COTR_R50_KWARGS = dict(
        cfg_path=R50_BASIC_BYOL_CFG,
        builder_class=CotrainSAYCamParamBuilder,
        batch_size=64, opt_update_interval=16,
        mix_weight=1.0, concat_batches=True,
        valid_initial=False,
        )

cotr_kwargs = dict(
        builder_class=CotrainSAYCamParamBuilder,
        mix_weight=1.0, concat_batches=True,
        batch_size=128,
        )
cnd_kwargs = copy.deepcopy(cotr_kwargs)
cnd_kwargs['use_cnd_hook'] = True


def get_typical_params(
        args, exp_id, cfg_func, 
        builder_class=SAYCamParamBuilder,
        cfg_path=BASIC_BYOL_EP300_CFG,
        opt_update_interval=8, batch_size=256, 
        num_epochs=100, post_cfg_func=None,
        need_ewc_hook=False, fx_up=False, 
        col_name='byol_samS',
        **kwargs):
    def _apply_func(cfg):
        cfg = cfg_func(cfg)
        cfg = gnrl_funcs.sequential_func(
                gnrl_funcs.res112,
                ep300_funcs.ep300_cfg_func,
                gnrl_funcs.set_total_epochs(num_epochs),
                )(cfg)
        if post_cfg_func is not None:
            cfg = post_cfg_func(cfg)
        if batch_size is not None:
            cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = builder_class(
            args=args, exp_id=exp_id, cfg_path=cfg_path, 
            add_svm_val=True, col_name=col_name,
            cfg_change_func=_apply_func,
            opt_update_interval=opt_update_interval,
            col_name_in_work_dir=True,
            need_ewc_hook=need_ewc_hook,
            model_find_unused=None,
            **kwargs)
    params = param_builder.build_params()
    params = add_byol_hook_to_params(
            params, opt_update_interval, 
            use_ewc_hook=need_ewc_hook,
            fx_up=fx_up)
    return params


def add_cotr_ct_wd_x_scale_y_func(
        func_name, x, y, 
        mlp=2, seed=None, exp_name=None,
        post_func=lambda cfg: cfg,
        concat='default',
        curr_kwargs=cotr_kwargs):
    def _func(args):
        local_kwargs = copy.copy(curr_kwargs)
        curr_bs = local_kwargs.pop('batch_size')
        if concat is not 'default':
            local_kwargs['concat_batches'] = concat
        params = get_typical_params(
                args, exp_name or func_name,
                gnrl_funcs.sequential_func(
                    saycam_funcs.cotrain_sam_only_half_ep100_cfg_func,
                    byol_funcs.mlp_layers_x_cfg_func(mlp),
                    saycam_funcs.cotr_cont_set_window_size(x),
                    saycam_funcs.eq_scale_datasets(y),
                    post_func),
                seed=seed, 
                fx_up=True,
                scale_ratio=y, batch_size=2 * curr_bs // (1+y),
                **local_kwargs)
        return params
    all_things = globals()
    all_things[func_name] = _func


add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m2_wd20m_eq3_aw5', 20 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5), mlp=2)
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m4_wd20m_eq3_aw5', 20 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5), mlp=4)

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m2_wd30s_eq3_aw5', 0.5 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5), mlp=2)
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m4_wd30s_eq3_aw5', 0.5 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5), mlp=4)

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m2_wd30s_eq1_aw5', 0.5 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5), mlp=2)
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m4_wd30s_eq1_aw5', 0.5 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5), mlp=4)

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m2_wd20m_eq1_aw5', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5), mlp=2)
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m4_wd20m_eq1_aw5', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5), mlp=4)

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m4_wd20m_eqd3_aw5', 20 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ), mlp=4)
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m4_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ), mlp=4)
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m2_wd20m_eqd3_aw5', 20 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ), mlp=2)
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m2_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ), mlp=2)


add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m2_wd30s_eqd3_aw5_smtm', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            byol_funcs.base_mtm_x_cfg_func(0.95),
            ), mlp=2)
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m2_wd30s_eqd3_aw5_lmtm', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            byol_funcs.base_mtm_x_cfg_func(0.995),
            ), mlp=2)

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m2_wd30s_eqd3_aw5_lwd', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            gnrl_funcs.get_update_opt_func(weight_decay=1e-4),
            ), mlp=2)

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m8_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ), mlp=8)
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_m12_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ), mlp=12)


def add_neg_cotr_ct_wd_x_scale_y_func(
        func_name, x, y, 
        mlp=2, seed=None, exp_name=None,
        post_add_func=lambda cfg: cfg,
        concat='default',
        which_cotr=saycam_funcs.cotrain_sam_only_half_ep100_cfg_func,
        ):
    def _func(args):
        local_kwargs = copy.copy(cotr_kwargs)
        curr_bs = local_kwargs.pop('batch_size')
        if concat is not 'default':
            local_kwargs['concat_batches'] = concat
        params = get_typical_params(
                args, exp_name or func_name,
                gnrl_funcs.sequential_func(
                    which_cotr,
                    byol_neg_funcs.byol_neg_cfg_func,
                    simclr_cfg_funcs.mlp_xlayers_cfg_func(mlp),
                    saycam_funcs.cotr_cont_set_window_size(x),
                    saycam_funcs.eq_scale_datasets(y),
                    post_add_func),
                seed=seed, 
                fx_up=True,
                scale_ratio=y, batch_size=2 * curr_bs // (1+y),
                **local_kwargs)
        return params
    all_things = globals()
    all_things[func_name] = _func

add_neg_cotr_ct_wd_x_scale_y_func(
        'r18_neg_cotr_ct_wd30s_eq3_aw5', 0.5 * 60 * 25, 3,
        post_add_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_neg_cotr_ct_wd_x_scale_y_func(
        'r18_neg_cotr_ct_wd20m_eq3_aw5', 20 * 60 * 25, 3,
        post_add_func=saycam_funcs.set_aggre_window_size_keep_type(5))

add_neg_cotr_ct_wd_x_scale_y_func(
        'r18_neg_cotr_ct_wd30s_eq1_aw5', 0.5 * 60 * 25, 1,
        post_add_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_neg_cotr_ct_wd_x_scale_y_func(
        'r18_neg_cotr_ct_wd20m_eq1_aw5', 20 * 60 * 25, 1,
        post_add_func=saycam_funcs.set_aggre_window_size_keep_type(5))

add_neg_cotr_ct_wd_x_scale_y_func(
        'r18_neg_cotr_ct_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_add_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ))
add_neg_cotr_ct_wd_x_scale_y_func(
        'r18_neg_cotr_ct_wd20m_eqd3_aw5', 20 * 60 * 25, 3,
        post_add_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ))
