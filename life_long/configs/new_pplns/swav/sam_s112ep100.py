from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
import openselfsup.config_related.swav_cfg_funcs as swav_cfg_funcs
from openselfsup.models.swav import QueueHook, OptimizerHook
import torch
import numpy as np
import copy


def add_queue_hook_to_params(params):
    queue_hook_params = {
            'builder': QueueHook,
            'builder_kwargs': {
                'args': swav_cfg_funcs.default_args,
                'save_dir': params['save_params']['ckpt_hook_kwargs']['out_dir'],
                },
            }
    if 'extra_hook_params' not in params:
        params['extra_hook_params'] = queue_hook_params
    else:
        if isinstance(params['extra_hook_params'], dict):
            params['extra_hook_params'] = [params['extra_hook_params']]
        params['extra_hook_params'].append(queue_hook_params)
    return params

BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'
cotr_kwargs = dict(
        builder_class=CotrainSAYCamParamBuilder,
        mix_weight=1.0, concat_batches=True,
        batch_size=128,
        )
cnd_kwargs = copy.deepcopy(cotr_kwargs)
cnd_kwargs['use_cnd_hook'] = True


def get_typical_params(
        args, exp_id, cfg_func,
        batch_size=256, opt_update_interval=8,
        seed=None, cfg_path=BASIC_SIMCLR_CFG, 
        builder_class=SAYCamParamBuilder,
        num_epochs=100, post_cfg_func=None,
        col_name='swav_samS',
        **kwargs):
    def _apply_func(cfg):
        if cfg_func is not None:
            cfg = cfg_func(cfg)
        cfg = gnrl_funcs.sequential_func(
                gnrl_funcs.res112,
                ep300_funcs.ep300_cfg_func,
                gnrl_funcs.set_total_epochs(num_epochs),
                )(cfg)
        if post_cfg_func is not None:
            cfg = post_cfg_func(cfg)
        if batch_size is not None:
            swav_cfg_funcs.default_args.batch_size = batch_size
            cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = builder_class(
            args=args, exp_id=exp_id, cfg_path=cfg_path, 
            add_svm_val=True, col_name=col_name,
            cfg_change_func=_apply_func,
            opt_update_interval=opt_update_interval,
            col_name_in_work_dir=True,
            model_find_unused=None,
            seed=seed, **kwargs)
    params = param_builder.build_params()
    params = add_queue_hook_to_params(params)
    return params


def add_cotr_ct_wd_x_scale_y_func(
        func_name, x, y, 
        mlp=2, seed=None, exp_name=None,
        post_add_func=lambda cfg: cfg,
        which_cotr=saycam_funcs.cotrain_sam_only_half_ep100_cfg_func,
        concat='default'):
    def _func(args):
        local_kwargs = copy.copy(cotr_kwargs)
        curr_bs = local_kwargs.pop('batch_size')
        if concat is not 'default':
            local_kwargs['concat_batches'] = concat
        params = get_typical_params(
                args, exp_name or func_name,
                gnrl_funcs.sequential_func(
                    swav_cfg_funcs.no_crop_cfg_func, # the first one as this changes the dataset type
                    which_cotr,
                    simclr_cfg_funcs.mlp_xlayers_cfg_func(mlp),
                    saycam_funcs.cotr_cont_set_window_size(x),
                    saycam_funcs.eq_scale_datasets(y),
                    post_add_func),
                seed=seed, 
                scale_ratio=y, batch_size=2 * curr_bs // (1+y),
                **local_kwargs)
        return params
    all_things = globals()
    all_things[func_name] = _func

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd20m_eq3_aw5', 20 * 60 * 25, 3,
        post_add_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd30s_eq3_aw5', 0.5 * 60 * 25, 3,
        post_add_func=saycam_funcs.set_aggre_window_size_keep_type(5))

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd30s_eq1_aw5', 0.5 * 60 * 25, 1,
        post_add_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd20m_eq1_aw5', 20 * 60 * 25, 1,
        post_add_func=saycam_funcs.set_aggre_window_size_keep_type(5))

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_add_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ))
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd20m_eqd3_aw5', 20 * 60 * 25, 3,
        post_add_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ))
