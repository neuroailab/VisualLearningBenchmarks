from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.moco.r18_funcs as r18_funcs
import copy
from .moco_hook import MoCoHook
BASIC_MOCO_CFG = './configs/selfsup/moco/r18_v2_with_val.py'


def add_moco_hook_to_params(
        params, update_interval=1,
        fix_up=False):
    moco_hook_params = {
            'builder': MoCoHook,
            'update_interval': update_interval,
            }
    if fix_up:
        moco_hook_params['builder_kwargs'] = {
                'update_interval': update_interval,
                }
    if 'extra_hook_params' not in params:
        params['extra_hook_params'] = moco_hook_params
    else:
        if isinstance(params['extra_hook_params'], dict):
            params['extra_hook_params'] = [params['extra_hook_params']]
        params['extra_hook_params'].append(moco_hook_params)
    return params

cotr_kwargs = dict(
        builder_class=CotrainSAYCamParamBuilder,
        mix_weight=1.0, concat_batches=True,
        batch_size=128,
        )

def get_typical_params(
        args, exp_id, cfg_func, 
        builder_class=SAYCamParamBuilder,
        cfg_path=BASIC_MOCO_CFG,
        opt_update_interval=8, batch_size=256, 
        num_epochs=100, post_cfg_func=None,
        fx_up=False, col_name='moco_samS', add_moco_hook=False,
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
            model_find_unused=None,
            **kwargs)
    params = param_builder.build_params()
    if add_moco_hook:
        params = add_moco_hook_to_params(
                params, opt_update_interval,
                fx_up)
    return params


def add_cotr_ct_wd_x_scale_y_func(
        func_name, x, y, 
        mlp=2, seed=None, exp_name=None,
        post_add_func=lambda cfg: cfg,
        concat='default'):
    def _func(args):
        local_kwargs = copy.copy(cotr_kwargs)
        curr_bs = local_kwargs.pop('batch_size')
        if concat is not 'default':
            local_kwargs['concat_batches'] = concat
        params = get_typical_params(
                args, exp_name or func_name,
                gnrl_funcs.sequential_func(
                    saycam_funcs.cotrain_sam_only_half_ep100_cfg_func,
                    saycam_funcs.cotr_cont_set_window_size(x),
                    saycam_funcs.eq_scale_datasets(y),
                    r18_funcs.use_sync_bn,
                    r18_funcs.avoid_update_in_forward,
                    post_add_func,
                    ),
                add_moco_hook=True,
                seed=seed, 
                scale_ratio=y, batch_size=2 * curr_bs // (1+y),
                **local_kwargs)
        return params
    all_things = globals()
    all_things[func_name] = _func

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd20m_eq3_aw5', 20 * 60 * 25, 3,
        post_add_func=saycam_funcs.set_aggre_window_size(5))
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd30s_eq3_aw5', 0.5 * 60 * 25, 3,
        post_add_func=saycam_funcs.set_aggre_window_size(5))

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd30s_eq1_aw5', 0.5 * 60 * 25, 1,
        post_add_func=saycam_funcs.set_aggre_window_size(5))
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd20m_eq1_aw5', 20 * 60 * 25, 1,
        post_add_func=saycam_funcs.set_aggre_window_size(5))

add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_add_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ))
add_cotr_ct_wd_x_scale_y_func(
        'r18_cotr_ct_wd20m_eqd3_aw5', 20 * 60 * 25, 3,
        post_add_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ))
