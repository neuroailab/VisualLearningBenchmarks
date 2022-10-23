from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
import torch
import numpy as np
import copy
import functools
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'
R50_BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r50_bs256_ep200.py'

R50_KWARGS = dict(
        cfg_path=R50_BASIC_SIMCLR_CFG,
        batch_size=128, opt_update_interval=16,
        builder_class=CotrainSAYCamParamBuilder,
        mix_weight=1.0, concat_batches=True,
        )
COTR_R50_KWARGS = copy.deepcopy(R50_KWARGS)
COTR_R50_KWARGS['batch_size'] = 64

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
        col_name='simclr_samS',
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
    return params

def r18_sam_storage_only_mlp4_ep100_aw5(args):
    return get_typical_params(
            args, 'r18_sam_storage_only_mlp4_ep100_aw5',
            gnrl_funcs.sequential_func(
                saycam_funcs.storage_sam_only_half_ep100_cfg_func,
                simclr_cfg_funcs.mlp_xlayers_cfg_func(4),
                saycam_funcs.set_aggre_window_size(5),
                ))


def add_cotr_ct_wd_x_eq_y_func(
        func_name, x, y, 
        mlp=2, seed=None, exp_name=None, concat='default',
        curr_kwargs=cotr_kwargs, post_func=lambda x: x):
    def _func(args):
        local_kwargs = copy.copy(curr_kwargs)
        curr_bs = local_kwargs.pop('batch_size')
        if concat is not 'default':
            local_kwargs['concat_batches'] = concat
        params = get_typical_params(
                args, exp_name or func_name,
                gnrl_funcs.sequential_func(
                    saycam_funcs.cotrain_sam_only_half_ep100_cfg_func,
                    simclr_cfg_funcs.mlp_xlayers_cfg_func(mlp),
                    saycam_funcs.cotr_cont_set_window_size(x),
                    saycam_funcs.eq_scale_datasets(y),
                    post_func),
                scale_ratio=y, batch_size=2 * curr_bs // (1+y),
                seed=seed, **local_kwargs)
        return params
    all_things = globals()
    all_things[func_name] = _func

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m2_wd20m_eq3_aw5', 20 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m2_wd30s_eq3_aw5', 0.5 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m2_wd30s_eq1_aw5', 0.5 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m2_wd20m_eq1_aw5', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m2_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ), 
        mlp=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m2_wd20m_eqd3_aw5', 20 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ), 
        mlp=2)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq3_aw5', 20 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd30s_eq3_aw5', 0.5 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd30s_eq1_aw5', 0.5 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw5', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ), 
        mlp=4)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eqd3_aw5', 20 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ), 
        mlp=4)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw10', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(10), mlp=4)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw15', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(15), mlp=4)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw25', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(25), mlp=4)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw10_s1', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(10), mlp=4,
        seed=1)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw15_s1', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(15), mlp=4,
        seed=1)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw25_s1', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(25), mlp=4,
        seed=1)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw10_s2', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(10), mlp=4,
        seed=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw15_s2', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(15), mlp=4,
        seed=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw25_s2', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(25), mlp=4,
        seed=2)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd40m_eq1_aw5', 40 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd80m_eq1_aw5', 80 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd30s_eq3_aw5_s1', 0.5 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4,
        seed=1)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd30s_eq1_aw5_s1', 0.5 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4,
        seed=1)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw5_s1', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4,
        seed=1)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd30s_eqd3_aw5_s1', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ),
        mlp=4, seed=1)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eqd3_aw5_s1', 20 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ), 
        mlp=4, seed=1)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd30s_eq3_aw5_s2', 0.5 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4,
        seed=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd30s_eq1_aw5_s2', 0.5 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4,
        seed=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq1_aw5_s2', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4,
        seed=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd30s_eqd3_aw5_s2', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ),
        mlp=4, seed=2)
add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eqd3_aw5_s2', 20 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ), 
        mlp=4, seed=2)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq15_aw5', 20 * 60 * 25, 15,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eq7_aw5', 20 * 60 * 25, 7,
        post_func=saycam_funcs.set_aggre_window_size(5), mlp=4)

add_cotr_ct_wd_x_eq_y_func(
        'r18_cotr_m4_wd20m_eqd7_aw5', 20 * 60 * 25, 7,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ), 
        mlp=4)

add_cotr_ct_wd_x_eq_y_func(
        'r50_cotr_ct_wd20m_eq3_aw5', 20 * 60 * 25, 3,
        curr_kwargs=COTR_R50_KWARGS,
        post_func=saycam_funcs.set_aggre_window_size(5))
add_cotr_ct_wd_x_eq_y_func(
        'r50_cotr_ct_wd30s_eq3_aw5', 0.5 * 60 * 25, 3,
        curr_kwargs=COTR_R50_KWARGS,
        post_func=saycam_funcs.set_aggre_window_size(5))
add_cotr_ct_wd_x_eq_y_func(
        'r50_cotr_ct_wd30s_eq1_aw5', 0.5 * 60 * 25, 1,
        curr_kwargs=COTR_R50_KWARGS,
        post_func=saycam_funcs.set_aggre_window_size(5))
add_cotr_ct_wd_x_eq_y_func(
        'r50_cotr_ct_wd20m_eq1_aw5', 20 * 60 * 25, 1,
        curr_kwargs=COTR_R50_KWARGS,
        post_func=saycam_funcs.set_aggre_window_size(5))
add_cotr_ct_wd_x_eq_y_func(
        'r50_cotr_ct_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        curr_kwargs=COTR_R50_KWARGS,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ))
add_cotr_ct_wd_x_eq_y_func(
        'r50_cotr_ct_wd20m_eqd3_aw5', 20 * 60 * 25, 3,
        curr_kwargs=COTR_R50_KWARGS,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ))
