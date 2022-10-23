from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.mae_cfg_funcs as mae_cfg_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
import copy
import os


BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'
def get_typical_params(
        args, exp_id, cfg_func=None, 
        cfg_path=BASIC_SIMCLR_CFG,
        batch_size=512, 
        builder_class=SAYCamParamBuilder,
        model_find_unused=None,
        valid_interval=10,
        **kwargs):
    def _apply_func(cfg):
        if cfg_func is not None:
            cfg = cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = builder_class(
            args=args, exp_id=exp_id, cfg_path=cfg_path, 
            add_svm_val=True, col_name='mae_sy_sep',
            cfg_change_func=_apply_func,
            col_name_in_work_dir=True,
            model_find_unused=model_find_unused,
            opt_use_fp16=True,
            valid_interval=valid_interval,
            **kwargs)
    params = param_builder.build_params()
    return params


cotr_kwargs = dict(
        builder_class=CotrainSAYCamParamBuilder,
        mix_weight=1.0, concat_batches=True,
        batch_size=256,
        )
def add_cotr_ct_wd_x_scale_y_func(
        func_name, x, y, 
        seed=None, exp_name=None,
        post_func=lambda cfg: cfg,
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
                    mae_cfg_funcs.small_vit_ep100_corr_wd,
                    mae_cfg_funcs.use_two_image,
                    mae_cfg_funcs.change_to_res112,
                    which_cotr,
                    gnrl_funcs.res112,
                    gnrl_funcs.change_to_same_aug_type,
                    saycam_funcs.cotr_cont_set_window_size(x),
                    saycam_funcs.eq_scale_datasets(y),
                    saycam_funcs.scale_both_datasets(4),
                    post_func),
                seed=seed, 
                scale_ratio=y, batch_size=2 * curr_bs // (1+y),
                valid_interval=1,
                **local_kwargs)
        return params
    all_things = globals()
    all_things[func_name] = _func

add_cotr_ct_wd_x_scale_y_func(
        'vits_cotr_wd30s_eq1_aw5', 0.5 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_cotr_ct_wd_x_scale_y_func(
        'vits_cotr_wd20m_eq1_aw5', 20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_cotr_ct_wd_x_scale_y_func(
        'vits_cotr_wd20m_eq3_aw5', 20 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5))
add_cotr_ct_wd_x_scale_y_func(
        'vits_cotr_wd30s_eq3_aw5', 0.5 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size_keep_type(5))

add_cotr_ct_wd_x_scale_y_func(
        'vits_cotr_wd30s_eqd3_aw5', 0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ))
add_cotr_ct_wd_x_scale_y_func(
        'vits_cotr_wd20m_eqd3_aw5', 20 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size_keep_type(5),
            saycam_funcs.cotr_switch_datasource,
            ))
