from ..saycam_param_setter import SAYCamParamBuilder, CotrainSAYCamParamBuilder
import openselfsup.config_related.saycam_funcs as saycam_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.simsiam_cfg_funcs as simsiam_cfg_funcs
import openselfsup.config_related.byol_funcs as byol_funcs
BASIC_SIMSIAM_CFG = './configs/selfsup/siamese/r18.py'


def get_typical_ep300_params(
        args, exp_id, cfg_func, lr=0.1,
        batch_size=256, cotrain_builder=False,
        add_ep300_func=True,
        **kwargs):
    cfg_path = BASIC_SIMSIAM_CFG
    builder_class = SAYCamParamBuilder
    if cotrain_builder:
        builder_class = CotrainSAYCamParamBuilder
        if 'mix_weight' not in kwargs:
            kwargs['mix_weight'] = 1.0
        batch_size = 128
    def cfg_func_then_change_bs128_ep300(cfg):
        cfg = cfg_func(cfg)
        if add_ep300_func:
            cfg = ep300_funcs.ep300_SGD_cfg_func(cfg)
        cfg.data['imgs_per_gpu'] = batch_size
        cfg.optimizer['lr'] = lr
        return cfg
    param_builder = builder_class(
            args=args, exp_id=exp_id, cfg_path=cfg_path, 
            add_svm_val=True, col_name='simsiam_sy_sep',
            cfg_change_func=cfg_func_then_change_bs128_ep300,
            col_name_in_work_dir=True,
            model_find_unused=None,
            **kwargs)
    params = param_builder.build_params()
    return params


def add_cotr_ct_wd_x_scale_y_func(
        func_name, x, y,
        exp_name=None,
        post_func=lambda cfg: cfg,
        cotr_func=saycam_funcs.cotrain_sam_only_half_ep100_cfg_func,
        ):
    def _func(args):
        params = get_typical_ep300_params(
                args, exp_name or func_name,
                gnrl_funcs.sequential_func(
                    cotr_func,
                    saycam_funcs.cotr_cont_set_window_size(x),
                    saycam_funcs.eq_scale_datasets(y),
                    gnrl_funcs.res112,
                    simsiam_cfg_funcs.sep_crop_sim_neck_2048_w_params(
                        neck_layers=3,
                        neck_with_last_bn=True),
                    ep300_funcs.ep300_SGD_cfg_func,
                    gnrl_funcs.set_total_epochs(100),
                    post_func,
                    ),
                cotrain_builder=True,
                concat_batches=True,
                add_ep300_func=False,
                scale_ratio=y, batch_size=256 * 2 // (1+y),
                )
        return params
    all_things = globals()
    all_things[func_name] = _func

add_cotr_ct_wd_x_scale_y_func(
        'r18_sam_cotr_wd20m_eq3_aw5',
        20 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size(5))
add_cotr_ct_wd_x_scale_y_func(
        'r18_sam_cotr_wd30s_eq3_aw5',
        0.5 * 60 * 25, 3,
        post_func=saycam_funcs.set_aggre_window_size(5))

add_cotr_ct_wd_x_scale_y_func(
        'r18_sam_cotr_wd30s_eq1_aw5',
        0.5 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5))
add_cotr_ct_wd_x_scale_y_func(
        'r18_sam_cotr_wd20m_eq1_aw5',
        20 * 60 * 25, 1,
        post_func=saycam_funcs.set_aggre_window_size(5))

add_cotr_ct_wd_x_scale_y_func(
        'r18_sam_cotr_wd20m_eqd3_aw5',
        20 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ))
add_cotr_ct_wd_x_scale_y_func(
        'r18_sam_cotr_wd30s_eqd3_aw5',
        0.5 * 60 * 25, 3,
        post_func=gnrl_funcs.sequential_func(
            saycam_funcs.set_aggre_window_size(5),
            saycam_funcs.cotr_switch_datasource,
            ))
