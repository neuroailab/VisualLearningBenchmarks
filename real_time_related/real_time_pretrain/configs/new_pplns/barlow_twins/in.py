from ..basic_param_setter import ParamsBuilder
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.barlow_twins_funcs as barlow_twins_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def get_typical_ep300_params(
        args, exp_id, cfg_func,
        batch_size=128, opt_update_interval=16,
        **kwargs):
    def _apply_ep300_func(cfg):
        cfg = cfg_func(cfg)
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        if batch_size is not None:
            cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = ParamsBuilder(
            args, exp_id, BASIC_SIMCLR_CFG, 
            add_svm_val=True,
            col_name='barlow_twins',
            cfg_change_func=_apply_ep300_func,
            col_name_in_work_dir=True,
            model_find_unused=None,
            opt_update_interval=opt_update_interval,
            **kwargs)
    params = param_builder.build_params()
    return params


def r18_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ep300', 
            cfg_func=barlow_twins_funcs.basic_r18_cfg_func)


def r18_ep300_face(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_face', 
            cfg_func=gnrl_funcs.sequential_func(
                barlow_twins_funcs.basic_r18_cfg_func,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ))
