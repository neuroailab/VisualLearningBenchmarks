from ..basic_param_setter import ParamsBuilder
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r50_bs256_ep200.py'


def get_typical_params(
        args, exp_id, cfg_func,
        opt_update_interval=16,
        seed=None):
    param_builder = ParamsBuilder(
            args, exp_id, BASIC_SIMCLR_CFG, 
            add_svm_val=True,
            col_name='simclr',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True,
            opt_update_interval=opt_update_interval,
            model_find_unused=None,
            seed=seed,
            )
    params = param_builder.build_params()
    return params


def r50_ep100(args):
    return get_typical_params(
            args, 'r50_ep100', 
            cfg_func=gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                gnrl_funcs.set_total_epochs(100),
                ))

def r50_ep100_face(args):
    return get_typical_params(
            args, 'r50_ep100_face', 
            cfg_func=gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                gnrl_funcs.set_total_epochs(100),
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ))
