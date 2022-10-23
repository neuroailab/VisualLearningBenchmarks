from ..basic_param_setter import ParamsBuilder
from openselfsup.config_related.moco.r18_funcs import \
        rdpd_sm_cfg_func, img_face_rdpd_sm_cfg_func
import openselfsup.config_related.moco.r18_funcs as r18_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.ws_gn_funcs as ws_gn_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
from .moco_hook import MoCoHook

BASIC_MOCO_CFG = './configs/selfsup/moco/r18_v2_with_val.py'


def r18_v2(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = ParamsBuilder(args, 'moco_r18_v2_2', cfg_path)
    params = param_builder.build_params()
    return params


def r18_v2_with_svm(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = ParamsBuilder(
            args, 'moco_r18_v2_svm', cfg_path, add_svm_val=True)
    params = param_builder.build_params()
    return params


def r18_v2_rdpd_sm(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = ParamsBuilder(
            args, 'r18_v2_rdpd_sm', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=rdpd_sm_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_v2_img_face_rdpd_sm(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = ParamsBuilder(
            args, 'r18_v2_img_face_rdpd_sm', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=img_face_rdpd_sm_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def r18_v2_img_face_rdpd_sm_s1(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = ParamsBuilder(
        args, 'r18_v2_img_face_rdpd_sm_s1', cfg_path, 
        add_svm_val=True, col_name='moco',
        cfg_change_func=img_face_rdpd_sm_cfg_func,
        col_name_in_work_dir=True,
        seed=1)
    params = param_builder.build_params()
    return params


def r18_v2_img_face_rdpd_sm_s2(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = ParamsBuilder(
        args, 'r18_v2_img_face_rdpd_sm_s2', cfg_path, 
        add_svm_val=True, col_name='moco',
        cfg_change_func=img_face_rdpd_sm_cfg_func,
        col_name_in_work_dir=True,
        seed=2)
    params = param_builder.build_params()
    return params


def r18_v2_img_face(args):
    cfg_path = BASIC_MOCO_CFG
    param_builder = ParamsBuilder(
            args, 'r18_v2_img_face', cfg_path, 
            add_svm_val=True, col_name='moco',
            cfg_change_func=r18_funcs.img_face_cfg_func,
            col_name_in_work_dir=True)
    params = param_builder.build_params()
    return params


def get_typical_ep300_params(
        args, exp_id, cfg_func,
        add_moco_hook=False,
        opt_update_interval=16,
        seed=None):
    param_builder = ParamsBuilder(
            args, exp_id, BASIC_MOCO_CFG, 
            add_svm_val=True,
            col_name='moco',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True,
            opt_update_interval=opt_update_interval,
            seed=seed,
            )
    params = param_builder.build_params()
    if add_moco_hook:
        params = add_moco_hook_to_params(params, 16)
    return params


def r18_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_2', 
            cfg_func=ep300_funcs.ep300_cfg_func)


def r18_is112_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_is112_ep300', 
            cfg_func=gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                gnrl_funcs.res112,
                ),
            opt_update_interval=8,
            )


def r18_ep300_s1(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_s1',
            cfg_func=ep300_funcs.ep300_cfg_func,
            seed=1)


def r18_ep300_s2(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_s2',
            cfg_func=ep300_funcs.ep300_cfg_func,
            seed=2)


def r18_simneck_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_simneck_ep300',
            cfg_func=gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                r18_funcs.simclr_neck_tau))


def r18_ws_gn_ep300(args):
    def _seq_cfg_func(cfg):
        cfg = ws_gn_funcs.moco_ws_gn_cfg_func(cfg)
        cfg = ep300_funcs.ep300_cfg_func(cfg)
        return cfg
    return get_typical_ep300_params(
            args, 'r18_ws_gn_ep300', 
            cfg_func=_seq_cfg_func)


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


def r18_up_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_up_ep300', 
            cfg_func=gnrl_funcs.sequential_func(
                ep300_funcs.ep300_cfg_func,
                r18_funcs.avoid_update_in_forward),
            add_moco_hook=True)
