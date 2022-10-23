from ..basic_param_setter import ParamsBuilder
from .byol_hook import BYOLHook, EWCBYOLHook
import openselfsup.config_related.ws_gn_funcs as ws_gn_funcs
import openselfsup.config_related.byol_neg_funcs as byol_neg_funcs
import openselfsup.config_related.byol_funcs as byol_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
BASIC_BYOL_CFG = './configs/selfsup/byol/r18.py'


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


def r18_with_svm(args):
    param_builder = ParamsBuilder(
            args, 'byol_r18', BASIC_BYOL_CFG, 
            add_svm_val=True)
    params = param_builder.build_params()
    params = add_byol_hook_to_params(params)
    return params


BASIC_BYOL_EP300_CFG = './configs/selfsup/byol/r18_ep300.py'
def get_typical_ep300_params(
        args, exp_id, cfg_func, 
        seed=None, fx_up=False):
    param_builder = ParamsBuilder(
            args, exp_id, BASIC_BYOL_EP300_CFG, 
            add_svm_val=True,
            col_name='byol',
            cfg_change_func=cfg_func,
            col_name_in_work_dir=True,
            opt_update_interval=16,
            seed=seed,
            model_find_unused=None,
            )
    params = param_builder.build_params()
    params = add_byol_hook_to_params(
            params, 16, fx_up=fx_up)
    return params


def r18_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ep300', 
            cfg_func=lambda cfg: cfg)


def r18_ep300_s1(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_s1',
            cfg_func=lambda cfg: cfg,
            seed=1)


def r18_ep300_s2(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_s2',
            cfg_func=lambda cfg: cfg,
            seed=2)

def r18_ep300_fx(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_fx', 
            cfg_func=lambda cfg: cfg,
            fx_up=True)


def r18_ep300_face(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_face', 
            cfg_func=gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func)

def r18_ep300_celeba_face(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_celeba_face', 
            cfg_func=gnrl_funcs.sequential_func(
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                gnrl_funcs.use_celeba_faces,
                ),
            )

def r18_ep300_celeba_1vs3_face(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_celeba_1vs3_face', 
            cfg_func=gnrl_funcs.sequential_func(
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                gnrl_funcs.use_celeba_1vs3_faces,
                ),
            )

def r18_ep300_celeba_align_face(args):
    return get_typical_ep300_params(
            args, 'r18_ep300_celeba_align_face', 
            cfg_func=gnrl_funcs.sequential_func(
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                gnrl_funcs.use_celeba_align_faces,
                ),
            )


def r18_mlp4_ep300(args):
    return get_typical_ep300_params(
        args, 'r18_mlp4_ep300',
        cfg_func=byol_funcs.mlp_4layers_cfg_func)


def r18_img_face_rdpd_sm(args):
    return get_typical_ep300_params(
        args, 'r18_img_face_rdpd_sm',
        cfg_func=byol_funcs.img_rdpd_face_cfg_func)


def r18_img_face_rdpd_sm_s1(args):
    return get_typical_ep300_params(
        args, 'r18_img_face_rdpd_sm_s1',
        cfg_func=byol_funcs.img_rdpd_face_cfg_func,
        seed=1)


def r18_img_face_rdpd_sm_s2(args):
    return get_typical_ep300_params(
        args, 'r18_img_face_rdpd_sm_s2',
        cfg_func=byol_funcs.img_rdpd_face_cfg_func,
        seed=2)


def r18_mlp3_img_face_rdpd_sm(args):
    return get_typical_ep300_params(
        args, 'r18_mlp3_img_face_rdpd_sm',
        cfg_func=byol_funcs.mlp_3layers_rdpd_face_cfg_func)


def r18_mlp3_img_face_rdpd_sm_s1(args):
    return get_typical_ep300_params(
        args, 'r18_mlp3_img_face_rdpd_sm_s1',
        cfg_func=byol_funcs.mlp_3layers_rdpd_face_cfg_func,
        seed=1)


def r18_mlp3_img_face_rdpd_sm_s2(args):
    return get_typical_ep300_params(
        args, 'r18_mlp3_img_face_rdpd_sm_s2',
        cfg_func=byol_funcs.mlp_3layers_rdpd_face_cfg_func,
        seed=2)


def r18_mlp4_img_face_rdpd_sm(args):
    return get_typical_ep300_params(
        args, 'r18_mlp4_img_face_rdpd_sm',
        cfg_func=byol_funcs.mlp_4layers_rdpd_face_cfg_func)


def r18_mlp4_img_face_rdpd_sm_s1(args):
    return get_typical_ep300_params(
        args, 'r18_mlp4_img_face_rdpd_sm_s1',
        cfg_func=byol_funcs.mlp_4layers_rdpd_face_cfg_func,
        seed=1)


def r18_mlp4_img_face_rdpd_sm_s2(args):
    return get_typical_ep300_params(
        args, 'r18_mlp4_img_face_rdpd_sm_s2',
        cfg_func=byol_funcs.mlp_4layers_rdpd_face_cfg_func,
        seed=2)

def r18_mlp4_ep300_face_s1(args):
    return get_typical_ep300_params(
            args, 'r18_mlp4_ep300_face_s1',
            cfg_func=gnrl_funcs.sequential_func(
                byol_funcs.mlp_4layers_cfg_func,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ),
            seed=1)

def r18_mlp4_ep100_s112_face(args):
    return get_typical_ep300_params(
            args, 'r18_mlp4_ep100_s112_face_2',
            cfg_func=gnrl_funcs.sequential_func(
                gnrl_funcs.res112,
                gnrl_funcs.set_total_epochs(100),
                byol_funcs.mlp_4layers_cfg_func,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ))

def r18_mlp4_ep100_s112_face_sm(args):
    return get_typical_ep300_params(
            args, 'r18_mlp4_ep100_s112_face_sm',
            cfg_func=gnrl_funcs.sequential_func(
                gnrl_funcs.res112,
                gnrl_funcs.set_total_epochs(100),
                byol_funcs.mlp_4layers_cfg_func,
                gnrl_funcs.img_face_rdpd_ssm_eqlen_cfg_func,
                ))

def r18_mlp6_ep300_face(args):
    return get_typical_ep300_params(
            args, 'r18_mlp6_ep300_face',
            cfg_func=gnrl_funcs.sequential_func(
                byol_funcs.mlp_6layers_cfg_func,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ))

def r18_mlp8_ep300_face(args):
    return get_typical_ep300_params(
            args, 'r18_mlp8_ep300_face',
            cfg_func=gnrl_funcs.sequential_func(
                byol_funcs.mlp_layers_x_cfg_func(8),
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ))

def r18_mlp8_ep300_face_fxup(args):
    return get_typical_ep300_params(
            args, 'r18_mlp8_ep300_face_fxup',
            cfg_func=gnrl_funcs.sequential_func(
                byol_funcs.mlp_layers_x_cfg_func(8),
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ),
            fx_up=True)


def r18_ws_gn_ep300(args):
    return get_typical_ep300_params(
            args, 'r18_ws_gn_ep300_2', 
            cfg_func=ws_gn_funcs.byol_ws_gn_cfg_func)


def r18_neg_ep300(args):
    return get_typical_ep300_params(
        args, 'r18_neg_ep300_3', 
        cfg_func=byol_neg_funcs.byol_neg_cfg_func)

def r18_neg_ep300_mlp4_face(args):
    return get_typical_ep300_params(
            args, 'r18_neg_ep300_mlp4_face', 
            cfg_func=gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                simclr_cfg_funcs.mlp_4layers_cfg_func,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ),
            )

def r18_neg_ep300_mlp8_face(args):
    return get_typical_ep300_params(
            args, 'r18_neg_ep300_mlp8_face',
            cfg_func=gnrl_funcs.sequential_func(
                byol_neg_funcs.byol_neg_cfg_func,
                simclr_cfg_funcs.mlp_xlayers_cfg_func(8),
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ))


def r18_neg_ep300_s1(args):
    return get_typical_ep300_params(
        args, 'r18_neg_ep300_s1',
        cfg_func=byol_neg_funcs.byol_neg_cfg_func,
        seed=1)


def r18_neg_ep300_s2(args):
    return get_typical_ep300_params(
        args, 'r18_neg_ep300_s2',
        cfg_func=byol_neg_funcs.byol_neg_cfg_func,
        seed=2)


def r18_neg_face_rdpd_sm(args):
    return get_typical_ep300_params(
        args, 'r18_neg_face_rdpd_sm', 
        cfg_func=byol_neg_funcs.byol_neg_face_rdpd_cfg_func)


def r18_neg_face_rdpd_sm_s1(args):
    return get_typical_ep300_params(
        args, 'r18_neg_face_rdpd_sm_s1', 
        cfg_func=byol_neg_funcs.byol_neg_face_rdpd_cfg_func,
        seed=1)


def r18_neg_face_rdpd_sm_s2(args):
    return get_typical_ep300_params(
        args, 'r18_neg_face_rdpd_sm_s2', 
        cfg_func=byol_neg_funcs.byol_neg_face_rdpd_cfg_func,
        seed=2)
