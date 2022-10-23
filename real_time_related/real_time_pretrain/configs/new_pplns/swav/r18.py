from ..basic_param_setter import ParamsBuilder

from openselfsup.models.swav import QueueHook, OptimizerHook
import openselfsup.config_related.swav_cfg_funcs as swav_cfg_funcs
import openselfsup.config_related.ep300_funcs as ep300_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
import openselfsup.config_related.simclr_cfg_funcs as simclr_cfg_funcs


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


def replace_opt_hook_in_params(params):
    optimizer_hook_params = {
            'builder': OptimizerHook,
            'builder_kwargs': {
                'use_fp16': False, 
                'freeze_prototypes_niters': 5004,
                },
            }
    params['optimizer_hook_params'] = optimizer_hook_params
    return params


BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'
def get_typical_ep300_params(
        args, exp_id, cfg_func, 
        opt_update_interval=8, batch_size=256, 
        add_ep300_func=True,
        replace_opt_hook=False,
        **kwargs):
    def _apply_ep300_func(cfg):
        cfg = cfg_func(cfg)
        if add_ep300_func:
            cfg = ep300_funcs.ep300_cfg_func(cfg)
        if batch_size is not None:
            swav_cfg_funcs.default_args.batch_size = batch_size
            cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = ParamsBuilder(
            args=args, exp_id=exp_id, cfg_path=BASIC_SIMCLR_CFG, 
            add_svm_val=True,
            col_name='swav_in',
            cfg_change_func=_apply_ep300_func,
            col_name_in_work_dir=True,
            opt_update_interval=opt_update_interval,
            model_find_unused=None,
            **kwargs)
    params = param_builder.build_params()
    params = add_queue_hook_to_params(params)
    if replace_opt_hook:
        params = replace_opt_hook_in_params(params)
    return params


def nocrop_r18_ep300(args):
    return get_typical_ep300_params(
            args, 'nocrop_r18_ep300', 
            cfg_func=swav_cfg_funcs.no_crop_cfg_func)


def nocrop_r18_ep300_face(args):
    return get_typical_ep300_params(
            args, 'nocrop_r18_ep300_face', 
            cfg_func=gnrl_funcs.sequential_func(
                swav_cfg_funcs.no_crop_cfg_func,
                gnrl_funcs.img_face_rdpd_sm_eqlen_cfg_func,
                ))


def wcrop_r18_ep300(args):
    return get_typical_ep300_params(
            args, 'wcrop_r18_ep300', 
            cfg_func=swav_cfg_funcs.w_crop_cfg_func,
            opt_update_interval=8, batch_size=128, 
            )


def wcrop_r18_no_outbn_ep300(args):
    return get_typical_ep300_params(
            args, 'wcrop_r18_no_outbn_ep300', 
            cfg_func=gnrl_funcs.sequential_func(
                swav_cfg_funcs.w_crop_cfg_func,
                simclr_cfg_funcs.remove_out_bn,
                )
            )


def wcrop_r18_no_outbn_sgd400(args):
    return get_typical_ep300_params(
            args, 'wcrop_r18_no_outbn_sgd400', 
            cfg_func=gnrl_funcs.sequential_func(
                swav_cfg_funcs.w_crop_cfg_func,
                simclr_cfg_funcs.remove_out_bn,
                swav_cfg_funcs.sgd400_cfg_func,
                ),
            add_ep300_func=False, opt_update_interval=1)


def wcrop_r18_no_outbn_res112_sgd100(args):
    return get_typical_ep300_params(
            args, 'wcrop_r18_no_outbn_res112_sgd100', 
            cfg_func=gnrl_funcs.sequential_func(
                swav_cfg_funcs.w_crop_cfg_func,
                simclr_cfg_funcs.remove_out_bn,
                swav_cfg_funcs.sgd100_cfg_func,
                gnrl_funcs.res112,
                ),
            add_ep300_func=False, opt_update_interval=1,
            replace_opt_hook=True)
