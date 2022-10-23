from ..saycam_param_setter import SAYCamParamBuilder
import openselfsup.config_related.vae_funcs as vae_funcs
import openselfsup.config_related.gnrl_funcs as gnrl_funcs
BASIC_SIMCLR_CFG = './configs/selfsup/simclr/r18.py'


def get_typical_params(
        args, exp_id, cfg_func,
        batch_size=None, opt_update_interval=1,
        seed=None, **kwargs):
    def _func(cfg):
        cfg = cfg_func(cfg)
        if batch_size is not None:
            cfg.data['imgs_per_gpu'] = batch_size
        return cfg
    param_builder = SAYCamParamBuilder(
            args=args, exp_id=exp_id, cfg_path=BASIC_SIMCLR_CFG, 
            add_svm_val=True, col_name='vae',
            cfg_change_func=_func,
            opt_update_interval=opt_update_interval,
            col_name_in_work_dir=True,
            seed=seed, **kwargs)
    params = param_builder.build_params()
    params['validation_params'] = {}
    return params


def default_vae(args):
    return get_typical_params(
            args, 'default_vae_2', 
            vae_funcs.ctl_saycam_vae)


def default_vae64(args):
    return get_typical_params(
            args, 'default_vae64_2', 
            vae_funcs.ctl_saycam_vae64)


def default_vae112(args):
    return get_typical_params(
            args, 'default_vae112', 
            vae_funcs.ctl_saycam_vae112)


def saycam_inter_vae(args):
    return get_typical_params(
            args, 'saycam_inter_vae', 
            vae_funcs.saycam_inter_vae)


def saycam_inter_vae_moreZ(args):
    return get_typical_params(
            args, 'saycam_inter_vae_moreZ', 
            vae_funcs.saycam_inter_vae_moreZ)


def saycam_inter_vae_c2(args):
    return get_typical_params(
            args, 'saycam_inter_vae_c2', 
            vae_funcs.saycam_inter_vae_c2)


def saycam_inter_vae_c2_lessZ(args):
    return get_typical_params(
            args, 'saycam_inter_vae_c2_lessZ', 
            vae_funcs.saycam_inter_vae_c2_lessZ)


def saycam_inter_vae_c2_moreZ(args):
    return get_typical_params(
            args, 'saycam_inter_vae_c2_moreZ', 
            vae_funcs.saycam_inter_vae_c2_moreZ)


def saycam_inter_vae_c1_moreZ(args):
    return get_typical_params(
            args, 'saycam_inter_vae_c1_moreZ',
            vae_funcs.saycam_inter_vae_c1_moreZ)


def saycam_inter_vae_c2_s112(args):
    return get_typical_params(
            args, 'saycam_inter_vae_c2_s112',
            vae_funcs.saycam_inter_vae_c2_s112)


def saycam_inter_vae_c2_s224m112(args):
    return get_typical_params(
            args, 'saycam_inter_vae_c2_s224m112',
            vae_funcs.saycam_inter_vae_c2_s224m112)


def saycam_interbn_vae_c2(args):
    return get_typical_params(
            args, 'saycam_interbn_vae_c2',
            vae_funcs.saycam_interbn_vae_c2)


def saycam_interbn_vae_c128(args):
    return get_typical_params(
            args, 'saycam_interbn_vae_c128',
            vae_funcs.saycam_interbn_vae_c128)


def saycam_interbn_vae_c2_s224m112(args):
    return get_typical_params(
            args, 'saycam_interbn_vae_c2_s224m112',
            vae_funcs.saycam_interbn_vae_c2_s224m112)
