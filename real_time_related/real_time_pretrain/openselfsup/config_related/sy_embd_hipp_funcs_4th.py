from .sy_embd_hipp_funcs import *


def sc_pft_vlen64_p300n_dynca1_4th_typ_ssslw(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg = osf_smq_4th_prcmp(cfg)
    cfg = selfgrad_func(cfg)
    cfg = ca1_hlf(cfg, 4)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['learning_rate'] = 5e-5
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['learning_rate'] = 5e-5
    return cfg


def mtm_sg(cfg):
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['type'] = 'MomentumSGDynamicMLP2L'
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['type'] = 'MomentumSGDynamicMLP2L'
    return cfg


def sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_slw(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg = osf_smq_4th_prcmp(cfg)
    cfg = selfgrad_func(cfg)
    cfg = ca1_hlf(cfg, 4)
    cfg = mtm_sg(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['learning_rate'] = 5e-4
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['learning_rate'] = 5e-4
    return cfg


def sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_sslw(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg = osf_smq_4th_prcmp(cfg)
    cfg = selfgrad_func(cfg)
    cfg = ca1_hlf(cfg, 4)
    cfg = mtm_sg(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['learning_rate'] = 1e-4
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['learning_rate'] = 1e-4
    return cfg


def sc_pft_vlen64_p300n_dynca1_4th_typ_mtm_ssslw(cfg):
    cfg = sc_pft_vlen64_reluth_samps_p300_nsdcy(cfg)
    cfg = dgdynca3ca1_r32_func(cfg)
    cfg = osf_smq_4th_prcmp(cfg)
    cfg = selfgrad_func(cfg)
    cfg = ca1_hlf(cfg, 4)
    cfg = mtm_sg(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['ca3_cfg']['learning_rate'] = 5e-5
    cfg.model['hipp_head']['rnn_kwargs']['ca3to1_cfg']['learning_rate'] = 5e-5
    return cfg
