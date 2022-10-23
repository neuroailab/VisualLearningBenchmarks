from . import basic


def moco_realsim(args):
    args = basic.moco_basic_setting(args)
    
    args.exp_setting = 'imgnt_exp_randomrealsim'
    args.num_steps = 3000
    args.eval_freq = 150
    args.real_sim_mem_len = 10
    return args


def moco_realsim_100(args):
    args = basic.moco_basic_setting(args)
    
    args.exp_setting = 'imgnt_exp_randomrealsim'
    args.num_steps = 1600
    args.eval_freq = 50
    args.real_sim_mem_len = 20
    args.switch_ratio = 0.9375
    return args


def moco_realsim_200(args):
    args = basic.moco_basic_setting(args)
    
    args.exp_setting = 'imgnt_exp_randomrealsim'
    args.num_steps = 1700
    args.eval_freq = 50
    args.real_sim_mem_len = 10
    args.switch_ratio = 0.8825
    return args


def moco_realsim_300(args):
    args = basic.moco_basic_setting(args)
    
    args.exp_setting = 'imgnt_exp_randomrealsim'
    args.num_steps = 1800
    args.eval_freq = 50
    args.real_sim_mem_len = 10
    args.switch_ratio = 0.8334
    return args


def moco_time_realsim_200(args):
    args = basic.moco_basic_setting(args)
    
    args.num_steps = 1700
    args.eval_freq = 50
    args.switch_ratio = 0.8825
    args.exp_setting = 'imgnt_exp_randomtimerealsim'
    return args


def moco_window_time_realsim_200(args):
    args = basic.moco_basic_setting(args)
    
    args.num_steps = 1700
    args.eval_freq = 50
    args.switch_ratio = 0.8825
    args.exp_setting = 'imgnt_exp_RndmWindowTimeRSim'
    return args


def moco_wtrealsim_randnorand_200(args):
    args = basic.moco_basic_setting(args)
    
    args.num_steps = 1700
    args.eval_freq = 50
    args.switch_ratio = 0.8825
    args.exp_setting = 'imgnt_exp_WindowTimeRSimRndmNRndm'
    args.disallow_self_transitions = True
    return args


def simclr_wtrealsim_randnorand_200(args):
    args = basic.simclr_basic_setting(args)
    
    args.num_steps = 1700
    args.eval_freq = 50
    args.switch_ratio = 0.8825
    args.exp_setting = 'imgnt_exp_WindowTimeRSimRndmNRndm'
    args.disallow_self_transitions = True
    return args


def simclr_wtrealsim_randnorand(args):
    args = basic.simclr_basic_setting(args)
    
    args.num_steps = 3000
    args.eval_freq = 150
    args.switch_ratio = 0.5
    args.exp_setting = 'imgnt_exp_WindowTimeRSimRndmNRndm'
    args.disallow_self_transitions = True
    return args


def simclr_wtrealsim_randnorand_300_200(args):
    args = basic.simclr_basic_setting(args)
    
    args.num_steps = 500
    args.eval_freq = 25
    args.switch_ratio = 0.6
    args.exp_setting = 'imgnt_exp_WindowTimeRSimRndmNRndm'
    args.disallow_self_transitions = True
    return args


def simclr_wtrealsim_randnorand_cmmnty_300_200(args):
    args = basic.simclr_basic_setting(args)
    
    args.num_steps = 500
    args.eval_freq = 25
    args.switch_ratio = 0.6
    args.exp_setting = 'imgnt_exp_WindowTimeRSimRndmNRndm_community'
    return args


def simclr_wtrealsim_cmmnty_fx_300_400(args):
    args = basic.simclr_basic_setting(args)
    
    args.num_steps = 700
    args.eval_freq = 25
    args.switch_ratio = 3.0 / 7.0
    args.exp_setting = 'imgnt_exp_win_realsim_community_fx'
    return args


def moco_window_time_realsim_cmmnty_200(args):
    args = basic.moco_basic_setting(args)
    
    args.num_steps = 1700
    args.eval_freq = 50
    args.switch_ratio = 0.8825
    args.exp_setting = 'imgnt_exp_WindowTimeRSimRndmNRndm_community'
    return args


def moco_window_time_realsim_cmmnty_200_singleQ(args):
    args = basic.moco_basic_setting(args)
    
    args.num_steps = 1700
    args.eval_freq = 50
    args.switch_ratio = 0.8825
    args.exp_setting = 'imgnt_exp_WindowTimeRSimRndmNRndm_community'
    args.queue_update_options = None
    return args


def simclr_wtrealsim_randnorand_cmmnty_300_1200(args):
    args = basic.simclr_basic_setting(args)
    
    args.num_steps = 1500
    args.eval_freq = 50
    args.switch_ratio = 0.2
    args.exp_setting = 'imgnt_exp_WindowTimeRSimRndmNRndm_community'
    return args


def moco_community_1500_1500(args):
    args = basic.moco_basic_setting(args)
    
    args.exp_setting = 'imgnt_exp_community'
    args.num_steps = 3000
    args.eval_freq = 150
    return args


def moco_realsim_cmmnty_1500_1500(args):
    args = basic.moco_basic_setting(args)
    
    args.num_steps = 3000
    args.eval_freq = 150
    args.exp_setting = 'imgnt_exp_WindowTimeRSimRndmNRndm_community'
    return args


def simclr_community_300_1200(args):
    args = basic.simclr_basic_setting(args)
    
    args.num_steps = 1500
    args.eval_freq = 50
    args.switch_ratio = 0.2
    args.exp_setting = 'imgnt_exp_community'
    return args
