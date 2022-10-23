import sys


def moco_basic_setting(args):
    args.loss_type = 'mix_ce_moco'
    args.which_model = 'moco_v2'
    args.include_self_pair = True
    args.queue_update_options = 'separate_queues'
    args.pseudorandom = True
    args.init_lr = 1e-4
    args.num_workers = 15
    return args


def simclr_basic_setting(args):
    args.loss_type = 'mix_ce_simclr'
    args.which_model = 'simclr_r18'
    args.include_self_pair = True
    args.pseudorandom = True
    args.init_lr = 1e-4
    args.num_workers = 15
    return args
