def step600_basic_setting(args):
    args.num_steps = 600
    args.batch_size = 64
    args.eval_freq = 30
    args.optimizer = 'from_cfg'
    args.use_distributed_samplr = True
    return args


def mix_build_basic_setting(args):
    args.exp_setting = 'imgnt_exp_noswap'
    args.loss_type = 'mix'
    return args


def mix_break_basic_setting(args):
    args.exp_setting = 'imgnt_exp_swap'
    args.loss_type = 'mix'
    return args


def mix_switch_basic_setting(args):
    args.exp_setting = 'imgnt_exp_noswapswap'
    args.loss_type = 'mix'
    return args


# SimCLR
def simclr_face_rdpd_basic_setting(args, seed=None):
    if seed is None:
        args.which_model = 'simclr_r18_face_rdpd'
    return args

def simclr_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def simclr_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def simclr_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# SimCLR-More-MLPs
def simclr_mlp4_early_face_rdpd_basic_setting(args, seed=None):
    args.which_model = 'simclr_mlp4_face_rdpd_early'
    return args

def simclr_mlp4_early_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_early_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_mlp4_early_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_early_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_mlp4_early_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_early_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# SimCLR-ResNet50
def simclr_r50_face_rdpd_basic_setting(args, seed=None):
    args.which_model = 'simclr_r50_face_rdpd'
    return args


def simclr_r50_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_r50_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_r50_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_r50_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_r50_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_r50_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# DINO
def dino_mlp3_face_rdpd_basic_setting(args):
    args.which_model = 'dino_mlp3_face_rdpd'
    args.batch_size = 128
    args.mc_update = True
    return args


def dino_mlp3_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = dino_mlp3_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def dino_mlp3_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = dino_mlp3_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def dino_mlp3_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = dino_mlp3_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# DINONeg
def dinoneg_mlp3_face_rdpd_basic_setting(args):
    args.which_model = 'dinoneg_mlp3_face_rdpd'
    args.batch_size = 128
    args.mc_update = True
    return args


def dinoneg_mlp3_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = dinoneg_mlp3_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def dinoneg_mlp3_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = dinoneg_mlp3_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def dinoneg_mlp3_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = dinoneg_mlp3_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# MAE
def mae_face_rdpd_crop_mask_d1_basic_setting(args):
    args.which_model = 'mae_vit_s_face_rdpd'
    args.batch_size = 64
    args.train_transforms = 'mae_same_crop'
    args.mae_mask_ratio = 0.1
    return args


def mae_face_rdpd_mae_same_crop_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = mae_face_rdpd_crop_mask_d1_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def mae_face_rdpd_mae_same_crop_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = mae_face_rdpd_crop_mask_d1_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def mae_face_rdpd_mae_same_crop_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = mae_face_rdpd_crop_mask_d1_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# BYOL
def byol_r18_face_rdpd_basic_setting(args, seed=None):
    if seed is None:
        args.which_model = 'byol_r18_face_rdpd'
    args.mc_update = True
    return args

def byol_r18_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = byol_r18_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def byol_r18_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = byol_r18_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byol_r18_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = byol_r18_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# BYOL-More-MLPs
def byol_mlp4_face_rdpd_basic_setting(args, seed=None):
    if seed is None:
        args.which_model = 'byol_mlp4_face_rdpd'
    args.mc_update = True
    return args


def byol_mlp4_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = byol_mlp4_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def byol_mlp4_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = byol_mlp4_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byol_mlp4_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = byol_mlp4_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# BYOLNeg
def byolneg_r18_face_rdpd_basic_setting(args, seed=None):
    if seed is None:
        args.which_model = 'byolneg_r18_face_rdpd'
    args.mc_update = True
    return args


def byolneg_r18_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = byolneg_r18_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def byolneg_r18_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = byolneg_r18_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byolneg_r18_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = byolneg_r18_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# SwAV
def swav_r18_face_rdpd_early_basic_setting(args, seed=None):
    args.which_model = 'swav_r18_face_rdpd_early'
    args.batch_size = 64
    return args


def swav_r18_face_rdpd_early_mix_build(args):
    args = step600_basic_setting(args)
    args = swav_r18_face_rdpd_early_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def swav_r18_face_rdpd_early_mix_break(args):
    args = step600_basic_setting(args)
    args = swav_r18_face_rdpd_early_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def swav_r18_face_rdpd_early_mix_switch(args):
    args = step600_basic_setting(args)
    args = swav_r18_face_rdpd_early_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# MoCo v2
def moco_face_rdpd_basic_setting(args, seed=None):
    if seed is None:
        args.which_model = 'moco_v2_face_rdpd'
    return args


def moco_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = moco_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)    
    return args


def moco_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = moco_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def moco_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = moco_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# Barlow-Twins
def barlow_twins_r18_face_rdpd_basic_setting(args):
    args.which_model = 'barlow_twins_r18_face_rdpd'
    return args


def barlow_twins_r18_face_mix_build(args):
    args = step600_basic_setting(args)
    args = barlow_twins_r18_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def barlow_twins_r18_face_mix_break(args):
    args = step600_basic_setting(args)
    args = barlow_twins_r18_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def barlow_twins_r18_face_mix_switch(args):
    args = step600_basic_setting(args)
    args = barlow_twins_r18_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# SimSiam
def siamese_face_rdpd_basic_setting(args, seed=None):
    if seed is None:
        args.which_model = 'siamese_r18_face_rdpd'
    return args


def siamese_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = siamese_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def siamese_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = siamese_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def siamese_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = siamese_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args







def byol_ep300_basic_setting(args):
    args.which_model = 'byol_ep300_r18'
    args.mc_update = True
    return args


def byol_r18_cb_face_rdpd_basic_setting(args, seed=None):
    args.which_model = 'byol_r18_cb_face_rdpd'
    args.mc_update = True
    return args


def byolneg_mlp4_face_rdpd_basic_setting(args):
    args.which_model = 'byolneg_mlp4_face_rdpd'
    args.mc_update = True
    return args


def byolneg_mlp4_early_face_rdpd_basic_setting(args):
    args.which_model = 'byolneg_mlp4_face_rdpd_early'
    args.mc_update = True
    return args


def byol_mlp3_face_rdpd_basic_setting(args, seed=None):
    if seed is None:
        args.which_model = 'byol_mlp3_face_rdpd'
    elif seed == 1:
        args.which_model = 'byol_mlp3_face_rdpd_s1'
    elif seed == 2:
        args.which_model = 'byol_mlp3_face_rdpd_s2'
    args.mc_update = True
    return args

def byol_mlp4_basic_setting(args):
    args.which_model = 'byol_mlp4'
    args.mc_update = True
    return args


def byol_mlp8_face_rdpd_basic_setting(args, seed=None):
    args.which_model = 'byol_mlp8_face_rdpd'
    args.mc_update = True
    return args


def simclr_ep300_basic_setting(args):
    args.which_model = 'simclr_ep300_r18'
    return args

def simclr_ep300_early_basic_setting(args):
    args.which_model = 'simclr_ep300_r18_early'
    return args


def simclr_rdpd_basic_setting(args, seed=None):
    args.which_model = 'simclr_r18_rdpd'        
    return args


def simclr_mlp4_face_rdpd_basic_setting(args, seed=None):
    args.which_model = 'simclr_mlp4_face_rdpd'
    return args

def simclr_mlp4_cb_face_rdpd_basic_setting(args, seed=None):
    args.which_model = 'simclr_mlp4_cb_face_rdpd'
    return args


def moco_ep300_basic_setting(args):
    args.which_model = 'moco_v2_ep300'
    return args


def moco_rdpd_basic_setting(args):
    args.which_model = 'moco_v2_rdpd'
    return args


def siamese_ep300_basic_setting(args):
    args.which_model = 'siamese_ep300_r18'
    return args


def ce_build_basic_setting(args):
    args.exp_setting = 'no_swap'
    args.loss_type = 'ce_only'
    return args

def ce_break_basic_setting(args):
    args.exp_setting = 'swap'
    args.loss_type = 'ce_only'
    return args

def ce_switch_basic_setting(args):
    args.exp_setting = 'noswap_swap'
    args.loss_type = 'ce_only'
    return args


'''Actual settings used in command line args'''
def swav_r18_ce_build(args):
    args = step600_basic_setting(args)
    args = swav_r18_basic_setting(args)
    args = ce_build_basic_setting(args)
    return args


def swav_r18_ce_break(args):
    args = step600_basic_setting(args)
    args = swav_r18_basic_setting(args)
    args = ce_break_basic_setting(args)
    return args


def swav_r18_ce_switch(args):
    args = step600_basic_setting(args)
    args = swav_r18_basic_setting(args)
    args = ce_switch_basic_setting(args)
    return args


def byol_mix_build(args):
    args = step600_basic_setting(args)
    args = byol_ep300_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def byol_mix_break(args):
    args = step600_basic_setting(args)
    args = byol_ep300_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byol_mix_switch(args):
    args = step600_basic_setting(args)
    args = byol_ep300_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def byol_r18_cb_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = byol_r18_cb_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def byol_r18_cb_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = byol_r18_cb_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byol_r18_cb_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = byol_r18_cb_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def byolneg_mlp4_face_rdpd_basic_setting(args, seed=None):
    args.which_model = 'byolneg_mlp4_face_rdpd'
    args.mc_update = True
    return args


def byolneg_mlp4_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = byolneg_mlp4_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def byolneg_mlp4_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = byolneg_mlp4_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byolneg_mlp4_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = byolneg_mlp4_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def byolneg_mlp4_face_rdpd_early_mix_build(args):
    args = step600_basic_setting(args)
    args = byolneg_mlp4_early_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def byolneg_mlp4_face_rdpd_early_mix_break(args):
    args = step600_basic_setting(args)
    args = byolneg_mlp4_early_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byolneg_mlp4_face_rdpd_early_mix_switch(args):
    args = step600_basic_setting(args)
    args = byolneg_mlp4_early_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def byol_mlp3_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = byol_mlp3_face_rdpd_basic_setting(
        args, seed=1)
    args = mix_build_basic_setting(args)
    return args


def byol_mlp3_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = byol_mlp3_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byol_mlp3_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = byol_mlp3_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def byol_mlp3_face_rdpd_s1_mix_build(args):
    args = step600_basic_setting(args)
    args = byol_mlp3_face_rdpd_basic_setting(
        args, seed=1)
    args = mix_build_basic_setting(args)
    return args


def byol_mlp3_face_rdpd_s1_mix_break(args):
    args = step600_basic_setting(args)
    args = byol_mlp3_face_rdpd_basic_setting(
        args, seed=1)
    args = mix_break_basic_setting(args)
    return args


def byol_mlp3_face_rdpd_s1_mix_switch(args):
    args = step600_basic_setting(args)
    args = byol_mlp3_face_rdpd_basic_setting(
        args, seed=1)
    args = mix_switch_basic_setting(args)
    return args


def byol_mlp3_face_rdpd_s2_mix_build(args):
    args = step600_basic_setting(args)
    args = byol_mlp3_face_rdpd_basic_setting(
        args, seed=2)
    args = mix_build_basic_setting(args)
    return args


def byol_mlp3_face_rdpd_s2_mix_break(args):
    args = step600_basic_setting(args)
    args = byol_mlp3_face_rdpd_basic_setting(
        args, seed=2)
    args = mix_break_basic_setting(args)
    return args


def byol_mlp3_face_rdpd_s2_mix_switch(args):
    args = step600_basic_setting(args)
    args = byol_mlp3_face_rdpd_basic_setting(
        args, seed=2)
    args = mix_switch_basic_setting(args)
    return args


def byol_mlp4_mix_build(args):
    args = step600_basic_setting(args)
    args = byol_mlp4_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def byol_mlp4_mix_break(args):
    args = step600_basic_setting(args)
    args = byol_mlp4_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byol_mlp4_mix_switch(args):
    args = step600_basic_setting(args)
    args = byol_mlp4_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def byol_mlp8_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = byol_mlp8_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def byol_mlp8_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = byol_mlp8_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def byol_mlp8_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = byol_mlp8_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def simclr_ce_build(args):
    args = step600_basic_setting(args)
    args = simclr_ep300_basic_setting(args)
    args = ce_build_basic_setting(args)
    return args


def simclr_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_ep300_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_ep300_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_ep300_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def simclr_early_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_ep300_early_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_early_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_ep300_early_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_early_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_ep300_early_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def simclr_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def simclr_face_rdpd_ep300_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_face_rdpd_ep300_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_face_rdpd_ep300_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_face_rdpd_ep300_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_face_rdpd_ep300_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_face_rdpd_ep300_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def simclr_mlp4_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_mlp4_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_mlp4_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def simclr_mlp4_early_face_rdpd_nomix_build(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_early_face_rdpd_basic_setting(args)
    args = ce_build_basic_setting(args)
    return args


def simclr_mlp4_early_face_rdpd_nomix_break(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_early_face_rdpd_basic_setting(args)
    args = ce_break_basic_setting(args)
    return args


def simclr_mlp4_early_face_rdpd_nomix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_early_face_rdpd_basic_setting(args)
    args = ce_switch_basic_setting(args)
    return args


def simclr_mlp4_cb_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_cb_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_mlp4_cb_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_cb_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_mlp4_cb_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_mlp4_cb_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def moco_mix_build(args):
    args = step600_basic_setting(args)
    args = moco_ep300_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def moco_mix_break(args):
    args = step600_basic_setting(args)
    args = moco_ep300_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def moco_mix_switch(args):
    args = step600_basic_setting(args)
    args = moco_ep300_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def moco_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = moco_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)    
    return args


def moco_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = moco_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def moco_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = moco_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def siamese_mix_build(args):
    args = step600_basic_setting(args)
    args = siamese_ep300_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def siamese_mix_break(args):
    args = step600_basic_setting(args)
    args = siamese_ep300_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def siamese_mix_switch(args):
    args = step600_basic_setting(args)
    args = siamese_ep300_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def step600_basic_setting_res112(args):
    args.num_steps = 600
    args.batch_size = 128
    args.eval_freq = 30
    args.optimizer = 'from_cfg'
    args.use_distributed_samplr = True
    return args


def simclr_r50_sam_ctl_basic_setting(args):
    args.which_model = 'simclr_r50_sy_sam_ctl'
    return args


def simclr_r18_sam_ctl_face_rdpd_mlp4_basic_setting(args):
    args.which_model = 'simclr_r18_sam_ctl_face_mlp4'
    args.train_transforms = 'from_cfg_fx'
    return args


def simclr_r18_sam_ctl_face_rdpd_mlp4_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = simclr_r18_sam_ctl_face_rdpd_mlp4_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_r18_sam_ctl_face_rdpd_mlp4_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = simclr_r18_sam_ctl_face_rdpd_mlp4_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_r18_sam_ctl_face_rdpd_mlp4_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = simclr_r18_sam_ctl_face_rdpd_mlp4_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def simclr_r18_sam_ctl_mlp1_basic_setting(args):
    args.which_model = 'simclr_rs18_sam_ctl_face'
    return args


def simclr_r18_sam_ctl_face_rdpd_mlp1_mix_build(args):
    args = step600_basic_setting(args)
    args = simclr_r18_sam_ctl_mlp1_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def simclr_r18_sam_ctl_face_rdpd_mlp1_mix_break(args):
    args = step600_basic_setting(args)
    args = simclr_r18_sam_ctl_mlp1_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def simclr_r18_sam_ctl_face_rdpd_mlp1_mix_switch(args):
    args = step600_basic_setting(args)
    args = simclr_r18_sam_ctl_mlp1_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def barlow_twins_r18_sam_cont_mlp6_basic_setting(args):
    args.which_model = 'barlow_twins_r18_sam_cont_mlp6'
    return args


def barlow_twins_r18_sam_cont_mlp6_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = barlow_twins_r18_sam_cont_mlp6_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def barlow_twins_r18_sam_cont_mlp6_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = barlow_twins_r18_sam_cont_mlp6_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def barlow_twins_r18_sam_cont_mlp6_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = barlow_twins_r18_sam_cont_mlp6_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args

    
def barlow_twins_r18_sam_cotr_mlp6_basic_setting(args):
    args.which_model = 'barlow_twins_r18_sam_cotr_mlp6'
    return args

def barlow_twins_r18_sam_cotr_mlp6_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = barlow_twins_r18_sam_cotr_mlp6_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def barlow_twins_r18_sam_cotr_mlp6_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = barlow_twins_r18_sam_cotr_mlp6_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def barlow_twins_r18_sam_cotr_mlp6_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = barlow_twins_r18_sam_cotr_mlp6_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def barlow_twins_r18_sam_ctl_mlp6_basic_setting(args):
    args.which_model = 'barlow_twins_r18_sam_ctl_mlp6'
    return args

def barlow_twins_r18_sam_ctl_mlp6_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = barlow_twins_r18_sam_ctl_mlp6_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def barlow_twins_r18_sam_ctl_mlp6_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = barlow_twins_r18_sam_ctl_mlp6_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def barlow_twins_r18_sam_ctl_mlp6_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = barlow_twins_r18_sam_ctl_mlp6_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


# hyperopt settings
def simclr_r18_sam_ctl_mlp4_basic_setting(args):
    args.which_model = 'simclr_r18_sam_ctl_mlp4'
    return args

def simclr_r18_sam_ctl_mlp4_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = simclr_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def simclr_r18_sam_ctl_mlp4_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = simclr_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def simclr_r18_sam_ctl_mlp4_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = simclr_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def byolneg_r18_sam_ctl_mlp4_basic_setting(args):
    args.which_model = 'byolneg_r18_sam_ctl_mlp4'
    return args

def byolneg_r18_sam_ctl_mlp4_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = byolneg_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def byolneg_r18_sam_ctl_mlp4_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = byolneg_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def byolneg_r18_sam_ctl_mlp4_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = byolneg_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def moco_v2_sam_ctl_mlp4_basic_setting(args):
    args.which_model = 'moco_v2_sam_ctl_mlp4'
    return args

def moco_v2_sam_ctl_mlp4_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = moco_v2_sam_ctl_mlp4_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def moco_v2_sam_ctl_mlp4_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = moco_v2_sam_ctl_mlp4_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def moco_v2_sam_ctl_mlp4_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = moco_v2_sam_ctl_mlp4_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def byol_r18_sam_ctl_mlp4_basic_setting(args):
    args.which_model = 'byol_r18_sam_ctl_mlp4'
    return args

def byol_r18_sam_ctl_mlp4_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = byol_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def byol_r18_sam_ctl_mlp4_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = byol_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def byol_r18_sam_ctl_mlp4_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = byol_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def simsiam_r18_sam_ctl_mlp4_basic_setting(args):
    args.which_model = 'simsiam_r18_sam_ctl_mlp4'
    return args

def simsiam_r18_sam_ctl_mlp4_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = simsiam_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def simsiam_r18_sam_ctl_mlp4_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = simsiam_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def simsiam_r18_sam_ctl_mlp4_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = simsiam_r18_sam_ctl_mlp4_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def swav_r18_basic_setting(args):
    args.which_model = 'swav_r18'
    return args

def swav_r18_mix_build(args):
    args = step600_basic_setting(args)
    args = swav_r18_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def swav_r18_mix_break(args):
    args = step600_basic_setting(args)
    args = swav_r18_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def swav_r18_mix_switch(args):
    args = step600_basic_setting(args)
    args = swav_r18_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def swav_r18_sam_ctl_mlp6_basic_setting(args):
    args.which_model = 'swav_r18_sam_ctl_mlp6'
    return args

def swav_r18_sam_ctl_mlp6_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = swav_r18_sam_ctl_mlp6_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def swav_r18_sam_ctl_mlp6_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = swav_r18_sam_ctl_mlp6_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def swav_r18_sam_ctl_mlp6_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = swav_r18_sam_ctl_mlp6_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def swav_r18_face_rdpd_basic_setting(args, seed=None):
    args.which_model = 'swav_r18_face_rdpd'
    args.batch_size = 64
    return args


def swav_r18_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = swav_r18_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def swav_r18_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = swav_r18_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def swav_r18_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = swav_r18_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def swav_r18_early_basic_setting(args):
    args.which_model = 'swav_r18_early'
    return args

def swav_r18_early_mix_build(args):
    args = step600_basic_setting(args)
    args = swav_r18_early_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def swav_r18_early_mix_break(args):
    args = step600_basic_setting(args)
    args = swav_r18_early_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def swav_r18_early_mix_switch(args):
    args = step600_basic_setting(args)
    args = swav_r18_early_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def dino_vit_s_basic_setting(args):
    args.which_model = 'dino_vit_s'
    args.batch_size = 64
    args.mc_update = True
    return args


def dino_mlp5_face_rdpd_basic_setting(args):
    args.which_model = 'dino_mlp5_face_rdpd'
    args.batch_size = 128
    args.mc_update = True
    return args


def dino_mlp5_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = dino_mlp5_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def dino_mlp5_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = dino_mlp5_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def dino_mlp5_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = dino_mlp5_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args
    

def dino_vit_s_imgnt_mix_build(args):
    args = step600_basic_setting(args)
    args = dino_vit_s_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def dino_vit_s_imgnt_mix_break(args):
    args = step600_basic_setting(args)
    args = dino_vit_s_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def dino_vit_s_imgnt_mix_switch(args):
    args = step600_basic_setting(args)
    args = dino_vit_s_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def dinoneg_vit_s_basic_setting(args):
    args.which_model = 'dinoneg_vit_s'
    args.batch_size = 32
    args.mc_update = True
    return args

def dinoneg_vit_s_imgnt_mix_build(args):
    args = step600_basic_setting(args)
    args = dinoneg_vit_s_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def dinoneg_vit_s_imgnt_mix_break(args):
    args = step600_basic_setting(args)
    args = dinoneg_vit_s_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def dinoneg_vit_s_imgnt_mix_switch(args):
    args = step600_basic_setting(args)
    args = dinoneg_vit_s_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def dino_vit_s_sam_ctl_basic_setting(args):
    args.which_model = 'dino_vit_s_sam_ctl'
    args.batch_size = 64
    args.mc_update = True
    return args

def dino_vit_s_sam_ctl_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = dino_vit_s_sam_ctl_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def dino_vit_s_sam_ctl_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = dino_vit_s_sam_ctl_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def dino_vit_s_sam_ctl_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = dino_vit_s_sam_ctl_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def mae_vit_s_face_rdpd_basic_setting(args):
    args.which_model = 'mae_vit_s_face_rdpd'
    args.batch_size = 64
    return args


def mae_vit_s_face_rdpd_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = mae_vit_s_face_rdpd_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def mae_vit_s_face_rdpd_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = mae_vit_s_face_rdpd_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def mae_vit_s_face_rdpd_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = mae_vit_s_face_rdpd_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def mae_vit_l_basic_setting(args):
    args.which_model = 'mae_vit_l_face_rdpd'
    args.batch_size = 64
    return args


def mae_vit_l_face_rdpd_mix_build(args):
    args = step600_basic_setting(args)
    args = mae_vit_l_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args


def mae_vit_l_face_rdpd_mix_break(args):
    args = step600_basic_setting(args)
    args = mae_vit_l_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args


def mae_vit_l_face_rdpd_mix_switch(args):
    args = step600_basic_setting(args)
    args = mae_vit_l_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args


def mae_vit_s_sam_ctl_basic_setting(args):
    args.which_model = 'mae_vit_s_sam_ctl'
    args.batch_size = 64
    return args

def mae_vit_s_sam_ctl_mix_build(args):
    args = step600_basic_setting_res112(args)
    args = mae_vit_s_sam_ctl_basic_setting(args)
    args = mix_build_basic_setting(args)
    return args

def mae_vit_s_sam_ctl_mix_break(args):
    args = step600_basic_setting_res112(args)
    args = mae_vit_s_sam_ctl_basic_setting(args)
    args = mix_break_basic_setting(args)
    return args

def mae_vit_s_sam_ctl_mix_switch(args):
    args = step600_basic_setting_res112(args)
    args = mae_vit_s_sam_ctl_basic_setting(args)
    args = mix_switch_basic_setting(args)
    return args
