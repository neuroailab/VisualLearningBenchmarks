import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import copy

import numpy as np
from . import utils

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    return parser


def get_default_args():
    parser = get_args_parser()
    args = parser.parse_args([])
    return args
default_args = get_default_args()


def add_multi_crop_dataset(args, cfg):
    cfg.data['train']['type'] = 'DINOMultiCropDataset'
    cfg.data['train']['size_crops'] = (224, 96)
    cfg.data['train']['min_scale_crops'] = (args.global_crops_scale[0], args.local_crops_scale[0])
    cfg.data['train']['max_scale_crops'] = (args.global_crops_scale[1], args.local_crops_scale[1])
    cfg.data['train']['local_pipeline'] = copy.deepcopy(cfg.data['train']['pipeline1'])
    cfg.data['train']['local_pipeline'][4]['p'] = 0.5
    return cfg

def add_model_setting(args, cfg):
    cfg.model = dict(
            type='DINO',
            dino_args=args,
            )
    return cfg

def add_AdamW_ep100(args, cfg):
    cfg.total_epochs = 100
    cfg.optimizer = dict(
            type='AdamW', lr=args.lr, 
            weight_decay=args.weight_decay)
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=1e-6,
            warmup='linear',
            warmup_iters=10,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg

def basic_settings(cfg):
    args = default_args
    args.local_crops_number = 0
    args.global_crops_scale = (0.14, 1.)

    cfg = add_multi_crop_dataset(args, cfg)
    cfg = add_model_setting(args, cfg)
    cfg = add_AdamW_ep100(args, cfg)
    cfg.data['train']['nmb_crops'] = [2, 0]
    return cfg


def vit_s_ncrp(cfg):
    cfg = basic_settings(cfg)
    return cfg

def vit_s_ncrp_corr(cfg):
    default_args.lr = 0.0005 * 2
    #default_args.norm_last_layer = False
    cfg = basic_settings(cfg)
    return cfg

def vit_s_ncrp_corr_wd(cfg):
    default_args.lr = 0.0005 * 2
    cfg = basic_settings(cfg)
    cfg.optimizer['paramwise_options']= {
            '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.)}
    return cfg

def vit_s_ncrp_p8(cfg):
    default_args.patch_size = 8
    cfg = basic_settings(cfg)
    return cfg

def vit_change_to_p8(cfg):
    default_args.patch_size = 8
    return cfg


def mlp_w_layer(nlayers):
    def _func(cfg):
        if 'dino_head_kwargs' not in cfg.model:
            cfg.model['dino_head_kwargs'] = dict(
                    nlayers=nlayers)
        else:
            cfg.model['dino_head_kwargs']['nlayers'] = nlayers
        return cfg
    return _func

def neg_vit_s_ncrp(cfg):
    cfg = basic_settings(cfg)
    cfg.model['neg_head'] = dict(type='ContrastiveHead', temperature=0.1)
    return cfg

def neg_nsf_vit_s_ncrp(cfg):
    cfg = basic_settings(cfg)
    cfg.model['neg_head'] = dict(type='ContrastiveHead', temperature=0.1)
    cfg.model['use_neg_loss'] = True
    if 'dino_head_kwargs' not in cfg.model:
        cfg.model['dino_head_kwargs'] = dict()
    cfg.model['dino_head_kwargs']['with_last_layer'] = False
    return cfg

def neg_nsf_vit_s_ncrp_corr_wd(cfg):
    default_args.lr = 0.0005 * 2
    cfg = neg_nsf_vit_s_ncrp(cfg)
    cfg.optimizer['paramwise_options']= {
            '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.)}
    return cfg
