import argparse
import math
import os
import shutil
import time
from openselfsup.framework.dist_utils import get_dist_info
from . import utils


def get_parser():
    parser = argparse.ArgumentParser(
            description="Implementation of SwAV")
    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                        help="path to dataset repository")
    parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                        help="list of number of crops (example: [2, 6])")
    parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")
    parser.add_argument("--use_pil_blur", type=utils.bool_flag, default=True,
                        help="""use PIL library to perform blur instead of opencv""")

    #########################
    ## swav specific params #
    #########################
    parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                        help="list of crops id used for computing assignments")
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="temperature parameter in training loss")
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--improve_numerical_stability", default=False, type=utils.bool_flag,
                        help="improves numerical stability in Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--feat_dim", default=128, type=int,
                        help="feature dimension")
    parser.add_argument("--nmb_prototypes", default=3000, type=int,
                        help="number of prototypes")
    parser.add_argument("--queue_length", type=int, default=0,
                        help="length of the queue (0 for no queue)")
    parser.add_argument("--epoch_queue_starts", type=int, default=15,
                        help="from this epoch, we start using a queue")

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--hidden_mlp", default=2048, type=int,
                        help="hidden layer dimension in projection head")
    parser.add_argument("--workers", default=10, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint_freq", type=int, default=25,
                        help="Save the model periodically")
    parser.add_argument("--use_fp16", type=utils.bool_flag, default=True,
                        help="whether to train with mixed precision or not")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
    parser.add_argument("--dump_path", type=str, default=".",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    return parser


def get_default_args():
    parser = get_parser()
    args = parser.parse_args([])

    args.size_crops = [224, 96]
    args.min_scale_crops = [0.14, 0.05]
    args.max_scale_crops = [1., 0.14]
    args.use_fp16 = True
    args.freeze_prototypes_niters = 5005
    args.queue_length = 3840
    args.epoch_queue_starts = 15
    args.hidden_mlp = 512

    args.rank, args.world_size = get_dist_info()
    return args
default_args = get_default_args()


def basic_settings(cfg):
    args = default_args

    cfg = add_multi_crop_dataset(args, cfg)
    cfg = add_model_setting(args, cfg)
    return cfg


def add_multi_crop_dataset(args, cfg):
    cfg.data['train']['type'] = 'MultiCropDataset'
    cfg.data['train']['size_crops'] = args.size_crops
    cfg.data['train']['min_scale_crops'] = args.min_scale_crops
    cfg.data['train']['max_scale_crops'] = args.max_scale_crops
    return cfg


def add_model_setting(args, cfg):
    cfg.model['type'] = 'SwAV'
    cfg.model.pop('head')
    cfg.model['args'] = args
    cfg.model['nmb_prototypes'] = args.nmb_prototypes
    return cfg


def no_crop_cfg_func(cfg):
    cfg = basic_settings(cfg)
    cfg.data['train']['nmb_crops'] = [2, 0]
    return cfg


def w_crop_cfg_func(cfg):
    cfg = basic_settings(cfg)
    cfg.data['train']['nmb_crops'] = [2, 6]
    return cfg


def sgd400_cfg_func(cfg):
    cfg.total_epochs = 400
    cfg.optimizer = dict(type='SGD', lr=0.6, weight_decay=1e-6, momentum=0.9)
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=0.0006,
            warmup='linear',
            warmup_iters=1,
            warmup_ratio=0.0001,
            warmup_by_epoch=True)
    return cfg


def sgd100_cfg_func(cfg):
    cfg.total_epochs = 100
    cfg.optimizer = dict(type='SGD', lr=0.6, weight_decay=1e-6, momentum=0.9)
    cfg.lr_config = dict(
            policy='CosineAnnealing',
            min_lr=0.0006,
            warmup=None)
    return cfg


def test():
    args = basic_settings()


if __name__ == '__main__':
    test()
