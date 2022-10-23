import os
import sys
import pdb
import h5py
import pickle
import numpy as np
import argparse
import copy
from tqdm import tqdm
from PIL import Image
import importlib
import warnings
warnings.filterwarnings("ignore")

import mmcv
import torch
torch.autograd.set_detect_anomaly(True)
import torch.distributed as dist
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
#import apex.amp as amp
#from apex.parallel import DistributedDataParallel as DDP


import dataset
import time_dataset
import time_video_dataset
import build_response
import eval_utils
import local_paths
from local_paths import RESULT_FOLDER, IMAGENET_FOLDER
from utils import check_input

import openselfsup
fwk_path = openselfsup.__path__[0]
sys.path.insert(1, fwk_path[: fwk_path.rfind('/')])


DEFAULT_VALUES = {'setting_func': None,
                  'optimizer': 'from_cfg',
                  'update_interval': None,
                  'init_lr': None,    # Flag to use lr from loaded optmizer state_dict
                  'num_steps': 10,
                  'exposure_num_steps': None,
                  'batch_size': 8,
                  'exp_bs_scale': 1,
                  'eval_freq': 1,
                  'loss_type': None,
                  'which_model': None,
                  'exp_setting': 'swap',
                  'mix_weight': '1.0',
                  'within_batch_ctr': None,
                  'rep': 0,
                  'switch_ratio': 0.5,
                  'uniform_nn': False,
                  'nn_num_mult': 1,
                  'byol_momentum': None,
                  'use_latest_ckpt': False,
                  'multiply_lr_by': 1,    # default is no change
                  'queue_update_options': None,    # default to None, meaning update queue with each input batch.
                  'momentum_encoder': True,
                  'add_train_noise': None,
                  'eval_in_exposure': False,
                  'add_noise': None,
                  'limit_context': None,
                  'context_switching': False,    # used to limit how many negative samples are used in one batch, batch_size stays unchanged (SimCLR only)
                  'vit_im_size': 224,
                  #'mc_update': True,
                  'mc_update': True,
                  'moco_mc_update_in_loss': True,
                  'train_transforms': 'from_cfg',
                  'which_stimuli': 'face',
                  'dist_samplr_seed': False,
                  'fill_in_ctx_queue': False,
                  'freeze_layers_conv': 0,
                  'mae_mask_ratio': None,
                  'save_embd': False,    # if not false, h5 files of embds will be saved
                  'use_apex': False,
                  'concat_batch': False,
                  'real_time_window_size': None,
                  'real_video_aggre_time': None,
                  'rv_min_aggre_time': None,
                  'filter_gray_back': False,
                  'filter_gray_diff': False,
                  'include_test': False,
                  'test_more_trials': False,
                  }
ONLY_EXPOSURE_LOSSES = [
        'ce_moco', 'ce_byol', 'ce_siamese', 'ce_simclr', 'ce_only']
MIX_EXP_IMGNT_LOSSES = [
        'mix_ce_moco', 'mix_ce_simclr', 
        'mix_ce_byol', 'mix_ce_siamese',
        'mix_ce_simclr_asy', 'mix']
DEBUG = os.environ.get('DEBUG', '0')=='1'


def add_general_argument(parser):
    parser.add_argument('--gpu', default='0', type=str,
                        action='store', help='GPU index')
    parser.add_argument('--result_folder', default=RESULT_FOLDER,
                        type=str, action='store',
                        help='Folder to host the results')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--setting_func', 
                        default=DEFAULT_VALUES['setting_func'], type=str,
                        action='store', help='path:func_name')
    return parser


def add_learning_argument(parser):
    parser.add_argument('--init_lr',
                        default=DEFAULT_VALUES['init_lr'],
                        type=float, action='store',
                        help='Start learning rate')
    parser.add_argument('--multiply_lr_by',
                        default=DEFAULT_VALUES['multiply_lr_by'],
                        type=float, action='store',
                        help='Reduce learning rate by this amount after switch_ratio, Noswap_swap Only.')
    parser.add_argument('--num_steps',
                        default=DEFAULT_VALUES['num_steps'],
                        type=int, action='store', help='Number of steps')
    parser.add_argument('--exposure_num_steps',
                        default=DEFAULT_VALUES['exposure_num_steps'],
                        type=int, action='store', help='Number of steps for exposure')
    parser.add_argument('--batch_size',
                        default=DEFAULT_VALUES['batch_size'], type=int,
                        action='store', help='Batch size')
    parser.add_argument('--exp_bs_scale',
                        default=DEFAULT_VALUES['exp_bs_scale'], type=float,
                        action='store', help='How to scale the exp bs')
    parser.add_argument('--eval_freq',
                        default=DEFAULT_VALUES['eval_freq'], type=int,
                        action='store', help='Evaluation frequency')
    parser.add_argument('--loss_type',
                        default=DEFAULT_VALUES['loss_type'], type=str,
                        action='store', help='Type of loss',
                        choices=['simclr_out_of_ctx', 'imgnt_only'] \
                        + MIX_EXP_IMGNT_LOSSES \
                        + ONLY_EXPOSURE_LOSSES)
    parser.add_argument('--which_model',
                        default=DEFAULT_VALUES['which_model'], type=str,
                        action='store', help='Which model to load from',
                        choices=build_response.ALL_MODEL_NAMES)
    parser.add_argument('--uniform_nn', action='store_true',
                        help='Whether background neighbors are sampled uniformly')
    parser.add_argument('--nn_num_mult', action='store', type=int,
                        default=DEFAULT_VALUES['nn_num_mult'],
                        help='Whether background neighbors are sampled from a larger NNs')
    parser.add_argument('--byol_momentum', action='store', type=float,
                        default=DEFAULT_VALUES['byol_momentum'],
                        help='Momentum for BYOL, default is not specified')
    parser.add_argument('--use_latest_ckpt', action='store_true',
                        help='Whether use the latest ckpt regardlessly')
    parser.add_argument('--mix_weight',
                        default=DEFAULT_VALUES['mix_weight'], type=str,
                        action='store', help='Weight for mixing the losses')
    parser.add_argument('--within_batch_ctr',
                        default=DEFAULT_VALUES['within_batch_ctr'], type=float,
                        action='store', help='Weight for within batch contrasting')
    parser.add_argument('--optimizer',
                        default=DEFAULT_VALUES['optimizer'], type=str,
                        action='store', help='Which optimizer to use')
    parser.add_argument('--update_interval',
                        default=DEFAULT_VALUES['update_interval'], type=int,
                        action='store', help='Number of steps to accumulate gradients')
    parser.add_argument('--rep',
                        default=DEFAULT_VALUES['rep'], type=int,
                        action='store', help='Unused parameter, just for repetitions')
    parser.add_argument('--resume', 
                        action='store_true', help='Resume from previous exps')
    parser.add_argument('--queue_update_options', action='store',
                        default=DEFAULT_VALUES['queue_update_options'],
                        choices=[DEFAULT_VALUES['queue_update_options'], 'pretrained_queue',
                                 'face_queue_only', 'separate_queues'],
                        help='For MoCo training only, different ways to update queues')
    parser.add_argument('--momentum_encoder', action='store_false',
                        default=DEFAULT_VALUES['momentum_encoder'],
                        help='Use this flag when not usin momentum encoder.')
    parser.add_argument('--add_noise',
                        default=DEFAULT_VALUES['add_noise'], type=float,
                        action='store', help='Norm of the noise added')
    parser.add_argument('--freeze_layers_conv',
                        default=DEFAULT_VALUES['freeze_layers_conv'], type=int,
                        action='store', help='Number of layers frozen. + starts from early, - starts from late')    
    parser.add_argument('--add_train_noise',
                        default=DEFAULT_VALUES['add_train_noise'], type=float,
                        action='store', help='Norm of the noise added in training')
    parser.add_argument('--eval_in_exposure',
                        action='store_true',
                        help='Changing models to eval phase in exposure phase')
    parser.add_argument('--train_transforms',
                        default=DEFAULT_VALUES['train_transforms'], type=str,
                        action='store', help='Types of transforms used during training')
    parser.add_argument('--mc_update', action='store_true',
                        default=DEFAULT_VALUES['mc_update'])
    parser.add_argument('--moco_mc_update_in_loss', action='store_true',
                        default=DEFAULT_VALUES['moco_mc_update_in_loss'])
    parser.add_argument('--dist_samplr_seed', action='store_true',
                        default=DEFAULT_VALUES['dist_samplr_seed'])
    parser.add_argument('--fill_in_ctx_queue', action='store_true',
                        default=DEFAULT_VALUES['fill_in_ctx_queue'])
    parser.add_argument('--mae_mask_ratio', action='store', type=float,
                        default=DEFAULT_VALUES['mae_mask_ratio'],
                        help='MAE only param, value more than 0.01')
    parser.add_argument('--use_apex', action='store_true',
                        default=DEFAULT_VALUES['use_apex'],
                        help='Use apex instead of torch DDP')
    parser.add_argument('--concat_batch', action='store_true',
                        default=DEFAULT_VALUES['concat_batch'],
                        help='concatenate exposure and pretraining batches')
    parser.add_argument('--real_time_window_size', action='store', type=float,
                        default=DEFAULT_VALUES['real_time_window_size'],
                        help='Learn window size for real time, unit is minute')
    parser.add_argument('--real_video_aggre_time', action='store', type=float,
                        default=DEFAULT_VALUES['real_video_aggre_time'],
                        help='Aggregating window size for real time, unit is second')
    parser.add_argument('--rv_min_aggre_time', action='store', type=float,
                        default=DEFAULT_VALUES['rv_min_aggre_time'],
                        help='Minimal aggregating window size for real time, unit is second')
    parser.add_argument('--filter_gray_back', action='store_true',
                        default=DEFAULT_VALUES['filter_gray_back'],
                        help='Remove the gray background pairs in the real video stim')
    parser.add_argument('--filter_gray_diff', action='store_true',
                        default=DEFAULT_VALUES['filter_gray_diff'],
                        help='Remove the pair including gray background in the real video stim')
    parser.add_argument('--vit_im_size', action='store',
                        default=DEFAULT_VALUES['vit_im_size'],
                        help='Not actually work right now!')
    parser.add_argument('--include_test', action='store_true',
                        default=DEFAULT_VALUES['include_test'],
                        help='Include learning in the test phase')
    parser.add_argument('--test_more_trials', action='store_true',
                        default=DEFAULT_VALUES['test_more_trials'],
                        help='More test trials in real video case')
    parser.add_argument('--torch_distributed', action='store_true',
                        help='deprecated!')
    return parser


def add_exp_argument(parser):
    parser.add_argument('--exp_setting',
                        default=DEFAULT_VALUES['exp_setting'], type=str,
                        action='store', help='Swap, No swap, Noswap swap',
                        choices=[DEFAULT_VALUES['exp_setting'], 'no_swap',
                                 'noswap_swap', 'imgnt_exp_swap',
                                 'imgnt_exp_noswap', 'imgnt_exp_noswapswap',
                                 'imgnt_only'])
    parser.add_argument('--switch_ratio',
                        default=DEFAULT_VALUES['switch_ratio'], type=float,
                        action='store', help='Ratio for switching the stimuli in some exps')
    parser.add_argument('--context_switching', action='store',
                        default=DEFAULT_VALUES['context_switching'],
                        choices=[DEFAULT_VALUES['context_switching'], 'face2object', 'object2face'],
                        help='Ctx switching experiments train with faces then with objects')
    parser.add_argument('--which_stimuli', action='store',
                        default=DEFAULT_VALUES['which_stimuli'],
                        choices=[DEFAULT_VALUES['which_stimuli'], 'objectome'],
                        help='Which experiment to use, objectome or face')
    parser.add_argument('--limit_context',
                        default=DEFAULT_VALUES['limit_context'], type=int,
                        action='store', help='Limit amount of context in one batch, SimCLR only')
    parser.add_argument('--save_embd',
                        default=DEFAULT_VALUES['save_embd'], action='store_true',
                        help='save embedding throughout training. Default False')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
        description='Unsupervised weight change')
    parser = add_general_argument(parser)
    parser = add_learning_argument(parser)
    parser = add_exp_argument(parser)
    return parser


def get_setting_func(setting):
    assert len(setting.split(':')) == 2, \
            'Setting should be "script_path:func_name"'
    script_path, func_name = setting.split(':')
    assert script_path.endswith('.py'), \
            'Script should end with ".py"'
    module_name = script_path.strip('.py').replace('/', '.')
    if module_name.startswith('.'):
        module_name = module_name[1:]
    load_setting_module = importlib.import_module(module_name)
    setting_func = getattr(load_setting_module, func_name)
    setting_func_name = module_name + '.' + func_name
    return setting_func, setting_func_name


import time
class ConductExp:
    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ['RANK'])
        self.num_gpus = torch.cuda.device_count()
        self.reload_args()
        self.__get_model_builder()
        self.__init_process()
        self.__compose_transforms()
        self.get_exposure_builders()
        self.__general_setup()

    def reload_args(self):
        args = self.args
        self.pure_func_args = argparse.Namespace()
        if args.setting_func is None:
            return
        # Change setting_func for later file name
        setting_func, args.setting_func = get_setting_func(args.setting_func)
        self.pure_func_args = setting_func(self.pure_func_args)
        for _key, _value in vars(self.pure_func_args).items():
            use = True
            for _argv in sys.argv[1:]:
                if not _argv.startswith('--'):
                    continue
                if _key.startswith(_argv[2:]):
                    use = False
                    break
            if use:
                setattr(self.args, _key, _value)
            else:
                print(_key + ' specified in both setting func and command line,'\
                           + ' command line value is used!')

    # turn off some layers in resnet backbone (currently doing this by
    # resnet blocks)
    def freeze_layers(self, model):
        ct = 0
        if self.args.freeze_layers_conv != 0:
            if 'byol_' in self.args.which_model \
               or 'siamese_' in self.args.which_model:
                target_layers = getattr(model.online_net, '0')
            elif 'simclr_' in self.args.which_model:
                target_layers = model.backbone
            elif 'moco_' in self.args.which_model:
                target_layers = getattr(model.encoder_q, '0')
            
            module_lists = list(target_layers.named_children())
            if self.args.freeze_layers_conv < 0:
                module_lists.reverse()
            n_layers_frozen = np.abs(self.args.freeze_layers_conv)            
            for name, child in module_lists:
                if ct < n_layers_frozen:
                    for param in child.parameters():
                        param.requires_grad = False
                    ct += 1
                
    def __init_process(self, backend='nccl'):
        """ Initialize the distributed environment. """
        args = self.args
        os.environ['MASTER_ADDR'] = '127.0.0.1'        
        self.multi_gpu = self.num_gpus > 1
        dist.init_process_group(
            backend=backend, world_size=-1, rank=self.rank)        
        # set gpu_id for each process
        self.current_gpu_id = self.rank % self.num_gpus        
        torch.cuda.set_device(self.current_gpu_id)
        model = self.model_builder.model        
        self.freeze_layers(model)
        model = model.cuda(self.current_gpu_id)
        if self.args.fill_in_ctx_queue:
            self.queue_len = model.queue_len
        self.load_model_weights_optimizer()
        
    def load_model_weights_optimizer(self):        
        optimizer_state_dict = self.model_builder.load_weights()
        model = self.model_builder.model
        if self.args.optimizer == 'momentum':
            assert args.init_lr is not None, \
                'init_lr cannot be None for SGD optimizer'
            optimizer = SGD(
                model.parameters(), lr=args.init_lr, momentum=0.9)
        elif self.args.optimizer == 'adam':
            assert args.init_lr is not None, \
                'init_lr cannot be None for Adam optimizer'
            optimizer = Adam(model.parameters(), lr=args.init_lr)
        elif self.args.optimizer == 'from_cfg':
            from openselfsup.apis.train import build_optimizer
            opt_cfg = copy.copy(self.model_args.loaded_cfg.optimizer)
            optimizer = build_optimizer(model, opt_cfg, verbose=False)
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            raise NotImplementedError()
        # setup Apex
        opt_level = 'O1'
        if self.args.use_apex:
            model, self.optimizer = amp.initialize(
                model, optimizer, opt_level=opt_level,
                max_loss_scale=2**13)
        else:            
            self.optimizer = optimizer
            
        self.model = DistributedDataParallel(
            model, broadcast_buffers=False,
            device_ids=[self.current_gpu_id],
            output_device=self.current_gpu_id)
            
    # if using separate queues for co-training (moco), create a copy
    # of initial queue with pre-trained ImageNet representations
    def __memory_queue_setting(self):
        args = self.args
        if args.queue_update_options == 'separate_queues':
            assert args.exp_setting.startswith('imgnt_exp') \
                and 'moco' in args.which_model, \
                "Need ImgNt exp and moco for separate_queue setting"
            self.model.module.copy_pretrain_queue()
            
    def __general_setup(self):
        self.all_results = []
        if self.is_main_process():
            pass

    def remove_vertical_flip_in_transform(self, loaded_cfg):
        try:
            loaded_cfg.data.train.pipeline.remove(
                {'type': 'RandomVerticalFlip', 'p': 1})
        except:
            loaded_cfg.data.train.pipeline1.remove(
                {'type': 'RandomVerticalFlip', 'p': 1})
            loaded_cfg.data.train.pipeline2.remove(
                {'type': 'RandomVerticalFlip', 'p': 1})
        return loaded_cfg
        
    # transform for train&eval images
    def __compose_transforms(self):
        if self.args.which_model in build_response.MODEL_RES112_NAMES:
            self.im_size = 112
        else:                
            self.im_size = self.args.vit_im_size
            
        if self.args.train_transforms == 'default':
            self.transform = dataset.compose_transforms(
                    is_train=True, size=self.im_size)
        elif self.args.train_transforms == 'from_cfg':
            self.transform = dataset.get_transforms_from_cfg(
                    self.model_args.loaded_cfg)
        elif self.args.train_transforms == 'from_cfg_fx':
            loaded_cfg = copy.deepcopy(self.model_args.loaded_cfg)
            if self.args.which_model in \
                    build_response.SAYCAM_MODEL_RES112_NAMES:                
                loaded_cfg = self.remove_vertical_flip_in_transform(loaded_cfg)
            self.transform = dataset.get_actual_transforms_from_cfg(loaded_cfg)
        elif self.args.train_transforms == 'from_cfg_ncj':
            self.transform = dataset.get_transforms_from_cfg_ncj(
                    self.model_args.loaded_cfg)
        elif self.args.train_transforms == 'mae_same_crop':
            self.transform = dataset.compose_mae_transform(size=self.im_size)            
        elif self.args.train_transforms == 'from_cfg_ctrlCJ':
            self.transform = dataset.get_transforms_from_cfg_ctrlCJ(
                    self.model_args.loaded_cfg)
        else:
            raise NotImplementedError
        self.eval_transform = dataset.compose_transforms(
                is_train=False, size=self.im_size)

    # only evaluate & save results in main process
    def is_main_process(self):
        return self.rank == 0

    def __get_change_time(self):
        return int(self.args.num_steps * self.args.switch_ratio)

    def get_exp_batch_size(self):
        args = self.args
        return int(args.batch_size * args.exp_bs_scale)

    def need_phase_from_exp_builder(self):
        return self.args.filter_gray_back or self.args.filter_gray_diff

    def __init_exposure_builder(self, ExposureBuilder, which_pair, which_stimuli):
        args = self.args
        num_steps = args.num_steps * self.num_gpus
        mae_transform = self.args.train_transforms == 'mae_same_crop'
        exp_kwargs = dict(
                which_pair=which_pair,
                num_steps=num_steps,    # extend length for DistSampler
                batch_size=self.get_exp_batch_size(),
                mae_transform=mae_transform,
                which_stimuli=which_stimuli,
                transform=self.transform,
                eval_transform=self.eval_transform,
                im_size=self.args.vit_im_size,
                )
        if args.real_time_window_size is not None:
            exp_kwargs['window_size'] = args.real_time_window_size
            if args.exposure_num_steps is not None:
                exp_kwargs['exposure_num_steps'] = args.exposure_num_steps * self.num_gpus
            if args.real_video_aggre_time is not None:
                exp_kwargs['aggre_time'] = args.real_video_aggre_time
                exp_kwargs['min_aggre_time'] = args.rv_min_aggre_time
            if self.need_phase_from_exp_builder():
                exp_kwargs['return_phases'] = True
            if args.include_test:
                exp_kwargs['include_test'] = args.include_test
            if args.test_more_trials:
                exp_kwargs['test_more_trials'] = args.test_more_trials
        exposure_builder = ExposureBuilder(**exp_kwargs)
        return exposure_builder
    
    def change_noswapswap_builder(self, step):
        assert self.args.exp_setting in ['noswap_swap', 'imgnt_exp_noswapswap'],\
            "exposure builder only changes in noswap_swap condition"
        if self.args.real_time_window_size is not None:
            return
        self.exposure_builder.switch_training()
        sampler = self.build_data_sampler(self.exposure_builder)
        exposure_loader = DataLoader(
                self.exposure_builder, sampler=sampler, **self.params)
        self.exposure_loader = iter(exposure_loader)

    # get stimuli type indicated by how context_switch is done
    def get_stimuli_type(self):
        if self.args.context_switching == 'face2object':
            self.stimuli1 = 'face'
            self.stimuli2 = 'objectome'
        elif self.args.context_switching == 'object2face':
            self.stimuli1 = 'objectome'
            self.stimuli2 = 'face'
            
    # get all experiments training exposure builder
    def get_all_exposure_builders(self, ExposureBuilder, which_stimuli):
        exposure_builder = self.__init_exposure_builder(
            ExposureBuilder, 0, which_stimuli)
        if exposure_builder.is_legal():
            all_exposure_builders = [exposure_builder]
        else:
            all_exposure_builders = []
        num_pairs = exposure_builder.num_pairs
        
        for which_pair in range(1, num_pairs):
            exposure_builder = self.__init_exposure_builder(
                ExposureBuilder, which_pair, which_stimuli)
            if exposure_builder.is_legal():
                all_exposure_builders.append(exposure_builder)
        return all_exposure_builders
        
    def get_exposure_builders(self):
        args = self.args
        # imgnt_only still uses eval images same across ExposureBuilders
        if args.exp_setting in ['swap', 'imgnt_exp_swap', 'imgnt_only']:
            ExposureBuilder = dataset.ExposureBuilder
            if args.real_time_window_size is not None:
                if args.real_video_aggre_time is None:
                    ExposureBuilder = time_dataset.TimeExposureBuilder
                else:
                    ExposureBuilder = time_video_dataset.TimeVideoExpBuilder
        elif args.exp_setting in ['no_swap', 'imgnt_exp_noswap']:
            ExposureBuilder = dataset.NoSwapExposureBuilder
            if args.real_time_window_size is not None:
                if args.real_video_aggre_time is None:
                    ExposureBuilder = time_dataset.NoSwapTimeExposureBuilder
                else:
                    ExposureBuilder = time_video_dataset.NoSwapTimeVideoExpBuilder
        elif args.exp_setting in ['noswap_swap', 'imgnt_exp_noswapswap']:
            ExposureBuilder = dataset.NoswapSwapExposureBuilder
            if args.real_time_window_size is not None:
                if args.real_video_aggre_time is None:
                    ExposureBuilder = time_dataset.NoswapSwapTimeExposureBuilder
                else:
                    ExposureBuilder = time_video_dataset.NoSwapSwapTimeVideoExpBuilder
        else:
            raise NotImplementedError('Wrong exp_setting')
        
        if self.args.context_switching:            
            self.get_stimuli_type()
            self.all_exposure_builders = self.get_all_exposure_builders(
                ExposureBuilder, self.stimuli1)
            self.all_second_exposure_builders = self.get_all_exposure_builders(
                ExposureBuilder, self.stimuli2)            
        else:            
            self.all_exposure_builders = self.get_all_exposure_builders(
                ExposureBuilder, self.args.which_stimuli)

    def get_shuffle_flag(self):
        if self.args.real_time_window_size is not None:
            return False
        return True
            
    def __get_model_builder(self):
        args = self.args
        model_args_func = getattr(
                build_response, 'get_{}_args'.format(args.which_model))
        self.model_args = model_args_func()        
        model_kwargs = {}
        if 'moco' in args.which_model:
            model_kwargs = dict(
                    uniform_nn=args.uniform_nn,
                    nn_num_mult=args.nn_num_mult)
            if args.moco_mc_update_in_loss \
               and 'moco' in args.which_model:
                model_kwargs['update_in_forward'] = False
            else:
                assert args.loss_type not in MIX_EXP_IMGNT_LOSSES, \
                        "MoCo update is wrong without moco_mc_update_in_loss"
        elif 'byol' in args.which_model:
            assert args.mc_update, \
                    "BYOL update is wrong without mc_update"
            if args.byol_momentum is not None:
                model_kwargs['base_momentum'] = args.byol_momentum
        elif 'dino' in args.which_model:
            assert args.mc_update, \
                "DINO is wrong without mc_update"
            if args.byol_momentum is not None:
                model_kwargs['base_momentum'] = args.byol_momentum
            if self.num_gpus == 4:
                self.args.batch_size = 32
                assert self.num_gpus == 4, \
                    'batch_size changed to 32, please use 4 GPUs'
            
        elif 'swav' in args.which_model:
            # swav models queue needs to be loaded according to rank of the process
            self.model_args.rank = self.rank
        elif 'mae' in args.which_model:
            if args.mae_mask_ratio is not None:
                self.model_args.loaded_cfg.model.mask_ratio \
                    = args.mae_mask_ratio
                if args.mae_mask_ratio < 0.3 and args.which_model == 'mae_vit_l_face_rdpd':
                    self.args.batch_size = 32
                    assert self.num_gpus == 4, \
                        'batch_size changed to 32, please use 4 GPUs'
        self.params = {'batch_size': self.get_exp_batch_size(),
                       'shuffle': self.get_shuffle_flag(),
                       'num_workers': self.args.num_workers}                
        if DEBUG:
            self.model_args.loaded_cfg.model['head']['loss_reduced'] = False
        self.model_builder = build_response.ModelBuilder(
                args=self.model_args,
                use_latest_ckpt=args.use_latest_ckpt,
                **model_kwargs)

    def add_perf_acc(self):
        pass

    def load_all_eval_images(self):
        self.all_eval_info = []
        if self.args.context_switching:
            self.second_all_eval_info = []            
        for which_exp in range(self.num_exps):
            exposure_builder = self.all_exposure_builders[which_exp]
            self.all_eval_info.append(
                exposure_builder.get_eval_images())
            
            if self.args.context_switching:
                second_builder = self.all_second_exposure_builders[which_exp]
                self.second_all_eval_info.append(
                    second_builder.get_eval_images())
        
    # only evaluate on main process
    def get_eval_info(self):
        if self.is_main_process():
            self.model.eval()
            smalls = []
            bigs = []
            self.big_succ_diffs = []
            self.big_fail_diffs = []
            self.face_sizes = []
            self.embds = []
            self.test_embds = []
            self.big_faces = []
            self.all_CFs = []
            
            if self.args.context_switching:
                smalls_second = []
                bigs_second = []
                self.big_succ_diffs_second = []
                self.big_fail_diffs_second = []
            
            for which_exp in range(self.num_exps):
                eval_info = self.all_eval_info[which_exp]
                if self.args.context_switching:
                    # first set of stimuli                    
                    small_dprime, big_dprime = self.eval_model(
                        eval_info, which_stimuli=self.stimuli1)
                    smalls.append(small_dprime)
                    bigs.append(big_dprime)
                    
                    # second set of stimuli
                    second_eval_info = self.second_all_eval_info[which_exp]
                    small_dprime_second, big_dprime_second = self.eval_model(
                        second_eval_info, which_stimuli=self.stimuli2)
                    smalls_second.append(small_dprime_second)
                    bigs_second.append(big_dprime_second)                    
                else:
                    small_dprime, big_dprime = self.eval_model(
                        eval_info, which_stimuli=self.args.which_stimuli)
                    smalls.append(small_dprime)
                    bigs.append(big_dprime)
                    
            # get the first stimuli or the only stimuli
            self.results['smalls'].append(smalls)
            self.results['bigs'].append(bigs)
            self.results['big_succs'].append(self.big_succ_diffs)
            self.results['big_fails'].append(self.big_fail_diffs)
            self.results['all_CFs'].append(self.all_CFs)
            if self.args.save_embd:                
                self.results['test_embds'].append(self.test_embds)
                self.results['embds'].append(self.embds)
                self.results['big_faces'].append(self.big_faces)
                
            self.add_perf_acc()
            # second stimuli 
            if self.args.context_switching:
                self.results['smalls_second'].append(smalls_second)
                self.results['bigs_second'].append(bigs_second)
                self.results['big_succs_second'].append(self.big_succ_diffs_second)
                self.results['big_fails_second'].append(self.big_fail_diffs_second)                        
        else:
            pass            

    def add_noise_to_resp(self, all_resp):
        add_noise = self.args.add_noise
        noise = np.random.randn(*all_resp.shape) * add_noise
        all_resp += noise
        all_resp = all_resp / np.linalg.norm(all_resp, axis=1, keepdims=True)
        return all_resp
        
    def eval_model(self, eval_info, which_stimuli):
        test_imgs = eval_info['test_imgs']
        test_imgs = test_imgs.cuda()
        test_faces = eval_info['test_faces']
        num_faces = len(test_faces)
        model_results = self.model(test_imgs, mode='test')
        test_resp = model_results['embd']
        test_resp = test_resp.detach().numpy()

        self.test_embds.append(test_resp)
        self.big_faces.append(eval_info['big_faces'])
        if self.args.which_model == 'moco_v2':            
            test_resp_m = model_results['embd_m']
            test_resp_m = test_resp_m.detach().numpy()
            self.results['embds_m'].append(test_resp_m)
        
        def _get_dprime(imgs, faces, size):
            imgs = imgs.cuda()
            resp = self.model(imgs, mode='test')['embd']            
            resp = resp.detach().numpy()
            if size == 'big':
                self.embds.append(resp)
                
            if self.args.add_noise is not None:
                resp = self.add_noise_to_resp(resp)

            num_stim = len(imgs)
            CF = np.zeros(shape=(num_faces, num_faces))
            succ_diff = []
            fail_diff = []
            for _img_idx in range(num_stim):
                _sims = []
                for _face_idx in range(num_faces):
                    _sims.append(
                        np.sum(resp[_img_idx] * test_resp[_face_idx]))
                gt_idx = 0 if faces[_img_idx]==test_faces[0] else 1
                if _sims[gt_idx] > _sims[1-gt_idx]:
                    CF[gt_idx, gt_idx] += 1
                    succ_diff.append(_sims[gt_idx] - _sims[1-gt_idx])
                else:
                    CF[1-gt_idx, gt_idx] += 1
                    fail_diff.append(_sims[gt_idx] - _sims[1-gt_idx])
                    
            dprime = build_response.d_prime2x2(CF)
            if size == 'big':
                self.all_CFs.append(CF)
            return dprime, np.mean(succ_diff), np.mean(fail_diff)
        
        small_dprime, _, _ = _get_dprime(
            eval_info['small_imgs'], eval_info['small_faces'], 'small')
        big_dprime, big_succ_diff, big_fail_diff = _get_dprime(
            eval_info['big_imgs'], eval_info['big_faces'], 'big')
        
        if not self.args.context_switching or which_stimuli == self.stimuli1:
            self.big_succ_diffs.append(big_succ_diff)
            self.big_fail_diffs.append(big_fail_diff)
        elif which_stimuli == self.stimuli2:
            self.big_succ_diffs_second.append(big_succ_diff)
            self.big_fail_diffs_second.append(big_fail_diff)
        return small_dprime, big_dprime
        
    # training setup for one experiment, including optimizer,
    # load model weights at the begining for each trail
    def setup(self):        
        args = self.args
        self.load_model_weights_optimizer()
            
        if args.optimizer == 'from_cfg':
            for param_group in self.optimizer.param_groups:
                if args.init_lr is not None:
                    param_group['lr'] = args.init_lr                    
                if args.mix_weight == 0:                    
                    param_group['weight_decay'] = 0    # sanity check with 0 weight decay

        if args.queue_update_options == 'separate_queues':
            self.__memory_queue_setting()
            
        if self.is_noswapswap():
            if args.multiply_lr_by != 1:
                def lr_lambda(step):
                    if step < self.__get_change_time():
                        return 1
                    else:
                        return args.multiply_lr_by
                self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            else:
                self.scheduler = None
            
    def is_noswapswap(self):
        return self.args.exp_setting in ['noswap_swap', 'imgnt_exp_noswapswap']

    def check_images(self, step, imgs):
        from torchvision.utils import save_image
        img1 = imgs[:, 0, :, :, :]
        img2 = imgs[:, 1, :, :, :]
        save_image(img1, f'input_imgs/step_{step}_img_0.png')
        save_image(img2, f'input_imgs/step_{step}_img_1.png')
        
    def fill_within_ctx_queue(self):
        assert 'moco' in self.args.which_model and \
            self.args.queue_update_options == 'separate_queues', \
            'Need moco model and separate_queues'
        
        effective_batch_size = self.num_gpus * self.args.batch_size
        n_steps = int(self.queue_len // effective_batch_size) + 1
        sampler = self.build_data_sampler(self.exposure_builder)        
        exposure_loader = DataLoader(
            self.exposure_builder, sampler=sampler, **self.params)
        with torch.no_grad():            
            for step, images in enumerate(exposure_loader):
                if step < n_steps:                    
                    images = images.cuda()
                    self.model(images)
                else:
                    break

    def get_next_exposure_batch(self, step, switch_context):
        if (self.args.exposure_num_steps is None)\
                or (step <= self.args.exposure_num_steps)\
                or (self.args.real_time_window_size is not None):
            if not switch_context:
                images = next(self.exposure_loader)
            else:
                images = next(self.second_loader)
            if self.need_phase_from_exp_builder():
                images, (phase_0, phase_1) = images
                idx_to_keep = []
                for idx, (_p0, _p1) in enumerate(zip(phase_0, phase_1)):
                    if self.args.filter_gray_back\
                            and (_p0 == 'g' and _p1 == 'g'):
                        continue
                    if self.args.filter_gray_diff\
                            and ((_p0 == 'g' and _p1 != 'g')\
                                 or (_p1 == 'g' and _p0 != 'g')):
                        continue
                    idx_to_keep.append(idx)
                images = images[idx_to_keep]
            images = images.cuda()
        else:
            images = self.get_next_imgnt_batch()['img']
            images = self.change_image_to_cuda(images)
        return images
        
    def do_one_exp(self, switch_context=False):
        args = self.args
        if not switch_context:    # during switching task, this is just the last results from previous stimuli
            self.get_eval_info()
            
        if self.is_main_process():
            step_iter = tqdm(range(args.num_steps))
        else:
            step_iter = range(args.num_steps)
            
        for step in step_iter:
            self.model.train()
            self.optimizer.zero_grad()
            
            images = self.get_next_exposure_batch(
                    step, switch_context)
            loss = self.get_loss(images)
            if args.update_interval is not None:
                loss /= args.update_interval
            if self.args.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()                

            if (args.update_interval is None) or ((step+1) % args.update_interval==0):
                self.optimizer.step()
            if self.args.mc_update \
               and ('byol' in args.which_model or 'dino' in args.which_model):
                self.model.module.momentum_update()
            if self.args.moco_mc_update_in_loss \
               and 'moco' in args.which_model:
                self.model.module._momentum_update_key_encoder()
            self.all_losses.append(loss.item())
            if (step+1) % self.args.eval_freq == 0 or \
               (switch_context and step < self.args.eval_freq
                and (step+1) % 5 == 0):    # record every 5 step before evaluation point after switching context
                self.get_eval_info()
                if self.args.save_embd:
                    self.results['training_embds'].append(self.training_embds)
                self.results['face_sizes'].append([
                    self.exposure_builder.face1_sizes, self.exposure_builder.face2_sizes])
                if 'swav' in self.args.which_model and self.args.save_embd:
                    self.results['prototypes'].append(self.prototype)
                    self.results['cluster_assignment'].append(self.cluster_assignment)                    
            # Noswap_swap condition at switch change face pairs and learning rate
            if self.is_noswapswap():
                if self.scheduler is not None:
                    self.scheduler.step()
                if step == self.__get_change_time() - 1:
                    self.change_noswapswap_builder(step)        
            
    def limit_current_context(self, images):
        limit_context = self.args.limit_context
        assert self.args.which_model == 'simclr_r18', \
            'Context varying experiments are only used in SimCLR experiments'
        assert self.args.batch_size % limit_context == 0, \
            'limit context not divisible by batch_size * num_gpu'
        num_repeats = self.args.batch_size // limit_context
        limited_images = images[ : limit_context]
        return limited_images.repeat(num_repeats, 1, 1, 1, 1)

    def get_imgnt_model_kwargs(self):
        kwargs = {}
        if 'moco' in self.args.which_model:
            kwargs['momentum_enc'] = self.args.momentum_encoder
            if self.args.queue_update_options in \
                    ['pretrained_queue', 'face_queue_only']:
                kwargs['mode'] = 'train_no_queue_update'
            elif self.args.queue_update_options == 'separate_queues':
                kwargs['mode'] = 'train_separate_queue'
        elif 'mae' in self.args.which_model:
            kwargs['exposure'] = False
        return kwargs
        
    ''' 
    __imgnt_loss & __exposure_loss used for all models following
    openselfsup format, models include byol, siamese, simclr, moco
    '''
    def get_next_imgnt_batch(self):
        try:
            images = next(self.imgnt_loader_iter)
        except StopIteration:    # full imgnt will run out in 600 steps
            self.epoch += 1
            self.imgnt_sampler.set_epoch(self.epoch)
            self.imgnt_loader_iter = iter(self.imgnt_loader)
            images = next(self.imgnt_loader_iter)
        return images

    def change_image_to_cuda(self, images):
        if 'swav' not in self.args.which_model and \
           'dino' not in self.args.which_model:            
            images = images.cuda()
        else:
            if self.args.use_apex:
                images = [
                    images[0].half().cuda(), images[1].half().cuda()]
            else:
                images = [
                    images[0].cuda(), images[1].cuda()]
        return images
        
    def __imgnt_loss(self):
        images = self.get_next_imgnt_batch()['img']
        images = self.change_image_to_cuda(images)
            
        # varying available negative samples can be used given fixed batch size
        if self.args.limit_context:
            images = self.limit_current_context(images)

        kwargs = self.get_imgnt_model_kwargs()
        loss = self.model(images, **kwargs)
        return loss['loss']

    def get_exposure_model_kwargs(self):
        kwargs = {}
        which_model = self.args.which_model
        if 'moco' in which_model:
            kwargs.update({
                    'within_batch_ctr': self.args.within_batch_ctr,
                    'more_metrics': True,
                    'momentum_enc': self.args.momentum_encoder,
                    'add_noise': self.args.add_train_noise,
                    })
            if self.args.queue_update_options == 'pretrained_queue':
                kwargs['mode'] = 'train_no_queue_update'
            if getattr(self, 'curr_time_diff', None) is not None \
                    and self.args.add_train_noise is not None:
                kwargs['add_noise'] = self.args.add_train_noise \
                        * self.curr_time_diff
        elif 'simclr' in which_model:
            kwargs['within_batch_ctr'] = self.args.within_batch_ctr
            kwargs['add_noise'] = self.args.add_train_noise
            if 'simclr_asy' in which_model:
                kwargs['mode'] = 'asy_train'
        elif 'siamese' in which_model:
            kwargs['add_noise'] = self.args.add_train_noise
        elif 'byol' in which_model:
            kwargs['add_noise'] = self.args.add_train_noise
        elif 'mae' in which_model:
            kwargs['exposure'] = True
        return kwargs

    def multi_crop_style_input(self):
        return ('swav' in self.args.which_model \
                or 'dino' in self.args.which_model)
    
    def __exposure_loss(self, inputs):
        # varying avaiu868lable negative samples can be used given fixed batch size
        if self.args.limit_context:
            inputs = self.limit_current_context(inputs)

        if self.multi_crop_style_input()\
           and (not isinstance(inputs, list)):
            inputs = [
                inputs[:, 0, ...], inputs[:, 1, ...]]
        if self.args.eval_in_exposure:
            self.model.eval()
        else:
            self.model.train()
        kwargs = self.get_exposure_model_kwargs()
        loss = self.model(inputs, **kwargs)
        self.model.train()
        loss_value = loss.pop('loss')
        if DEBUG:
            pdb.set_trace()
            loss_value = torch.mean(loss_value)
            pass
        #self.results['other_metrics'].append(loss)
        if self.args.save_embd:
            self.training_embds = loss['embds']
            
        if 'swav' in self.args.which_model:
            self.prototype = loss['prototype']
            self.cluster_assignment = loss['cluster_assignment']            
            return loss_value#, loss['cluster_assignment'], loss['prototype']
        else:
            return loss_value

    # simclr loss using out-of-context negative images
    def __simclr_cotrain_loss(self, inputs):
        imgnt_batch = self.get_next_imgnt_batch()
        imgnt_batch = imgnt_batch['img'].cuda()
        imgs = torch.cat((inputs, imgnt_batch), dim=0)
        loss = self.model(inputs, mode='exposure_cotrain')
        return loss['loss']

    def get_concat_loss(self, inputs):
        images = self.get_next_imgnt_batch()['img']
        images = self.change_image_to_cuda(images)
        
        if self.multi_crop_style_input():
            inputs1 = torch.vstack([images[0], inputs[:, 0, ...]])
            inputs2 = torch.vstack([images[1], inputs[:, 1, ...]])
            concat_inputs = [inputs1, inputs2]
        else:
            if 'mae' in self.args.which_model:
                images = torch.stack([images, images], dim=1)
            concat_inputs = torch.cat((inputs, images), 0)
        return self.__exposure_loss(concat_inputs)
                
    def get_loss(self, inputs):
        #check_input(inputs)
        loss_type = self.args.loss_type
        if loss_type in ONLY_EXPOSURE_LOSSES:
            return self.__exposure_loss(inputs)
        elif loss_type in MIX_EXP_IMGNT_LOSSES:
            assert self.args.exp_setting.startswith('imgnt_exp'), \
                "Need ImgNt exp"
            if self.args.concat_batch:
                return self.get_concat_loss(inputs)
            else:                
                imgnt_loss = self.__imgnt_loss()
                exposure_loss = self.__exposure_loss(inputs)
                return exposure_loss * float(self.args.mix_weight) + imgnt_loss
        elif loss_type == 'simclr_out_of_ctx':
            assert self.args.which_model in ['simclr_r18', 'simclr_r18_objectome'], \
                'simclr_out_of_ctx loss is only used for simclr models'
            return self.__simclr_cotrain_loss(inputs)
        elif loss_type == 'imgnt_only':
            assert self.args.exp_setting == 'imgnt_only', \
                'exp_setting need to be imgnt_only'
            return self.__imgnt_loss()
        else:
            raise NotImplementedError

    def build_data_sampler(self, builder, shuffle=None):
        sampler = None
        if self.multi_gpu:
            if self.args.dist_samplr_seed:
                sampler = DistributedSampler(
                    builder, shuffle=shuffle or self.get_shuffle_flag(), 
                    seed=self.which_exp)
            else:
                sampler = DistributedSampler(
                        builder, shuffle=shuffle or self.get_shuffle_flag())
            
            self.params['shuffle'] = False    # distributedSampler doesn't work with shuffle in dataloader
        return sampler

    def build_imgnt_load_iter(self):
        imgnt_builder = dataset.build_imgnt_dataset(
            self.model_args.loaded_cfg)
        # adding faces to the saycam loader
        if self.args.real_time_window_size is not None:
            self.exposure_builder.set_pretrain_fns(
                    imgnt_builder.data_source.fns)

        if self.args.exp_setting.startswith('imgnt_exp')\
                or self.args.exp_setting == 'imgnt_only':
            
            self.imgnt_sampler = self.build_data_sampler(
                    imgnt_builder, shuffle=True)
            self.epoch = 0    # used to update seed for imgnt_samplr
            imgnt_params = copy.copy(self.params)
            imgnt_params['batch_size'] = self.args.batch_size*2 - self.params['batch_size']
            self.imgnt_loader = DataLoader(
                imgnt_builder, sampler=self.imgnt_sampler, **imgnt_params)
                
            self.imgnt_loader_iter = iter(self.imgnt_loader)
            
    def set_exp_of_interest(self, which_exp):
        self.load_all_eval_images()
        self.which_exp = which_exp
        self.exposure_builder = self.all_exposure_builders[which_exp]        

        # First build imgnt load iter so that 
        # the pretrain fns can be set if needed
        self.build_imgnt_load_iter()

        sampler = self.build_data_sampler(self.exposure_builder)        
        exposure_loader = DataLoader(
            self.exposure_builder, sampler=sampler, **self.params)        
        self.exposure_loader = iter(exposure_loader)
        
        if self.args.context_switching:
            assert self.args.real_time_window_size is None,\
                    "Real time builder for context switch not implemented"
            self.second_builder = self.all_second_exposure_builders[which_exp]
            second_sampler = self.build_data_sampler(self.second_builder)
            second_loader = DataLoader(
                self.second_builder, sampler=second_sampler, **self.params)            
            self.second_loader = iter(second_loader)
        
        self.init_results()
            
    def get_command(self):
        copy_argv = copy.copy(sys.argv)
        for _arg in sys.argv:
            if _arg.startswith('--local_rank='):
                copy_argv.remove(_arg)
        command = 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU_IDS} ' \
                  + 'python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=$RANDOM ' \
                  + ' '.join(copy_argv)
        return command

    def init_results(self):
        self.all_losses = []
        command = self.get_command()
        self.results = {
            # first set of results
            'smalls': [], 'bigs': [],
            'big_succs': [], 'big_fails': [],
            # second set of results (face or objects)
            'smalls_second': [], 'bigs_second': [],
            'big_succs_second': [], 'big_fails_second': [],
            # common stuff for both objects
            'which_exp': self.which_exp,
            'faces': [self.exposure_builder.face0, \
                      self.exposure_builder.face1],
            'all_losses': self.all_losses,
            'command': command,
            'Perf_acc': [],
            'args': copy.copy(self.args),
            'all_CFs': [],
            #'other_metrics': [],
            'training_embds': [],
            'face_sizes': [],
            'embds': [],    # embding of choice face
            'embds_m': [],
            'test_embds': [],    # embding of test faces with natural background
            'big_faces': [],    # record faces for embedding
            # SwAV Only
            'prototypes': [],
            'cluster_assignment': []
        }
        ctl_pair = getattr(
                self.exposure_builder, 'chosen_other_pair_path', None)
        if ctl_pair is not None:
            self.results['ctl_pair'] = ctl_pair

    def do_exps(self):        
        start_idx = 0
        if self.args.resume:
            results_path = self.get_results_path()
            if os.path.exists(results_path):
                old_results = pickle.load(open(results_path, 'rb'))
                start_idx = len(old_results)
                self.all_results = old_results

        if self.args.context_switching:    # face only has 15 because two faces were exclueded
            
            self.num_exps = min(
                len(self.all_exposure_builders),
                len(self.all_second_exposure_builders))
        else:    
            self.num_exps = len(self.all_exposure_builders)

        if self.is_main_process():
            exp_iter = tqdm(
                range(start_idx, self.num_exps),
                desc='Experiments')
        else:
            exp_iter = range(start_idx, self.num_exps)
        for which_exp in exp_iter:
            self.setup()
            self.set_exp_of_interest(which_exp)
            if self.args.fill_in_ctx_queue:
                self.fill_within_ctx_queue()
                
            self.do_one_exp(switch_context=False) 
            # switching context to objects
            if self.args.context_switching:                
                self.do_one_exp(switch_context=True)
                
            if self.is_main_process():
                self.all_results.append(self.results)
            self.store_results()
            
    def get_results_path(self):
        args = self.args        
        results_basename = 'results%s.pkl'
        suffix = ''
        for key, default_value in DEFAULT_VALUES.items():
            value = getattr(args, key, default_value)
            if value==default_value:
                continue
            # Only log the keys not specified by the setting_func
            if getattr(self.pure_func_args, key, default_value) == value:
                continue
            suffix += f'_{key}{value}'
        
        suffix += f'_num_gpus{self.num_gpus}'        
        results_basename = results_basename % suffix
        if args.save_embd:
            idx = results_basename.rfind('.')
            h5_basename = results_basename[:idx]
            self.h5_path = os.path.join(
                args.result_folder, h5_basename + '.h5')
        results_path = os.path.join(args.result_folder, results_basename)        
        return results_path
    
    def store_results(self):
        if self.is_main_process():
            if not os.path.exists(self.args.result_folder):
                os.system('mkdir -p ' + self.args.result_folder)
            results_path = self.get_results_path()
            if self.args.save_embd:
                all_result_copy = copy.deepcopy(self.all_results)
                hf = h5py.File(self.h5_path, 'w')                
                for i, result in enumerate(all_result_copy):
                    exp_group = hf.create_group(f'exp{i}')
                    for k in [
                            'embds', 'embds_m', 'test_embds', 'prototypes', 
                            'cluster_assignment', 'training_embds']:
                        exp_group[k] = np.array(result[k])
                        result.pop(k)
                hf.close()
                pickle.dump(all_result_copy, open(results_path, 'wb'))
                print(self.h5_path)
                print(results_path)
            else:
                pickle.dump(self.all_results, open(results_path, 'wb'))
                print(results_path)

    def check_imgnt_distributed_loader(self):
        from torchvision.utils import save_image
        imgnt_builder = dataset.build_imgnt_dataset(
            self.model_args.loaded_cfg)
            
        self.imgnt_sampler = self.build_data_sampler(imgnt_builder, shuffle=True)
        
        self.imgnt_loader = DataLoader(
            imgnt_builder, sampler=self.imgnt_sampler, **self.params)
        self.imgnt_loader_iter = iter(self.imgnt_loader)
        epoch = 0
        
        for step in tqdm(range(120)):
            images = next(self.imgnt_loader_iter)            
            if self.is_main_process():
                images = next(self.imgnt_loader_iter)
                images = images['img']
                if step % 30 == 0:
                    self.imgnt_sampler.set_epoch(epoch)
                    self.imgnt_loader_iter = iter(self.imgnt_loader)
                    out_path = f'input_imgs/imgnt_epoch_{epoch}_step_{step}_shuffle.png'
                    #out_path = f'input_imgs/imgnt_epoch_{epoch}_step_{step}_no_shuffle.png'     
                    save_image(images[:, 0, :, :, :], out_path)
                    epoch += 1


def main():
    parser = get_parser()
    args = parser.parse_args()
    conduct_exp = ConductExp(args)
    conduct_exp.do_exps()
    
    
if __name__ == '__main__':
    main()

