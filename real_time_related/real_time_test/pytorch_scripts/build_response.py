import os
import sys
import pdb
import json
import scipy
import pickle
import numpy as np
from argparse import Namespace
import argparse
from scipy.stats import norm
import importlib
import dataset
import local_paths
from openselfsup.models import build_model

import mmcv
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

import openselfsup
FRMWK_REPO_PATH = os.path.dirname(openselfsup.__path__[0])
sys.path.insert(1, FRMWK_REPO_PATH)


ALL_MODEL_NAMES = [
    'simclr_r18_face_rdpd',
    'simclr_mlp4_face_rdpd_early',
    'simclr_r50_face_rdpd',
    'dino_mlp3_face_rdpd',
    'dinoneg_mlp3_face_rdpd',
    'mae_vit_s_face_rdpd',
    'byol_r18_face_rdpd',
    'byol_mlp4_face_rdpd',
    'byolneg_r18_face_rdpd',
    'swav_r18_face_rdpd_early',
    'moco_v2_face_rdpd',
    'barlow_twins_r18_face_rdpd',
    'siamese_r18_face_rdpd',
]
MODEL_RES112_NAMES = []
CONFIG_FRWK_SETUP = dict(
    simclr_r18_face_rdpd='configs/new_pplns/simclr/r18.py:r18_img_face_rdpd_sm',
    simclr_mlp4_face_rdpd_early='configs/new_pplns/simclr/r18.py:r18_ep300_mlp4_face',
    simclr_r50_face_rdpd='configs/new_pplns/simclr/r50.py:r50_ep100_face',
    dino_mlp3_face_rdpd='configs/new_pplns/dino/imgnt.py:vit_s_ncrp_corr_face_s0',
    dinoneg_mlp3_face_rdpd='configs/new_pplns/dino/imgnt.py:neg_nsf_vit_s_ncrp_corr_wd_face',
    mae_vit_s_face_rdpd='configs/new_pplns/mae/imgnt.py:small_vit_ep800_corr_wd_face',
    byol_r18_face_rdpd='configs/new_pplns/byol/r18.py:r18_img_face_rdpd_sm',
    byol_mlp4_face_rdpd='configs/new_pplns/byol/r18.py:r18_mlp4_img_face_rdpd_sm',
    byolneg_r18_face_rdpd='configs/new_pplns/byol/r18.py:r18_neg_face_rdpd_sm',
    swav_r18_face_rdpd_early='configs/new_pplns/swav/r18.py:nocrop_r18_ep300_face',
    moco_v2_face_rdpd='configs/new_pplns/moco/r18.py:r18_v2_img_face_rdpd_sm',
    barlow_twins_r18_face_rdpd='configs/new_pplns/barlow_twins/in.py:r18_ep300_face',
    siamese_r18_face_rdpd='configs/new_pplns/simsiam/r18.py:r18_img_face_rdpd_sm',
)


MODEL_KWARGS = {}

def get_setting_func(setting):
    assert len(setting.split(':')) == 2, \
            'Setting should be "script_path:func_name"'
    script_path, func_name = setting.split(':')
    assert script_path.endswith('.py'), \
            'Script should end with ".py"'
    module_name = script_path[:-3].replace('/', '.')
    while module_name.startswith('.'):
        module_name = module_name[1:]
    load_setting_module = importlib.import_module(module_name)
    setting_func = getattr(load_setting_module, func_name)
    return setting_func


def load_cfg_from_funcs(which_model):
    param_func = get_setting_func(
        CONFIG_FRWK_SETUP[which_model])
    params = param_func(None)
    cfg = params['batch_processor_params']['func'].__self__.cfg
    ckpt_path = os.path.join(
        params['save_params']['ckpt_hook_kwargs']['out_dir'],
        'epoch_300.pth')
    assert which_model not in MODEL_KWARGS
    MODEL_KWARGS[which_model] = dict(
        ckpt_path=ckpt_path,
        loaded_cfg=cfg,
        cfg_path=None, cfg_func=None)


def add_model_args(model_name):
    def __curr_func():
        args = Namespace()
        args.model_type = model_name
        up_model_name = model_name.upper()
        args.ckpt_path = getattr(local_paths, up_model_name)        
        if '_CKPT' in up_model_name:
            up_model_name = up_model_name[
                : up_model_name.rfind('_CKPT')]
                
        load_cfg_from_funcs(model_name)
        args.loaded_cfg = MODEL_KWARGS[model_name]['loaded_cfg']
        return args
    
    all_things = globals()
    all_things['get_{}_args'.format(model_name)] = __curr_func
        

for model_name in ALL_MODEL_NAMES:
    add_model_args(model_name)

    
def d_prime2x2(CF):
    H = CF[0,0]/(CF[0,0]+CF[1,0]) # H = hit/(hit+miss)
    F = CF[0,1]/(CF[0,1]+CF[1,1]) # F =  False alarm/(false alarm+correct rejection)
    if H == 1:
        H = 1-1/(2*(CF[0,0]+CF[1,0]))
    if H == 0:
        H = 0+1/(2*(CF[0,0]+CF[1,0]))
    if F == 0:
        F = 0+1/(2*(CF[0,1]+CF[1,1]))
    if F == 1:
        F = 1-1/(2*(CF[0,1]+CF[1,1]))
    d = norm.ppf(H)-norm.ppf(F)
    if np.isnan(d):
        d = 0
    return d


def get_dprime(resp, faces, test_faces, test_resp):
    num_stim = len(resp)
    num_faces = len(test_faces)
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
                
    dprime = d_prime2x2(CF)
    return dprime, np.mean(succ_diff), np.mean(fail_diff)
    

# build network based on model args
class ModelBuilder:
    '''
    kwargs in the init function will be used to update the model cfg
    '''
    def __init__(self, args, use_latest_ckpt=False, **kwargs):
        self.args = args
        self.kwargs = kwargs    # for model_cfg files and build from cfg
        if use_latest_ckpt:
            latest_ckpt = os.path.join(
                    os.path.dirname(args.ckpt_path), 'latest.pth')
            if os.path.exists(latest_ckpt):
                self.args.ckpt_path = latest_ckp
        self.__build_network()
    
    def __build_network(self):
        cfg = self.args.loaded_cfg
        model_cfg = cfg.model
        for key, value in self.kwargs.items():
            model_cfg[key] = value
        if 'mocosimclr' in self.args.model_type:
            model_cfg['type'] = 'MOCOSimCLR'
            model_cfg['neck']['type'] = 'NonLinearNeckV2'
        model = build_model(model_cfg)
        self.model = model
        
    def load_weights(self, apex=False, ckpt=None, test_only=False):
        # fixing multigpu issue by https://naga-karthik.github.io/post/pytorch-ddp/
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu')
        else:
            checkpoint = torch.load(
                self.args.ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            # MAE repo ckpts loading
            old_keys = list(checkpoint['model'].keys())
            for _key in old_keys:
                _value = checkpoint['model'].pop(_key)
                checkpoint['model']['backbone.'+_key] = _value
            self.model.load_state_dict(checkpoint['model'])
        if 'swav' in self.args.ckpt_path and not test_only:
            from openselfsup.models.swav import QueueHook
            rank = self.args.rank
            queue_dir = os.path.dirname(self.args.ckpt_path)
            queue_path = os.path.join(
                queue_dir, f'queue{rank}.pth')
            self.model.queue_hook = QueueHook(self.args, queue_dir)
            self.model.queue_hook.use_the_queue = True
            self.model.queue_hook.queue = torch.load(queue_path)['queue']
            
        if apex:
            import apex.amp as amp
            assert 'amp' in checkpoint, \
                'ckpt is not trained with apex!!'
            amp.load_state_dict(checkpoint['amp'])
            
        return checkpoint['optimizer']
        
        
def evaluate_resps_corr(all_resp, all_test_resp,
                        stim_faces, test_faces,
                        metric='dot_product',
                        add_noise=None):
    if add_noise is not None:
        print(f'adding {add_noise} ratio')
        noise = np.random.randn(*all_resp.shape) * add_noise
        all_resp += noise        
        all_resp = all_resp / np.linalg.norm(all_resp, axis=1, keepdims=True)

    corr_nums = []
    num_stim = len(stim_faces)
    num_faces = len(test_faces)
    confusion_matrix = np.zeros(shape=(num_faces, num_faces))
    for _img_idx in range(num_stim):
        _sims = []
        for _face_idx in range(num_faces):
            if metric == 'dot_product':
                _sims.append(np.sum(all_resp[_img_idx] * all_test_resp[_face_idx]))
            elif metric == 'abs_diff':
                _sims.append(-np.sum(np.abs(all_resp[_img_idx] - all_test_resp[_face_idx])))
            else:
                raise NotImplementedError
        _sims = np.asarray(_sims)
        gt_sim = _sims[test_faces == stim_faces[_img_idx]]
        gt_idx = np.where(test_faces == stim_faces[_img_idx])[0]
        for _face_idx in range(num_faces):
            if _sims[_face_idx] < gt_sim:
                confusion_matrix[gt_idx, gt_idx] += 1
            if _sims[_face_idx] > gt_sim:
                confusion_matrix[gt_idx, _face_idx] += 1
        corr_nums.append(np.sum(_sims < gt_sim))

    return confusion_matrix, \
        np.sum(corr_nums) / (num_stim * (num_faces-1))


def add_noise_to_resp(all_resp, add_noise):
    noise = np.random.randn(*all_resp.shape) * add_noise
    all_resp += noise
    all_resp = all_resp / np.linalg.norm(all_resp, axis=1, keepdims=True)
    return all_resp


# get performance for all checkpoint in a folder
def get_ckpt_perf(
        model_type, all_eval_images_builders=None,
        which_stimuli='face', **kwargs):
    results = {}
    if 'objectome' in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])

    if not all_eval_images_builders:
        all_eval_images, all_exposure_builders = get_all_eval_images(
            which_stimuli=which_stimuli)
    else:
        all_eval_images, all_exposure_builders = all_eval_images_builders

    ckpt_base = getattr(local_paths, model_type.upper() + '_BASE')    
    ckpts = [
        f for f in os.listdir(ckpt_base)
        if f.endswith('pth') and f.startswith('epoch_')]
    ckpts = sorted(
            ckpts, 
            key=lambda x: int(x.split('_')[1].split('.')[0]))
    # model args
    all_things = globals()
    func_name = 'get_{}_args'.format(model_type)
    args = all_things[func_name]()
    
    for ckpt in ckpts:
        ckpt_path = os.path.join(ckpt_base, ckpt)
        perf, dprime, pairwise_dprimes = test_build_resp(
            args, model_type, which_stimuli, ckpt=ckpt_path,
            all_eval_images_builders=(
                all_eval_images, all_exposure_builders))
        
        results[ckpt] = [perf, dprime, pairwise_dprimes]
        print(f'{ckpt} perf: {perf}, dprime: {dprime}')
    return results


def get_all_eval_images(which_stimuli='face', im_size=224, **kwargs):
    eval_transform = dataset.compose_transforms(
        False, size=im_size)
    exposure_builder = dataset.ExposureBuilder(
        0, num_steps=1, batch_size=1,
        which_stimuli=which_stimuli,
        eval_transform=eval_transform,
        im_size=im_size)
    
    if exposure_builder.is_legal():
        all_eval_images = [exposure_builder.get_eval_images()]
        all_exposure_builders = [exposure_builder]
    else:
        all_eval_images = []
        all_exposure_builders = []
    num_pairs = exposure_builder.num_pairs
    
    for which_pair in range(1, num_pairs):
        exposure_builder = dataset.ExposureBuilder(
            which_pair, num_steps=1, batch_size=1,
            which_stimuli=which_stimuli,
            eval_transform=eval_transform)
        if exposure_builder.is_legal():
            eval_imgs = exposure_builder.get_eval_images()
            all_eval_images.append(eval_imgs)
            all_exposure_builders.append(exposure_builder)
    return all_eval_images, all_exposure_builders
        
# test specific checkpoint
def test_build_resp(
        args, model_type, which_stimuli='face', ckpt=None,
        all_eval_images_builders=None, **kwargs):    
    if not all_eval_images_builders:    # for testing a single ckpt
        all_eval_images, all_exposure_builders \
            = get_all_eval_images(which_stimuli=which_stimuli)
    else:
        all_eval_images, all_exposure_builders \
            = all_eval_images_builders
        
    model_builder = ModelBuilder(args)
    model_builder.load_weights(ckpt=ckpt, test_only=True)
    model = model_builder.model
    model.cuda()
    model.eval()

    num_exps = len(all_eval_images)
    perfs = []
    dprimes = []
    pairwise_perfs = {}
    pairwise_dprimes = {}
    
    for which_exp in range(num_exps):
        exposure_builder = all_exposure_builders[which_exp]
        pair_name = ' & '.join(
            [str(exposure_builder.face0),
             str(exposure_builder.face1)])
        eval_imgs = all_eval_images[which_exp]
        big_imgs = eval_imgs['big_imgs']
        big_imgs = big_imgs.type(torch.cuda.FloatTensor)
        small_imgs = eval_imgs['small_imgs']
        small_imgs = small_imgs.type(torch.cuda.FloatTensor)
        
        if which_stimuli == 'face':
            all_resp = model(big_imgs, mode='test')['embd']
            stim_faces = eval_imgs['big_faces']
        elif which_stimuli == 'objectome':
            all_resp = model(small_imgs, mode='test')['embd']
            stim_faces = eval_imgs['small_faces']
        all_resp = all_resp.detach().numpy()

        test_faces = exposure_builder.all_test_objs
        test_imgs = torch.stack(exposure_builder._load_transform_imgs_from_paths(
            exposure_builder.all_test_stim_paths))
        test_imgs = test_imgs.type(torch.cuda.FloatTensor)
        all_test_resp = model(test_imgs, mode='test')['embd']
        all_test_resp = all_test_resp.detach().numpy()
        
        test_faces = exposure_builder.all_test_objs
        cm, perf = evaluate_resps_corr(
            all_resp, all_test_resp, stim_faces, test_faces, **kwargs)
        perfs.append(perf)
        
        dprime_test_faces = eval_imgs['test_faces']
        dprime_test_imgs = eval_imgs['test_imgs'].type(torch.cuda.FloatTensor)    
        dprime_test_resp = model(dprime_test_imgs, mode='test')['embd']
        dprime_test_resp = dprime_test_resp.detach().numpy()
        num_stimuli = len(dprime_test_faces)

        big_dprime, _, _ = get_dprime(
            all_resp, stim_faces, dprime_test_faces, dprime_test_resp)
        pairwise_dprimes[pair_name] = big_dprime
        
        dprimes.append(big_dprime)
    if ckpt:
       return np.mean(perfs), np.mean(dprimes), pairwise_dprimes
    else:
        output = f'{model_type} {which_stimuli} '
        output += f'accuracy {np.mean(perfs)} dprime {np.mean(dprimes)}'
        print(output)

def get_model_im_size(which_model):
    return 224


def test():
    model_type = os.environ.get(
        'MODEL', 'general')
    all_things = globals()
    func_name = 'get_{}_args'.format(model_type)
    args = all_things[func_name]()
    model_builder = ModelBuilder(args)

    
if __name__ == '__main__':
    pass
