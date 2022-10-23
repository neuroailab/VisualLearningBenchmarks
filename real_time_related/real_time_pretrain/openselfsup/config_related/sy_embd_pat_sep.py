import copy
import os
SY_DATASET_DIR = os.environ.get(
        'SY_DATASET_DIR',
        '/mnt/fs1/Dataset')


def basic_func(cfg):
    cfg.data['train'] = {
            'type': 'SAYCamSeqVecDataset',
            'seq_len': 64,
            'root': os.path.join(SY_DATASET_DIR, 'infant_headcam/embeddings/'),
            'list_file': os.path.join(SY_DATASET_DIR, 'infant_headcam/embd_train_meta.txt'),
            }
    cfg.data['train']['which_model'] = 'simclr_mst_pair_ft_in'
    cfg.data['imgs_per_gpu'] = 128
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    return cfg


NAIVE_MLP_CFG = dict(
        type='NaiveMLP',
        in_channels=128, 
        hid_channels=256, 
        out_channels=256,
        )
def naive_mlp_dg(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSep',
            'ps_neck': copy.deepcopy(NAIVE_MLP_CFG),
            }
    return cfg


def naive_mlp_dw_dg(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSep',
            'ps_neck': dict(
                type='NaiveMLP',
                in_channels=128, 
                hid_channels=[512, 512], 
                out_channels=256,
                ),
            }
    return cfg


def naive_mlp_ww_dg(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSep',
            'ps_neck': dict(
                type='NaiveMLP',
                in_channels=128, 
                hid_channels=2048, 
                out_channels=256,
                ),
            }
    return cfg


def naive_mlp_s10_ww_dg(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSep',
            'ps_neck': dict(
                type='SparseMLP',
                in_channels=128, 
                hid_channels=2048, 
                out_channels=256,
                sparsity=10,
                ),
            }
    return cfg


def naive_mlp_s20_ww_dg(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSep',
            'ps_neck': dict(
                type='SparseMLP',
                in_channels=128, 
                hid_channels=2048, 
                out_channels=256,
                sparsity=20,
                ),
            }
    return cfg


NAIVE_S30_WW_CFG = dict(
        type='SparseMLP',
        in_channels=128, 
        hid_channels=2048, 
        out_channels=256,
        sparsity=30,
        )
def naive_mlp_s30_ww_dg(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSep',
            'ps_neck': copy.deepcopy(NAIVE_S30_WW_CFG),
            }
    return cfg


NAIVE_S40_WW_CFG = dict(
        type='SparseMLP',
        in_channels=128, 
        hid_channels=2048, 
        out_channels=256,
        sparsity=40,
        )
def naive_mlp_s40_ww_dg(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSep',
            'ps_neck': copy.deepcopy(NAIVE_S40_WW_CFG),
            }
    return cfg


def naive_mlp_s40_dw_dg(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSep',
            'ps_neck': dict(
                type='SparseMLP',
                in_channels=128, 
                hid_channels=[512, 512], 
                out_channels=256,
                sparsity=40,
                ),
            }
    return cfg


def naive_mlp_s30_dw_dg(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSep',
            'ps_neck': dict(
                type='SparseMLP',
                in_channels=128, 
                hid_channels=[512, 512], 
                out_channels=256,
                sparsity=30,
                ),
            }
    return cfg


TYPICAL_PR = dict(
        type='NaiveMLP',
        in_channels=256, 
        hid_channels=256, 
        out_channels=128,
        )
def naive_mlp_dg_rec(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSepRec',
            'ps_neck': copy.deepcopy(NAIVE_MLP_CFG),
            'pr_neck': copy.deepcopy(TYPICAL_PR),
            }
    return cfg


WW_PR = dict(
        type='NaiveMLP',
        in_channels=256, 
        hid_channels=2048, 
        out_channels=128,
        )
def naive_mlp_s30_ww_dg_rec(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSepRec',
            'ps_neck': copy.deepcopy(NAIVE_S30_WW_CFG),
            'pr_neck': copy.deepcopy(WW_PR),
            }
    return cfg


def naive_mlp_s40_ww_dg_rec(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSepRec',
            'ps_neck': copy.deepcopy(NAIVE_S40_WW_CFG),
            'pr_neck': copy.deepcopy(WW_PR),
            }
    return cfg


H512_S30_WW_CFG = dict(
        type='SparseMLP',
        in_channels=128,
        hid_channels=2048,
        out_channels=512,
        sparsity=30,
        )
H512_WW_PR = dict(
        type='NaiveMLP',
        in_channels=512,
        hid_channels=2048, 
        out_channels=128,
        )
def h512_mlp_s30_ww_dg_rec(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'PatSepRec',
            'ps_neck': copy.deepcopy(H512_S30_WW_CFG),
            'pr_neck': copy.deepcopy(H512_WW_PR),
            }
    return cfg


DynamicMLP2L_CFG = dict(
        type='DynamicMLP2L',
        in_channels=128, 
        hid_channels=256, 
        out_channels=256,
        learning_rate=1e-3,
        )
def dgca3_pat_sep_rec(cfg):
    cfg = basic_func(cfg)
    cfg.model = {
            'type': 'DgCA3PatSepRec',
            'dg_neck': copy.deepcopy(NAIVE_S30_WW_CFG),
            'ca3_neck': copy.deepcopy(DynamicMLP2L_CFG),
            }
    return cfg


NAIVE_S60_W_CFG = dict(
        type='SparseMLP',
        in_channels=128, 
        hid_channels=1024,
        out_channels=256,
        sparsity=60,
        )


DynamicMLP2L_S8R16_CFG = dict(
        type='NStepRepDynamicMLP2L',
        in_channels=128, 
        hid_channels=256, 
        out_channels=256,
        learning_rate=1e-3,
        rep_num=16,
        step_num=8,
        )


DynamicMLP2L_R32_CFG = dict(
        type='RepDynamicMLP2L',
        in_channels=128, 
        hid_channels=256, 
        out_channels=256,
        learning_rate=1e-3,
        rep_num=32,
        )


SelfGradDynamicMLP2L_CFG = dict(
        type='SelfGradDynamicMLP2L',
        in_channels=128, 
        hid_channels=256, 
        out_channels=256,
        learning_rate=1e-3,
        )
