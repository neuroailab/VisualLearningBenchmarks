import numpy as np
import copy


def remove_out_bn(cfg):
    cfg.model['neck']['with_last_bn'] = False
    return cfg


def lower_tau(cfg):
    cfg.model['head']['temperature'] = 0.07
    return cfg

def higher_tau(cfg):
    cfg.model['head']['temperature'] = 0.13
    return cfg


def neg_fn_num(cfg):
    cfg.model['head']['neg_fn_num'] = 400
    return cfg


def mneg_fn_num(cfg):
    cfg.model['head']['neg_fn_num'] = 470
    return cfg


def mlp_3layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 3
    return cfg


def mlp_4layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    return cfg


def mlp_5layers_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 5
    return cfg


def mlp_res_inter_cfg_func(cfg):
    cfg.model['neck']['res_inter'] = True
    return cfg


def mlp_xlayers_cfg_func(x):
    def _func(cfg):
        cfg.model['neck']['num_layers'] = x
        return cfg
    return _func


def mlp_4L1bn_cfg_func(cfg):
    cfg.model['neck']['num_layers'] = 4
    cfg.model['neck']['bn_settings'] = [True, False, False]
    return cfg


def add_sm_backbone_neck(cfg):
    cfg.model['backbone']['add_neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=1024,
            out_channels=1024,
            num_layers=2,
            with_avg_pool=True)
    cfg.model['neck']['in_channels'] = 1024
    cfg.model['neck']['with_avg_pool'] = False
    return cfg


def add_backbone_neck(cfg):
    cfg.model['backbone']['add_neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512,
            hid_channels=2048,
            out_channels=2048,
            num_layers=2,
            with_avg_pool=True)
    cfg.model['neck']['in_channels'] = 2048
    cfg.model['neck']['with_avg_pool'] = False
    return cfg


def add_backbone_neck_np(cfg):
    cfg.model['backbone']['add_neck'] = dict(
            type='NonLinearNeckSimCLR',
            in_channels=512*4*4,
            hid_channels=2048,
            out_channels=2048,
            num_layers=2,
            with_avg_pool=False)
    cfg.model['neck']['in_channels'] = 2048
    cfg.model['neck']['with_avg_pool'] = False
    return cfg


def neg_th_value_d7(cfg):
    cfg.model['head']['neg_th_value'] = 0.7
    return cfg


def neg_th_value_d5(cfg):
    cfg.model['head']['neg_th_value'] = 0.5
    return cfg


def neg_th_value_d9(cfg):
    cfg.model['head']['neg_th_value'] = 0.9
    return cfg


def neg_sub16(cfg):
    cfg.model['head']['margin'] = np.log(510.0/16.0)
    cfg.model['neg_subsamples'] = 16
    return cfg


def less_neg_cont_16(cfg):
    cfg.model['head']['margin'] = np.log(510.0/16.0)
    cfg.model['type'] = 'LessNegSimCLR'
    cfg.model['less_method'] = 'cont-16'
    return cfg


def less_neg_with_param(k=16, margin=np.log(510.0/16.0)):
    def _func(cfg):
        cfg.model['head']['margin'] = margin
        cfg.model['type'] = 'LessNegSimCLR'
        cfg.model['less_method'] = 'cont-{}'.format(k)
        return cfg
    return _func


def corr_opt_head_w_param(mix_weight=1.0):
    def _func(cfg):
        cfg.model['head'] = {
                'type': 'CorrOptHead',
                'mix_weight': mix_weight,
                }
        return cfg
    return _func


def group_w_param(group_no=32):
    def _func(cfg):
        cfg.model['type'] = 'GroupSimCLR'
        cfg.model['group_no'] = group_no
        return cfg
    return _func


def sep_batch_w_param(**kwargs):
    def _func(cfg):
        cfg.model['type'] = 'SepBatchSimCLR'
        cfg.model.update(kwargs)
        return cfg
    return _func


def use_inter_out_simclr(
        out_indices=[2, 3, 4],
        **kwargs):
    def _func(cfg):
        cfg.model['type'] = 'InterOutSimCLR'
        cfg.model['backbone']['out_indices'] = out_indices
        neck_4 = cfg.model.pop('neck')
        neck_3 = copy.deepcopy(neck_4)
        neck_3['in_channels'] = 256
        neck_2 = copy.deepcopy(neck_4)
        neck_2['in_channels'] = 128
        cfg.model['all_necks'] = [neck_2, neck_3, neck_4]
        cfg.model.update(kwargs)
        return cfg
    return _func

def sparse_inter_out_simclr(
        out_indices=[3, 4],
        neck_t=0.1,
        **kwargs):
    def _func(cfg):
        cfg.model['type'] = 'InterOutSimCLR'
        cfg.model['backbone']['out_indices'] = out_indices
        neck = cfg.model.pop('neck')
        sp_neck_4 = dict(type='SmplSprsMsk', t=neck_t)
        sp_neck_3 = copy.deepcopy(sp_neck_4)
        cfg.model['all_necks'] = [sp_neck_3, sp_neck_4, neck]
        cfg.model.update(kwargs)
        return cfg
    return _func

def add_SmplSprsMsk_neck(cfg, neck_t, out_indices, zero_offset):
    neck = cfg.model.pop('neck')
    sp_neck = dict(type='SmplSprsMsk', t=neck_t, zero_offset=zero_offset)
    cfg.model['all_necks'] = []
    for _ in out_indices:
        cfg.model['all_necks'].append(copy.deepcopy(sp_neck))
    cfg.model['all_necks'].append(neck)
    return cfg

def add_CorrOptHead(
        cfg, head_mix_weight, out_indices,
        **kwargs):
    head = cfg.model.pop('head')
    sp_head = dict(type='CorrOptHead', mix_weight=head_mix_weight)
    sp_head.update(kwargs)
    cfg.model['head'] = []
    for _ in out_indices:
        cfg.model['head'].append(copy.deepcopy(sp_head))
    cfg.model['head'].append(head)
    return cfg

def sparse_corr_inter_out_simclr(
        out_indices=[4],
        head_mix_weight=1.0,
        neck_t=0.1,
        zero_offset=False,
        **kwargs):
    def _func(cfg):
        cfg.model['type'] = 'InterOutSimCLR'
        cfg.model['backbone']['out_indices'] = out_indices
        cfg = add_SmplSprsMsk_neck(cfg, neck_t, out_indices, zero_offset)
        cfg = add_CorrOptHead(cfg, head_mix_weight, out_indices, **kwargs)
        return cfg
    return _func

def add_AvgPool_neck(cfg, out_indices):
    neck = cfg.model.pop('neck')
    sp_neck = dict(type='AvgPoolNeck')
    cfg.model['all_necks'] = []
    for _ in out_indices:
        cfg.model['all_necks'].append(copy.deepcopy(sp_neck))
    cfg.model['all_necks'].append(neck)
    return cfg

def add_Identity_neck(cfg, out_indices):
    neck = cfg.model.pop('neck')
    sp_neck = dict(type='Identity')
    cfg.model['all_necks'] = []
    for _ in out_indices:
        cfg.model['all_necks'].append(copy.deepcopy(sp_neck))
    cfg.model['all_necks'].append(neck)
    return cfg

def add_L1SparseHead(
        cfg, head_weight, out_indices,
        **kwargs):
    head = cfg.model.pop('head')
    sp_head = dict(type='L1SparseHead', weight=head_weight)
    sp_head.update(kwargs)
    cfg.model['head'] = []
    for _ in out_indices:
        cfg.model['head'].append(copy.deepcopy(sp_head))
    cfg.model['head'].append(head)
    return cfg

def l1_sparse_inter_out_simclr(
        out_indices=[4],
        head_weight=1.0,
        avg_neck=True,
        ):
    def _func(cfg):
        cfg.model['type'] = 'InterOutSimCLR'
        cfg.model['backbone']['out_indices'] = out_indices
        cfg.model['head_mode'] = ['embd_head']*len(out_indices) + ['contrastive']
        if avg_neck:
            cfg = add_AvgPool_neck(cfg, out_indices)
        else:
            cfg = add_Identity_neck(cfg, out_indices)
        cfg = add_L1SparseHead(cfg, head_weight, out_indices)
        return cfg
    return _func

def remove_color_gray_gaussian(cfg):
    cfg.data['train']['pipeline'].pop(2)
    cfg.data['train']['pipeline'].pop(2)
    cfg.data['train']['pipeline'].pop(2)
    return cfg

def change_to_momentum(cfg):
    cfg.model['type'] = 'SimCLRMomentum'
    return cfg

def base_change_to_torch_gaussian(_transform):
    _transform['type'] = 'GaussianBlurTorch'
    _transform['kernel_size'] = 23
    _transform['sigma'] = (_transform.pop('sigma_min'), _transform.pop('sigma_max'))
    return _transform

def change_gaussian_to_torch(cfg):
    assert cfg.data['train']['pipeline'][4]['transforms'][0]['type'] == 'GaussianBlur'
    _transform = cfg.data['train']['pipeline'][4]['transforms'][0]
    _transform = base_change_to_torch_gaussian(_transform)
    cfg.data['train']['pipeline'][4]['transforms'][0] = _transform
    return cfg

def remove_gaussian(cfg):
    cfg.data['train']['pipeline'].pop(4)
    return cfg

def move_gaussian_to_model(multi_trans):
    def _func(cfg):
        assert cfg.data['train']['pipeline'][4]['transforms'][0]['type'] == 'GaussianBlur'
        _pipeline = cfg.data['train']['pipeline'].pop(4)
        _transform = _pipeline['transforms'][0]
        _transform = base_change_to_torch_gaussian(_transform)
        _pipeline['transforms'][0] = _transform
        _pipeline = dict(
                type='MultiTransform',
                num_imgs_one_sp=multi_trans,
                transforms=[_pipeline])
        cfg.model['add_transform'] = [_pipeline]
        return cfg
    return _func
