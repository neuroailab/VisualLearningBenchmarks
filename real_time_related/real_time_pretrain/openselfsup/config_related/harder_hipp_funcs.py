import copy


def mmax_inner_func(cfg, hidden_size=256):
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    cfg.model = {
            'type': 'HippRNN',
            'seq_len': 32,
            'num_negs': 3,
            'add_random_target_to_others': True,
            'randomize_noise_norm': True,
            'add_target_to_others': True,
            'loss_type': 'just_pos',
            'hipp_head': dict(
                type='HippRNNHead',
                rnn_type='gnrl_pat',
                rnn_tile=3,
                rnn_kwargs=dict(
                    input_size=128, 
                    hidden_size=hidden_size,
                    naive_hidden=False, hand_coded_softmax=False,
                    pattern_size=100),
                pred_mlp=dict(
                    type='NonLinearNeckV1',
                    in_channels=hidden_size, hid_channels=hidden_size,
                    out_channels=512, with_avg_pool=False),
                ),
            }
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    return cfg


def msim32_mns(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.model['add_target_range'] = '1.3,3.5'
    cfg.model['noise_norm'] = 0.2
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def msim32_mns_s(cfg):
    cfg = mmax_inner_func(cfg)
    cfg.model['add_target_range'] = '0,2.5'
    cfg.model['noise_norm'] = 0.3
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def mns_s_mask_func(cfg):
    cfg.model['add_target_range'] = '0,2.5'
    cfg.model['noise_norm'] = 0.3
    cfg.model['mask_use_kNN'] = True
    return cfg


def msim32_mns_s_mask(cfg):
    cfg = mmax_inner_func(cfg)
    cfg = mns_s_mask_func(cfg)
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def msim32_mns_s_mask_stpmlp(cfg):
    cfg = mmax_inner_func(cfg)
    cfg = mns_s_mask_func(cfg)
    cfg.data['imgs_per_gpu'] = 32
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_layer_io'
    return cfg


def msim32_mns_s_mask_gate_stpmlp(cfg):
    cfg = mmax_inner_func(cfg)
    cfg = mns_s_mask_func(cfg)
    cfg.data['imgs_per_gpu'] = 32
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_layer_io'
    cfg.model['hipp_head']['rnn_kwargs']['stpmlp_kwargs'] = {
            'gate_update': True,
            }
    return cfg
