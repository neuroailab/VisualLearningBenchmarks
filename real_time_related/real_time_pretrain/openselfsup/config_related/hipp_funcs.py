import copy


def hipp_rnn_test_cfg_func(cfg, hidden_size=512):
    cfg.data['train'] = {
            'type': 'NaiveVectorDataset',
            }
    cfg.model = {
            'type': 'HippRNN',
            'seq_len': 8,
            'noise_norm': 0.1,
            'num_negs': 3,
            'hipp_head': dict(
                type='HippRNNHead',
                rnn_kwargs=dict(
                    input_size=128, 
                    hidden_size=hidden_size, num_layers=1),
                pred_mlp=dict(
                    type='NonLinearNeckV1',
                    in_channels=hidden_size, hid_channels=hidden_size,
                    out_channels=512, with_avg_pool=False),
                )
            }
    cfg.optimizer = dict(
            type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
    cfg.data['imgs_per_gpu'] = 256
    return cfg


def hipp_rnn_adam_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    return cfg


def hipp_rnn_adam_self_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['rnn_type'] = 'self'
    return cfg


def hipp_rnn_adam_self_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_self_cfg_func(cfg)
    cfg.model['loss_type'] = 'just_pos'
    return cfg


def hipp_rnn_adam_pat_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_self_cfg_func(cfg)
    cfg.model['loss_type'] = 'just_pos'
    cfg.model['hipp_head']['rnn_type'] = 'pat'
    cfg.model['hipp_head']['rnn_kwargs'].pop('num_layers')
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 100
    cfg.model['hipp_head']['rnn_kwargs']['hidden_size'] = 128
    cfg.model['hipp_head']['pred_mlp']['in_channels'] = 128
    cfg.model['hipp_head']['rnn_tile'] = 3
    return cfg


def hipp_rnn_adam_pat_cfg_func(cfg):
    cfg = hipp_rnn_adam_pat_jp_cfg_func(cfg)
    cfg.model['loss_type'] = 'default'
    return cfg


def hipp_rnn_adam_gnrl_pat_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_pat_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'gnrl_pat'
    return cfg


def hipp_rnn_adam_mmax_gnrl_pat_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_pat_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'gnrl_pat'
    cfg.model['hipp_head']['rnn_kwargs']['hand_coded_softmax'] = False
    return cfg


def hipp_rnn_adam_mh_gnrl_pat_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_pat_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'gnrl_pat'
    cfg.model['hipp_head']['rnn_kwargs']['naive_hidden'] = False
    return cfg


def hipp_rnn_adam_mh_mmax_gnrl_pat_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_pat_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'gnrl_pat'
    cfg.model['hipp_head']['rnn_kwargs']['naive_hidden'] = False
    cfg.model['hipp_head']['rnn_kwargs']['hand_coded_softmax'] = False
    return cfg


def hipp_rnn_adam_mh_mmax_gnrl_pat_cfg_func(cfg):
    cfg = hipp_rnn_adam_pat_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'gnrl_pat'
    cfg.model['hipp_head']['rnn_kwargs']['naive_hidden'] = False
    cfg.model['hipp_head']['rnn_kwargs']['hand_coded_softmax'] = False
    cfg.model['loss_type'] = 'default'
    return cfg


def hipp_rnn_adam_mh_pr_mmax_gnrl_pat_cfg_func(cfg):
    cfg = hipp_rnn_adam_mh_mmax_gnrl_pat_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['parallel_mmax'] = 3
    cfg.model['hipp_head']['pred_mlp']['in_channels'] = 512
    return cfg


def hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_mh_mmax_gnrl_pat_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['hidden_size'] = 256
    cfg.model['hipp_head']['pred_mlp']['in_channels'] = 256
    cfg.model['add_target_to_others'] = True
    cfg.model['loss_type'] = 'just_pos'
    return cfg


def hipp_rnn_adam_atD_mh_mmax_gnrl_pat_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func(cfg)
    cfg.model['add_target_range'] = '0.8,1.2'
    return cfg


def hipp_rnn_adam_atDD_mh_mmax_gnrl_pat_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func(cfg)
    cfg.model['add_target_range'] = '2.6,3.0'
    cfg.model['noise_norm'] = 0.07
    return cfg


def hipp_rnn_adam_atDDH_mh_mmax_gnrl_pat_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func(cfg)
    cfg.model['add_target_range'] = '2.6,3.0'
    cfg.model['noise_norm'] = 0.07
    cfg.model['add_random_target_to_others'] = True
    cfg.model['randomize_noise_norm'] = True
    return cfg


def hipp_rnn_adam_atDDH_lng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func(cfg)
    cfg.model['add_target_range'] = '2.6,3.0'
    cfg.model['noise_norm'] = 0.07
    cfg.model['add_random_target_to_others'] = True
    cfg.model['randomize_noise_norm'] = True
    cfg.model['seq_len'] = 16
    return cfg


def hipp_rnn_adam_atLDDHH_lng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func(cfg)
    cfg.model['add_target_range'] = '1.3,5.0'
    cfg.model['noise_norm'] = 0.07
    cfg.model['add_random_target_to_others'] = True
    cfg.model['randomize_noise_norm'] = True
    cfg.model['seq_len'] = 16
    return cfg


def hipp_rnn_adam_atDDH_lnglng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func(cfg)
    cfg.model['add_target_range'] = '2.6,3.0'
    cfg.model['noise_norm'] = 0.07
    cfg.model['add_random_target_to_others'] = True
    cfg.model['randomize_noise_norm'] = True
    cfg.model['seq_len'] = 32
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func(cfg)
    cfg.model['add_target_range'] = '4.6,5.0'
    cfg.model['noise_norm'] = 0.07
    cfg.model['add_random_target_to_others'] = True
    cfg.model['randomize_noise_norm'] = True
    cfg.model['seq_len'] = 32
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def hipp_rnn_adam_seqctl_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    return cfg


def hipp_seqctl_gate_hsim32_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    return cfg


def hipp_seqctl200_gate_hsim32_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['pattern_size'] = 200
    cfg.data['imgs_per_gpu'] = 64
    return cfg


def hipp_seqctl_cntx_hsim32_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    cfg.model['context_remove_test'] = True
    return cfg


def hipp_rc_seqctl_gate_hsim32_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    cfg.model['hipp_head']['rnn_kwargs']['new_h_cat_input'] = True
    return cfg


def hipp_rnn_adam_stpmlp_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_layer_io'
    cfg.data['imgs_per_gpu'] = 32
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    return cfg


def hipp_stpmlp_gate_hsim32_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_layer_io'
    cfg.data['imgs_per_gpu'] = 32
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    return cfg


def hipp_gate_stpmlp_gate_hsim32_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_layer_io'
    cfg.model['hipp_head']['rnn_kwargs']['stpmlp_kwargs'] = {
            'gate_update': True,
            }
    cfg.data['imgs_per_gpu'] = 32
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    return cfg


def hipp_lstm_stpmlp_gate_hsim32_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_lstm_layer_io'
    cfg.data['imgs_per_gpu'] = 32
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    cfg.model['hipp_head']['rnn_kwargs']['gate_update'] = True
    return cfg


def hipp_rnn_adam_atLDDHH_lnglng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_at_mh_mmax_gnrl_pat_jp_cfg_func(cfg)
    cfg.model['add_target_range'] = '1.3,5.0'
    cfg.model['noise_norm'] = 0.07
    cfg.model['add_random_target_to_others'] = True
    cfg.model['randomize_noise_norm'] = True
    cfg.model['seq_len'] = 32
    cfg.data['imgs_per_gpu'] = 128
    return cfg


def hipp_rnn_adam_seqctl_atLDDHH_lnglng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atLDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    return cfg


def hipp_rnn_adam_stpmlp_atLDDHH_lnglng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atLDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_layer_io'
    cfg.data['imgs_per_gpu'] = 32
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    return cfg


def hipp_rnn_adam_2simstpmlp_atLDDHH_lnglng_mlpmax_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_atLDDHH_lnglng_mlpmax_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['use_stpmlp'] = 'one_sim_layer_io'
    cfg.data['imgs_per_gpu'] = 32
    cfg.data['train'] = {
            'type': 'SeqVectorDataset',
            'seq_len': 32,
            }
    return cfg


def hipp_rnn_adam_at_mh_pr_mmax_gnrl_pat_cfg_func(cfg):
    cfg = hipp_rnn_adam_mh_mmax_gnrl_pat_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['parallel_mmax'] = 3
    cfg.model['hipp_head']['pred_mlp']['in_channels'] = 512
    cfg.model['add_target_to_others'] = True
    return cfg


def hipp_rnn_adam_mh_pr_nh_mmax_gnrl_pat_cfg_func(cfg):
    cfg = hipp_rnn_adam_mh_mmax_gnrl_pat_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['parallel_mmax'] = 3
    cfg.model['hipp_head']['rnn_kwargs']['parallel_use_newh'] = True
    cfg.model['hipp_head']['pred_mlp']['in_channels'] = 512
    return cfg


def hipp_rnn_adam_at_mh_pr_nh_mmax_gnrl_pat_cfg_func(cfg):
    cfg = hipp_rnn_adam_mh_mmax_gnrl_pat_cfg_func(cfg)
    cfg.model['add_target_to_others'] = True
    cfg.model['hipp_head']['rnn_kwargs']['parallel_mmax'] = 3
    cfg.model['hipp_head']['rnn_kwargs']['parallel_use_newh'] = True
    cfg.model['hipp_head']['pred_mlp']['in_channels'] = 512
    return cfg


def hipp_rnn_adam_dw_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['rnn_type'] = 'dw'
    cfg.data['imgs_per_gpu'] = 64
    return cfg


def hipp_rnn_adam_dw_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_dw_cfg_func(cfg)
    cfg.model['loss_type'] = 'just_pos'
    return cfg


def hipp_rnn_adam_dw_l2_cfg_func(cfg):
    cfg = hipp_rnn_adam_dw_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['num_layers'] = 2
    cfg.data['imgs_per_gpu'] = 32
    return cfg


def hipp_rnn_adam_simple_rnn_jp_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg, 2048)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, 
            weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['rnn_type'] = 'simple_rnn'
    cfg.model['loss_type'] = 'just_pos'
    return cfg


def hipp_rnn_adam_dw_simple_rnn_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_simple_rnn_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'dw_simple_rnn'
    cfg.data['imgs_per_gpu'] = 32
    return cfg


def hipp_rnn_adam_dwt_simple_rnn_jp_cfg_func(cfg):
    cfg = hipp_rnn_adam_simple_rnn_jp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_type'] = 'dwt_simple_rnn'
    cfg.data['imgs_per_gpu'] = 32
    return cfg


def hipp_rnn_adam_wdr_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg, 2048)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    return cfg


def hipp_rnn_adam_wdr_jp_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg, 2048)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['loss_type'] = 'just_pos'
    return cfg


def hipp_rnn_adam_dwt_wdr_jp_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg, 1024)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['loss_type'] = 'just_pos'
    cfg.model['hipp_head']['rnn_type'] = 'dwt'
    cfg.data['imgs_per_gpu'] = 32
    return cfg


def hipp_rnn_adam_id_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['pred_mlp'] = dict(type='Identity')
    return cfg


def hipp_rnn_adam_id_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['pred_mlp'] = dict(type='Identity')
    return cfg


def hipp_rnn_adam_id_l2_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['pred_mlp'] = dict(type='Identity')
    cfg.model['hipp_head']['rnn_kwargs']['num_layers'] = 2
    return cfg


def hipp_rnn_adam_id_l4_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['pred_mlp'] = dict(type='Identity')
    cfg.model['hipp_head']['rnn_kwargs']['num_layers'] = 4
    return cfg


def hipp_rnn_adam_l4_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['rnn_kwargs']['num_layers'] = 4
    return cfg


def hipp_rnn_adam_id_n0_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['pred_mlp'] = dict(type='Identity')
    cfg.model['noise_norm'] = 0.
    return cfg


def hipp_rnn_adam_id_relu_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['pred_mlp'] = dict(type='Identity')
    cfg.model['hipp_head']['rnn_type'] = 'relu'
    return cfg


def hipp_rnn_adam_id_relu_l2_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['pred_mlp'] = dict(type='Identity')
    cfg.model['hipp_head']['rnn_type'] = 'relu'
    cfg.model['hipp_head']['rnn_kwargs']['num_layers'] = 2
    return cfg


def hipp_rnn_adam_id_t3_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['pred_mlp'] = dict(type='Identity')
    cfg.model['hipp_head']['rnn_tile'] = 3
    return cfg


def hipp_rnn_adam_hp_cfg_func(cfg):
    hidden_size = 1024
    cfg = hipp_rnn_test_cfg_func(cfg, hidden_size)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['rnn_type'] = 'hopfield'
    cfg.model['hipp_head']['rnn_kwargs'] = dict(
            input_size=128, 
            hidden_size=hidden_size,
            output_size=hidden_size,
            batch_first=False,
            num_heads=8,
            update_steps_max=3,
            scaling=0.25,
            )
    return cfg


def hipp_rnn_adam_hp_u0_cfg_func(cfg):
    cfg = hipp_rnn_adam_hp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['update_steps_max'] = 0
    return cfg


def hipp_rnn_adam_hp_st3_u0_cfg_func(cfg):
    cfg = hipp_rnn_adam_hp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['update_steps_max'] = 0
    cfg.model['hipp_head']['rnn_type'] = 'stacked_hopfield_3'
    first_kwargs = copy.deepcopy(cfg.model['hipp_head']['rnn_kwargs'])
    other_kwargs = copy.deepcopy(first_kwargs)
    other_kwargs['input_size'] = first_kwargs['output_size']
    cfg.model['hipp_head']['rnn_kwargs'] \
            = (first_kwargs, other_kwargs, other_kwargs)
    return cfg


def hipp_rnn_adam_hp_u0_n0_cfg_func(cfg):
    cfg = hipp_rnn_adam_hp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['update_steps_max'] = 0
    cfg.model['noise_norm'] = 0.
    return cfg


def hipp_rnn_adam_hp_pr4_u0_n0_cfg_func(cfg):
    cfg = hipp_rnn_adam_hp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['update_steps_max'] = 0
    cfg.model['noise_norm'] = 0.
    cfg.model['hipp_head']['rnn_type'] = 'parallel_hopfield_4'
    first_kwargs = copy.deepcopy(cfg.model['hipp_head']['rnn_kwargs'])
    cfg.model['hipp_head']['rnn_kwargs'] \
            = (first_kwargs, first_kwargs, first_kwargs, first_kwargs)
    cfg.model['hipp_head']['pred_mlp']['in_channels'] = 1024 * 4
    cfg.model['hipp_head']['pred_mlp']['hid_channels'] = 1024 * 4
    return cfg


def hipp_rnn_adam_hp_pr4_s3_u0_n0_cfg_func(cfg):
    cfg = hipp_rnn_adam_hp_cfg_func(cfg)
    cfg.model['hipp_head']['rnn_kwargs']['update_steps_max'] = 3
    cfg.model['noise_norm'] = 0.
    cfg.model['hipp_head']['rnn_type'] = 'parallel_hopfield_4'
    first_kwargs = copy.deepcopy(cfg.model['hipp_head']['rnn_kwargs'])
    cfg.model['hipp_head']['rnn_kwargs'] \
            = (first_kwargs, first_kwargs, first_kwargs, first_kwargs)
    cfg.model['hipp_head']['pred_mlp']['in_channels'] = 1024 * 4
    cfg.model['hipp_head']['pred_mlp']['hid_channels'] = 1024 * 4
    return cfg


def hipp_rnn_adam_mlp_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['rnn_type'] = 'none'
    cfg.model['hipp_head']['rnn_kwargs'] = {}
    cfg.model['hipp_head']['pred_mlp'] = dict(
            type='NonLinearNeckV1',
            in_channels=128, hid_channels=512,
            out_channels=512, with_avg_pool=False)
    return cfg


def hipp_adam_mlp_cfg_func(cfg):
    cfg = hipp_rnn_test_cfg_func(cfg)
    cfg.optimizer = dict(
            type='Adam', lr=1e-4, weight_decay=0.0001)
    cfg.lr_config = dict(policy='Fixed')
    cfg.model['hipp_head']['type'] = 'HippMLPHead'
    cfg.model['hipp_head']['pred_mlp'] = dict(
            type='NonLinearNeckV1',
            in_channels=128*9, hid_channels=2048,
            out_channels=512, with_avg_pool=False)
    cfg.model['hipp_head'].pop('rnn_kwargs')
    return cfg
