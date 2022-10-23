from hyperopt import hp


def get_default_exp_t():
    return [0.3, 1.0, 3.0, 5.0, 7.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 30.0]

def get_default_power():
    return [0.1, 0.3, 0.5, 0.75, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0]

def get_default_func_and_params(name_prefix):
    func_and_params = [
            {
                'type': 'exp',
                't': hp.choice(f'{name_prefix}_exp_t', get_default_exp_t()),
            },
            {
                'type': 'power',
                'p': hp.choice(f'{name_prefix}_power_p', get_default_power()),
            },
            ]
    return func_and_params

def get_default_ms_len_list():
    return [50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]

def get_default_ms_center_ratio():
    return [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

def get_default_ms_non_zero_sta():
    return [5000, 7000, 10000, 15000, 20000, 30000, 40000, 60000, 80000]

def get_default_ms_related_params(prefix):
    params = {
            'ms_amp_func_and_params': hp.choice(
                f'{prefix}_ms_amp_func_and_params', 
                get_default_func_and_params(f'{prefix}_ms_amp')),
            'ms_len': hp.choice(f'{prefix}_ms_len', get_default_ms_len_list()),
            'ms_center_ratio': hp.choice(
                f'{prefix}_ms_center_ratio', get_default_ms_center_ratio()),
            'ms_sta': hp.choice(
                f'{prefix}_ms_sta', [0, hp.choice(f'{prefix}_ms_non_zero_sta', 
                                                  get_default_ms_non_zero_sta())]),
            }
    return params

def get_default_loss_max_turn():
    return [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 100.0]

def get_default_neg_img_nums():
    return [512, 1024, 1536, 2048]

def get_default_neg_img_mix_ratio():
    return [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

def get_default_loss_related_params(prefix):
    params = {
            'loss_amp_func_and_params': hp.choice(
                f'{prefix}_loss_amp_func_and_params', 
                get_default_func_and_params(f'{prefix}_loss_amp')),
            'loss_max_turn': hp.choice(
                f'{prefix}_loss_max_turn',
                get_default_loss_max_turn()),
            'loss_neg_img_nums': hp.choice(
                f'{prefix}_loss_neg_img_nums',
                get_default_neg_img_nums()),
            'loss_neg_sample_method': hp.choice(
                f'{prefix}_loss_neg_sample_method',
                [
                    {'type': 'from_curr'}, 
                    {
                        'type': 'mix_curr_st',
                        'mix_ratio': hp.choice(f'{prefix}_neg_mix_ratio', get_default_neg_img_mix_ratio()),
                    },
                ]),
            }
    return params

def get_default_mix_weight():
    return [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def set_param_space_dict():
    ms_only_params = {
            'loss_mix_w': 0,
            'gmm_loss_mix_w': 0,
            'ms_mix_w': 1.0,
            }
    loss_only_params = {
            'loss_mix_w': 1.0,
            'gmm_loss_mix_w': 0,
            'ms_mix_w': 0,
            }
    ms_only_params.update(get_default_ms_related_params('ms_only'))
    loss_only_params.update(get_default_loss_related_params('loss_only'))

    ms1_lv_params = {
            'loss_mix_w': hp.choice('ms1_lv_mix_w', get_default_mix_weight()),
            'gmm_loss_mix_w': 0,
            'ms_mix_w': 1.0,
            }
    ms1_lv_params.update(get_default_ms_related_params('ms1_lv_ms'))
    ms1_lv_params.update(get_default_loss_related_params('ms1_lv_loss'))

    l1_msv_params = {
            'ms_mix_w': hp.choice('l1_msv_mix_w', get_default_mix_weight()),
            'gmm_loss_mix_w': 0,
            'loss_mix_w': 1.0,
            }
    l1_msv_params.update(get_default_ms_related_params('l1_msv_ms'))
    l1_msv_params.update(get_default_loss_related_params('l1_msv_loss'))
    space = hp.choice('how_to_mix', [
        ms_only_params,
        loss_only_params,
        hp.choice('which_one_1', 
                  [ms1_lv_params, l1_msv_params]),
        ])
    return space


def set_param_space_dict_nl():
    ms_only_params = {
            'loss_mix_w': 0,
            'gmm_loss_mix_w': 0,
            'ms_mix_w': 1.0,
            }
    ms_only_params.update(get_default_ms_related_params('ms_only'))

    ms1_lv_params = {
            'loss_mix_w': hp.choice('ms1_lv_mix_w', get_default_mix_weight()),
            'gmm_loss_mix_w': 0,
            'ms_mix_w': 1.0,
            }
    ms1_lv_params.update(get_default_ms_related_params('ms1_lv_ms'))
    ms1_lv_params.update(get_default_loss_related_params('ms1_lv_loss'))

    l1_msv_params = {
            'ms_mix_w': hp.choice('l1_msv_mix_w', get_default_mix_weight()),
            'gmm_loss_mix_w': 0,
            'loss_mix_w': 1.0,
            }
    l1_msv_params.update(get_default_ms_related_params('l1_msv_ms'))
    l1_msv_params.update(get_default_loss_related_params('l1_msv_loss'))
    space = hp.choice('how_to_mix', [
        ms_only_params,
        hp.choice('which_one_1', 
                  [ms1_lv_params, l1_msv_params]),
        ])
    return space
