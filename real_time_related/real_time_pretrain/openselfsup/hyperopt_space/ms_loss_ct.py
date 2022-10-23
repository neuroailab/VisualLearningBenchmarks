from hyperopt import hp

def get_default_func_and_params(name_prefix, ms_or_loss='ms'):
    exp_kwargs = {
            'type': 'power',
            'p': hp.uniform(f'{name_prefix}_power_p', 0.1, 30),
            }
    if ms_or_loss == 'ms':
        exp_kwargs['center_ratio'] = hp.uniform(
                f'{name_prefix}_center_ratio', 0, 0.99)
    elif ms_or_loss == 'loss':
        exp_kwargs['max_turn'] = hp.uniform(
                f'{name_prefix}_max_turn',
                3.0, 10.0)
    else:
        raise NotImplementedError
    cont_kwargs = {
            'type': 'cont',
            }
    x_values = [
            0, 1, 2, 4, 6, 8,
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for x_value in x_values:
        cont_kwargs[f'cont_{x_value}'] = hp.uniform(
                f'{name_prefix}_cont_{x_value}', 0, 1)
    func_and_params = [exp_kwargs, cont_kwargs]
    return func_and_params

def get_default_ms_related_params(prefix):
    params = {
            'ms_amp_func_and_params': hp.choice(
                f'{prefix}_ms_amp_func_and_params', 
                get_default_func_and_params(f'{prefix}_ms_amp')),
            'ms_len': hp.quniform(f'{prefix}_ms_len', 50, 3000, 10),
            'ms_sta': hp.choice(
                f'{prefix}_ms_sta', [0, hp.quniform(f'{prefix}_ms_non_zero_sta', 
                                                   5000, 80000, 2000)]),
            }
    return params

def get_default_loss_related_params(prefix):
    params = {
            'loss_amp_func_and_params': hp.choice(
                f'{prefix}_loss_amp_func_and_params', 
                get_default_func_and_params(
                    f'{prefix}_loss_amp', ms_or_loss='loss')),
            'loss_neg_img_nums': hp.choice(
                f'{prefix}_loss_neg_img_nums',
                [1024]),
            'loss_neg_sample_method': hp.choice(
                f'{prefix}_loss_neg_sample_method',
                [{'type': 'from_curr'}]),
            }
    return params

def get_default_mix_weight(label):
    return hp.uniform(label, 0.05, 0.99)

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
            'loss_mix_w': get_default_mix_weight('ms1_lv_mix_w'),
            'gmm_loss_mix_w': 0,
            'ms_mix_w': 1.0,
            }
    ms1_lv_params.update(get_default_ms_related_params('ms1_lv_ms'))
    ms1_lv_params.update(get_default_loss_related_params('ms1_lv_loss'))

    l1_msv_params = {
            'ms_mix_w': get_default_mix_weight('l1_msv_mix_w'),
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
