from hyperopt import hp

def get_default_ms_len_list():
    return [50, 100, 200, 400, 800, 1500, 3000]

def get_default_ms_center_ratio():
    return [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

def get_default_mix_weight():
    return [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def get_default_exp_t():
    return [0.3, 1.0, 3.0, 10.0, 20.0, 30.0]

def get_default_power():
    return [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

def get_default_loss_max_turn():
    return [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 100.0]

def get_default_gmm_loss_max_turn():
    return [-30.0, -10.0, -3.0, -1.0, 0, 1.0, 3.0, 10.0, 30.0]

def set_param_space_dict():
    space = dict(
            ms_len_0=hp.choice('ms_len_0', get_default_ms_len_list()),
            ms_len_1=hp.choice('ms_len_1', get_default_ms_len_list()),
            ms_second_sta=hp.choice('ms_second_sta', [5000, 10000, 20000, 40000, 80000]),
            )
    for _idx in range(4):
        name = f'ms_center_ratio_{_idx}'
        space[name] = hp.choice(name, get_default_ms_center_ratio())
        name = f'ms_mix_weight_{_idx}'
        space[name] = hp.choice(name, get_default_mix_weight())
    for _idx in range(2):
        name = f'ms_exp_t_{_idx}'
        space[name] = hp.choice(name, get_default_exp_t())
        name = f'ms_power_{_idx}'
        space[name] = hp.choice(name, get_default_power())

    for l_prefix in ['loss', 'gmm_loss']:
        for _idx in range(2):
            name = f'{l_prefix}_mix_weight_{_idx}'
            space[name] = hp.choice(name, get_default_mix_weight())
            name = f'{l_prefix}_max_turn_{_idx}'
            if l_prefix == 'loss':
                space[name] = hp.choice(name, get_default_loss_max_turn())
            else:
                space[name] = hp.choice(name, get_default_gmm_loss_max_turn())
        name = f'{l_prefix}_exp_t'
        space[name] = hp.choice(name, get_default_exp_t())
        name = f'{l_prefix}_power'
        space[name] = hp.choice(name, get_default_power())

    name = 'gmml_sub_w'
    space[name] = hp.choice(name, [0.1, 0.3, 1.0, 2.0, 4.0, 8.0])
    return space
