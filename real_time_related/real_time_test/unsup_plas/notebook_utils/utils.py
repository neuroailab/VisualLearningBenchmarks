import copy
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pylab
from matplotlib.backends.backend_pdf import PdfPages

from scipy import misc
from scipy.stats import norm
import os
import time
import importlib
import pickle
import pdb
USER_NAME = os.getlogin()
DATA_ROOT = f'/data1/{USER_NAME}/pub_clean_related/real_time_related'


TEST_FACE_DICT = {
        '22.pkl': [b'face0005', b'face0006'],
        '9.pkl': [b'face0002', b'face0005'],
        '15.pkl': [b'face0003', b'face0006'],
        '19.pkl': [b'face0004', b'face0006'],
        '8.pkl': [b'face0002', b'face0004'],
        '14.pkl': [b'face0003', b'face0005'],
        '2.pkl': [b'face0001', b'face0004'],
        '10.pkl': [b'face0002', b'face0006'],
        '3.pkl': [b'face0001', b'face0005'],
        '0.pkl': [b'face0001', b'face0002'],
        }
EXP_IDX_TO_FACES = [
        [b'face0001', b'face0002'],
        [b'face0001', b'face0003'],
        [b'face0001', b'face0004'],
        [b'face0001', b'face0005'],
        [b'face0001', b'face0006'],
        [b'face0002', b'face0003'],
        [b'face0002', b'face0004'],
        [b'face0002', b'face0005'],
        [b'face0002', b'face0006'],
        [b'face0003', b'face0004'],
        [b'face0003', b'face0005'],
        [b'face0003', b'face0006'],
        [b'face0004', b'face0005'],
        [b'face0004', b'face0006'],
        [b'face0005', b'face0006'],
        ]


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

def add_lapse_to_CF(CF, lapse_rate):
    new_CF = np.zeros(CF.shape)
    all_times = np.sum(CF)
    lapse_times = int(all_times * lapse_rate)
    lapse_idxs = np.random.permutation(int(all_times))
    lapse_idxs = lapse_idxs[:lapse_times]

    sta_idx = 0
    all_pos = [[0, 0], [1, 0], [0, 1], [1, 1]]
    for pos_x, pos_y in all_pos:
        end_idx = sta_idx + CF[pos_x, pos_y]
        curr_lapse = np.sum(np.logical_and(
            lapse_idxs >= sta_idx, lapse_idxs < end_idx))
        new_CF[pos_x, pos_y] += CF[pos_x, pos_y] - curr_lapse
        for _ in range(curr_lapse):
            choice = 0 if np.random.uniform() < 0.5 else 1
            new_CF[choice, pos_y] += 1
        sta_idx = end_idx
    return new_CF

def get_measures(
        all_dprimes, ignore_faces, lapse_rate=None, filter_by_init_dp=None):
    big_dprimes = []
    small_dprimes = []
    ctl_big_dprimes = []
    ctl_small_dprimes = []
    for dprimes in all_dprimes:
        which_exp = dprimes['which_exp']
        which_faces = dprimes['faces']
        if (which_faces[0] in ignore_faces) or (which_faces[1] in ignore_faces):
            continue
        all_big_dprime = np.asarray(dprimes['bigs'])
        if lapse_rate is not None:
            all_big_dprime = np.zeros_like(all_big_dprime)
            all_CFs = np.asarray(dprimes['all_CFs'])
            for x in range(all_big_dprime.shape[0]):
                for y in range(all_big_dprime.shape[1]):
                    all_big_dprime[x, y] = d_prime2x2(add_lapse_to_CF(
                            all_CFs[x, y], lapse_rate))
        all_small_dprime = np.asarray(dprimes['smalls'])
        big_dprime = all_big_dprime[:, which_exp]
        if filter_by_init_dp is not None:
            if filter_by_init_dp(big_dprime[0]):
                continue
        big_dprime = big_dprime - big_dprime[0]
        big_dprimes.append(big_dprime)

        if 'ctl_pair' in dprimes:
            ctl_faces = TEST_FACE_DICT[dprimes['ctl_pair']]
        else:
            ctl_faces = None
        num_ctls = 0
        ctl_big_dprime = []
        ctl_small_dprime = []
        for exp_idx in range(all_big_dprime.shape[1]):
            _which_faces = EXP_IDX_TO_FACES[exp_idx]
            if _which_faces[0] in which_faces:
                continue
            if _which_faces[1] in which_faces:
                continue
            if (_which_faces[0] in ignore_faces) or (_which_faces[1] in ignore_faces):
                continue
            if ctl_faces is not None:
                if _which_faces[0] not in ctl_faces:
                    continue
                if _which_faces[1] not in ctl_faces:
                    continue
            _ctl_big_dprime = all_big_dprime[:, exp_idx]
            _ctl_big_dprime = _ctl_big_dprime - _ctl_big_dprime[0]
            ctl_big_dprime.append(_ctl_big_dprime)
            _ctl_small_dprime = all_small_dprime[:, exp_idx]
            _ctl_small_dprime = _ctl_small_dprime - _ctl_small_dprime[0]
            ctl_small_dprime.append(_ctl_small_dprime)

        ctl_big_dprimes.extend(ctl_big_dprime)
        ctl_small_dprimes.extend(ctl_small_dprime)

        small_dprime = np.asarray(dprimes['smalls'])
        small_dprime = small_dprime[:, which_exp]
        small_dprime = small_dprime - small_dprime[0]
        small_dprimes.append(small_dprime)

    big_dprimes = np.asarray(big_dprimes)
    small_dprimes = np.asarray(small_dprimes)
    ctl_big_dprimes = np.asarray(ctl_big_dprimes)
    ctl_small_dprimes = np.asarray(ctl_small_dprimes)
    return big_dprimes, small_dprimes, ctl_big_dprimes, ctl_small_dprimes


def get_init_dp_and_last_delta_dp(
        result_path=None, lapse_rate=None, all_dprimes=None):
    if all_dprimes is None:
        assert result_path is not None
        all_dprimes = pickle.load(open(result_path, 'rb'))
    big_dprimes, _, ctl_big_dprimes, _ = get_measures(
            all_dprimes, ignore_faces=[], lapse_rate=lapse_rate)

    ctl_big_dprimes = np.reshape(
            ctl_big_dprimes, 
            [len(all_dprimes), -1, ctl_big_dprimes.shape[-1]])
    ctl_big_dprimes = np.mean(ctl_big_dprimes, axis=1)
    effects = big_dprimes - ctl_big_dprimes

    if ('args' in all_dprimes[0])\
            and getattr(all_dprimes[0]['args'], 'include_test', None):
        include_test = True
    else:
        include_test = False

    initial_dps = []
    for dprimes in all_dprimes:
        which_exp = dprimes['which_exp']
        if lapse_rate is None:
            all_big_dprime = np.asarray(dprimes['bigs'])
            big_dprime = all_big_dprime[:, which_exp]
            if not include_test:
                initial_dps.append(big_dprime[0])
            else:
                if effects.shape[1] == 19:
                    mean_inter = 2
                elif effects.shape[1] == 37:
                    mean_inter = 4
                else:
                    raise NotImplementedError
                _init_dp = np.mean(big_dprime[1 : (1+mean_inter)])
                initial_dps.append(_init_dp)
        else:
            if include_test:
                raise NotImplementedError
            all_CFs = np.asarray(dprimes['all_CFs'])
            CFs = all_CFs[:, which_exp]
            init_dp = d_prime2x2(add_lapse_to_CF(CFs[0], lapse_rate))
            initial_dps.append(init_dp)
    return effects[:, -1], initial_dps


def align_model_effects_format_to_human(effects):
    if effects.shape[1] == 19:
        freq = 4
        mean_inter = 2
    elif effects.shape[1] == 37:
        freq = 8
        mean_inter = 4
    else:
        raise NotImplementedError
    new_effects = []
    sta_idx = 1
    for test_block in range(5):
        new_effects.append(effects[:, np.newaxis, sta_idx:sta_idx+mean_inter])
        sta_idx += freq
    effects = np.concatenate(new_effects, axis=1).mean(axis=2)
    effects -= effects[:, :1]
    return effects

def get_value_from_path(path, key, default_value, data_type=int):
    path = path.strip('.pkl')
    if key not in path:
        return default_value
    pos = path.find(key) + len(key)
    value = data_type(path[pos:].split('_')[0])
    return value


def save_pdf_to_default_folder(pdf_dir, filename):
    if filename is None:
        return
    pp = PdfPages(os.path.join(pdf_dir, filename))
    plt.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()


def general_format_setting(axes):
    # Hide the right and top spines
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    linewidth=3
    axes.spines['left'].set_linewidth(linewidth)
    axes.spines['bottom'].set_linewidth(linewidth)

    tick_length = 6
    axes.tick_params(direction='in', axis='y', length=tick_length, width=linewidth)
    axes.tick_params(direction='in', axis='x', length=tick_length, width=linewidth)


def plot_three_cond_five_point_effects(
        result_path, figsize=(3.5, 12),
        *args, **kwargs):
    plt.figure(figsize=figsize)
    plt.subplot(311)
    try:
        plot_five_point_effects(
                result_path.format('build'), None,
                'noswap', None, False, *args, **kwargs)
    except:
        pass
    plt.subplot(312)
    try:
        plot_five_point_effects(
                result_path.format('break'), None,
                'swap', None, False, *args, **kwargs)
    except:
        pass
    plt.subplot(313)
    try:
        plot_five_point_effects(
                result_path.format('switch'), None,
                'noswapswap', None, False, *args, **kwargs)
    except:
        pass


def load_human_effects(plot_format, data_source='xiaoxuan'):
    if data_source == 'xiaoxuan':
        xx_data_root = os.path.join(DATA_ROOT, 'data/human_behavioral_data')
        if plot_format == 'swap':
            big = np.load(os.path.join(
                xx_data_root, 'break/break_mb_b.npy'))
            ctl_big = np.load(os.path.join(
                xx_data_root, 'break/break_mb_b_control.npy'))
        elif plot_format == 'noswap':
            big = np.load(os.path.join(xx_data_root, 'build/build_mb_b.npy'))
            ctl_big = np.load(os.path.join(
                xx_data_root, 'build/build_mb_b_other.npy'))
        else:
            big = np.load(os.path.join(xx_data_root, 'switch/switch_mb_b.npy'))
            ctl_big = np.load(os.path.join(
                xx_data_root, 'switch/switch_mb_b_other.npy'))
        human_effects = big - ctl_big
    else:
        raise NotImplementedError
    return human_effects


def plot_five_point_effects(
        result_path, figsize=(3.5, 3.5), 
        plot_format='swap', model_id=None,
        new_figure=True, plot_human=True, set_default_ylim=True,
        human_data_source='xiaoxuan', verbose=True, check_finish=False,
        *args, **kwargs):
    all_dprimes = pickle.load(open(result_path, 'rb'))
    big_dprimes, _, ctl_big_dprimes, _ \
            = get_measures(all_dprimes, [], *args, **kwargs)
    if check_finish:
        assert len(all_dprimes) == 15
    if len(all_dprimes) < 15:
        print('Number of exps: {}'.format(len(all_dprimes)))
    if 'command' in all_dprimes[0] and verbose:
        print(all_dprimes[0]['command'])

    if 'eval_freq50' in result_path:
        freq = 3
    else:
        freq = 5
    if 'args' in all_dprimes[0]:
        args = all_dprimes[0]['args']
        eval_freq = args.eval_freq
        num_steps = args.num_steps
        freq = (num_steps // 4) // eval_freq
    temp_drop = ctl_big_dprimes.shape[0] % len(big_dprimes)
    if temp_drop != 0:
        print('Control Big Dprimes are only estimates!')
        ctl_big_dprimes = ctl_big_dprimes[:-temp_drop]
    ctl_big_dprimes = np.reshape(
            ctl_big_dprimes, 
            [len(big_dprimes), -1, ctl_big_dprimes.shape[-1]])
        
    ctl_big_dprimes = np.mean(ctl_big_dprimes, axis=1)
    effects = big_dprimes - ctl_big_dprimes
    if ('args' in all_dprimes[0])\
            and getattr(all_dprimes[0]['args'], 'include_test', None):
        effects = align_model_effects_format_to_human(effects)
    else:
        effects = effects[:, ::freq]
    x = np.asarray([0, 1, 2, 3, 4])
    #offset = np.asarray([0, 0.1, 0.1, 0.1, 0.1])
    offset = 0.1
    
    swap_big_kwargs = {
            'c': '#ED1C24',
            'linewidth': 3}
    noswap_big_kwargs = {
            'c': '#0000FF',
            'linewidth': 3}

    if plot_format == 'swap':
        big_kwargs = swap_big_kwargs
    elif plot_format == 'noswap':
        big_kwargs = noswap_big_kwargs

    if new_figure:
        fig = plt.figure(figsize=figsize)
        axes = plt.subplot(111)
    else:
        axes = plt.gca()
    if plot_format == 'noswapswap':
        axes.errorbar(
                x[:3]-offset, np.mean(effects[:, :3], axis=0),
                yerr=np.std(effects[:, :3], axis=0)/np.sqrt(effects[:, :3].shape[0]),
                **noswap_big_kwargs)
        axes.errorbar(
                x[2:]-offset, np.mean(effects[:, 2:], axis=0),
                yerr=np.std(effects[:, 2:], axis=0)/np.sqrt(effects[:, 2:].shape[0]),
                **swap_big_kwargs)
    else:
        axes.errorbar(
                x-offset, np.mean(effects, axis=0),
                yerr=np.std(effects, axis=0)/np.sqrt(effects.shape[0]),
                **big_kwargs)

    general_format_setting(axes)
    axes.plot(
            [-0.3, 4.3], [0, 0], 
            linewidth=2, linestyle='dashed',
            c='#7F7F7F')
    axes.set_xticklabels(('', '', '', ''))
    #axes.set_yticklabels((''))

    human_effects = load_human_effects(plot_format, human_data_source)
    if (plot_format == 'noswapswap') and plot_human:
        eb1 = axes.errorbar(
                x[:3]+offset, np.mean(human_effects[:3], axis=1),
                yerr=np.std(human_effects[:3], axis=1),
                linestyle='dashed',
                **noswap_big_kwargs)
        eb1[-1][0].set_linestyle('--')
        eb1 = axes.errorbar(
                x[2:]+offset, np.mean(human_effects[2:], axis=1),
                yerr=np.std(human_effects[2:], axis=1),
                linestyle='dashed',
                **swap_big_kwargs)
        eb1[-1][0].set_linestyle('--')
    elif plot_human:
        eb1 = axes.errorbar(
                x+offset, np.mean(human_effects, axis=1),
                yerr=np.std(human_effects, axis=1),
                linestyle='dashed',
                **big_kwargs)
        eb1[-1][0].set_linestyle('--')
    axes.set_xlim([-0.3, 4.3])
    if set_default_ylim:
        if plot_format == 'swap':
            axes.set_ylim([-3.0, 0.7])
        elif plot_format == 'noswap':
            axes.set_ylim([-0.1, 2.5])
        else:
            axes.set_ylim([-1.5, 1.5])


def plot_multi_effects_for_paths(
        paths, figsize=(8, 9),
        build_ylim=(-0.6, 1.2), build_yticks=[0, 1],
        break_ylim=(-1.8, 2.6), break_yticks=[-1, 0, 1, 2],
        switch_ylim=(-0.7, 2.2), switch_yticks=[0, 1, 2],
        ):
    plt.figure(figsize=figsize)
    num_algs = len(paths)
    wspace = 0.1
    for idx, _paths in enumerate(paths):
        plt.subplot(3, num_algs, idx+1)
        plot_five_point_effects(
                _paths.format('build'), None, 'noswap', 
                None, False, verbose=False)
        plt.ylim(*build_ylim)
        plt.yticks(build_yticks, [''] * len(build_yticks))
        plt.xticks([0, 2, 4], ['', '', ''])
        plt.subplots_adjust(hspace = .1, wspace=wspace)
        
        plt.subplot(3, num_algs, idx+1+num_algs)
        plot_five_point_effects(
                _paths.format('break'), None, 'swap', 
                None, False, verbose=False)
        plt.ylim(*break_ylim)
        plt.yticks(break_yticks, [''] * len(break_yticks))
        plt.xticks([0, 2, 4], ['', '', ''])
        plt.subplots_adjust(hspace = .1, wspace=wspace)
        
        plt.subplot(3, num_algs, idx+1+num_algs*2)
        plot_five_point_effects(
                _paths.format('switch'), None, 'noswapswap', 
                None, False, verbose=False)
        plt.ylim(*switch_ylim)
        plt.yticks(switch_yticks, [''] * len(switch_yticks))
        plt.xticks([0, 2, 4], ['', '', ''])
        plt.subplots_adjust(hspace = .1, wspace=wspace)


def plot_initial_dp_vs_delta_dp(
        result_path_pat, figsize=(7, 7), lapse_rate=None,
        **kwargs):
    build_last_dp, init_dp = get_init_dp_and_last_delta_dp(
            result_path_pat.format('build'), lapse_rate)
    break_last_dp, init_dp = get_init_dp_and_last_delta_dp(
            result_path_pat.format('break'), lapse_rate)
    
    fig = plt.figure(figsize=figsize)
    axes = plt.subplot(111)
    sc_kwargs = dict(s=100, alpha=0.7)
    sc_kwargs.update(kwargs)
    axes.scatter(init_dp, break_last_dp, c='tab:blue', **sc_kwargs)
    axes.scatter(init_dp, build_last_dp, c='tab:red', **sc_kwargs)
    axes.plot([0, 4], [0, 0], linestyle='dashed', color='k')

    axes.tick_params(
        direction='in',
        axis='y', labelsize=20, length=6, width=3)
    axes.tick_params(
        direction='in',
        axis='x', labelsize=20, length=6, width=3)
    
    # Hide the right and top spines
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['left'].set_linewidth(3)
    axes.spines['bottom'].set_linewidth(3)
    axes.set_xlabel('Initial D-prime')
    axes.set_ylabel('Learning Effects')
