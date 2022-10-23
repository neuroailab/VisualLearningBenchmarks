from . import gnrl_funcs, simclr_cfg_funcs, ep300_funcs
import os
from ..datasets import svm_eval as svm_eval
import copy


g0_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        gnrl_funcs.use_IN_class_g0,
        )
sup_g0_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.use_IN_gx_meta('0'),
        gnrl_funcs.res112,
        gnrl_funcs.change_to_r18,
        )

g1_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        gnrl_funcs.use_IN_class_g1,
        )
sup_g1_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.use_IN_gx_meta('1'),
        gnrl_funcs.res112,
        gnrl_funcs.change_to_r18,
        )

g01_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        gnrl_funcs.use_IN_class_g01,
        )
sup_g01_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.use_IN_gx_meta('01', 200),
        gnrl_funcs.res112,
        gnrl_funcs.change_to_r18,
        )

g01_1of2_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        gnrl_funcs.use_IN_class_g01_1of2,
        )

g01_2of2_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        gnrl_funcs.use_IN_class_g01_2of2,
        )

g23_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        gnrl_funcs.use_IN_class_g23,
        )
sup_g23_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.use_IN_gx_meta('23', 200),
        gnrl_funcs.res112,
        gnrl_funcs.change_to_r18,
        )


def use_IN_vary_meta(
        switch_epoch, meta_suffix='',
        metas=None,
        ):
    def _func(cfg):
        cfg.data['train']['data_source'].pop('list_file')
        if metas is None:
            cfg.data['train']['data_source']['list_files'] = \
                    [os.path.join(
                         gnrl_funcs.meta_dir, 
                         'train_w_dogs{}.txt'.format(meta_suffix)), 
                     os.path.join(
                         gnrl_funcs.meta_dir, 
                         'train_w_birds{}.txt'.format(meta_suffix))]
        else:
            cfg.data['train']['data_source']['list_files'] = \
                    [os.path.join(gnrl_funcs.meta_dir, metas[0]), 
                     os.path.join(gnrl_funcs.meta_dir, metas[1])]
        cfg.data['train']['data_source']['switch_epochs'] = [switch_epoch]
        cfg.data['train']['data_source']['type'] = 'ImageNetVaryMetas'
        return cfg
    return _func
vary_ep200_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(100),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(200),
        )

vary_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(50),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )
vary_to_nodogs_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(
            50, metas=['train_w_dogs.txt', 'train_wo_dogs_or_birds.txt']),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )

vary_frn_brd_100_20_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(
            50, metas=['train_w_frn_wo_brd_100_20.txt', 'train_wo_frn_w_brd_100_20.txt']),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )

vary_dog_brd_100_20_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(
            50, metas=['train_w_dog_wo_brd_100_20.txt', 'train_wo_dog_w_brd_100_20.txt']),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )
vary_dog_brd_100_20_toRply_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(
            50, metas=['train_w_dog_wo_brd_100_20.txt', 'train_dog_brd_RdRply_100_20.txt']),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )
vary_dog_brd_100_20_OrclRply_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(
            50, metas=['train_w_dog_wo_brd_100_20.txt', 'train_dog_brd_OrclRply_100_20.txt']),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )
vary_db20_toNo_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(
            50, metas=['train_w_dog_wo_brd_100_20.txt', 'train_wo_dog_or_brd_100_20.txt']),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )

vary_dog_brd_100_20_io_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        simclr_cfg_funcs.use_inter_out_simclr(),
        use_IN_vary_meta(
            50, metas=['train_w_dog_wo_brd_100_20.txt', 'train_wo_dog_w_brd_100_20.txt']),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )

vary_30_30_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(50, '_30_30'),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )

vary_10_10_ep100_cfg_func = gnrl_funcs.sequential_func(
        gnrl_funcs.res112,
        simclr_cfg_funcs.mlp_4layers_cfg_func,
        use_IN_vary_meta(50, '_10_10'),
        ep300_funcs.ep300_cfg_func,
        gnrl_funcs.set_total_epochs(100),
        )


def add_typical_vary_datasource(cfg, metas):
    cfg.data['train']['data_source']['data_len'] = 1281167 // 2
    cfg.data['train']['data_source']['list_file'] = metas[1]
    cfg.data['train1'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2'] = copy.deepcopy(cfg.data['train'])
    cfg.data['train2']['data_source']['prev_list_file'] = metas[0]
    return cfg


def cotrain_IN_vary_meta(metas, cond_method, **kwargs):
    metas = \
            [os.path.join(gnrl_funcs.meta_dir, metas[0]), 
             os.path.join(gnrl_funcs.meta_dir, metas[1])]
    def _func(cfg):
        cfg = add_typical_vary_datasource(cfg, metas)
        cfg.data['train2']['data_source']['type'] = 'ImageNetVaryCnd'
        input_size = svm_eval.get_input_size_from_cfg(cfg)
        pipeline = svm_eval.get_typical_svm_dataset_cfg(
                input_size=input_size)['pipeline']
        cfg.data['train2']['data_source']['pipeline'] = pipeline
        cfg.data['train2']['data_source']['cond_method'] = cond_method
        cfg.data['train2']['data_source'].update(kwargs)
        return cfg
    return _func

def db20_ep100_cnd_w_method(cond_method, **kwargs):
    _cfg_func = gnrl_funcs.sequential_func(
            gnrl_funcs.res112,
            simclr_cfg_funcs.mlp_4layers_cfg_func,
            cotrain_IN_vary_meta(
                metas=['train_w_dog_wo_brd_100_20.txt', 'train_wo_dog_w_brd_100_20.txt'],
                cond_method=cond_method, **kwargs),
            ep300_funcs.ep300_cfg_func,
            gnrl_funcs.set_total_epochs(100),
            )
    return _cfg_func


def cotrain_IN_vary_loss_meta(metas, **kwargs):
    metas = \
            [os.path.join(gnrl_funcs.meta_dir, metas[0]), 
             os.path.join(gnrl_funcs.meta_dir, metas[1])]
    def _func(cfg):
        cfg = add_typical_vary_datasource(cfg, metas)
        cfg.data['train2']['data_source']['type'] = 'ImageNetVaryCndLoss'
        cfg.data['train2']['data_source']['pipeline'] = cfg.data['train2']['pipeline']
        cfg.data['train2']['data_source']['loss_head'] = dict(
                type='ContrastiveHead', temperature=0.1,
                loss_reduced=False)
        cfg.data['train2']['data_source'].update(kwargs)
        return cfg
    return _func

def db20_ep100_cnd_loss(**kwargs):
    _cfg_func = gnrl_funcs.sequential_func(
            gnrl_funcs.res112,
            simclr_cfg_funcs.mlp_4layers_cfg_func,
            cotrain_IN_vary_loss_meta(
                metas=['train_w_dog_wo_brd_100_20.txt', 'train_wo_dog_w_brd_100_20.txt'],
                **kwargs),
            ep300_funcs.ep300_cfg_func,
            gnrl_funcs.set_total_epochs(100),
            )
    return _cfg_func

def _sample_p_func(mean_sim):
    return (1 - mean_sim) ** 1.4

def cotrain_IN_vary_mean_sim_meta(metas, **kwargs):
    metas = \
            [os.path.join(gnrl_funcs.meta_dir, metas[0]), 
             os.path.join(gnrl_funcs.meta_dir, metas[1])]
    def _func(cfg):
        cfg = add_typical_vary_datasource(cfg, metas)
        cfg.data['train2']['data_source']['type'] = 'ImageNetVaryCndMeanSim'
        input_size = svm_eval.get_input_size_from_cfg(cfg)
        pipeline = svm_eval.get_typical_svm_dataset_cfg(
                input_size=input_size)['pipeline']
        cfg.data['train2']['data_source']['pipeline'] = pipeline
        cfg.data['train2']['data_source']['mean_sim_metric'] = 'max_20'
        cfg.data['train2']['data_source']['sample_p_func'] = _sample_p_func
        cfg.data['train2']['data_source'].update(kwargs)
        return cfg
    return _func

def db20_ep100_cnd_mean_sim(**kwargs):
    _cfg_func = gnrl_funcs.sequential_func(
            gnrl_funcs.res112,
            simclr_cfg_funcs.mlp_4layers_cfg_func,
            cotrain_IN_vary_mean_sim_meta(
                metas=['train_w_dog_wo_brd_100_20.txt', 'train_wo_dog_w_brd_100_20.txt'],
                **kwargs),
            ep300_funcs.ep300_cfg_func,
            gnrl_funcs.set_total_epochs(100),
            )
    return _cfg_func


def cotrain_IN_vary_smask_meta(metas, **kwargs):
    metas = \
            [os.path.join(gnrl_funcs.meta_dir, metas[0]), 
             os.path.join(gnrl_funcs.meta_dir, metas[1])]
    def _func(cfg):
        cfg = add_typical_vary_datasource(cfg, metas)
        cfg.data['train2']['data_source']['type'] = 'ImageNetVaryCndSparse'
        input_size = svm_eval.get_input_size_from_cfg(cfg)
        pipeline = svm_eval.get_typical_svm_dataset_cfg(
                input_size=input_size)['pipeline']
        cfg.data['train2']['data_source']['pipeline'] = pipeline
        cfg.data['train2']['data_source'].update(kwargs)
        return cfg
    return _func

def db20_ep100_cnd_smask(**kwargs):
    _cfg_func = gnrl_funcs.sequential_func(
            gnrl_funcs.res112,
            simclr_cfg_funcs.mlp_4layers_cfg_func,
            cotrain_IN_vary_smask_meta(
                metas=['train_w_dog_wo_brd_100_20.txt', 'train_wo_dog_w_brd_100_20.txt'],
                **kwargs),
            ep300_funcs.ep300_cfg_func,
            gnrl_funcs.set_total_epochs(100),
            )
    return _cfg_func
