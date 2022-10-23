from .basic_param_setter import ParamsBuilder
from openselfsup.framework.hooks.hook import Hook
from openselfsup.framework.hooks.validate_hook import ValidateHook
from openselfsup.apis.train import batch_processor
from openselfsup.datasets import build_dataset, build_ffcvloader
import openselfsup.datasets.concat_datasets as concat_datasets
import torch
import pdb
import numpy as np
import os
USER_NAME = os.getlogin()
PERSISTENT_WORKERS = int(os.environ.get('PERSISTENT_WORKERS', 0))==1


def add_face_data(
        data_source,
        add_num=-1,
        meta_path=f'/mnt/fs4/{USER_NAME}/openselfsup_models/metas/face_train.txt', 
        data_dir=f'/data5/{USER_NAME}/Dataset'):
    with open(meta_path, 'r') as fin:
        all_paths = fin.readlines()
    all_paths = [l.strip() for l in all_paths]
    all_paths = [
            os.path.join(data_dir, l) 
            for l in all_paths]
    if add_num > 0:
        all_paths = all_paths[:add_num]
    curr_len = len(data_source.fns)
    new_fns = np.concatenate([data_source.fns, all_paths])
    new_fns = np.random.choice(new_fns, curr_len, replace=False)
    data_source.fns = new_fns
    return data_source


class SetEpochHook(Hook):
    def __init__(
            self, need_face_data=False,
            add_face_num=-1):
        self.need_face_data = need_face_data
        self.add_face_num = add_face_num
    
    def before_epoch(self, runner):
        data_source = runner.data_loader.dataset.data_source
        assert hasattr(data_source, 'set_epoch')
        data_source.set_epoch(runner.epoch)
        if self.need_face_data:
            runner.data_loader.dataset.data_source \
                    = add_face_data(
                            data_source, self.add_face_num)


class FFCVLoaderSetEpochHook(Hook):
    def before_epoch(self, runner):
        assert hasattr(runner.data_loader, 'set_epoch')
        runner.data_loader.set_epoch(runner.epoch)
            

class EWCHook(Hook):
    def after_epoch(self, runner):
        assert hasattr(runner.model.module, 'update_buffer')
        runner.model.module.update_buffer(runner.data_loader)


class InnerIterValidHook(ValidateHook):
    def before_run(self, runner):
        pass

    def after_train_epoch(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.data_loader.dataset.data_source.set_epoch(runner.epoch)

    def before_train_iter(self, runner):
        if runner.inner_iter % self.interval == 0:
            self._run_validate(runner)

    def after_train_iter(self, runner):
        pass


class SAYCamParamBuilder(ParamsBuilder):
    def __init__(
            self, need_ewc_hook=False, 
            need_IN_cont_eval=False,
            need_face_data=False,
            *args, **kwargs):
        self.need_ewc_hook = need_ewc_hook
        self.need_IN_cont_eval = need_IN_cont_eval
        self.need_face_data = need_face_data
        super().__init__(*args, **kwargs)

    def add_one_hook_params(self, one_hook_params):
        if 'extra_hook_params' not in self.params:
            self.params['extra_hook_params'] = []
        self.params['extra_hook_params'].append(one_hook_params)

    def add_set_epoch_hook(self):
        set_epoch_hook_params = {
                'builder': SetEpochHook,
                'builder_kwargs': {
                    'need_face_data': self.need_face_data,
                    },
                }
        if self.use_ffcvloader:
            assert not self.need_face_data
            set_epoch_hook_params = {'builder': FFCVLoaderSetEpochHook}
        self.add_one_hook_params(set_epoch_hook_params)

    def add_ewc_hook(self):
        set_epoch_hook_params = {'builder': EWCHook}
        self.add_one_hook_params(set_epoch_hook_params)

    def build_IN_cont_eval_data_loader(self):
        IN_cont_val_cfg = self.get_IN_eval_data_cfg()
        IN_cont_val_cfg['data_source']['type'] = 'ImageNetContVal'
        IN_cont_val_cfg['data_source']['list_file'] = 'data/imagenet/meta/train.txt'
        IN_cont_val_cfg['data_source']['val_keep_prev_epoch_num'] = 4
        IN_cont_val_cfg['data_source']['num_epochs'] = 300
        IN_cont_val_cfg['data_source']['keep_prev_epoch_num'] = 2
        self.IN_cont_val_dataset = build_dataset(IN_cont_val_cfg)
        return self.build_val_loader_from_dataset(val_dataset=self.IN_cont_val_dataset)

    def get_IN_cont_svm_acc(self, results):
        return self.IN_cont_val_dataset.data_source.get_svm_perf(results)

    def add_IN_cont_eval(self):
        IN_cont_eval_hook_params = {
                'builder': InnerIterValidHook,
                'builder_kwargs': {
                    'name': 'in_cont_svm',
                    'data_loader_builder': self.build_IN_cont_eval_data_loader,
                    'batch_processor': self.run_model_extract,
                    'agg_func': self.get_IN_cont_svm_acc,
                    'interval': 2501,
                    },
                }
        self.add_one_hook_params(IN_cont_eval_hook_params)

    def build_params(self):
        super().build_params()
        self.add_set_epoch_hook()
        if self.need_ewc_hook:
            self.add_ewc_hook()
        if self.need_IN_cont_eval:
            self.add_IN_cont_eval()
        self.params['train_data_params']['shuffle'] = False
        return self.params


class ConcatSetEpochHook(Hook):
    def __init__(self, epoch_offset=0, *args, **kwargs):
        self.epoch_offset = epoch_offset
        super().__init__(*args, **kwargs)

    def before_epoch(self, runner):
        datasets = runner.data_loader.dataset.datasets
        for dataset in datasets:
            data_source = dataset.data_source
            assert hasattr(data_source, 'set_epoch')
            data_source.set_epoch(
                    runner.epoch + self.epoch_offset)

class FFCVConcatLoaderSetEpochHook(Hook):
    def before_epoch(self, runner):
        loaders = runner.data_loader.loaders
        for loader in loaders:
            loader.set_epoch(runner.epoch)


class CndSetEpochHook(Hook):
    def __init__(self, *args, **kwargs):
        self.cnd_loader = None
        self.cont_loader = None
        super().__init__(*args, **kwargs)

    def before_epoch(self, runner):
        datasets = runner.data_loader.dataset.datasets
        assert len(datasets) == 2
        cont_data_source = datasets[0].data_source
        cont_data_source.set_epoch(runner.epoch)
        cnd_acc_data_source = datasets[1].data_source
        if PERSISTENT_WORKERS and (self.cnd_loader is not None):
            cnd_acc_data_source.cnd_loader = self.cnd_loader
            cnd_acc_data_source.cont_loader = self.cont_loader
        cnd_acc_data_source.cnd_set_epoch(
                runner.epoch, runner.model, 
                datasets[0].data_source)
        if PERSISTENT_WORKERS:
            if self.cnd_loader is None:
                self.cnd_loader = cnd_acc_data_source.cnd_loader
                self.cont_loader = cnd_acc_data_source.cont_loader
            cnd_acc_data_source.cnd_loader = None
            cnd_acc_data_source.cont_loader = None


class FFCVCndSetEpochHook(Hook):
    def before_epoch(self, runner):
        loaders = runner.data_loader.loaders
        assert len(loaders) == 2
        loaders[0].set_epoch(runner.epoch)
        loaders[1].cnd_set_epoch(
                runner.epoch, runner.model, 
                loaders[0])


class INVryCndSetEpochHook(Hook):
    def before_epoch(self, runner):
        datasets = runner.data_loader.dataset.datasets
        assert len(datasets) == 2
        cont_data_source = datasets[0].data_source
        storage_data_source = datasets[1].data_source
        storage_data_source.cnd_set_epoch(
                runner.epoch, runner.model, 
                cont_data_source)


class CotrainSAYCamParamBuilder(SAYCamParamBuilder):
    def __init__(
            self, mix_weight, use_cnd_hook=False, 
            concat_batches=False,
            scale_ratio=None,
            separate_batch_kwargs=[{}, {}],
            set_epoch_offset=None,
            *args, **kwargs):
        self.mix_weight = mix_weight
        self.use_cnd_hook = use_cnd_hook
        self.concat_batches = concat_batches
        self.scale_ratio = scale_ratio
        self.separate_batch_kwargs = separate_batch_kwargs
        self.set_epoch_offset = set_epoch_offset
        super().__init__(*args, **kwargs)

    def add_set_epoch_hook(self):
        if not self.use_cnd_hook:
            set_epoch_hook_params = {'builder': ConcatSetEpochHook}
            if self.use_ffcvloader:
                set_epoch_hook_params = {'builder': FFCVConcatLoaderSetEpochHook}
        elif self.use_cnd_hook == True:
            set_epoch_hook_params = {'builder': CndSetEpochHook}
            if self.use_ffcvloader:
                set_epoch_hook_params = {'builder': FFCVCndSetEpochHook}
        elif self.use_cnd_hook == 'INVry':
            set_epoch_hook_params = {'builder': INVryCndSetEpochHook}
        else:
            raise NotImplementedError
        if self.set_epoch_offset is not None:
            set_epoch_hook_params['builder_kwargs'] = {
                    'epoch_offset': self.set_epoch_offset}
        self.add_one_hook_params(set_epoch_hook_params)

    def build_train_ffcvloader(self, **kwargs):
        self.cfg.data['train1'].update(kwargs)
        self.cfg.data['train2'].update(kwargs)
        train_ffcvloader1 = build_ffcvloader(self.cfg.data['train1'])
        train_ffcvloader2 = build_ffcvloader(self.cfg.data['train2'])
        train_ffcvloader = concat_datasets.ConcatLoader(
                train_ffcvloader1, train_ffcvloader2)
        return train_ffcvloader

    def build_train_dataset(self):
        data_cfgs = [
                self.cfg.data['train1'],
                self.cfg.data['train2']]
        datasets = []
        for _data_cfg in data_cfgs:
            if 'data_source' in _data_cfg:
                _data_cfg['data_source']['memcached'] = False
            datasets.append(build_dataset(_data_cfg))
        if self.scale_ratio is None:
            train_dataset = concat_datasets.ConcatDataset(*datasets)
        else:
            train_dataset = concat_datasets.ScaledConcatDataset(
                    self.scale_ratio, *datasets)
        return train_dataset

    def get_concat_batches(self, data_batch):
        new_data_batch = {}
        for key in data_batch[0].keys():
            if isinstance(data_batch[0][key], torch.Tensor):
                new_data_batch[key] = torch.cat(
                        [_data_batch[key] for _data_batch in data_batch], 
                        dim=0)
            elif isinstance(data_batch[0][key], list) and isinstance(data_batch[0][key][0], torch.Tensor):
                new_data_batch[key] = [
                        torch.cat([_data_batch[key][_idx] for _data_batch in data_batch]) \
                        for _idx in range(len(data_batch[0][key]))]
            else:
                raise NotImplementedError
        return new_data_batch

    def naive_processor(self, model, loss_func, data_batch):
        if not self.concat_batches:
            if self.scale_ratio is None:
                assert len(data_batch) == 2
            else:
                assert len(data_batch) == self.scale_ratio + 1
                data_batch = [
                        data_batch[0], 
                        self.get_concat_batches(data_batch[1:])]
            # Tricky way to set per-run kwargs
            data_batch[0].update(self.separate_batch_kwargs[0])
            data_batch[1].update(self.separate_batch_kwargs[1])
            model_outputs0 = batch_processor(model, data_batch[0], True)
            model_outputs1 = batch_processor(model, data_batch[1], True)
            model_outputs = dict(
                    loss=model_outputs0['loss']*self.mix_weight \
                         + model_outputs1['loss'],
                    log_vars0=model_outputs0['log_vars'],
                    log_vars1=model_outputs1['log_vars'],
                    num_samples=model_outputs0['num_samples'],
                    )
        else:
            new_data_batch = self.get_concat_batches(data_batch)
            model_outputs = batch_processor(model, new_data_batch, True)
        return model_outputs
