import logging
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
import torch

from . import defaults
from .defaults import DEFAULT_VALUES
from .hooks.hook import Hook
from .hooks.validate_hook import ValidateHook
from .hooks.checkpoint import CheckpointHook
from .hooks.record_saver import BaseSaver
from .hooks.sampler_seed import DistSamplerSeedHook
from .priority import get_priority
from .checkpoint import load_checkpoint
from .dist_utils import get_dist_info


class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``train()``
    - ``test()``

    Args:
    """

    LATEST_CKPT_NAME = 'latest_cached.pth'
    def __init__(
            self,
            save_params,
            train_data_params,
            model_optimizer_params,
            loss_params,
            learning_rate_params,
            batch_processor_params=DEFAULT_VALUES['batch_processor_params'],
            extra_hook_params=DEFAULT_VALUES['extra_hook_params'],
            load_params=DEFAULT_VALUES['load_params'],
            optimizer_hook_params=DEFAULT_VALUES['optimizer_hook_params'],
            logger_hook_params=DEFAULT_VALUES['logger_hook_params'],
            validation_params=DEFAULT_VALUES['validation_params'],
            max_iters=DEFAULT_VALUES['max_iters'],
            max_epochs=DEFAULT_VALUES['max_epochs'],
            logger=DEFAULT_VALUES['logger'],
            validate_hook=ValidateHook,
            *args, **kwargs):
        self.save_params = save_params
        self.load_params = load_params
        self.train_data_params = train_data_params
        self.model_optimizer_params = model_optimizer_params
        self.loss_params = loss_params
        self.learning_rate_params = learning_rate_params
        self.extra_hook_params = extra_hook_params
        self.batch_processor_params = batch_processor_params
        self.optimizer_hook_params = optimizer_hook_params
        self.logger_hook_params = logger_hook_params
        self.validation_params = validation_params
        self.validate_hook = validate_hook
        if logger is None:
            self.logger = defaults.get_default_logger()
        else:
            if not isinstance(logger, logging.Logger):
                raise TypeError(f'logger must be a logging.Logger object, '
                                f'but got {type(logger)}')
            self.logger = logger

        self.check_params()
        self.add_defaults_of_certain_keys_to_params()

        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')
        self._max_epochs = max_epochs
        self._max_iters = max_iters

        self._rank, self._world_size = get_dist_info()
        self.setup()

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl,
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        pass

    def check_params(self):
        for _param_name, _required_keys in defaults.ALL_REQUIRED_KEYS.items():
            curr_params = getattr(self, _param_name)
            for each_key in _required_keys:
                assert each_key in curr_params, \
                        "Key {} missing in {}".format(each_key, _param_name)

    def add_defaults_of_certain_keys_to_params(self):
        defaults_of_certain_keys = defaults.DEFAULT_VALUES_FOR_CERTAIN_KEYS
        for _param_name in defaults.PARAM_NAMES:
            default_params = defaults_of_certain_keys.get(_param_name, None)
            if default_params:
                curr_params = getattr(self, _param_name)
                for key, value in default_params.items():
                    if key not in curr_params:
                        curr_params[key] = value

    def build_train_dataset_then_loader(self):
        batch_size = self.train_data_params['batch_size']
        num_workers = self.train_data_params['num_workers']
        shuffle = self.train_data_params['shuffle']
        dist = self.train_data_params['distributed']

        dataset = self.train_data_params['dataset_builder'](
                **self.train_data_params['dataset_builder_kwargs'])

        if not dist:
            sampler = torch.utils.data.RandomSampler(dataset) \
                    if shuffle else None
        else:
            rank, world_size = get_dist_info()
            sampler = torch.utils.data.DistributedSampler(
                dataset, world_size, rank, 
                shuffle=shuffle)
            self.register_hook(DistSamplerSeedHook())

        self.data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=False,
                **self.train_data_params['data_loader_kwargs'])

    def build_train_loader_directly(self):
        self.data_loader = self.train_data_params['data_loader_builder'](
                **self.train_data_params['data_loader_kwargs'])

    def build_data_loader(self):
        if self.train_data_params['build_dataset_then_loader']:
            self.build_train_dataset_then_loader()
        else:
            self.build_train_loader_directly()

    def build_model_optimizer(self):
        self.model, self.optimizer = self.model_optimizer_params['builder'](
                **self.model_optimizer_params['builder_kwargs'])

    def build_loss(self):
        self.loss_func = self.loss_params['builder'](
                **self.loss_params['builder_kwargs'])

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            try:
                lr = [group['lr'] for group in self.optimizer.param_groups]
            except:
                raise RuntimeError(
                    'lr is not applicable, you need to overwrite function current_lr.')
        return lr

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def register_lr_hook(self):
        lr_hook = self.learning_rate_params['builder'](
                **self.learning_rate_params['builder_kwargs'])
        assert isinstance(lr_hook, Hook), \
                "learning_rate_params must build a hook"
        self.register_hook(lr_hook)

    def register_optimizer_hook(self):
        optimizer_hook = self.optimizer_hook_params['builder'](
                **self.optimizer_hook_params['builder_kwargs'])
        assert isinstance(optimizer_hook, Hook), \
                "optimizer_hook_params must build a hook"
        self.register_hook(optimizer_hook)

    def register_logger_hook(self):
        logger_hook = self.logger_hook_params['builder'](
                **self.logger_hook_params['builder_kwargs'])
        assert isinstance(logger_hook, Hook), \
                "logger_hook_params must build a hook"
        self.register_hook(logger_hook, priority='LOWEST')

    def register_save_ckpt_hook(self):
        save_ckpt_hook = self.save_params['ckpt_hook_builder'](
                **self.save_params['ckpt_hook_kwargs'])
        assert isinstance(save_ckpt_hook, Hook), \
                "ckpt_hook in save_params must build a hook"
        self.register_hook(save_ckpt_hook, priority='LOWEST')
        self.save_ckpt_hook = save_ckpt_hook # for later loading purpose

    def register_record_saver_hook(self):
        params = {
                _name: getattr(self, _name) 
                for _name in defaults.PARAM_NAMES}
        record_saver_hook = self.save_params['record_saver_builder'](
                params=params, logger=self.logger, 
                **self.save_params['record_saver_kwargs'])
        assert isinstance(record_saver_hook, BaseSaver), \
                "record_saver in save_params must build a saver"
        self.register_hook(record_saver_hook, priority='VERY_LOW')
        self.record_saver = record_saver_hook # for later saving purpose

    def register_training_hooks(self):
        self.register_save_ckpt_hook()
        self.register_record_saver_hook()
        self.register_lr_hook()
        self.register_optimizer_hook()
        self.register_logger_hook()

    def register_validation_hooks(self):
        if self.validation_params is None:
            return
        for valid_name, valid_hook_params in self.validation_params.items():
            valid_hook = self.validate_hook(
                    name=valid_name, **valid_hook_params)
            self.register_hook(valid_hook)

    def register_one_extra_hook(self, hook_params):
        assert 'builder' in hook_params, \
                "Builder not found in extra_hook_params"
        builder_kwargs = hook_params.get('builder_kwargs', {})
        extra_hook = hook_params['builder'](**builder_kwargs)
        self.register_hook(extra_hook)

    def register_extra_hooks(self):
        if self.extra_hook_params is None:
            return
        elif isinstance(self.extra_hook_params, dict):
            self.register_one_extra_hook(self.extra_hook_params)
        elif isinstance(self.extra_hook_params, list):
            for one_extra_hook_param in self.extra_hook_params:
                self.register_one_extra_hook(one_extra_hook_param)
        else:
            raise NotImplementedError(
                    'Extra_hook_param should be list or dict!')

    def get_latest_checkpoint(self):
        if isinstance(self.save_ckpt_hook, CheckpointHook):
            ckpt = osp.join(self.save_ckpt_hook.out_dir, self.LATEST_CKPT_NAME)
            if osp.exists(ckpt):
                return ckpt
        else:
            self.logger.info(
                    'Save ckpt hook not recognized, resume cannot be done')
        return None

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def resume_from_ckpt(
            self,
            checkpoint,
            resume_optimizer=True,
            map_location='default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, torch.optim.Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'amp' in checkpoint:
            import apex
            apex.amp.load_state_dict(checkpoint['amp'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def load(self):
        ckpt_path = self.load_params.get('from_checkpoint', None)
        resume = self.load_params.get('resume', True)
        if resume:
            ckpt_path = self.get_latest_checkpoint() or ckpt_path

        if ckpt_path and osp.exists(ckpt_path):
            self.resume_from_ckpt(ckpt_path)
            return

    def setup(self):
        self.build_data_loader()
        self.build_model_optimizer()
        self.build_loss()
        self.register_training_hooks()
        self.register_validation_hooks()
        self.register_extra_hooks()
        self.load()
