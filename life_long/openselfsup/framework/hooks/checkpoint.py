import os

from ..dist_utils import allreduce_params, master_only
from .hook import Hook


class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        out_dir (str, required): The directory to save checkpoints.
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        sync_buffer (bool): Whether to synchronize buffers in different
            gpus. Default: False.
    """

    def __init__(self,
                 out_dir,
                 interval=-1,
                 cache_interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 sync_buffer=False,
                 ):
        self.out_dir = out_dir
        self.interval = interval
        self.cache_interval = cache_interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.sync_buffer = sync_buffer

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        to_cache = self.every_n_epochs(runner, self.cache_interval)
        to_save = self.every_n_epochs(runner, self.interval)
        skip_save = not (to_cache or to_save)
        if skip_save:
            return

        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(runner, only_as_cache=to_cache and (not to_save))
        runner.logger.info(f'Saved checkpoint at {runner.epoch + 1} epochs')

    @master_only
    def _save_checkpoint(self, runner, only_as_cache, **kwargs):
        """Save the current checkpoint and delete unwanted checkpoint."""
        runner.save_checkpoint(
            self.out_dir, 
            save_optimizer=self.save_optimizer, 
            only_as_cache=only_as_cache, **kwargs)

    def after_train_iter(self, runner):
        if self.by_epoch:
            return
        to_cache = self.every_n_iters(runner, self.cache_interval)
        to_save = self.every_n_iters(runner, self.interval)
        skip_save = not (to_cache or to_save)
        if skip_save:
            return

        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(runner, only_as_cache=to_cache and (not to_save))
        runner.logger.info(
            f'Saved checkpoint at {runner.iter + 1} iterations')
