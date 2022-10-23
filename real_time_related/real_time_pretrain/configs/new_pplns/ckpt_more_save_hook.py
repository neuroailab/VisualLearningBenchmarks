from openselfsup.framework.hooks.checkpoint import CheckpointHook


class CkptSpecifySaveHook(CheckpointHook):
    def __init__(self,
                 specify_iter=[], specify_epoch=[],
                 *args, **kwargs):
        self.specify_iter = specify_iter
        self.specify_epoch = specify_epoch
        super().__init__(*args, **kwargs)

    def after_train_epoch(self, runner):
        if (not self.by_epoch) and ((runner.epoch+1) not in self.specify_epoch):
            return
        to_cache = self.every_n_epochs(runner, self.cache_interval)
        to_save = self.every_n_epochs(runner, self.interval)
        if (runner.epoch+1) in self.specify_epoch:
            to_save = True
        skip_save = not (to_cache or to_save)
        if skip_save:
            return

        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(runner, only_as_cache=to_cache and (not to_save))
        runner.logger.info(f'Saved checkpoint at {runner.epoch + 1} epochs')

    def after_train_iter(self, runner):
        if (self.by_epoch) and ((runner.iter+1) not in self.specify_iter):
            return
        to_cache = self.every_n_iters(runner, self.cache_interval)
        to_save = self.every_n_iters(runner, self.interval)
        if (runner.iter+1) in self.specify_iter:
            to_save = True
        skip_save = not (to_cache or to_save)
        if skip_save:
            return

        if self.sync_buffer:
            allreduce_params(runner.model.buffers())
        self._save_checkpoint(
                runner, only_as_cache=to_cache and (not to_save),
                filename_tmpl='epoch_{}_iter_' + str(runner.iter+1) + '.pth',
                )
        runner.logger.info(
            f'Saved checkpoint at {runner.iter + 1} iterations')


def make_ckpt_hook_more_save(params, base_bs=256):
    params['save_params']['ckpt_hook_builder'] = CkptSpecifySaveHook
    params['save_params']['ckpt_hook_kwargs']['interval'] = 10
    specify_iter = list(range(100, 1001, 100)) + [1500, 2000, 3000, 4000]
    if base_bs != 256:
        specify_iter = [int(_step * 256 / base_bs) for _step in specify_iter]
    params['save_params']['ckpt_hook_kwargs']['specify_iter'] = specify_iter
    params['save_params']['ckpt_hook_kwargs']['specify_epoch'] = list(range(1, 10))
    return params


def make_ckpt_hook_even_more_save(params, base_bs=256):
    params['save_params']['ckpt_hook_builder'] = CkptSpecifySaveHook
    params['save_params']['ckpt_hook_kwargs']['interval'] = 10
    specify_iter = list(range(100, 1001, 100)) + [1500, 2000, 3000, 4000]
    for ep in range(1, 10):
        specify_iter.append(ep * 5004 + 5004 // 3)
        specify_iter.append(ep * 5004 + 5004 // 3 * 2)
    if base_bs != 256:
        specify_iter = [int(_step * 256 / base_bs) for _step in specify_iter]
    params['save_params']['ckpt_hook_kwargs']['specify_iter'] = specify_iter
    params['save_params']['ckpt_hook_kwargs']['specify_epoch'] = list(range(1, 10))
    return params
