from math import cos, pi
from openselfsup.framework.hooks.hook import Hook
from openselfsup.framework.checkpoint import is_module_wrapper


class MoCoHook(Hook):
    """Hook for MoCo.
    """

    def __init__(self, update_interval=1, **kwargs):
        self.update_interval = update_interval

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                assert not runner.model.module.update_in_forward
                runner.model.module._momentum_update_key_encoder()
            else:
                assert not runner.model.update_in_forward
                runner.model._momentum_update_key_encoder()
