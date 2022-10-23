from .hook import Hook
import time
import numpy as np


class NaiveLogger(Hook):
    def __init__(self, interval=50):
        self.interval = interval
        self.buffer = {}

    def before_train_epoch(self, runner):
        self.buffer = {
                'loss': [],
                'time': [],
                'data_time': [],
                }
        self.last_time = time.time()

    def before_train_iter(self, runner):
        self.buffer['data_time'].append(time.time() - self.last_time)

    def after_train_iter(self, runner):
        self.buffer['time'].append(time.time() - self.last_time)
        self.last_time = time.time()

        if self.every_n_inner_iters(runner, self.interval):
            self.buffer['loss'].append(
                    runner.iter_outputs['loss'].detach().item())
            message_pattern = \
                    'Epoch: [{0}/{3}][{1}/{2}]\t' + \
                    'Time {batch_time:.3f}, \t' + \
                    'Data {data_time:.3f}, \t' + \
                    'LR {learning_rate:.5f}, \t' + \
                    'Loss {loss:.4f} ({run_loss:.4f})'
            message = message_pattern.format(
                    runner.epoch+1, runner.inner_iter+1, 
                    len(runner.data_loader), runner.max_epochs,
                    batch_time=np.mean(self.buffer['time']),
                    data_time=np.mean(self.buffer['data_time']), 
                    learning_rate=np.mean(runner.current_lr()),
                    loss=np.mean(self.buffer['loss']),
                    run_loss=np.mean(self.buffer['loss'][-1:]))
            runner.logger.info(message)
