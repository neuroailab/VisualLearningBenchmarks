import torch
from tqdm import tqdm
import numpy as np
from .hook import Hook
from ..dist_utils import gather_tensors_batch


class ValidateHook(Hook):
    def __init__(
            self,
            name,
            data_loader_builder,
            batch_processor,
            agg_func,
            data_loader_builder_kwargs={},
            batch_processor_kwargs={},
            agg_func_kwargs={},
            initial=True,
            interval=1,
            by_epoch=True,
            **kwargs):
        self.data_loader = data_loader_builder(
                **data_loader_builder_kwargs)
        self.batch_processor = batch_processor
        self.batch_processor_kwargs = batch_processor_kwargs
        self.agg_func = agg_func
        self.agg_func_kwargs = agg_func_kwargs

        self.name = name
        self.initial = initial
        self.by_epoch = by_epoch
        self.interval = interval

    def before_run(self, runner):
        if self.initial and (runner.epoch==0 and runner.iter==0):
            self._run_validate(runner)

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        if not self.every_n_epochs(runner, self.interval):
            return
        self._run_validate(runner)

    def after_train_iter(self, runner):
        if self.by_epoch:
            return
        if self.every_n_iters(runner, self.interval):
            self._run_validate(runner)

    def nondist_forward_collect(self, runner):
        '''Forward and collect network outputs.

        This function performs forward propagation and collects outputs.
        It can be used to collect results, features, losses, etc.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        '''
        results = []
        for data_batch in tqdm(self.data_loader, desc=self.name):
            with torch.no_grad():
                result = self.batch_processor(
                        runner.model, data_batch, 
                        **self.batch_processor_kwargs)
            results.append(result)

        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate(
                [batch[k].cpu().numpy() for batch in results], axis=0)
        return results_all

    def dist_forward_collect(self, runner):
        results = []
        to_enum = self.data_loader
        if runner.rank == 0:
            to_enum = tqdm(self.data_loader, desc=self.name)
        for data_batch in to_enum:
            with torch.no_grad():
                result = self.batch_processor(
                        runner.model, data_batch, 
                        **self.batch_processor_kwargs)
            results.append(result)

        results_all = {}
        length = len(self.data_loader.dataset)
        for k in results[0].keys():
            results_cat = np.concatenate(
                    [batch[k].cpu().numpy() for batch in results],
                    axis=0)
            results_gathered = gather_tensors_batch(results_cat, part_size=20)
            results_gathered = np.concatenate(results_gathered, axis=0)
            assert len(results_gathered) >= length, \
                    "Please make sure the results of your validation processor "\
                    + "have batch size dimension as the first dimsion"\
                    + (", {}, {}".format(len(results_gathered), length))
            results_strip = results_gathered[:length]
            results_all[k] = results_strip
        return results_all

    def _get_valid_results(self, runner):
        in_dist = (runner.world_size > 1) \
                and isinstance(self.data_loader.sampler, 
                               torch.utils.data.DistributedSampler)
        results = None
        if in_dist:
            results = self.dist_forward_collect(runner)
        else:
            if runner.rank == 0:
                results = self.nondist_forward_collect(runner)
        return results

    def _run_validate(self, runner):
        runner.model.eval()

        results = self._get_valid_results(runner)
        if runner.rank == 0:
            agg_res = self.agg_func(results, **self.agg_func_kwargs)
            runner.logger.info({self.name: agg_res})
            runner.record_saver.save(
                    {'validation_results': {self.name: agg_res}})

        runner.model.train()
