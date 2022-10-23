from .hooks.optimizer import OptimizerHook
from .hooks.checkpoint import CheckpointHook
from .hooks.logger import NaiveLogger
from .hooks.record_saver import FileSaver
import logging


__logger = None
def get_default_logger():
    global __logger
    if __logger is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(format=format_str, level=logging.INFO)
        __logger = logging.getLogger('ptutils')
    return __logger


def run_model_get_loss(model, loss_func, data_batch):
    model_outputs = model(data_batch)
    loss_value = loss_func(data_batch, model_outputs)
    iter_outputs = {'loss': loss_value}
    return iter_outputs

DEFAULT_VALUES = {
        'logger_hook_params': {
            'builder': NaiveLogger,
            'builder_kwargs': {},
            },
        'optimizer_hook_params': {
            'builder': OptimizerHook,
            'builder_kwargs': {},
            },
        'batch_processor_params': {
            'func': run_model_get_loss,
            'func_kwargs': {},
            },
        'extra_hook_params': None,
        'load_params': {
            'resume': True,
            'from_checkpoint': None,
            },
        'max_iters': None,
        'max_epochs': None,
        'validation_params': None,
        'logger': None,
        }

DEFAULT_VALUES_FOR_CERTAIN_KEYS = {
        'train_data_params': {
            'shuffle': True,
            'distributed': False,
            'data_loader_kwargs': {},
            'dataset_builder_kwargs': {},
            'build_dataset_then_loader': True,
            },
        'model_optimizer_params': {
            'builder_kwargs': {},
            },
        'loss_params': {
            'builder_kwargs': {},
            },
        'learning_rate_params': {
            'builder_kwargs': {},
            },
        'optimizer_hook_params': {
            'builder_kwargs': {},
            },
        'logger_hook_params': {
            'builder_kwargs': {},
            },
        'batch_processor_params': {
            'func_kwargs': {},
            },
        'save_params': {
            'record_saver_builder': FileSaver,
            'ckpt_hook_builder': CheckpointHook,
            },
        }


ALL_REQUIRED_KEYS = {
        'train_data_params': [
            'num_workers',
            'batch_size',
            'dataset_builder',
            ],
        'model_optimizer_params': ['builder'],
        'loss_params': ['builder'],
        'learning_rate_params': ['builder'],
        'batch_processor_params': ['func'],
        'optimizer_hook_params': ['builder'],
        'logger_hook_params': ['builder'],
        'save_params': ['record_saver_kwargs', 'ckpt_hook_kwargs'],
        }

PARAM_NAMES = [
        'save_params', 'load_params',
        'train_data_params', 'model_optimizer_params', 
        'loss_params', 'learning_rate_params',
        'extra_hook_params', 'optimizer_hook_params',
        'batch_processor_params', 'logger_hook_params',
        'validation_params', 'max_epochs', 'max_iters',
        ]
