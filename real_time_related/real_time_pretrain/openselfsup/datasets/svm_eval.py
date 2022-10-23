from PIL import Image
from .registry import DATASETS
from .base import BaseDataset
from openselfsup.utils import print_log
from sklearn.linear_model import SGDClassifier
import torch
from tqdm import tqdm
import numpy as np

from .builder import build_dataset
from .loader.sampler import DistributedSampler
from ..framework.dist_utils import get_dist_info


def get_input_size_from_cfg(cfg):
    # Assuming first one is RandomResizedCrop
    if 'pipeline' in cfg.data['train']:
        input_size = cfg.data['train']['pipeline'][0]['size']
    elif 'pipeline1' in cfg.data['train']:
        input_size = cfg.data['train']['pipeline1'][0]['size']
    else:
        input_size = 224
    return input_size


def get_typical_svm_dataset_cfg(
        input_size,
        meta_path='data/imagenet/meta/part_train_val_labeled.txt'):
    data_root = 'data/imagenet'
    data_source_cfg = dict(
            type='ImageNet',
            memcached=False,
            mclient_path=None)
    img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize_size = int(256.0 * input_size / 224)
    test_pipeline = [
        dict(type='Resize', size=resize_size),
        dict(type='CenterCrop', size=input_size),
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
    ]
    val_svm_cfg=dict(
            type='ExtractDataset',
            data_source=dict(
                list_file=meta_path, 
                root=data_root, **data_source_cfg),
            pipeline=test_pipeline)
    return val_svm_cfg


def build_data_loader_from_dataset(
        val_dataset, batch_size=32, num_workers=10):
    rank, world_size = get_dist_info()
    sampler = DistributedSampler(
            val_dataset, world_size, rank, 
            shuffle=False)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=10,
            sampler=sampler,
            )
    return val_loader


class GroupSVMEval:
    def __init__(
            self, input_size, meta_path, train_len, group_info):
        self.input_size = input_size
        self.meta_path = meta_path
        self.train_len = train_len
        self.group_info = group_info

    def get_eval_data_loader(self):
        self.eval_dataset = build_dataset(
                get_typical_svm_dataset_cfg(
                    self.input_size, self.meta_path))
        val_loader = build_data_loader_from_dataset(self.eval_dataset)
        return val_loader

    def get_svm_acc(self, results):
        features = results['embd']
        target = self.eval_dataset.data_source.labels

        alpha_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]

        train_features = features[:self.train_len]
        train_labels = target[:self.train_len]

        test_features = features[self.train_len:]
        test_labels = target[self.train_len:]

        best_perf = None
        best_lbls = None
        for _alpha in tqdm(alpha_list, desc='SVM Param Search'):
            clf = SGDClassifier(alpha=_alpha, n_jobs=25)
            clf.fit(train_features, train_labels)
            _perf = clf.score(test_features, test_labels)
            if (best_perf is None) or (_perf > best_perf):
                best_perf = _perf
                best_lbls = clf.predict(test_features)

        eval_res = {}
        all_correct = best_lbls == test_labels
        for group_name, (start_idx, end_idx) in self.group_info.items():
            curr_perf = np.mean(all_correct[start_idx:end_idx])
            eval_res[group_name] = curr_perf
        return eval_res
