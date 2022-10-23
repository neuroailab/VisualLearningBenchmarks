import torch
import torch.nn as nn
import os
import numpy as np
import pdb
from tqdm import tqdm
import pickle
import copy
from .registry import DATASETS
from torch.utils.data import Dataset
from openselfsup.framework.dist_utils import get_dist_info
import time


@DATASETS.register_module
class SAYCamSeqVecDataset(Dataset):
    def __init__(
            self, root, list_file, which_model, 
            data_len=256*5000, seq_len=32, sub_dim=None):
        self.seq_len = seq_len
        self.data_len = data_len
        self.root = root
        self.list_file = list_file
        self.which_model = which_model
        self.sub_dim = sub_dim
        self.load_video_list()
        self.last_set_epoch = None
        self.set_epoch(0)

    def __len__(self):
        return self.data_len

    def get_curr_seq_len(self, idx):
        return self.seq_len

    def get_curr_which_aug_target_idx(self, idx, seq_len):
        which_aug = np.random.choice([0, 1], seq_len, replace=True)
        target_idx = np.random.choice(range(seq_len))
        return which_aug, target_idx

    def l2_normalize(self, vector):
        _norm = np.linalg.norm(vector, axis=-1, keepdims=True)
        vector = vector / _norm
        return vector

    def __getitem__(self, idx):
        sta_pos = self.all_sta_pos[idx]
        vector = []
        seq_len = self.get_curr_seq_len(idx)
        which_aug, target_idx = self.get_curr_which_aug_target_idx(idx, seq_len)
        for embd_idx, _aug in enumerate(which_aug):
            vector.append(self.all_embds[sta_pos+embd_idx, _aug])
        vector.append(
                self.all_embds[sta_pos+target_idx, 1-which_aug[target_idx]])
        vector = np.stack(vector, axis=0)
        target = self.all_embds[sta_pos+target_idx, which_aug[target_idx]]
        if self.sub_dim is not None:
            vector = self.l2_normalize(vector[:, :self.sub_dim])
            target = self.l2_normalize(target[:self.sub_dim])
        return dict(img=vector, target=target)

    def load_video_list(self):
        with open(self.list_file, 'r') as fin:
            all_lines = fin.readlines()
            self.video_list = [_line[:-1] for _line in all_lines]

    def load_embds(self):
        possible_sta_pos = []
        embds = []
        curr_len = 0

        def _get_offset_num(name):
            return int(name.split('_')[1])
        def _get_aug_num(name):
            return int(name.split('_')[3].split('.')[0])

        for video_name in self.video_list:
            embd_dir = os.path.join(self.root, self.which_model, video_name)
            embd_files = os.listdir(embd_dir)
            one_embd_file = np.random.choice(embd_files)
            offset_num = _get_offset_num(one_embd_file)
            aug_num = _get_aug_num(one_embd_file)
            filtered_embd_files = filter(
                    lambda x: (_get_offset_num(x) == offset_num) \
                              & (_get_aug_num(x) != aug_num),
                    embd_files, 
                    )
            another_embd_file = np.random.choice(list(filtered_embd_files))
            one_embd = np.load(os.path.join(embd_dir, one_embd_file))
            another_embd = np.load(os.path.join(embd_dir, another_embd_file))
            embds.append(np.stack([one_embd, another_embd], axis=1))
            possible_sta_pos.extend(
                    list(range(curr_len, 
                               curr_len + len(one_embd) - self.seq_len+1)))
            curr_len += len(one_embd)
        self.all_embds = np.concatenate(embds, axis=0)
        self.all_sta_pos = np.random.choice(
                possible_sta_pos, self.data_len, replace=True)

    def set_epoch(self, epoch):
        if self.last_set_epoch == epoch:
            return
        np.random.seed(epoch)
        self.load_embds()
        self.last_set_epoch = epoch


@DATASETS.register_module
class VaryLenSAYCamSeqVec(SAYCamSeqVecDataset):
    def __init__(
            self, seq_len, min_seq_len, batch_size,
            *args, **kwargs):
        self.min_seq_len = min_seq_len
        self.batch_size = batch_size
        super().__init__(seq_len=seq_len, *args, **kwargs)

    def get_seq_len_array(self):
        assert self.min_seq_len < self.seq_len
        assert self.data_len % self.batch_size == 0, \
                (self.data_len, self.batch_size)
        seq_len_array = []
        for _ in range(0, self.data_len, self.batch_size):
            curr_seq_len = np.random.randint(self.min_seq_len, self.seq_len+1)
            seq_len_array.extend([curr_seq_len]*self.batch_size)
        self.seq_len_array = np.asarray(seq_len_array)

    def get_curr_seq_len(self, idx):
        return self.seq_len_array[idx]

    def set_epoch(self, epoch):
        if self.last_set_epoch == epoch:
            return
        np.random.seed(epoch)
        self.load_embds()
        self.get_seq_len_array()
        self.last_set_epoch = epoch


@DATASETS.register_module
class OSFilterVLenSCSeqVec(VaryLenSAYCamSeqVec):
    def __init__(
            self, over_sample_ratio=3, cache_folder=None, 
            *args, **kwargs):
        self.over_sample_ratio = over_sample_ratio
        self.cache_folder = cache_folder
        kwargs = copy.deepcopy(kwargs)
        self.true_batch_size = kwargs['batch_size']
        self.true_data_len = kwargs.get('data_len', 256*5000)
        kwargs['batch_size'] = over_sample_ratio * self.true_batch_size
        kwargs['data_len'] = over_sample_ratio * self.true_data_len
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch):
        if self.last_set_epoch == epoch:
            return
        np.random.seed(epoch)
        self.curr_epoch = epoch
        self.load_embds()
        self.get_seq_len_array()
        self.get_which_aug_target_idx_array()
        self.filter_over_sampled_data()
        self.last_set_epoch = epoch

    def get_which_aug_target_idx_array(self):
        which_aug_arr = []
        target_idx_arr = []
        rank, world_size = get_dist_info()
        to_iter = range(0, self.data_len)
        if rank == 0:
            to_iter = tqdm(to_iter, desc='Generate Meta Info')
        for idx in to_iter:
            seq_len = self.get_curr_seq_len(idx)
            which_aug, target_idx \
                    = SAYCamSeqVecDataset.get_curr_which_aug_target_idx(
                            self, idx, seq_len)
            which_aug_arr.append(which_aug)
            target_idx_arr.append(target_idx)
        self.which_aug_arr = which_aug_arr
        self.target_idx_arr = target_idx_arr

    def get_curr_which_aug_target_idx(self, idx, seq_len):
        return self.which_aug_arr[idx], self.target_idx_arr[idx]

    def __len__(self):
        return self.true_data_len

    def get_all_right_d_idx(self):
        rank, world_size = get_dist_info()
        to_iter = range(self.data_len // self.batch_size)
        if rank == 0:
            to_iter = tqdm(to_iter, desc='Filter Dataset')
        wrong_batches = []
        all_right_d_idx = []
        for idx in to_iter:
            d_sta = idx * self.batch_size
            d_end = (idx+1) * self.batch_size
            right_d_idx = []
            for d_idx in range(d_sta, d_end):
                curr_data = self.__getitem__(d_idx)
                seqs = curr_data['img']
                target = curr_data['target']
                cue = seqs[-1:]
                target = target[np.newaxis, :]
                cue_tar_sim = np.sum(target * cue, axis=-1)
                cue_seq_sim = np.sum(seqs[:-1] * cue, axis=-1)
                NN_right = np.all(cue_tar_sim >= cue_seq_sim)
                if NN_right:
                    right_d_idx.append(d_idx)
                    if len(right_d_idx) >= self.true_batch_size:
                        break
            sample_replace = False
            if len(right_d_idx) < self.true_batch_size:
                wrong_batches.append(
                        self.true_batch_size - len(right_d_idx))
                sample_replace = True
            sampled_d_idx = np.random.choice(
                    right_d_idx, self.true_batch_size, 
                    replace=sample_replace)
            all_right_d_idx.append(sampled_d_idx)
        all_right_d_idx = np.concatenate(all_right_d_idx)
        return all_right_d_idx

    def get_cache_path(self):
        return os.path.join(
                self.cache_folder, 
                'ep{}.npy'.format(self.curr_epoch))

    def load_or_get_all_right_d_idx(self):
        if self.cache_folder is None:
            return self.get_all_right_d_idx()

        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            return np.load(cache_path)
        rank, world_size = get_dist_info()
        if rank == 0:
            all_right_d_idx = self.get_all_right_d_idx()
            os.system('mkdir -p ' + self.cache_folder)
            np.save(cache_path, all_right_d_idx)
            return all_right_d_idx
        else:
            while not os.path.exists(cache_path):
                # Wait for the main process to adjust
                time.sleep(1.0)
            time.sleep(15.0)
            return np.load(cache_path)

    def filter_over_sampled_data(self):
        all_right_d_idx = self.load_or_get_all_right_d_idx()
        new_all_sta_pos = [self.all_sta_pos[_i] for _i in all_right_d_idx]
        new_which_aug_arr = [self.which_aug_arr[_i] for _i in all_right_d_idx]
        new_target_idx_arr = [self.target_idx_arr[_i] for _i in all_right_d_idx]
        new_seq_len_array = [self.seq_len_array[_i] for _i in all_right_d_idx]

        assert len(new_all_sta_pos) == self.true_data_len
        self.all_sta_pos = np.asarray(new_all_sta_pos)
        self.which_aug_arr = new_which_aug_arr
        self.target_idx_arr = new_target_idx_arr
        self.seq_len_array = new_seq_len_array


@DATASETS.register_module
class FilterVLenSCSeqVec(VaryLenSAYCamSeqVec):
    def __init__(self, cache_folder=None, *args, **kwargs):
        self.cache_folder = cache_folder
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch):
        if self.last_set_epoch == epoch:
            return
        np.random.seed(epoch)
        self.load_embds()
        self.get_seq_len_array()
        if self.has_cached(epoch):
            self.load_cached_aug_target(epoch)
        else:
            if self.cache_folder is None:
                self.get_which_aug_target_idx_array()
                self.adjust_aug_target()
            else:
                rank, world_size = get_dist_info()
                if rank == 0:
                    self.get_which_aug_target_idx_array()
                    self.adjust_aug_target()
                    self.cache_aug_target(epoch)
                else:
                    while not self.has_cached(epoch):
                        # Wait for the main process to adjust
                        time.sleep(1.0)
                    time.sleep(15.0)
                    self.load_cached_aug_target(epoch)
        self.last_set_epoch = epoch

    def get_cache_path(self, epoch):
        return os.path.join(self.cache_folder, 'ep{}.pkl'.format(epoch))

    def has_cached(self, epoch):
        if self.cache_folder is None:
            return False
        cache_path = self.get_cache_path(epoch)
        return os.path.exists(cache_path)

    def load_cached_aug_target(self, epoch):
        cache_path = self.get_cache_path(epoch)
        cached_data = pickle.load(open(cache_path, 'rb'))
        self.which_aug_arr = cached_data['which_aug_arr']
        self.target_idx_arr = cached_data['target_idx_arr']

    def cache_aug_target(self, epoch):
        os.system('mkdir -p ' + self.cache_folder)
        cache_path = self.get_cache_path(epoch)
        data_to_cache = {
                'which_aug_arr': self.which_aug_arr,
                'target_idx_arr': self.target_idx_arr,
                }
        pickle.dump(data_to_cache, open(cache_path, 'wb'))

    def get_curr_which_aug_target_idx(self, idx, seq_len):
        return self.which_aug_arr[idx], self.target_idx_arr[idx]

    def get_which_aug_target_idx_array(self):
        which_aug_arr = []
        target_idx_arr = []
        for idx in range(0, self.data_len):
            seq_len = self.get_curr_seq_len(idx)
            which_aug, target_idx \
                    = SAYCamSeqVecDataset.get_curr_which_aug_target_idx(
                            self, idx, seq_len)
            which_aug_arr.append(which_aug)
            target_idx_arr.append(target_idx)
        self.which_aug_arr = which_aug_arr
        self.target_idx_arr = target_idx_arr

    def adjust_aug_target(self):
        mask = []

        rank, world_size = get_dist_info()
        if rank == 0:
            to_iter = tqdm(range(self.data_len), desc='Adjust Dataset Metas')
        else:
            to_iter = range(self.data_len)
        for idx in to_iter:
            NN_right = False
            _attemps = 0
            while (NN_right == False) and (_attemps < 50):
                curr_data = self.__getitem__(idx)
                seqs = curr_data['img']
                target = curr_data['target']
                cue = seqs[-1:]
                target = target[np.newaxis, :]
                cue_tar_sim = np.sum(target * cue, axis=-1)
                cue_seq_sim = np.sum(seqs[:-1] * cue, axis=-1)
                NN_right = np.all(cue_tar_sim >= cue_seq_sim)
                _attemps += 1
                if NN_right == False:
                    seq_len = self.get_curr_seq_len(idx)
                    which_aug, target_idx \
                            = SAYCamSeqVecDataset.get_curr_which_aug_target_idx(
                                    self, idx, seq_len)
                    self.which_aug_arr[idx] = which_aug
                    self.target_idx_arr[idx] = target_idx
            mask.append(NN_right)
