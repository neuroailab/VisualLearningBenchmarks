from ..registry import DATASOURCES
from .image_list import ImageList
import copy
import numpy as np
import os
from tqdm import tqdm

from .saycam import SAYCamCndCont, SAYCamCndLoss, SAYCamCndMeanSim
from .utils import compute_mean_sim
from ..builder import build_datasource
from sklearn.linear_model import SGDClassifier

from openselfsup.framework.dist_utils import get_dist_info
import openselfsup.models.builder as model_builder
import torch
import pdb


@DATASOURCES.register_module
class ImageNet(ImageList):

    def __init__(
            self, root, list_file, memcached, mclient_path,
            data_len=None):
        super(ImageNet, self).__init__(
            root, list_file, memcached, mclient_path)
        if data_len is not None:
            np.random.seed(0)
            self.fns = np.random.choice(
                    self.fns, data_len, replace=True)


@DATASOURCES.register_module
class ImageNetVaryMetas(ImageList):
    def __init__(
            self, root, switch_epochs, list_files,
            memcached=False, mclient_path=None):
        self.root = root
        self.switch_epochs = switch_epochs
        self.all_fns, self.all_labels = [], []
        for _list_file in list_files:
            _fns, _labels = self.load_one_meta(_list_file)
            self.all_fns.append(_fns)
            self.all_labels.append(_labels)
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False
        self.set_epoch(0)

    def load_one_meta(self, list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.has_labels = len(lines[0].split()) == 2
        if self.has_labels:
            fns, labels = zip(*[l.strip().split() for l in lines])
            labels = [int(l) for l in labels]
        else:
            fns = [l.strip() for l in lines]
            labels = None
        fns = [os.path.join(self.root, fn) for fn in fns]
        return fns, labels

    def set_epoch(self, epoch):
        meta_idx = 0
        for _switch_epoch in self.switch_epochs:
            if epoch >= _switch_epoch:
                meta_idx += 1
        print('which meta: {}'.format(meta_idx))
        self.fns = self.all_fns[meta_idx]
        self.labels = self.all_labels[meta_idx]


@DATASOURCES.register_module
class ImageNetCont(ImageList):
    def __init__(
            self, num_epochs, keep_prev_epoch_num, 
            root, list_file, memcached, mclient_path, 
            data_len=None, accu=False, fix_end_idx=False):
        super().__init__(
            root, list_file, memcached, mclient_path)
        self.raw_fns = copy.copy(self.fns)
        self.num_epochs = num_epochs
        self.keep_prev_epoch_num = keep_prev_epoch_num
        if data_len is None:
            self.data_len = len(self.raw_fns)
        else:
            self.data_len = data_len
        self.accu = accu
        self.fix_end_idx = fix_end_idx
        self.set_epoch(0)

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        len_data = len(self.raw_fns)
        div_num = self.num_epochs + self.keep_prev_epoch_num
        if not self.accu:
            start_idx = int(len_data * (epoch * 1.0 / div_num))
        else:
            start_idx = 0
        if not self.fix_end_idx:
            # This misses the last segment
            end_idx = int(len_data * ((epoch+self.keep_prev_epoch_num) \
                                      * 1.0 / div_num))
        else:
            end_idx = int(len_data * ((epoch+self.keep_prev_epoch_num+1) \
                                      * 1.0 / div_num))
        new_fns = self.raw_fns[start_idx : end_idx]
        self.fns = np.random.choice(
                new_fns, self.data_len, replace=True)


@DATASOURCES.register_module
class ImageNetContVal(ImageNetCont):
    def __init__(
            self, val_keep_prev_epoch_num, root,
            *args, **kwargs):
        self.val_keep_prev_epoch_num = val_keep_prev_epoch_num
        self.val_img_fns = None
        self.root = root
        super().__init__(
                root=os.path.join(root, 'train'), 
                *args, **kwargs)
        assert not self.accu

    def get_all_val_img_fns(self):
        val_root = os.path.join(self.root, 'val')
        all_cates = os.listdir(val_root)
        val_img_fns = {}
        for _cate in all_cates:
            _curr_root = os.path.join(val_root, _cate)
            _img_fns = os.listdir(_curr_root)
            _img_fns = [
                    os.path.join(_curr_root, _img_fn) 
                    for _img_fn in _img_fns]
            val_img_fns[_cate] = _img_fns
        self.val_img_fns = val_img_fns

    def get_start_end_idx(self, seg_start, seg_end):
        len_data = len(self.raw_fns)
        div_num = self.num_epochs + self.keep_prev_epoch_num
        start_idx = int(len_data * (seg_start * 1.0 / div_num))
        end_idx = int(len_data * (seg_end * 1.0 / div_num))
        return start_idx, end_idx

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        if self.val_img_fns is None:
            self.get_all_val_img_fns()
        self.curr_epoch = epoch
        seg_start = max(epoch - self.val_keep_prev_epoch_num, 0)
        seg_end = epoch + self.keep_prev_epoch_num
        start_idx, end_idx = self.get_start_end_idx(seg_start, seg_end)
        train_fns = self.raw_fns[start_idx : end_idx]
        max_class_num = \
                int(np.ceil(\
                    1000.0 / (self.num_epochs + self.keep_prev_epoch_num))) * \
                (self.val_keep_prev_epoch_num + self.keep_prev_epoch_num) 
        train_fns = np.random.choice(train_fns, max_class_num*200)
        train_lbls = [_fn.split('/')[-2] for _fn in train_fns]
        unique_lbls = np.unique(train_lbls)
        val_fns = np.concatenate(
                [self.val_img_fns[_lbl] for _lbl in unique_lbls])
        val_fns = np.random.choice(val_fns, max_class_num*50)
        val_lbls = [_fn.split('/')[-2] for _fn in val_fns]
        self.fns = np.concatenate([train_fns, val_fns])
        self.labels = np.concatenate([train_lbls, val_lbls])
        self.train_len = len(train_fns)

    def get_svm_perf(self, results):
        features = results['embd']
        target = self.labels

        alpha_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]

        train_features = features[:self.train_len]
        train_labels = target[:self.train_len]

        test_features = features[self.train_len:]
        test_labels = target[self.train_len:]

        best_perf = None
        best_lbls = None
        for _alpha in tqdm(alpha_list, desc='SVM Param Search'):
            clf = SGDClassifier(alpha=_alpha, n_jobs=15)
            clf.fit(train_features, train_labels)
            _perf = clf.score(test_features, test_labels)
            if (best_perf is None) or (_perf > best_perf):
                best_perf = _perf
                best_lbls = clf.predict(test_features)

        unique_lbls_per_seg = {}
        curr_lbls = []
        min_seg_idx = max(0, self.curr_epoch - self.val_keep_prev_epoch_num)
        max_seg_idx = self.curr_epoch + self.keep_prev_epoch_num
        for seg_idx in range(min_seg_idx, max_seg_idx):
            _start_idx, _end_idx = self.get_start_end_idx(seg_idx, seg_idx+1)
            _train_fns = self.raw_fns[_start_idx : _end_idx]
            _train_lbls = [_fn.split('/')[-2] for _fn in _train_fns]
            unique_lbls = np.unique(_train_lbls)
            _rel_seg_idx = max_seg_idx - seg_idx
            unique_lbls_per_seg[_rel_seg_idx] = [_lbl for _lbl in unique_lbls if _lbl not in curr_lbls]
            curr_lbls.extend(unique_lbls_per_seg[_rel_seg_idx])

        eval_perf = {}
        for _idx in unique_lbls_per_seg:
            wanted_lbls = unique_lbls_per_seg[_idx]
            _test_idx = [_idx for _idx, _lbl in enumerate(test_labels) if _lbl in wanted_lbls]
            perf_on_seg = best_lbls[_test_idx] == test_labels[_test_idx]
            eval_perf['perf_on_seg_{}'.format(_idx)] = np.mean(perf_on_seg)
        return eval_perf


@DATASOURCES.register_module
class ImageNetCndCont(ImageNetCont):
    def __init__(
            self, pipeline, batch_size=128,
            cond_method='max',
            *args, **kwargs):
        self.cnd_loader = None
        self.cont_loader = None
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.cond_method = cond_method
        super().__init__(*args, **kwargs)

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        self.set_epoch(epoch)

        if self.cont_loader is None:
            SAYCamCndCont.build_loaders(self, cont_data_source)

        storage_embds = SAYCamCndCont.get_storage_embds(self, model)
        self.fns = SAYCamCndCont.get_new_fns(self, model, storage_embds)

    def get_cond_idx(self, _storage_sim):
        return SAYCamCndCont.get_cond_idx(self, _storage_sim)

    def cross_device_gather(self, arr, data_len):
        return SAYCamCndCont.cross_device_gather(self, arr, data_len)


@DATASOURCES.register_module
class ImageNetCndMeanSim(ImageNetCndCont):
    def __init__(
            self, mean_sim_metric,
            sample_p_func, sample_p_func_kwargs={},
            use_sparse_mask=False,
            real_data_len=None,
            *args, **kwargs):
        self.mean_sim_metric = mean_sim_metric
        self.sample_p_func = sample_p_func
        self.sample_p_func_kwargs = sample_p_func_kwargs
        self.use_sparse_mask = use_sparse_mask
        super().__init__(*args, **kwargs)
        self.real_data_len = real_data_len or self.data_len

    def get_storage_embds(self, *args, **kwargs):
        return SAYCamCndMeanSim.get_storage_embds(self, *args, **kwargs)

    def get_embds_from_enum(self, *args, **kwargs):
        return SAYCamCndMeanSim.get_embds_from_enum(self, *args, **kwargs)

    def build_loaders(self, *args, **kwargs):
        return SAYCamCndMeanSim.build_loaders(self, *args, **kwargs)

    def build_loaders_from_datasets(self, *args, **kwargs):
        return SAYCamCndMeanSim.build_loaders_from_datasets(self, *args, **kwargs)

    def get_embds_from_loader(self, *args, **kwargs):
        return SAYCamCndMeanSim.get_embds_from_loader(self, *args, **kwargs)

    def need_np_normalize(self):
        return self.use_sparse_mask

    def get_mean_sim(self, model):
        return SAYCamCndMeanSim.get_mean_sim(self, model)

    def get_sample_prob(self):
        return SAYCamCndMeanSim.get_sample_prob(self)

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        self.set_epoch(epoch)
        self.curr_epoch = epoch

        if self.cont_loader is None:
            self.build_loaders(cont_data_source)

        self.get_mean_sim(model)
        sample_prob = self.get_sample_prob()
        del self.mean_sim
        fns = self.cnd_loader.dataset.data_source.fns
        smpl_idx = np.random.choice(
                len(fns), self.real_data_len, 
                p=sample_prob)
        self.fns = fns[smpl_idx]


@DATASOURCES.register_module
class ImageNetCndLoss(ImageNetCndCont):
    def __init__(
            self, neg_img_nums=511, neg_sample_method='from_curr',
            loss_head=None, sample_batches=5, delta_loss=False,
            accu_lambda=None, sample_p_func=None, sample_p_func_kwargs={},
            real_data_len=None, forward_mode='test',
            *args, **kwargs):
        self.neg_img_nums = neg_img_nums
        self.neg_sample_method = neg_sample_method
        self.loss_head = model_builder.build_head(loss_head)
        self.loss_head = self.loss_head.cuda()
        self.sample_batches = sample_batches
        self.delta_loss = delta_loss
        self.sample_p_func = sample_p_func
        self.sample_p_func_kwargs = sample_p_func_kwargs
        self.accu_lambda = accu_lambda
        if self.delta_loss:
            self.prev_losses = 0
        if self.accu_lambda:
            self.accu_losses = 0
        self.forward_mode = forward_mode
        super().__init__(*args, **kwargs)
        self.real_data_len = real_data_len or self.data_len

    def build_loaders_from_datasets(self, *args, **kwargs):
        return SAYCamCndLoss.build_loaders_from_datasets(self, *args, **kwargs)

    def build_loaders(self, *args, **kwargs):
        return SAYCamCndLoss.build_loaders(self, *args, **kwargs)

    def get_embds_from_loader(self, *args, **kwargs):
        return SAYCamCndLoss.get_embds_from_loader(self, *args, **kwargs)

    def get_all_embds(self, model):
        return SAYCamCndLoss.get_all_embds(self, model)

    def reset_presampled_idxs(self, *args, **kwargs):
        return SAYCamCndLoss.reset_presampled_idxs(self, *args, **kwargs)

    def get_neg_smpl_idxs(self, *args, **kwargs):
        return SAYCamCndLoss.get_neg_smpl_idxs(self, *args, **kwargs)

    def get_neg_embds_for_one_batch(self):
        return SAYCamCndLoss.get_neg_embds_for_one_batch(self)

    def get_one_batch_loss(self, sta_idx, end_idx):
        return SAYCamCndLoss.get_one_batch_loss(self, sta_idx, end_idx)

    def compute_raw_loss_values(self, del_embds=True):
        return SAYCamCndLoss.compute_raw_loss_values(self, del_embds)

    def get_loss_values(self, model):
        return SAYCamCndLoss.get_loss_values(self, model)

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        self.set_epoch(epoch)
        self.curr_epoch = epoch

        if self.cont_loader is None:
            self.build_loaders(cont_data_source)

        loss_values = self.get_loss_values(model)

        fns = self.cnd_loader.dataset.data_source.fns
        p = self.sample_p_func(loss_values, **self.sample_p_func_kwargs)
        p = p / np.sum(p)
        smpl_idx = np.random.choice(
                len(fns), self.real_data_len, p=p)
        self.fns = fns[smpl_idx]


@DATASOURCES.register_module
class ImageNetVaryCnd(ImageNet):
    def __init__(
            self, prev_list_file, pipeline, 
            batch_size=128, cond_method='max',
            data_len=1281167 // 2,
            scale_ratio=5,
            *args, **kwargs):
        self.cnd_loader = None
        self.cont_loader = None
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.cond_method = cond_method
        self.prev_list_file = prev_list_file
        self.list_file = kwargs['list_file']
        self.root = kwargs['root']
        self.data_len = data_len
        self.scale_ratio = scale_ratio
        super().__init__(*args, **kwargs)

    def build_loaders(self):
        from ..extraction import ExtractDatasetWidx
        from openselfsup.datasets.loader.sampler import DistributedSampler

        base_data_source_cfg = dict(
                type='ImageNet',
                memcached=False,
                mclient_path=None,
                root=self.root, 
                )
        prev_cntxt_source = dict(list_file=self.prev_list_file)
        prev_cntxt_source.update(base_data_source_cfg)
        curr_cntxt_source = dict(list_file=self.list_file)
        curr_cntxt_source.update(base_data_source_cfg)
        
        prev_cntxt_source = build_datasource(prev_cntxt_source)
        prev_cntxt_source.fns = np.unique(prev_cntxt_source.fns)
        curr_cntxt_source = build_datasource(curr_cntxt_source)
        curr_cntxt_source.fns = np.unique(curr_cntxt_source.fns)

        prev_cntxt_dataset = ExtractDatasetWidx(prev_cntxt_source, self.pipeline)
        curr_cntxt_dataset = ExtractDatasetWidx(curr_cntxt_source, self.pipeline)

        rank, world_size = get_dist_info()
        prev_sampler = DistributedSampler(
                prev_cntxt_dataset, world_size, rank, 
                shuffle=False)
        self.cnd_loader = torch.utils.data.DataLoader(
                prev_cntxt_dataset,
                batch_size=self.batch_size,
                num_workers=10,
                sampler=prev_sampler)
        curr_sampler = DistributedSampler(
                curr_cntxt_dataset, world_size, rank, 
                shuffle=False)
        self.cont_loader = torch.utils.data.DataLoader(
                curr_cntxt_dataset,
                batch_size=self.batch_size,
                num_workers=10,
                sampler=curr_sampler)

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        np.random.seed(epoch)

        if self.cont_loader is None:
            self.build_loaders()

        storage_embds = SAYCamCndCont.get_storage_embds(self, model)
        self.fns = SAYCamCndCont.get_new_fns(self, model, storage_embds)
        np.random.seed(epoch)
        cont_fns = self.cont_loader.dataset.data_source.fns.copy()
        cont_fns = np.tile(cont_fns[:, np.newaxis], [1, self.scale_ratio])
        cont_fns = np.reshape(cont_fns, [-1])
        assert len(cont_fns) == len(self.fns)
        assert self.data_len == len(cont_data_source.fns)
        smpl_idx = np.random.choice(len(cont_fns), self.data_len)
        cont_data_source.fns = cont_fns[smpl_idx]
        self.fns = self.fns[smpl_idx]

    def get_cond_idx(self, _storage_sim):
        return SAYCamCndCont.get_cond_idx(self, _storage_sim)

    def cross_device_gather(self, arr, data_len):
        return SAYCamCndCont.cross_device_gather(self, arr, data_len)


@DATASOURCES.register_module
class ImageNetVaryCndLoss(ImageNetVaryCnd):
    def __init__(
            self, sample_loss_ratio=0.5, neg_img_nums=511, 
            loss_head=None, sample_batches=5, delta_loss=False,
            fix_loss_th=None, accu_lambda=None,
            sample_p_func=None, sample_p_func_kwargs={},
            *args, **kwargs):
        self.sample_loss_ratio = sample_loss_ratio
        self.fix_loss_th = fix_loss_th
        self.neg_img_nums = neg_img_nums
        self.loss_head = model_builder.build_head(loss_head)
        self.loss_head = self.loss_head.cuda()
        self.sample_batches = sample_batches
        self.delta_loss = delta_loss
        self.sample_p_func = sample_p_func
        self.sample_p_func_kwargs = sample_p_func_kwargs
        self.accu_lambda = accu_lambda
        if self.delta_loss:
            self.prev_losses = 0
        if self.accu_lambda:
            self.accu_losses = 0
        super().__init__(*args, **kwargs)

    def get_one_batch_loss(self, sta_idx, end_idx):
        pos_embd_0 = self.storage_embds[0][sta_idx:end_idx]
        pos_embd_1 = self.storage_embds[1][sta_idx:end_idx]
        pos = torch.sum(pos_embd_0 * pos_embd_1, dim=-1, keepdim=True)

        losses = 0
        for _ in range(self.sample_batches):
            neg_smpl_idxs = np.random.choice(
                    len(self.curr_embds[0]), self.neg_img_nums,
                    replace=False)
            neg_embds = self.curr_embds[:, neg_smpl_idxs, :]
            neg_embds = neg_embds.reshape(-1, neg_embds.size(-1))
            neg_embds = neg_embds.transpose(1, 0)
            neg0 = torch.matmul(pos_embd_0, neg_embds)
            neg1 = torch.matmul(pos_embd_1, neg_embds)
            _loss = self.loss_head(pos, neg0)['loss'] \
                    + self.loss_head(pos, neg1)['loss']
            _loss = _loss / 2
            losses = losses + _loss
        losses = losses / self.sample_batches
        return losses.detach().cpu()

    def get_all_embds(self):
        torch.manual_seed(0)
        storage_embds_0 = SAYCamCndCont.get_storage_embds(self, self.model)
        curr_embds_0 = SAYCamCndCont.get_storage_embds(
                self, self.model, self.cont_loader, 'Get Current Embeds')
        torch.manual_seed(1)
        storage_embds_1 = SAYCamCndCont.get_storage_embds(self, self.model)
        curr_embds_1 = SAYCamCndCont.get_storage_embds(
                self, self.model, self.cont_loader, 'Get Current Embeds')

        self.storage_embds = np.asarray([storage_embds_0, storage_embds_1])
        self.storage_embds = torch.from_numpy(self.storage_embds).cuda()
        self.curr_embds = np.asarray([curr_embds_0, curr_embds_1])
        self.curr_embds = torch.from_numpy(self.curr_embds).cuda()

    def compute_raw_loss_values(self):
        loss_values = []
        rank, world_size = get_dist_info()
        len_storage = len(self.storage_embds[0])
        to_enum = range(
                0, len_storage,
                self.batch_size)
        if rank == 0:
            to_enum = tqdm(to_enum, desc='Loss Computation')
        for sta_idx in to_enum:
            end_idx = min(sta_idx+self.batch_size, len_storage)
            _loss_value = self.get_one_batch_loss(sta_idx, end_idx)
            loss_values.append(_loss_value)
        del self.storage_embds
        del self.curr_embds
        loss_values = np.concatenate(loss_values)
        return loss_values

    def get_loss_values(self):
        self.get_all_embds()

        loss_values = self.compute_raw_loss_values()
        if self.delta_loss:
            loss_values = loss_values - self.prev_losses
            self.prev_losses = loss_values
        if self.accu_lambda:
            loss_values = self.accu_losses * self.accu_lambda + loss_values
            self.accu_losses = loss_values
        return loss_values

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        np.random.seed(epoch)
        self.model = model

        if self.cont_loader is None:
            self.build_loaders()

        loss_values = self.get_loss_values()
        if self.sample_p_func is None:
            if self.fix_loss_th is None:
                sort_loss_values = sorted(loss_values)
                loss_th_value = sort_loss_values[
                        int(len(sort_loss_values) * (1-self.sample_loss_ratio))]
            else:
                loss_th_value = self.fix_loss_th
            fns = self.cnd_loader.dataset.data_source.fns[loss_values > loss_th_value]
            p = None
        else:
            fns = self.cnd_loader.dataset.data_source.fns
            p = self.sample_p_func(loss_values, **self.sample_p_func_kwargs)
            p = p / np.sum(p)
        smpl_idx = np.random.choice(len(fns), self.data_len, p=p)
        self.fns = fns[smpl_idx]
        self.model = None


@DATASOURCES.register_module
class ImageNetVaryCndMeanSim(ImageNetVaryCnd):
    def __init__(
            self, mean_sim_metric,
            sample_p_func, sample_p_func_kwargs={},
            *args, **kwargs):
        self.mean_sim_metric = mean_sim_metric
        self.sample_p_func = sample_p_func
        self.sample_p_func_kwargs = sample_p_func_kwargs
        super().__init__(*args, **kwargs)

    def get_mean_sim(self, model):
        self.storage_embds = SAYCamCndCont.get_storage_embds(self, model)
        self.curr_embds = SAYCamCndCont.get_storage_embds(
                self, model, self.cont_loader, 'Get Current Embeds')
        self.mean_sim = self.compute_mean_sim(self.storage_embds, self.curr_embds)
        del self.storage_embds
        del self.curr_embds

    def compute_mean_sim(
            self, group_embds_A, group_embds_B, batch_size=128):
        # Mean of similarity to B from A (output in the same shape of A)
        metric = self.mean_sim_metric
        return compute_mean_sim(
                group_embds_A, group_embds_B, metric,
                batch_size=128)

    def get_sample_prob(self):
        sample_prob = self.sample_p_func(
                self.mean_sim, **self.sample_p_func_kwargs)
        sample_prob = sample_prob / np.sum(sample_prob)
        return sample_prob

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        np.random.seed(epoch)

        if self.cont_loader is None:
            self.build_loaders()

        self.get_mean_sim(model)
        sample_prob = self.get_sample_prob()
        del self.mean_sim
        fns = self.cnd_loader.dataset.data_source.fns
        smpl_idx = np.random.choice(len(fns), self.data_len, p=sample_prob)
        self.fns = fns[smpl_idx]


@DATASOURCES.register_module
class ImageNetVaryCndSparse(ImageNetVaryCnd):
    def __init__(
            self, sparse_rel_th=(0.1, 0.9),
            corr_method='dice',
            *args, **kwargs):
        self.sparse_rel_th = sparse_rel_th
        self.corr_method = corr_method
        super().__init__(*args, **kwargs)

    def get_one_batch_corr(self, sta_idx, end_idx):
        _storage_masks = self.storage_masks[sta_idx:end_idx]
        inter_res = torch.matmul(_storage_masks, self.curr_masks)
        if self.corr_method == 'cos':
            corr_values = inter_res
        elif self.corr_method == 'dice':
            sum_res = torch.sum(_storage_masks, dim=1, keepdims=True) + self.curr_masks_sum
            corr_values = inter_res * 2 / sum_res
        else:
            raise NotImplementedError
        corr_values = torch.mean(corr_values, dim=-1)
        return corr_values.detach().cpu()

    def get_sparse_corr(self, model):
        forward_mode = 'sparse_mask'
        storage_masks = SAYCamCndCont.get_storage_embds(
                self, model, tqdm_desc='Get Storage Masks',
                forward_mode=forward_mode)
        curr_masks = SAYCamCndCont.get_storage_embds(
                self, model, self.cont_loader, 'Get Current Masks',
                forward_mode=forward_mode)

        corr_values = []

        self.storage_masks = torch.from_numpy(storage_masks.astype(np.float)).cuda()
        self.curr_masks = torch.from_numpy(curr_masks.astype(np.float)).cuda()
        self.curr_masks = self.curr_masks.transpose(1, 0)
        self.curr_masks_sum = torch.sum(self.curr_masks, dim=0, keepdims=True)
        if self.corr_method == 'cos':
            self.storage_masks = \
                    self.storage_masks \
                    / torch.sqrt(torch.sum(
                        self.storage_masks, dim=-1, 
                        keepdims=True))
            self.curr_masks = self.curr_masks / torch.sqrt(self.curr_masks_sum)

        rank, world_size = get_dist_info()
        len_storage = len(storage_masks)
        to_enum = range(
                0, len_storage,
                self.batch_size)
        if rank == 0:
            to_enum = tqdm(to_enum, desc='Mask Corr Computation')
        for sta_idx in to_enum:
            end_idx = min(sta_idx+self.batch_size, len_storage)
            _corr_value = self.get_one_batch_corr(sta_idx, end_idx)
            corr_values.append(_corr_value)
        del self.storage_masks
        del self.curr_masks
        del self.curr_masks_sum
        return np.concatenate(corr_values)

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        np.random.seed(epoch)

        if self.cont_loader is None:
            self.build_loaders()

        sparse_corr = self.get_sparse_corr(model)
        sort_idx = np.argsort(sparse_corr)
        low_idx = int(len(sparse_corr) * self.sparse_rel_th[0])
        high_idx = int(len(sparse_corr) * self.sparse_rel_th[1])
        idx_to_use = sort_idx[low_idx : high_idx]
        fns = self.cnd_loader.dataset.data_source.fns[idx_to_use]
        smpl_idx = np.random.choice(len(fns), self.data_len)
        self.fns = fns[smpl_idx]


@DATASOURCES.register_module
class ImageNetCntAccuSY(ImageList):
    def __init__(
            self, num_epochs, keep_prev_epoch_num, 
            root, list_file, memcached, mclient_path, 
            sy_root, sy_epoch_meta_path, sy_file_num_meta_path, sy_end_epoch,
            data_len=None,
            ):
        super().__init__(
            root, list_file, memcached, mclient_path)
        self.raw_fns = copy.copy(self.fns)
        self.num_epochs = num_epochs
        self.keep_prev_epoch_num = keep_prev_epoch_num
        if data_len is None:
            self.data_len = len(self.raw_fns)
        else:
            self.data_len = data_len
        self.sy_root = sy_root
        self.sy_epoch_meta_path = sy_epoch_meta_path
        self.sy_file_num_meta_path = sy_file_num_meta_path
        self.sy_end_epoch = sy_end_epoch
        self.load_sy_metas()

    def load_sy_metas(self):
        self.sy_num_frames_meta = {}
        with open(self.sy_file_num_meta_path, 'r') as fin:
            all_lines = fin.readlines()
        for each_line in all_lines:
            video_name, num_frames = each_line.split(',')
            num_frames = int(num_frames)
            self.sy_num_frames_meta[video_name] = num_frames

        with open(self.sy_epoch_meta_path, 'r') as fin:
            all_lines = fin.readlines()
        self.sy_epoch_meta = [l.strip() for l in all_lines]

        sy_fns_up_to_end = []
        for _ep in range(self.sy_end_epoch):
            video_dirs = self.sy_epoch_meta[_ep].split(',')
            for _dir in video_dirs:
                frames = [
                        os.path.join(
                            self.sy_root, _dir, '%06i.jpg' % (_frame+1))
                        for _frame in range(self.sy_num_frames_meta[_dir])]
                sy_fns_up_to_end.extend(frames)
        np.random.seed(self.sy_end_epoch)
        self.sy_fns = np.random.choice(
                sy_fns_up_to_end, self.data_len, replace=True)

    def set_epoch(self, epoch):
        assert epoch >= self.sy_end_epoch
        np.random.seed(epoch)
        len_data = len(self.raw_fns)
        div_num = self.num_epochs + self.keep_prev_epoch_num
        start_idx = int(len_data * (self.sy_end_epoch * 1.0 / div_num))
        end_idx = int(len_data * ((epoch+self.keep_prev_epoch_num) \
                                  * 1.0 / div_num))
        if start_idx < end_idx:
            new_fns = self.raw_fns[start_idx : end_idx]
        else:
            new_fns = []
        new_fns = np.concatenate([new_fns, self.sy_fns])
        self.fns = np.random.choice(
                new_fns, self.data_len, replace=True)


@DATASOURCES.register_module
class ImageNetBatchLD(ImageList):
    def __init__(
            self, batch_size, no_cate_per_batch,
            root, list_file, memcached, mclient_path):
        super().__init__(
            root, list_file, memcached, mclient_path)
        self.raw_fns = np.asarray(copy.copy(self.fns))
        self.batch_size = batch_size
        self.no_cate_per_batch = no_cate_per_batch
        self.get_class_metas()
        self.set_epoch(0)

    def get_class_metas(self):
        class_raw_names = [
                os.path.basename(os.path.dirname(_fn)) 
                for _fn in self.raw_fns]
        class_raw_names = np.asarray(class_raw_names)
        unique_class_names = np.unique(class_raw_names)
        class_indexes = {}
        class_no_imgs = {}
        for each_class in unique_class_names:
            class_indexes[each_class] = set(
                    np.where(class_raw_names==each_class)[0])
            class_no_imgs[each_class] = len(class_indexes[each_class])

        self.class_indexes = class_indexes
        self.class_no_imgs = class_no_imgs

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        len_data = len(self.raw_fns)
        left_imgs = copy.deepcopy(self.class_no_imgs)
        left_indexes = copy.deepcopy(self.class_indexes)
        new_fns = []

        class_not_enough = 0
        img_not_enough = 0
        for _ in range(len_data // self.batch_size):
            all_possible_classes = list(left_imgs.keys())
            cls_replace = len(all_possible_classes) < self.no_cate_per_batch
            batch_classes = np.random.choice(
                    all_possible_classes, self.no_cate_per_batch,
                    replace=cls_replace)
            img_index = set([])
            for each_class in batch_classes:
                img_index = img_index.union(left_indexes[each_class])
            img_replace = len(img_index) < self.batch_size
            img_index = np.random.choice(
                    np.asarray(list(img_index)), 
                    self.batch_size, replace=img_replace)
            for each_class in batch_classes:
                left_indexes[each_class] \
                        = left_indexes[each_class].difference(img_index)
                left_imgs[each_class] = len(left_indexes[each_class])
                if left_imgs[each_class] == 0:
                    left_imgs.pop(each_class)

            if cls_replace:
                class_not_enough += 1
            if img_replace:
                img_not_enough += 1
            new_fns.append(img_index)
        new_fns = np.asarray(new_fns)
        new_fns = new_fns[np.random.permutation(new_fns.shape[0])]
        new_fns = new_fns.reshape([-1])
        self.fns = self.raw_fns[new_fns]
