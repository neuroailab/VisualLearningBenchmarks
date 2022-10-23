import numpy as np
import pdb
from .registry import FFCVLOADER, PIPELINES
from .contrastive_ffcv_loader import ContrastiveFFCVLoader
from torch.utils.data import DistributedSampler
from openselfsup.datasets.loader.sampler import DistributedSampler as self_DSampler
import openselfsup.models.builder as model_builder
from .data_sources import saycam
from .data_sources.utils import compute_mean_sim, np_l2_normalize
from openselfsup.framework.dist_utils import get_dist_info
from tqdm import tqdm
import copy
import torch


@FFCVLOADER.register_module
class SAYCamCtlCntrstFFCVLoader(ContrastiveFFCVLoader):
    def __init__(
            self, overall_len, one_epoch_img_num, 
            use_self_sampler=False,
            *args, **kwargs):
        self.one_epoch_img_num = one_epoch_img_num
        self.overall_len = overall_len
        self.use_self_sampler = use_self_sampler
        super().__init__(
                indices=range(self.one_epoch_img_num), #get dataset length right
                *args, **kwargs)
        self.set_epoch(0)

    def set_idxs_in_loaders(self, curr_idxs):
        for _loader in self.loaders:
            _loader.indices = curr_idxs
            _loader.traversal_order.indices = curr_idxs
            sampler_builder = DistributedSampler
            if self.use_self_sampler:
                sampler_builder = self_DSampler
            _loader.traversal_order.sampler = sampler_builder(
                    curr_idxs,
                    shuffle=False)

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        curr_idxs = np.random.choice(
                self.overall_len, self.one_epoch_img_num,
                replace=False)
        self.set_idxs_in_loaders(curr_idxs)


@FFCVLOADER.register_module
class SAYCamContCntrstFFCVLoader(SAYCamCtlCntrstFFCVLoader):
    def __init__(
            self, list_file, num_frames_meta_file,
            per_video_fps_range_rate=1,
            scale_total_epochs=None,
            per_seg_sample_nums=None,
            *args, **kwargs):
        with open(list_file, 'r') as f:
            lines = f.readlines()
            self.epoch_meta = [l.strip() for l in lines]
        self.num_frames_meta_file = num_frames_meta_file
        self.per_video_fps_range_rate = per_video_fps_range_rate
        self.per_seg_sample_nums = per_seg_sample_nums
        self.scale_epoch_meta(scale_total_epochs)
        self.load_num_frames_meta()
        self.load_start_indices()
        super().__init__(*args, **kwargs)
        assert self.start_indices[len(self.epoch_meta)] == self.overall_len, \
                "Meta or overall length might be wrong!"

    def scale_epoch_meta(self, scale_total_epochs):
        if scale_total_epochs is not None:
            assert isinstance(scale_total_epochs, int)
            new_epoch_meta = []
            for ep_idx in range(scale_total_epochs):
                prj_ep_idx = int((ep_idx/scale_total_epochs) * len(self.epoch_meta))
                new_epoch_meta.append(self.epoch_meta[prj_ep_idx])
            self.epoch_meta = np.asarray(new_epoch_meta)

    def load_num_frames_meta(self):
        self.num_frames_meta = {}
        with open(self.num_frames_meta_file, 'r') as fin:
            all_lines = fin.readlines()
        for each_line in all_lines:
            video_name, num_frames = each_line.split(',')
            num_frames = int(num_frames)
            if self.per_video_fps_range_rate > 0:
                num_frames += num_frames % self.per_video_fps_range_rate
                num_frames //= self.per_video_fps_range_rate
            else:
                num_frames -= len(range(0, num_frames, -self.per_video_fps_range_rate))
            self.num_frames_meta[video_name] = num_frames

    def load_start_indices(self):
        start_indices = {0: 0}
        curr_start_idx = 0
        included_dir = []
        length_maps = {}
        for idx in range(len(self.epoch_meta)):
            video_dirs = self.epoch_meta[idx].split(',')
            curr_len = 0
            for _dir in video_dirs:
                if _dir not in included_dir:
                    curr_start_idx += self.num_frames_meta[_dir]
                    included_dir.append(_dir)
                curr_len += self.num_frames_meta[_dir]
            start_indices[idx+1] = curr_start_idx
            length_maps[idx] = curr_len
        self.start_indices = start_indices
        self.length_maps = length_maps

    def get_curr_idxs(self, epoch):
        np.random.seed(epoch)
        sample_range = self.length_maps[epoch]
        if self.per_seg_sample_nums is not None:
            sample_range = np.arange(
                    0, self.length_maps[epoch],
                    self.length_maps[epoch] // self.per_seg_sample_nums)
        curr_idxs = np.random.choice(
                sample_range,
                self.one_epoch_img_num,
                replace=True)
        return curr_idxs

    def set_epoch(self, epoch):
        curr_idxs = self.get_curr_idxs(epoch)
        self.set_idxs_in_loaders(
                curr_idxs + self.start_indices[epoch+1]\
                          - self.length_maps[epoch])

    def set_epoch_unique(self):
        curr_idxs = SAYCamContCntrstFFCVLoader.get_curr_idxs(
                self, self.curr_epoch)
        unique_idxs, indices = np.unique(curr_idxs, return_inverse=True)
        self.set_idxs_in_loaders(
                unique_idxs + self.start_indices[self.curr_epoch+1]\
                            - self.length_maps[self.curr_epoch])
        return unique_idxs, indices


@FFCVLOADER.register_module
class SAYCamCotrainCntrstFFCVLoader(SAYCamContCntrstFFCVLoader):
    def get_curr_idxs(self, epoch):
        np.random.seed(epoch)
        sample_range = self.start_indices[epoch+1]
        if self.per_seg_sample_nums is not None:
            sample_range = []
            for _ep in range(0, epoch+1):
                if self.start_indices[_ep+1] > self.start_indices[_ep]:
                    curr_len = self.start_indices[_ep+1] - self.start_indices[_ep]
                    sample_range.append(
                            np.arange(
                                0, curr_len,
                                curr_len // self.per_seg_sample_nums)\
                            + self.start_indices[_ep])
            sample_range = np.concatenate(sample_range)
        curr_idxs = np.random.choice(
                sample_range,
                self.one_epoch_img_num,
                replace=True)
        return curr_idxs

    def set_epoch(self, epoch):
        curr_idxs = self.get_curr_idxs(epoch)
        self.set_idxs_in_loaders(curr_idxs)


class SAYCamCndBaseCntrstFFCVLoader(SAYCamCotrainCntrstFFCVLoader):
    def __init__(self, real_data_len=None, *args, **kwargs):
        self.real_data_len = real_data_len
        super().__init__(*args, **kwargs)

    def cross_device_gather(self, arr, data_len):
        return saycam.SAYCamCndCont.cross_device_gather(
                self, arr, data_len)

    def get_embds_from_loader(
            self, model, loader=None, 
            tqdm_desc='Get Storage Embds', num_crops=2,
            data_len=None):
        rank, world_size = get_dist_info()
        if loader is None:
            loader = self.cnd_loader
        if rank == 0:
            to_enum = tqdm(loader, desc=tqdm_desc)
        else:
            to_enum = loader
        all_embds = [[] for _ in range(num_crops)]
        model.eval()
        with torch.no_grad():
            for _idx, storage_batch in enumerate(to_enum):
                img = storage_batch['img'].cuda()
                img = img.reshape(
                        img.size(0) * img.size(1),
                        img.size(2), img.size(3), img.size(4))
                model_res = model(img=img, mode=self.forward_mode)
                curr_embd = model_res['embd'].reshape(
                        -1, num_crops, model_res['embd'].size(-1))
                curr_embd = curr_embd.detach().cpu().numpy()
                for crp_idx in range(num_crops):
                    all_embds[crp_idx].append(curr_embd[:, crp_idx])
                del storage_batch
                del model_res
        model.train()
        data_len = data_len or self.one_epoch_img_num
        for crp_idx in range(num_crops):
            all_embds[crp_idx] = self.cross_device_gather(
                    all_embds[crp_idx], data_len)
        return all_embds

    def sample_set_wrt_p(self, epoch, p):
        curr_idxs = self.get_curr_idxs(epoch)
        smpl_idx = np.random.choice(
                len(curr_idxs), 
                self.real_data_len or self.one_epoch_img_num, p=p)
        curr_idxs = curr_idxs[smpl_idx]
        self.set_idxs_in_loaders(curr_idxs)


@FFCVLOADER.register_module
class SAYCamCndLossCntrstFFCVLoader(SAYCamCndBaseCntrstFFCVLoader):
    def __init__(
            self, neg_img_nums=511, neg_sample_method='from_curr',
            loss_head=None, sample_batches=5, delta_loss=False,
            accu_lambda=None, sample_p_func=None, sample_p_func_kwargs={},
            forward_mode='test', embd_batch_size=512,
            *args, **kwargs):
        self.neg_img_nums = neg_img_nums
        self.neg_sample_method = neg_sample_method
        self.loss_head = model_builder.build_head(loss_head)
        self.loss_head = self.loss_head.cuda()
        self.sample_batches = sample_batches
        self.delta_loss = delta_loss
        self.sample_p_func = sample_p_func
        self.sample_p_func_kwargs = sample_p_func_kwargs
        self.forward_mode = forward_mode
        self.cont_loader = self
        self.cnd_loader = self
        self.batch_size = embd_batch_size
        super().__init__(*args, **kwargs)

    def get_all_embds(self, model):
        self.use_self_sampler = True
        unique_idxs, indices = SAYCamContCntrstFFCVLoader.set_epoch_unique(self)
        curr_embds = self.get_embds_from_loader(
                model, self.cont_loader, 'Get Current Embeds',
                data_len=len(unique_idxs))
        unique_curr_embds = np.asarray(curr_embds)
        curr_embds = unique_curr_embds[:, indices]

        SAYCamCotrainCntrstFFCVLoader.set_epoch(
                self, self.curr_epoch)
        storage_embds = self.get_embds_from_loader(model)
        self.use_self_sampler = False

        self.storage_embds = np.asarray(storage_embds)
        self.storage_embds = torch.from_numpy(self.storage_embds).cuda()
        self.curr_embds = np.asarray(curr_embds)
        self.curr_embds = torch.from_numpy(self.curr_embds).cuda()
        self.unique_curr_embds = np.asarray(unique_curr_embds)
        self.unique_curr_embds = torch.from_numpy(self.unique_curr_embds).cuda()

    def reset_presampled_idxs(
            self, *args, **kwargs):
        return saycam.SAYCamCndLoss.reset_presampled_idxs(self, *args, **kwargs)

    def get_neg_smpl_idxs(
            self, *args, **kwargs):
        return saycam.SAYCamCndLoss.get_neg_smpl_idxs(self, *args, **kwargs)

    def get_neg_embds_for_one_batch(self):
        return saycam.SAYCamCndLoss.get_neg_embds_for_one_batch(self)

    def get_one_batch_loss(self, sta_idx, end_idx):
        return saycam.SAYCamCndLoss.get_one_batch_loss(self, sta_idx, end_idx)

    def get_loss_values(self, model):
        self.get_all_embds(model)
        loss_values = saycam.SAYCamCndLoss.compute_raw_loss_values(self)
        return loss_values

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        self.curr_epoch = epoch
        loss_values = self.get_loss_values(model)
        p = self.sample_p_func(loss_values, **self.sample_p_func_kwargs)
        p = p / np.sum(p)
        self.sample_set_wrt_p(epoch, p)
        del self.unique_curr_embds


@FFCVLOADER.register_module
class SAYCamCndBothLossCntrst(SAYCamCndLossCntrstFFCVLoader):
    def __init__(
            self, curr_kwargs={}, update_batch_size=1024, 
            *args, **kwargs):
        self.curr_kwargs = curr_kwargs
        self.update_batch_size = update_batch_size
        super().__init__(*args, **kwargs)

    def get_loss_values(self, model):
        self.get_all_embds(model)
        loss_values = saycam.SAYCamCndLoss.compute_raw_loss_values(
                self, del_embds=False)
        return loss_values

    def get_curr_loss_values(self):
        # Tricky way to get the loss for the current images
        self.curr_embds = self.storage_embds
        self.storage_embds = self.unique_curr_embds
        loss_values_for_curr = saycam.SAYCamCndLoss.compute_raw_loss_values(
                self, del_embds=False)
        return loss_values_for_curr

    def update_kwargs(self):
        kwargs_to_recover = {}
        for key, value in self.curr_kwargs.items():
            if getattr(self, key, None) is not None:
                kwargs_to_recover[key] = getattr(self, key, None)
            setattr(self, key, value)
        self.kwargs_to_recover = kwargs_to_recover

    def recover_kwargs(self):
        for key, value in self.kwargs_to_recover.items():
            setattr(self, key, value)
        self.kwargs_to_recover = None

    def set_idx_for_curr_from_p(self, p, cont_data_source):
        cont_data_source.curr_epoch = self.curr_epoch
        back_up_data_len = cont_data_source.one_epoch_img_num
        cont_data_source.one_epoch_img_num = self.one_epoch_img_num
        unique_idxs, _ = cont_data_source.set_epoch_unique()
        unique_idxs += \
                cont_data_source.start_indices[
                    cont_data_source.curr_epoch+1]\
                - cont_data_source.length_maps[
                    cont_data_source.curr_epoch]
        new_idxs = []
        for sta_idx in range(
                0, back_up_data_len, 
                self.update_batch_size):
            end_idx = min(back_up_data_len, 
                          sta_idx+self.update_batch_size)
            _idxs = np.random.choice(
                    unique_idxs,
                    end_idx - sta_idx, p=p, replace=False)
            new_idxs.append(_idxs)
        new_idxs = np.concatenate(new_idxs)
        cont_data_source.set_idxs_in_loaders(new_idxs)
        cont_data_source.one_epoch_img_num = back_up_data_len

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        self.curr_epoch = epoch
        loss_values = self.get_loss_values(model)
        p = self.sample_p_func(
                loss_values,
                **self.sample_p_func_kwargs)
        p = p / np.sum(p)
        self.sample_set_wrt_p(epoch, p)

        self.update_kwargs()
        loss_values_for_curr = self.get_curr_loss_values()
        p = self.sample_p_func(
                loss_values_for_curr,
                **self.sample_p_func_kwargs)
        p = p / np.sum(p)
        self.set_idx_for_curr_from_p(p, cont_data_source)
        self.recover_kwargs()
        del self.unique_curr_embds
        del self.curr_embds
        del self.storage_embds


@FFCVLOADER.register_module
class SAYCamCndGammaLoss(SAYCamCndLossCntrstFFCVLoader):
    def compute_raw_loss_values(self):
        return saycam.SAYCamCndLoss.compute_raw_loss_values(self)

    def get_embds_from_loader(
            self, model, loader=None, 
            tqdm_desc='Get Storage Embds', num_crops=2,
            data_len=None):
        rank, world_size = get_dist_info()
        if loader is None:
            loader = self.cnd_loader
        if rank == 0:
            to_enum = tqdm(loader, desc=tqdm_desc)
        else:
            to_enum = loader
        all_embds = [[] for _ in range(num_crops)]
        all_momentum_embds = [[] for _ in range(num_crops)]

        def parse_embd_append(model_res, all_embds_to_ap):
            curr_embd = model_res['embd'].reshape(
                    -1, num_crops, model_res['embd'].size(-1))
            curr_embd = curr_embd.detach().cpu().numpy()
            for crp_idx in range(num_crops):
                all_embds_to_ap[crp_idx].append(curr_embd[:, crp_idx])
            del model_res
            return all_embds_to_ap

        model.eval()
        with torch.no_grad():
            for _idx, storage_batch in enumerate(to_enum):
                img = storage_batch['img'].cuda()
                img = img.reshape(
                        img.size(0) * img.size(1),
                        img.size(2), img.size(3), img.size(4))
                all_embds = parse_embd_append(
                        model(img=img, mode='test'),
                        all_embds)
                all_momentum_embds = parse_embd_append(
                        model(img=img, mode='momentum_test'),
                        all_momentum_embds)
                del storage_batch
        model.train()
        data_len = data_len or self.one_epoch_img_num
        for crp_idx in range(num_crops):
            all_embds[crp_idx] = self.cross_device_gather(
                    all_embds[crp_idx], data_len)
            all_momentum_embds[crp_idx] = self.cross_device_gather(
                    all_momentum_embds[crp_idx], data_len)
        return all_embds, all_momentum_embds

    def get_all_embds(self, model):
        self.use_self_sampler = True
        unique_idxs, indices = SAYCamContCntrstFFCVLoader.set_epoch_unique(self)
        curr_embds, curr_mtm_embds = self.get_embds_from_loader(
                model, self.cont_loader, 'Get Current Embeds',
                data_len=len(unique_idxs))
        unique_curr_embds = np.asarray(curr_embds)
        curr_embds = unique_curr_embds[:, indices]
        curr_mtm_embds = np.asarray(curr_mtm_embds)[:, indices]

        SAYCamCotrainCntrstFFCVLoader.set_epoch(
                self, self.curr_epoch)
        storage_embds, storage_mtm_embds = self.get_embds_from_loader(model)
        self.use_self_sampler = False

        self.storage_embds = np.asarray(storage_embds)
        self.storage_embds = torch.from_numpy(self.storage_embds).cuda()
        self.storage_mtm_embds = np.asarray(storage_mtm_embds)
        self.storage_mtm_embds = torch.from_numpy(self.storage_mtm_embds).cuda()
        self.curr_embds = np.asarray(curr_embds)
        self.curr_embds = torch.from_numpy(self.curr_embds).cuda()
        self.curr_mtm_embds = np.asarray(curr_mtm_embds)
        self.curr_mtm_embds = torch.from_numpy(self.curr_mtm_embds).cuda()
        self.unique_curr_embds = np.asarray(unique_curr_embds)
        self.unique_curr_embds = torch.from_numpy(self.unique_curr_embds).cuda()

    def compute_gamma_loss_values(self):
        return saycam.SAYCamCndGammaLoss.compute_gamma_loss_values(self)

    def get_loss_values(self, model):
        self.get_all_embds(model)
        loss_values = self.compute_gamma_loss_values()
        return loss_values


@FFCVLOADER.register_module
class SAYCamCndMeanSimCntrstFFCVLoader(SAYCamCndBaseCntrstFFCVLoader):
    def __init__(
            self, mean_sim_metric,
            sample_p_func, sample_p_func_kwargs={},
            embd_batch_size=512,
            *args, **kwargs):
        self.mean_sim_metric = mean_sim_metric
        self.sample_p_func = sample_p_func
        self.sample_p_func_kwargs = sample_p_func_kwargs
        self.cont_loader = self
        self.cnd_loader = self
        self.batch_size = embd_batch_size
        self.forward_mode = 'test'
        super().__init__(*args, **kwargs)

    def get_mean_sim(self, model):
        self.use_self_sampler = True
        SAYCamCotrainCntrstFFCVLoader.set_epoch(
                self, self.curr_epoch)
        self.storage_embds = self.get_embds_from_loader(model)
        self.storage_embds = np.mean(self.storage_embds, axis=0)

        unique_idxs, _ = SAYCamContCntrstFFCVLoader.set_epoch_unique(self)
        self.curr_embds = self.get_embds_from_loader(
                model, self.cont_loader, 'Get Current Embeds',
                data_len=len(unique_idxs))
        self.curr_embds = np.mean(self.curr_embds, axis=0)
        self.use_self_sampler = False

        self.storage_embds = np_l2_normalize(self.storage_embds)
        self.curr_embds = np_l2_normalize(self.curr_embds)
        self.mean_sim = compute_mean_sim(
                self.storage_embds, self.curr_embds,
                metric=self.mean_sim_metric)
        self.mean_sim = np.minimum(self.mean_sim, 0.99)
        del self.storage_embds
        del self.curr_embds

    def get_sample_prob(self):
        sample_prob = self.sample_p_func(
                self.mean_sim, **self.sample_p_func_kwargs)
        sample_prob = sample_prob / np.sum(sample_prob)
        return sample_prob

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        self.curr_epoch = epoch

        self.get_mean_sim(model)
        sample_prob = self.get_sample_prob()
        del self.mean_sim
        self.sample_set_wrt_p(epoch, sample_prob)


@FFCVLOADER.register_module
class SAYCamCndMixFFCVLoader(SAYCamCndGammaLoss):
    def __init__(
            self, neg_img_nums=511, neg_sample_method='from_curr',
            loss_head=None, sample_batches=5,
            l_sample_p_func=None, l_sample_p_func_kwargs={},
            ms_metric=None, ms_sample_p_func=None, ms_sample_p_func_kwargs={},
            gmml_sub_w=None, gmml_sample_p_func=None, gmml_sample_p_func_kwargs={},
            ml_sample_p_func=None, ml_sample_p_func_kwargs={},
            embd_batch_size=512, mix_weights={},
            *args, **kwargs):
        self.ms_metric = ms_metric
        self.ms_sample_p_func = ms_sample_p_func
        self.ms_sample_p_func_kwargs = ms_sample_p_func_kwargs
        self.neg_img_nums = neg_img_nums
        self.neg_sample_method = neg_sample_method
        self.loss_head = model_builder.build_head(loss_head)
        self.loss_head = self.loss_head.cuda()
        self.sample_batches = sample_batches
        self.l_sample_p_func = l_sample_p_func
        self.l_sample_p_func_kwargs = l_sample_p_func_kwargs
        self.ml_sample_p_func = ml_sample_p_func
        self.ml_sample_p_func_kwargs = ml_sample_p_func_kwargs
        self.gmml_sub_w = gmml_sub_w
        self.gmml_sample_p_func = gmml_sample_p_func
        self.gmml_sample_p_func_kwargs = gmml_sample_p_func_kwargs
        self.mix_weights = mix_weights
        self.forward_mode = 'test'

        epsilon = 1e-5
        self.need_ms = np.max(np.abs(self.mix_weights['ms'])) > epsilon
        self.need_loss = np.max(np.abs(self.mix_weights['loss'])) > epsilon
        self.need_gmm_loss = np.max(np.abs(self.mix_weights['gmm_loss'])) > epsilon
        assert (self.need_ms or self.need_loss or self.need_gmm_loss),\
                "Need to have at least one policy used!"

        self.cont_loader = self
        self.cnd_loader = self
        self.batch_size = embd_batch_size
        SAYCamCndBaseCntrstFFCVLoader.__init__(self, *args, **kwargs)

    def clean_states(self):
        del self.unique_curr_embds
        del self.storage_embds
        del self.curr_embds
        if self.need_ms:
            del self.mean_sims
        if self.need_gmm_loss:
            del self.storage_mtm_embds
            del self.curr_mtm_embds

    def get_loss_values(self):
        np.random.seed(self.curr_epoch)
        self.loss_values = saycam.SAYCamCndLoss.compute_raw_loss_values(
                self, del_embds=False)

    def get_mtm_loss_values(self):
        back_storage_embds = self.storage_embds
        back_curr_embds = self.curr_embds
        self.storage_embds = self.storage_mtm_embds
        self.curr_embds = self.curr_mtm_embds
        np.random.seed(self.curr_epoch)
        self.mtm_loss_values = saycam.SAYCamCndLoss.compute_raw_loss_values(
                self, del_embds=False)
        self.storage_embds = back_storage_embds
        self.curr_embds = back_curr_embds

    def l2_normalize(self, z):
        z = z / (torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-10)
        return z

    def get_mean_sim(self):
        storage_embds = torch.mean(self.storage_embds, axis=0)
        curr_embds = torch.mean(self.unique_curr_embds, axis=0)
        storage_embds = self.l2_normalize(storage_embds)
        curr_embds = self.l2_normalize(curr_embds)
        assert isinstance(self.ms_metric, list)
        mean_sims = []
        cache_mean_sims = {}
        for _ms_metric in self.ms_metric:
            if _ms_metric in cache_mean_sims:
                _mean_sim = cache_mean_sims[_ms_metric]
            else:
                _mean_sim = compute_mean_sim(
                        storage_embds, curr_embds,
                        metric=_ms_metric)
                _mean_sim = np.minimum(_mean_sim, 0.99)
                cache_mean_sims[_ms_metric] = _mean_sim
            mean_sims.append(_mean_sim)
        self.mean_sims = mean_sims
        del storage_embds
        del curr_embds

    def get_one_prob(self, metric, p_func, p_func_kwargs):
        p = p_func(metric, **p_func_kwargs)
        if np.min(p) < 0:
            p = p - np.min(p)
        p = p / np.sum(p)
        return p

    def get_gamma_loss_values(self):
        self.get_mtm_loss_values()
        self.gamma_loss_values = self.mtm_loss_values - self.loss_values * self.gmml_sub_w

    def get_mean_sim_prob(self):
        sample_prob = 0
        assert isinstance(self.ms_sample_p_func, list)
        assert isinstance(self.ms_sample_p_func_kwargs, list)
        assert isinstance(self.mix_weights['ms'], list)
        for _mean_sim, _func, _kwargs, _w in\
                zip(self.mean_sims, self.ms_sample_p_func, 
                    self.ms_sample_p_func_kwargs, self.mix_weights['ms']):
            ms_prob = self.get_one_prob(_mean_sim, _func, _kwargs)
            sample_prob += ms_prob * _w
        return sample_prob

    def get_loss_prob(self):
        sample_prob = 0
        assert isinstance(self.l_sample_p_func, list)
        assert isinstance(self.l_sample_p_func_kwargs, list)
        assert isinstance(self.mix_weights['loss'], list)
        for _func, _kwargs, _w in zip(
                self.l_sample_p_func, self.l_sample_p_func_kwargs,
                self.mix_weights['loss']):
            l_prob = self.get_one_prob(self.loss_values, _func, _kwargs)
            sample_prob += l_prob * _w
        return sample_prob

    def get_gamma_loss_prob(self):
        sample_prob = 0
        assert isinstance(self.gmml_sample_p_func, list)
        assert isinstance(self.gmml_sample_p_func_kwargs, list)
        assert isinstance(self.mix_weights['gmm_loss'], list)
        for _func, _kwargs, _w in zip(
                self.gmml_sample_p_func, self.gmml_sample_p_func_kwargs,
                self.mix_weights['gmm_loss']):
            gmml_prob = self.get_one_prob(self.gamma_loss_values, _func, _kwargs)
            sample_prob += gmml_prob * _w
        return sample_prob

    def get_sample_prob(self):
        sample_prob = 0
        if self.need_ms:
            self.get_mean_sim()
            sample_prob = sample_prob + self.get_mean_sim_prob()
        if (self.need_loss or self.need_gmm_loss):
            self.get_loss_values()
            if self.need_loss:
                sample_prob = sample_prob + self.get_loss_prob()
            if self.need_gmm_loss:
                self.get_gamma_loss_values()
                sample_prob = sample_prob + self.get_gamma_loss_prob()
        sample_prob /= np.sum(sample_prob)
        return sample_prob

    def get_all_embds(self, model):
        if self.need_gmm_loss:
            SAYCamCndGammaLoss.get_all_embds(self, model)
        else:
            SAYCamCndLossCntrstFFCVLoader.get_all_embds(self, model)

    def get_embds_from_loader(self, *args, **kwargs):
        if self.need_gmm_loss:
            return SAYCamCndGammaLoss.get_embds_from_loader(self, *args, **kwargs)
        else:
            return SAYCamCndLossCntrstFFCVLoader.get_embds_from_loader(self, *args, **kwargs)

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        self.curr_epoch = epoch

        self.get_all_embds(model)
        sample_prob = self.get_sample_prob()
        self.sample_set_wrt_p(epoch, sample_prob)
        self.clean_states()
