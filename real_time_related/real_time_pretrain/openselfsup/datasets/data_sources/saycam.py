from ..registry import DATASOURCES
from .image_list import ImageList
import copy
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm
import pdb

from .utils import compute_mean_sim, np_l2_normalize
from openselfsup.framework.dist_utils import get_dist_info, gather_tensors_batch
import openselfsup.models.builder as model_builder
PERSISTENT_WORKERS = int(os.environ.get('PERSISTENT_WORKERS', 0))==1

FPS_RATE = 25

@DATASOURCES.register_module
class SAYCam(ImageList):
    def __init__(
            self, root, list_file, batch_size, all_random=False,
            set_len=None,
            **kwargs):
        super().__init__(
            root, list_file, 
            **kwargs)
        self.base_fns = copy.deepcopy(self.fns)
        self.batch_size = batch_size
        self.all_random = all_random
        self.set_len = set_len
        self.set_epoch(0)

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        _, world_size = get_dist_info()
        assert self.batch_size % world_size == 0
        sub_batch_size = self.batch_size // world_size
        fns = []
        curr_order = np.random.permutation(self.base_fns)

        for _curr_start_frame in curr_order:
            frame_idx = os.path.basename(_curr_start_frame)
            frame_idx = frame_idx.split('.')[0]
            sta_frame_idx = int(frame_idx) + np.random.randint(FPS_RATE)
            all_frame_idxs = [
                    sta_frame_idx + _batch_idx*FPS_RATE \
                    for _batch_idx in range(self.batch_size)]
            frame_dir = os.path.dirname(_curr_start_frame)
            all_frame_path = [
                    os.path.join(frame_dir, '%06i.jpg' % _frame_idx) \
                    for _frame_idx in all_frame_idxs]
            all_frame_path = np.asarray(all_frame_path)
            # To address the distributed sampler
            all_frame_path = np.transpose(
                    all_frame_path.reshape([world_size, sub_batch_size]),
                    [1, 0]).reshape(-1)
            fns.append(all_frame_path)
        self.fns = np.concatenate(fns)
        if self.all_random:
            self.fns = np.random.permutation(self.fns)
        if self.set_len is not None:
            self.fns = self.fns[:self.set_len]


@DATASOURCES.register_module
class SAYCamTwoImage(SAYCam):
    def set_epoch(self, epoch):
        super().set_epoch(epoch)

        self.fns1 = self.fns
        fns2 = []
        for each_jpg in self.fns1:
            frame_dir = os.path.dirname(each_jpg)
            frame_idx = int(os.path.basename(each_jpg).split('.')[0])
            frame_idx += 2
            new_jpg = os.path.join(frame_dir, '%06i.jpg' % frame_idx)
            fns2.append(new_jpg)
        self.fns2 = np.asarray(fns2)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img1 = self.mc_loader(self.fns1[idx])
            img2 = self.mc_loader(self.fns2[idx])
        else:
            img1 = Image.open(self.fns1[idx])
            img2 = Image.open(self.fns2[idx])
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')
        assert not self.has_labels
        return img1, img2


@DATASOURCES.register_module
class SAYCamTwoImageRandom(SAYCamTwoImage):
    def set_epoch(self, epoch):
        SAYCam.set_epoch(self, epoch)

        if getattr(self, 'cache_dir_content', None) is None:
            self.cache_dir_content = {}

        self.fns1 = self.fns
        fns2 = []
        for each_jpg in self.fns1:
            frame_dir = os.path.dirname(each_jpg)
            if frame_dir not in self.cache_dir_content:
                self.cache_dir_content[frame_dir] = len(os.listdir(frame_dir))
            jpg_len = self.cache_dir_content[frame_dir]
            new_jpg = os.path.join(
                    frame_dir, 
                    '%06i.jpg' % (np.random.randint(jpg_len)+1))
            fns2.append(new_jpg)
        self.fns2 = np.asarray(fns2)


@DATASOURCES.register_module
class SAYCamCont(ImageList):
    def __init__(
            self, root, list_file, num_frames_meta_file,
            one_epoch_img_num=1281167,
            per_video_sample_rate=None,
            reverse_epoch_metas=False,
            randomize_epoch_metas=False,
            scale_total_epochs=None,
            group_metas=None,
            per_video_fps_range_rate=1,
            sort_fns=False,
            learn_window_size=None,
            aggre_window=None,
            **kwargs):
        super().__init__(
            '', list_file, 
            **kwargs)
        self.video_root = root
        self.epoch_meta = copy.deepcopy(self.fns)
        if reverse_epoch_metas:
            self.epoch_meta = list(reversed(self.epoch_meta))
        if randomize_epoch_metas:
            np.random.seed(0)
            self.epoch_meta = np.random.permutation(self.epoch_meta)
        if scale_total_epochs is not None:
            assert isinstance(scale_total_epochs, int)
            new_epoch_meta = []
            for ep_idx in range(scale_total_epochs):
                prj_ep_idx = int((ep_idx/scale_total_epochs) * len(self.epoch_meta))
                new_epoch_meta.append(self.epoch_meta[prj_ep_idx])
            self.epoch_meta = np.asarray(new_epoch_meta)
        self.num_frames_meta_file = num_frames_meta_file
        self.one_epoch_img_num = one_epoch_img_num
        self.per_video_sample_rate = per_video_sample_rate
        self.group_metas = group_metas
        self.per_video_fps_range_rate = per_video_fps_range_rate
        self.sort_fns = sort_fns
        self.learn_window_size = learn_window_size
        self.aggre_window = aggre_window
        self.load_num_frames_meta()
        self.set_epoch(0)

    def load_num_frames_meta(self):
        self.num_frames_meta = {}
        with open(self.num_frames_meta_file, 'r') as fin:
            all_lines = fin.readlines()
        for each_line in all_lines:
            video_name, num_frames = each_line.split(',')
            num_frames = int(num_frames)
            self.num_frames_meta[video_name] = num_frames
            if self.per_video_sample_rate is not None:
                self.num_frames_meta[video_name] = num_frames // self.per_video_sample_rate
            if '/' in video_name:
                self.num_frame_meta_key_type = 'saycam'
            else:
                self.num_frame_meta_key_type = 'ego4d'

    def get_base_fns(self, epoch):
        base_fns = []
        video_dirs = self.epoch_meta[epoch].split(',')
        for _dir in video_dirs:
            frames = [
                    os.path.join(self.video_root, _dir, '%06i.jpg' % (_frame+1))
                    for _frame in range(
                        0, self.num_frames_meta[_dir],
                        self.per_video_fps_range_rate)]
            base_fns.extend(frames)
        return base_fns

    def get_sample(self, idx):
        if self.aggre_window is None:
            return super().get_sample(idx)
        else:
            each_jpg = self.fns[idx]
            frame_dir = os.path.dirname(each_jpg)
            frame_idx = int(os.path.basename(each_jpg).split('.')[0])
            frame_idx_min = max(0, frame_idx - self.aggre_window)
            if self.num_frame_meta_key_type == 'saycam':
                _dir = os.path.join(
                        os.path.basename(os.path.dirname(frame_dir)),
                        os.path.basename(frame_dir))
            else:
                _dir = os.path.basename(frame_dir)
            frame_idx_max = min(
                    self.num_frames_meta[_dir],
                    frame_idx + self.aggre_window)
            frame_idx_2 = np.random.randint(
                    low=frame_idx_min, high=frame_idx_max)
            frame_2 = os.path.join(frame_dir, f'{frame_idx_2 + 1:06}.jpg')
            img_1 = self.load_one_img_from_path(self.fns[idx])
            img_2 = self.load_one_img_from_path(frame_2)
            return img_1, img_2

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        if self.group_metas is None:
            base_fns = self.get_base_fns(epoch)
        else:
            base_fns = []
            start_epoch = epoch - epoch % self.group_metas
            end_epoch = min(start_epoch + self.group_metas, len(self.epoch_meta))
            for _ep in range(start_epoch, end_epoch):
                base_fns.extend(self.get_base_fns(_ep))

        if self.learn_window_size is not None:
            learn_window_size = min(self.learn_window_size, len(base_fns))
            all_rel_pos = np.random.uniform(size=self.one_epoch_img_num)
            fns = []
            len_base = len(base_fns)
            for idx, _rel_pos in enumerate(all_rel_pos):
                base_pos = int(idx / self.one_epoch_img_num * len_base)
                sta_idx = max(0, base_pos - learn_window_size // 2)
                end_idx = min(len_base, base_pos + learn_window_size // 2)
                curr_idx = int((end_idx - sta_idx) * _rel_pos + sta_idx)
                fns.append(base_fns[curr_idx])
            self.fns = fns
        else:
            self.fns = np.random.choice(
                    base_fns, self.one_epoch_img_num, replace=True)
            if self.sort_fns:
                self.fns = np.sort(self.fns)


@DATASOURCES.register_module
class SAYCamContAccu(SAYCamCont):
    def __init__(
            self, accu_range=None, strict_accu=False, 
            **kwargs):
        self.accu_range = accu_range
        self.strict_accu = strict_accu
        self.all_fns = {}
        super().__init__(**kwargs)

    def get_base_fns(self, epoch):
        base_fns = []
        start_epoch = 0
        if self.accu_range is not None:
            start_epoch = max(0, epoch+1-self.accu_range)
        end_epoch = epoch+1
        if self.strict_accu:
            end_epoch = max(1, epoch)
        for _tmp_epoch in range(start_epoch, end_epoch):
            if _tmp_epoch not in self.all_fns:
                self.all_fns[_tmp_epoch] = SAYCamCont.get_base_fns(
                        self, _tmp_epoch)
            base_fns.append(self.all_fns[_tmp_epoch])
        base_fns = np.concatenate(base_fns)
        return base_fns


@DATASOURCES.register_module
class SAYCamContAccuMix(SAYCamCont):
    def __init__(
            self, 
            cont_one_epoch_img_num,
            accu_one_epoch_img_num,
            **kwargs):
        kwargs['one_epoch_img_num'] = cont_one_epoch_img_num
        self.cont_source = SAYCamCont(**kwargs)
        if 'learn_window_size' in kwargs:
            kwargs.pop('learn_window_size')
        kwargs['one_epoch_img_num'] = accu_one_epoch_img_num
        self.accu_source = SAYCamContAccu(**kwargs)
        super().__init__(**kwargs)

    def set_epoch(self, epoch):
        self.cont_source.set_epoch(epoch)
        cont_fns = self.cont_source.fns
        self.accu_source.set_epoch(epoch)
        accu_fns = self.accu_source.fns

        final_num = len(cont_fns) + len(accu_fns)
        insert_pos = np.random.choice(
                final_num, len(cont_fns), replace=False)
        insert_pos = sorted(insert_pos)
        fns = []
        insert_pos_idx = 0
        cont_idx = 0
        accu_idx = 0
        for idx in range(final_num):
            if (insert_pos_idx < len(insert_pos))\
                    and (idx == insert_pos[insert_pos_idx]):
                fns.append(cont_fns[cont_idx])
                cont_idx += 1
                insert_pos_idx += 1
            else:
                fns.append(accu_fns[accu_idx])
                accu_idx += 1
        self.fns = fns


@DATASOURCES.register_module
class SAYCamCndCont(SAYCamContAccu):
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

    def build_loaders_from_datasets(
            self, cnd_dataset, cont_dataset):
        from openselfsup.datasets.loader.sampler import DistributedSampler
        rank, world_size = get_dist_info()
        cnd_sampler = DistributedSampler(
                cnd_dataset, world_size, rank, 
                shuffle=False)
        num_workers = int(os.environ.get('NUM_WORKERS', 16))
        self.cnd_loader = torch.utils.data.DataLoader(
                cnd_dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                sampler=cnd_sampler,
                persistent_workers=PERSISTENT_WORKERS,
                )
        cont_sampler = DistributedSampler(
                cont_dataset, world_size, rank, 
                shuffle=False)
        self.cont_loader = torch.utils.data.DataLoader(
                cont_dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                sampler=cont_sampler,
                persistent_workers=PERSISTENT_WORKERS,
                )

    def build_loaders(self, cont_data_source):
        from ..extraction import ExtractDatasetWidx
        cnd_dataset = ExtractDatasetWidx(self, self.pipeline)
        cont_dataset = ExtractDatasetWidx(cont_data_source, self.pipeline)

        self.build_loaders_from_datasets(cnd_dataset, cont_dataset)

        #assert len(cnd_dataset) % len(cont_dataset) == 0
        self.scale_ratio = len(cnd_dataset) // len(cont_dataset)

    def cross_device_gather(self, arr, data_len):
        _, world_size = get_dist_info()
        if world_size > 1:
            arr = np.concatenate(arr, axis=0)
            arr = gather_tensors_batch(arr, part_size=20)
        arr = np.concatenate(arr, axis=0)
        assert len(arr) >= data_len, \
                "{} is smaller than {}".format(len(arr), data_len)
        arr = arr[:data_len]
        return arr

    def get_embds_from_enum(
            self, model, to_enum, forward_mode):
        all_embds = []
        all_idxs = []
        model.eval()
        with torch.no_grad():
            for _idx, storage_batch in enumerate(to_enum):
                storage_batch_cp = copy.deepcopy(storage_batch)
                del storage_batch
                all_idxs.append(storage_batch_cp['idx'].cpu().numpy())
                img = storage_batch_cp['img'].cuda()
                model_res = model(img=img, mode=forward_mode)
                all_embds.append(
                        model_res['embd'].detach().cpu().numpy())
                del storage_batch_cp
                del model_res
        model.train()
        return all_embds, all_idxs

    def get_storage_embds(
            self, model,
            loader=None, tqdm_desc='Get Storage Embds',
            forward_mode='test'):
        rank, world_size = get_dist_info()
        if loader is None:
            loader = self.cnd_loader
        if rank == 0:
            to_enum = tqdm(loader, desc=tqdm_desc)
        else:
            to_enum = loader
        all_embds, all_idxs = self.get_embds_from_enum(
                model, to_enum, forward_mode)
        data_len = len(loader.dataset)
        all_idxs = self.cross_device_gather(all_idxs, data_len)
        all_embds = self.cross_device_gather(all_embds, data_len)
        assert np.allclose(all_idxs, np.arange(data_len))
        return all_embds

    def get_cond_idx(self, _storage_sim):
        if self.cond_method == 'max':
            assert self.scale_ratio == 1
            _fn_idx = torch.argmax(_storage_sim, dim=-1)
        elif self.cond_method == 'min':
            assert self.scale_ratio == 1
            _fn_idx = torch.argmax(-_storage_sim, dim=-1)
        elif self.cond_method.startswith('max_'):
            _, _fn_idx = torch.topk(
                    _storage_sim, k=int(self.cond_method[4:]), 
                    largest=True, sorted=True,
                    dim=-1)
            if self.scale_ratio == 1:
                _fn_idx = _fn_idx[:, -1]
            else:
                _fn_idx = _fn_idx[:, -self.scale_ratio:]
                _fn_idx = _fn_idx.reshape(-1)
        elif self.cond_method.startswith('sum_max_'):
            _, _fn_idx = torch.topk(
                    torch.sum(_storage_sim, dim=0), 
                    k=int(self.cond_method[8:]), 
                    largest=True, sorted=True,
                    dim=-1)
            _fn_idx = _fn_idx[-_storage_sim.size(0)*self.scale_ratio:]
        elif self.cond_method.startswith('range_max_'):
            assert self.scale_ratio > 1
            large_k = int(self.cond_method.split('_')[-1])
            small_k = int(self.cond_method.split('_')[-2])
            _, _fn_idx = torch.topk(
                    _storage_sim, k=large_k, 
                    largest=True, sorted=True,
                    dim=-1)
            _fn_idx = _fn_idx[:, small_k::((large_k-small_k)//self.scale_ratio)]
            _fn_idx = _fn_idx.reshape(-1)
        elif self.cond_method.startswith('rand_corr_'):
            large_c = float(self.cond_method.split('_')[-1])
            small_c = float(self.cond_method.split('_')[-2])
            _between_flags = torch.logical_and(
                    _storage_sim > small_c,
                    _storage_sim < large_c).float()
            rand_idx = torch.randperm(_between_flags.size(1))
            _between_flags = _between_flags[:, rand_idx]
            if getattr(self, 'added_noise', None) is None:
                self.added_noise = 0.5 * torch.rand(
                        _between_flags.size(0), _between_flags.size(1)).cuda()
            _between_flags = _between_flags + self.added_noise[:_between_flags.size(0)]
            _, _fn_idx = torch.topk(
                    _between_flags, k=self.scale_ratio,
                    largest=True, sorted=False,
                    dim=-1)
            _fn_idx = _fn_idx.reshape(-1)
            _fn_idx = rand_idx[_fn_idx]
        elif self.cond_method.startswith('sum_range_max_'):
            large_k = int(self.cond_method.split('_')[-1])
            small_k = int(self.cond_method.split('_')[-2])
            _, _fn_idx = torch.topk(
                    torch.sum(_storage_sim, dim=0), 
                    k=large_k,
                    largest=True, sorted=True,
                    dim=-1)
            want_len = _storage_sim.size(0)*self.scale_ratio
            _fn_idx = _fn_idx[small_k::((large_k-small_k)//want_len)]
        elif self.cond_method.startswith('groupsum_range_max_'):
            large_k = int(self.cond_method.split('_')[-1])
            small_k = int(self.cond_method.split('_')[-2])
            group_no = int(self.cond_method.split('_')[-3])
            org_len = _storage_sim.size(0)
            if org_len % group_no != 0:
                _storage_sim = torch.cat(
                        [_storage_sim, torch.flip(_storage_sim, dims=[0])], 
                        dim=0)
                _storage_sim = _storage_sim[:(org_len // group_no + 1)*group_no]
            _storage_sim = _storage_sim.reshape(
                    -1, group_no, _storage_sim.size(-1))
            _, _fn_idx = torch.topk(
                    torch.sum(_storage_sim, dim=1), 
                    k=large_k,
                    largest=True, sorted=True,
                    dim=-1)
            want_len = group_no * self.scale_ratio
            _fn_idx = _fn_idx[:, small_k::((large_k-small_k)//want_len)]
            _fn_idx = _fn_idx.reshape(-1)
            if org_len % group_no != 0:
                _fn_idx = _fn_idx[:org_len]
        else:
            raise NotImplementedError
        return _fn_idx

    def get_new_fns(self, model, storage_embds):
        storage_embds = torch.from_numpy(storage_embds).cuda()
        storage_embds = storage_embds.permute(1, 0)
        rank, world_size = get_dist_info()
        if rank == 0:
            to_enum = tqdm(self.cont_loader, desc='Get New Fns')
        else:
            to_enum = self.cont_loader
        fn_idxs = []
        all_idxs = []
        model.eval()
        with torch.no_grad():
            for _idx, cont_batch in enumerate(to_enum):
                all_idxs.append(cont_batch['idx'].cpu().numpy())
                img = cont_batch['img'].cuda()
                _embd = model(img=img, mode='test')['embd'].detach().cuda()
                _storage_sim = torch.matmul(_embd, storage_embds)
                _fn_idx = self.get_cond_idx(_storage_sim)
                if not isinstance(_fn_idx, np.ndarray):
                    _fn_idx = _fn_idx.detach().cpu().numpy()
                fn_idxs.append(_fn_idx)
                del img
        model.train()
        data_len = len(self.cont_loader.dataset)
        all_idxs = self.cross_device_gather(all_idxs, data_len)
        fn_idxs = self.cross_device_gather(fn_idxs, data_len * self.scale_ratio)
        assert np.allclose(all_idxs, np.arange(data_len))
        fns = self.cnd_loader.dataset.data_source.fns[fn_idxs.astype(int)]
        if getattr(self, 'added_noise', None) is not None:
            del self.added_noise
            self.added_noise = None
        return fns

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        self.set_epoch(epoch)

        if self.cont_loader is None:
            self.build_loaders(cont_data_source)

        storage_embds = self.get_storage_embds(model)
        self.fns = self.get_new_fns(model, storage_embds)


@DATASOURCES.register_module
class SAYCamCndMeanSim(SAYCamCndCont):
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
        self.real_data_len = real_data_len
        super().__init__(*args, **kwargs)

    def need_np_normalize(self):
        return self.use_sparse_mask

    def get_mean_sim(self, model):
        embd_kwargs = {}
        if self.use_sparse_mask:
            embd_kwargs = {
                    'forward_mode': 'sparse_mask'}
        self.storage_embds = self.get_storage_embds(model, **embd_kwargs)
        self.curr_embds = self.get_storage_embds(
                model, self.cont_loader, 
                'Get Current Embeds', **embd_kwargs)
        _, unique_idx = np.unique(
                self.cont_loader.dataset.data_source.fns,
                return_index=True)
        self.curr_embds = self.curr_embds[unique_idx]
        if self.need_np_normalize():
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
        SAYCamContAccu.set_epoch(self, epoch)

        if self.cont_loader is None:
            self.build_loaders(cont_data_source)

        self.get_mean_sim(model)
        sample_prob = self.get_sample_prob()
        del self.mean_sim
        smpl_idx = np.random.choice(
                len(self.fns), self.real_data_len or self.one_epoch_img_num, 
                p=sample_prob)
        self.fns = self.fns[smpl_idx]


@DATASOURCES.register_module
class SAYCamCndMeanSimSpLayer(SAYCamCndMeanSim):
    def __init__(
            self, layer_name,
            *args, **kwargs):
        self.layer_name = layer_name
        super().__init__(*args, **kwargs)

    def need_np_normalize(self):
        return True

    def get_layer(self, model, layer_name):
        module = model.module
        for part in layer_name.split('.'):
            module = module._modules.get(part)
            assert module is not None, \
                    f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def register_hook(self, model):
        self.target_dict = {}
        hooks = []
        layer_name = self.layer_name
        layer = self.get_layer(model, layer_name)
        hooks.append(
                self.register_one_hook(
                    layer, layer_name, self.target_dict))
        return hooks

    def register_one_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            output_bool = output > 0
            target_dict[name] = output_bool.cpu().data.numpy()

        hook = layer.register_forward_hook(hook_function)
        return hook

    def get_embds_from_enum(
            self, model, to_enum, forward_mode):
        all_embds = []
        all_idxs = []
        model.eval()
        hooks = self.register_hook(model)
        with torch.no_grad():
            for _idx, storage_batch in enumerate(to_enum):
                storage_batch_cp = copy.deepcopy(storage_batch)
                del storage_batch
                all_idxs.append(storage_batch_cp['idx'].cpu().numpy())
                img = storage_batch_cp['img'].cuda()
                model_res = model(img=img, mode=forward_mode)
                all_embds.append(self.target_dict[self.layer_name])
                del storage_batch_cp
                del model_res
        model.train()
        for hook in hooks:
            hook.remove()
        return all_embds, all_idxs


@DATASOURCES.register_module
class SAYCamCndLoss(SAYCamCndCont):
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
        self.real_data_len = real_data_len or self.one_epoch_img_num

    def build_loaders(self, cont_data_source):
        from ..extraction import ExtractDatasetWidx
        cnd_dataset = ExtractDatasetWidx(self, self.pipeline, num_crops=2)
        cont_dataset = ExtractDatasetWidx(
                cont_data_source, self.pipeline, num_crops=2)
        self.build_loaders_from_datasets(cnd_dataset, cont_dataset)

    def get_embds_from_loader(
            self, model, loader=None, 
            tqdm_desc='Get Storage Embds', num_crops=2):
        rank, world_size = get_dist_info()
        if loader is None:
            loader = self.cnd_loader
        if rank == 0:
            to_enum = tqdm(loader, desc=tqdm_desc)
        else:
            to_enum = loader
        all_embds = [[] for _ in range(num_crops)]
        all_idxs = []
        model.eval()
        with torch.no_grad():
            for _idx, storage_batch in enumerate(to_enum):
                storage_batch_cp = copy.deepcopy(storage_batch)
                del storage_batch
                all_idxs.append(storage_batch_cp['idx'].cpu().numpy())
                img = storage_batch_cp['img'].cuda()
                img = img.reshape(
                        img.size(0) * img.size(1),
                        img.size(2), img.size(3), img.size(4))
                model_res = model(img=img, mode=self.forward_mode)
                curr_embd = model_res['embd'].reshape(
                        -1, num_crops, model_res['embd'].size(-1))
                curr_embd = curr_embd.detach().cpu().numpy()
                for crp_idx in range(num_crops):
                    all_embds[crp_idx].append(curr_embd[:, crp_idx])
                del storage_batch_cp
                del model_res
        model.train()
        data_len = len(loader.dataset)
        all_idxs = self.cross_device_gather(all_idxs, data_len)
        for crp_idx in range(num_crops):
            all_embds[crp_idx] = self.cross_device_gather(
                    all_embds[crp_idx], data_len)
        assert np.allclose(all_idxs, np.arange(data_len))
        return all_embds

    def get_all_embds(self, model):
        storage_embds = self.get_embds_from_loader(model)
        curr_embds = self.get_embds_from_loader(
                model, self.cont_loader, 'Get Current Embeds')

        self.storage_embds = np.asarray(storage_embds)
        self.storage_embds = torch.from_numpy(self.storage_embds).cuda()
        self.curr_embds = np.asarray(curr_embds)
        self.curr_embds = torch.from_numpy(self.curr_embds).cuda()

    def reset_presampled_idxs(self, presampled_idx_name, total_len):
        self.all_presampled_idxs[presampled_idx_name] \
                = np.random.permutation(total_len)
        self.all_neg_idx_heads[presampled_idx_name] = 0

    def get_neg_smpl_idxs(
            self, sample_len=None, 
            presampled_idx_name='curr', total_len=None):
        if total_len is None:
            total_len = len(self.curr_embds[0])
        if sample_len is None:
            sample_len = self.neg_img_nums
        if getattr(self, f'all_presampled_idxs', None) is None:
            self.all_presampled_idxs = {}
            self.all_neg_idx_heads = {}
        if presampled_idx_name not in self.all_presampled_idxs:
            self.reset_presampled_idxs(presampled_idx_name, total_len)
        neg_idx_head = self.all_neg_idx_heads[presampled_idx_name]
        neg_idx_end = neg_idx_head + sample_len
        if neg_idx_end > total_len:
            self.reset_presampled_idxs(presampled_idx_name, total_len)
            neg_idx_head, neg_idx_end = 0, sample_len
        presampled_idxs = self.all_presampled_idxs[presampled_idx_name]
        neg_smpl_idxs = presampled_idxs[neg_idx_head : neg_idx_end]
        self.all_neg_idx_heads[presampled_idx_name] = neg_idx_end
        return neg_smpl_idxs

    def get_neg_embds_for_one_batch(self):
        if self.neg_sample_method == 'from_curr':
            neg_smpl_idxs = self.get_neg_smpl_idxs()
            neg_embds = self.curr_embds[:, neg_smpl_idxs, :]
        elif self.neg_sample_method.startswith('mix_curr_st'):
            if self.neg_sample_method == 'mix_curr_st':
                curr_num = self.neg_img_nums // 2
            else:
                curr_num = int(
                        self.neg_img_nums \
                        * float(self.neg_sample_method.split('_')[-1]))
            curr_neg_smpl_idxs = self.get_neg_smpl_idxs(
                    sample_len=curr_num)
            curr_neg_embds = self.curr_embds[:, curr_neg_smpl_idxs, :]
            st_neg_embds_idxs = self.get_neg_smpl_idxs(
                    sample_len=self.neg_img_nums - curr_num,
                    total_len=len(self.storage_embds[0]),
                    presampled_idx_name='storage')
            st_neg_embds = self.storage_embds[:, st_neg_embds_idxs, :]
            neg_embds = torch.cat([curr_neg_embds, st_neg_embds], dim=1)
        else:
            raise NotImplementedError
        neg_embds = neg_embds.reshape(-1, neg_embds.size(-1))
        neg_embds = neg_embds.transpose(1, 0)
        return neg_embds

    def get_one_batch_loss(self, sta_idx, end_idx):
        pos_embd_0 = self.storage_embds[0][sta_idx:end_idx]
        pos_embd_1 = self.storage_embds[1][sta_idx:end_idx]
        pos = torch.sum(pos_embd_0 * pos_embd_1, dim=-1, keepdim=True)

        losses = 0
        for _ in range(self.sample_batches):
            neg_embds = self.get_neg_embds_for_one_batch()
            neg0 = torch.matmul(pos_embd_0, neg_embds)
            neg1 = torch.matmul(pos_embd_1, neg_embds)
            _loss = self.loss_head(pos, neg0)['loss'] \
                    + self.loss_head(pos, neg1)['loss']
            _loss = _loss / 2
            losses = losses + _loss
        losses = losses / self.sample_batches
        return losses.detach().cpu()

    def compute_raw_loss_values(self, del_embds=True):
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
        if del_embds:
            del self.storage_embds
            del self.curr_embds
        loss_values = np.concatenate(loss_values)
        return loss_values

    def get_loss_for_next_epoch(self, model):
        if self.curr_epoch + 1 >= len(self.epoch_meta):
            return None
        self.cont_loader.dataset.data_source.set_epoch(self.curr_epoch + 1)
        SAYCamContAccu.set_epoch(self, self.curr_epoch + 1)
        self.get_all_embds(model)
        loss_values = self.compute_raw_loss_values()
        self.cont_loader.dataset.data_source.set_epoch(self.curr_epoch)
        SAYCamContAccu.set_epoch(self, self.curr_epoch)
        return loss_values

    def get_loss_values(self, model):
        self.get_all_embds(model)

        loss_values = self.compute_raw_loss_values()
        if self.delta_loss:
            loss_values = loss_values - self.prev_losses
            self.prev_losses = self.get_loss_for_next_epoch(model)
        if self.accu_lambda:
            loss_values = self.accu_losses * self.accu_lambda + loss_values
            self.accu_losses = loss_values
        return loss_values

    def cnd_set_epoch(self, epoch, model, cont_data_source):
        SAYCamContAccu.set_epoch(self, epoch)
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
class SAYCamCndGammaLoss(SAYCamCndLoss):
    def get_embds_from_loader(
            self, model, loader=None, 
            tqdm_desc='Get Storage Embds', num_crops=2):
        rank, world_size = get_dist_info()
        if loader is None:
            loader = self.cnd_loader
        if rank == 0:
            to_enum = tqdm(loader, desc=tqdm_desc)
        else:
            to_enum = loader
        all_embds = [[] for _ in range(num_crops)]
        all_momentum_embds = [[] for _ in range(num_crops)]
        all_idxs = []

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
                storage_batch_cp = copy.deepcopy(storage_batch)
                del storage_batch
                all_idxs.append(storage_batch_cp['idx'].cpu().numpy())
                img = storage_batch_cp['img'].cuda()
                img = img.reshape(
                        img.size(0) * img.size(1),
                        img.size(2), img.size(3), img.size(4))
                all_embds = parse_embd_append(
                        model(img=img, mode='test'),
                        all_embds)
                all_momentum_embds = parse_embd_append(
                        model(img=img, mode='momentum_test'),
                        all_momentum_embds)
                del storage_batch_cp
        model.train()
        data_len = len(loader.dataset)
        all_idxs = self.cross_device_gather(all_idxs, data_len)
        for crp_idx in range(num_crops):
            all_embds[crp_idx] = self.cross_device_gather(
                    all_embds[crp_idx], data_len)
            all_momentum_embds[crp_idx] = self.cross_device_gather(
                    all_momentum_embds[crp_idx], data_len)
        assert np.allclose(all_idxs, np.arange(data_len))
        return all_embds, all_momentum_embds

    def get_all_embds(self, model):
        storage_embds, storage_mtm_embds = self.get_embds_from_loader(model)
        curr_embds, curr_mtm_embds = self.get_embds_from_loader(
                model, self.cont_loader, 'Get Current Embeds')

        self.storage_embds = np.asarray(storage_embds)
        self.storage_embds = torch.from_numpy(self.storage_embds).cuda()
        self.storage_mtm_embds = np.asarray(storage_mtm_embds)
        self.storage_mtm_embds = torch.from_numpy(self.storage_mtm_embds).cuda()
        self.curr_embds = np.asarray(curr_embds)
        self.curr_embds = torch.from_numpy(self.curr_embds).cuda()
        self.curr_mtm_embds = np.asarray(curr_mtm_embds)
        self.curr_mtm_embds = torch.from_numpy(self.curr_mtm_embds).cuda()

    def compute_gamma_loss_values(self):
        np.random.seed(self.curr_epoch)
        loss_values = self.compute_raw_loss_values()
        self.storage_embds = self.storage_mtm_embds
        self.curr_embds = self.curr_mtm_embds
        np.random.seed(self.curr_epoch)
        mtm_loss_values = self.compute_raw_loss_values()
        del self.storage_mtm_embds
        del self.curr_mtm_embds
        return loss_values - mtm_loss_values

    def get_loss_values(self, model):
        self.get_all_embds(model)

        loss_values = self.compute_gamma_loss_values()
        return loss_values


@DATASOURCES.register_module
class SAYCamFromList(ImageList):
    def __init__(
            self, root, list_file, num_frames_meta_file,
            one_epoch_img_num=1281167, video_filter_func=None,
            per_video_sample_rate=None, 
            per_video_fps_range_rate=1,
            **kwargs):
        super().__init__(
            '', list_file, 
            **kwargs)
        self.video_root = root
        self.all_videos = copy.deepcopy(self.fns)
        if video_filter_func is not None:
            self.all_videos = list(filter(video_filter_func, self.all_videos))
        self.num_frames_meta_file = num_frames_meta_file
        self.one_epoch_img_num = one_epoch_img_num
        self.per_video_sample_rate = per_video_sample_rate
        self.per_video_fps_range_rate = per_video_fps_range_rate
        self.load_num_frames_meta()
        self.load_all_jpgs()
        self.set_epoch(0)

    def load_num_frames_meta(self):
        self.num_frames_meta = {}
        with open(self.num_frames_meta_file, 'r') as fin:
            all_lines = fin.readlines()
        for each_line in all_lines:
            video_name, num_frames = each_line.split(',')
            num_frames = int(num_frames)
            self.num_frames_meta[video_name] = num_frames
            if self.per_video_sample_rate is not None:
                self.num_frames_meta[video_name] = \
                        num_frames // self.per_video_sample_rate

    def load_all_jpgs(self):
        all_jpgs = []
        for _dir in self.all_videos:
            if self.per_video_fps_range_rate > 0:
                frame_no_list = range(
                        0, self.num_frames_meta[_dir],
                        self.per_video_fps_range_rate)
            elif self.per_video_fps_range_rate < 0:
                frame_no_list = [
                        _frame for _frame in range(
                            0, self.num_frames_meta[_dir]) \
                            if _frame % -self.per_video_fps_range_rate != 0]
            else:
                raise NotImplementedError
            frames = [
                    os.path.join(
                        self.video_root, _dir, 
                        '%06i.jpg' % (_frame+1))
                    for _frame in frame_no_list]
            all_jpgs.extend(frames)
        self.all_jpgs = all_jpgs

    def set_epoch(self, epoch):
        if self.one_epoch_img_num is None:
            self.fns = np.asarray(self.all_jpgs)
            return
        np.random.seed(epoch)
        self.fns = np.random.choice(
                self.all_jpgs, self.one_epoch_img_num, 
                replace=True)
