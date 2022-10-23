import io
from PIL import Image
import torch
import numpy as np
try:
    import mc
except ImportError as E:
    pass


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff)


class McLoader(object):

    def __init__(self, mclient_path):
        assert mclient_path is not None, \
            "Please specify 'data_mclient_path' in the config."
        self.mclient_path = mclient_path
        server_list_config_file = "{}/server_list.conf".format(
            self.mclient_path)
        client_config_file = "{}/client.conf".format(self.mclient_path)
        self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                      client_config_file)

    def __call__(self, fn):
        try:
            img_value = mc.pyvector()
            self.mclient.Get(fn, img_value)
            img_value_str = mc.ConvertBuffer(img_value)
            img = pil_loader(img_value_str)
        except:
            print('Read image failed ({})'.format(fn))
            return None
        else:
            return img


def np_l2_normalize(mask):
    mask = mask / np.sqrt(np.sum(mask ** 2, axis=-1, keepdims=True))
    return mask


def compute_mean_sim(
        group_embds_A, group_embds_B, metric,
        batch_size=128):
    # Mean of similarity to B from A (output in the same shape of A)
    group_embds_B = group_embds_B.transpose(1, 0)
    if isinstance(group_embds_B, np.ndarray):
        group_embds_B = torch.from_numpy(group_embds_B).cuda()
    mean_sim = torch.zeros(len(group_embds_A)).cuda()
    for start_idx in range(0, len(group_embds_A), batch_size):
        end_idx = min(start_idx + batch_size, len(group_embds_A))
        if isinstance(group_embds_A, np.ndarray):
            _curr_embds = torch.from_numpy(group_embds_A[start_idx:end_idx]).cuda()
        else:
            _curr_embds = group_embds_A[start_idx:end_idx]
        _sim = torch.matmul(_curr_embds, group_embds_B)
        if metric == 'mean':
            mean_sim[start_idx:end_idx] = torch.mean(_sim, dim=-1)
        elif metric.startswith('max_'):
            split_metric = metric.split('_')
            if len(split_metric) == 2:
                max_k_sta = 0
                max_k = int(metric.split('_')[1])
            elif len(split_metric) == 3:
                max_k_sta = int(metric.split('_')[1])
                max_k = int(metric.split('_')[2])
                max_len = _curr_embds.shape[1]
                if max_k > max_len:
                    max_k_sta = max_len - (max_k - max_k_sta)
                    max_k = max_len
            else:
                raise NotImplementedError
            topk_sim, _ = torch.topk(
                    _sim, k=max_k, dim=-1,
                    largest=True, sorted=True)
            topk_sim = topk_sim.detach()
            mean_sim[start_idx:end_idx] = torch.mean(topk_sim[:, max_k_sta:], dim=-1)
        else:
            raise NotImplementedError
    del group_embds_B
    del group_embds_A
    return mean_sim.detach().cpu().numpy()
