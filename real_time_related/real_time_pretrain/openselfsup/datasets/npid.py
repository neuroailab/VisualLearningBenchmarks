from PIL import Image
from .registry import DATASETS
from .base import BaseDataset
from openselfsup.utils import print_log
import torch


@DATASETS.register_module
class NPIDDataset(BaseDataset):
    """Dataset for NPID.
    """

    def __init__(self, data_source, pipeline):
        super(NPIDDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        if not isinstance(img, Image.Image):
            img, gt_label = img
            img = self.pipeline(img)
            return dict(img=img, idx=idx, gt_label=gt_label)
        else:
            img = self.pipeline(img)
            return dict(img=img, idx=idx)

    def evaluate(self, scores, keyword, logger=None):
        num = scores.size(0)
        correct_k = scores.view(-1).float().sum(0).item()
        acc = correct_k * 100.0 / num
        eval_res = {}
        eval_res["{}_top1".format(keyword)] = acc
        if logger is not None and logger != 'silent':
            print_log(
                "{}_top1: {:.03f}".format(keyword, acc),
                logger=logger)
        return eval_res


@DATASETS.register_module
class NPIDNNDataset(NPIDDataset):
    """Dataset for NPID, with NN validation implemented
    """

    def __init__(self, data_source, pipeline):
        super(NPIDNNDataset, self).__init__(data_source, pipeline)

    def evaluate(self, scores, keyword, logger=None):
        target = torch.LongTensor(self.data_source.labels)

        dp_results = torch.mm(scores[20000:], scores[:20000].transpose(0, 1))
        _, pred_nn = dp_results.topk(1, dim=1, largest=True, sorted=True)
        pred_nn = pred_nn.squeeze(1)
        pred_label = torch.index_select(target[:20000], 0, pred_nn)
        corr = pred_label.eq(target[20000:]).float().sum(0).item()
        acc = corr * 100.0 / 5000

        eval_res = {}
        eval_res["{}_nn".format(keyword)] = acc
        if logger is not None and logger != 'silent':
            print_log(
                "{}_nn: {:.03f}".format(keyword, acc),
                logger=logger)
        return eval_res
