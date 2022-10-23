import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class NPID(nn.Module):
    """NPID.

    Implementation of "Unsupervised Feature Learning via Non-parametric
    Instance Discrimination (https://arxiv.org/abs/1805.01978)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        memory_bank (dict): Config dict for module of memory banks. Default: None.
        neg_num (int): Number of negative samples for each image. Default: 65536.
        ensure_neg (bool): If False, there is a small probability
            that negative samples contain positive ones. Default: False.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 memory_bank=None,
                 neg_num=65536,
                 ensure_neg=False,
                 pretrained=None,
                 save_bank_labels=False):
        super(NPID, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)

        if save_bank_labels:
            self.register_buffer(
                    "feature_bank", 
                    torch.randn(memory_bank['length'], memory_bank['feat_dim']).cuda())
            memory_bank['feature_bank'] = self.feature_bank
        self.memory_bank = builder.build_memory(memory_bank)

        if not save_bank_labels:
            self.train_labels = torch.arange(0, memory_bank['length']).cuda()
        else:
            self.register_buffer(
                    "train_labels", 
                    torch.arange(0, memory_bank['length']).cuda())
        self.init_weights(pretrained=pretrained)

        self.neg_num = neg_num
        self.ensure_neg = ensure_neg

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, idx, gt_label=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            idx (Tensor): Index corresponding to each image.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.forward_backbone(img)
        idx = idx.cuda()
        feature = self.neck(x)[0]
        feature = nn.functional.normalize(feature)  # BxC
        bs, feat_dim = feature.shape[:2]
        neg_idx = self.memory_bank.multinomial.draw(bs * self.neg_num)
        if self.ensure_neg:
            neg_idx = neg_idx.view(bs, -1)
            while True:
                wrong = (neg_idx == idx.view(-1, 1))
                if wrong.sum().item() > 0:
                    neg_idx[wrong] = self.memory_bank.multinomial.draw(
                        wrong.sum().item())
                else:
                    break
            neg_idx = neg_idx.flatten()

        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      idx)  # BXC
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs, self.neg_num,
                                                    feat_dim)  # BxKxC

        pos_logits = torch.einsum('nc,nc->n',
                                  [pos_feat, feature]).unsqueeze(-1)
        neg_logits = torch.bmm(neg_feat, feature.unsqueeze(2)).squeeze(2)

        losses = self.head(pos_logits, neg_logits)

        # update memory bank
        with torch.no_grad():
            self.memory_bank.update(idx, feature.detach())
            if gt_label is not None:
                self.train_labels[idx] = gt_label

        return losses

    def forward_test(self, img, idx, gt_label, **kwargs):
        x = self.forward_backbone(img)
        idx = idx.cuda()
        feature = self.neck(x)[0]
        feature = nn.functional.normalize(feature)  # BxC

        all_logits = torch.mm(
                feature, 
                self.memory_bank.feature_bank.transpose(0, 1))
        _, nn_idx = all_logits.topk(1, dim=1, largest=True, sorted=True)
        nn_idx = nn_idx.squeeze(1)
        pred = torch.index_select(self.train_labels, 0, nn_idx)
        corr = pred.eq(gt_label)
        return {'nn': corr.cpu()}

    def forward_embdng(self, img):
        x = self.forward_backbone(img)
        feature = self.neck(x)[0]
        feature = nn.functional.normalize(feature)  # BxC
        return {'embd': feature.cpu()}

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        elif mode == 'get_embdng':
            return self.forward_embdng(img)
        else:
            raise Exception("No such mode: {}".format(mode))
