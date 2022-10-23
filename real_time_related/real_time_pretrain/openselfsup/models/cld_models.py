# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

from . import builder
from .registry import MODELS
from .cld_modules import ResNet


def KMeans(x, K=10, Niters=10, verbose=False):

    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = x[:, None, :]  # (Npoints, 1, D)

    for i in range(Niters):
        c_j = c[None, :, :]  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = cl.view(cl.size(0), 1).expand(-1, D)
        unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
        # As some clusters don't contain any samples, manually assign count as 1
        labels_count_all = torch.ones([K]).long().cuda()
        labels_count_all[unique_labels[:,0]] = labels_count
        c = torch.zeros([K, D], dtype=torch.float).cuda().scatter_add_(0, Ncl, x)
        c = c / labels_count_all.float().unsqueeze(1)

    return cl, c


@MODELS.register_module
class CLDMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(
            self, base_encoder_arch='resnet18', 
            dim=128, K=65536, m=0.999, T=0.2, 
            mlp=True, two_branch=True, normlinear=True,
            clusters=120, num_iters=5, cld_T=0.4, Lambda=0.25,
            ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.two_branch = two_branch
        self.clusters = clusters
        self.num_iters = num_iters
        self.cld_T = cld_T
        self.Lambda = Lambda

        # create the encoders
        # num_classes is the output fc dimension
        base_encoder = ResNet.__dict__[base_encoder_arch]
        self.encoder_q = base_encoder(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)
        self.encoder_k = base_encoder(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)

        if mlp and not two_branch:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.criterion = nn.CrossEntropyLoss()

    def get_cld_loss(self, features_groupDis1, features_groupDis2, use_kmeans=True):
        if use_kmeans:
            cluster_label1, centroids1 = KMeans(features_groupDis1, K=self.clusters, Niters=self.num_iters)
            cluster_label2, centroids2 = KMeans(features_groupDis2, K=self.clusters, Niters=self.num_iters)
        else:
            raise NotImplementedError

        # group discriminative learning
        affnity1 = torch.mm(features_groupDis1, centroids2.t())
        CLD_loss = self.criterion(affnity1.div_(self.cld_T), cluster_label2)

        affnity2 = torch.mm(features_groupDis2, centroids1.t())
        CLD_loss = (CLD_loss + self.criterion(affnity2.div_(self.cld_T), cluster_label1))/2

        return CLD_loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def generate_logits_labels(self, q, k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        return logits

    def get_logits_labels(self, im_q, im_k, im_q2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        if self.two_branch:
            eq1 = nn.functional.normalize(q[1], dim=1) # branch 2
            q = q[0]                                   # branch 1
            # Generate logits for im_q2
            q2 = self.encoder_q(im_q2)  # queries: NxC
            eq2 = nn.functional.normalize(q2[1], dim=1) # branch 2
            q2 = nn.functional.normalize(q2[0], dim=1)  # branch 1

        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            if self.two_branch:
                ek1 = nn.functional.normalize(k[1], dim=1)
                k = k[0]
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits.
        # labels: positive key indicators
        logits = self.generate_logits_labels(q, k)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        if self.two_branch:
            logits2 = self.generate_logits_labels(q2, k)
            labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        if self.two_branch:
            return logits, labels, logits2, labels2, eq1, ek1, eq2
        return logits, labels
    
    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        outputs = self.get_logits_labels(
                im_q=img[:, 0].contiguous(), 
                im_k=img[:, 1].contiguous(), 
                im_q2=img[:, 2].contiguous())
        logits1, labels1, logits2, labels2, eq1, ek1, eq2 = outputs
        loss = self.criterion(logits1, labels1)/2 + self.criterion(logits2, labels2)/2
        loss += self.Lambda*self.get_cld_loss(eq1, eq2)
        losses = {'loss': loss}
        return losses

    def forward_test(self, img, **kwargs):
        assert img.dim() == 4, \
            "Input must have 4 dims, got: {}".format(img.dim())
        feature = self.encoder_q(img)[0]
        feature = nn.functional.normalize(feature)  # BxC
        feature_m = self.encoder_k(img)[0]
        feature_m = nn.functional.normalize(feature_m)  # BxC
        return {'embd': feature.cpu(), 'embd_m': feature_m.cpu()}

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return [self.encoder_q.forward_extract(img)]
        else:
            raise Exception("No such mode: {}".format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GroupQue(nn.Module):
    """
    Build a queue for Group Discriminative Branch
    """
    def __init__(self, dim=128, K=256*15, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(GroupQue, self).__init__()

        self.K = K
        self.T = T

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
