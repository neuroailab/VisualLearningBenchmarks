import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .simclr import SimCLR
from .utils import GatherLayer


@MODELS.register_module
class MOCO(nn.Module):
    """MOCO.

    Implementation of "Momentum Contrast for Unsupervised Visual
    Representation Learning (https://arxiv.org/abs/1911.05722)".
    Part of the code is borrowed from:
    "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        queue_len (int): Number of negative keys maintained in the queue.
            Default: 65536.
        feat_dim (int): Dimension of compact feature vectors. Default: 128.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 nn_num=None,
                 uniform_nn=False,
                 nn_num_mult=1,
                 remove_momentum_encoder=False,
                 sync_bn=False,
                 shared_bn=False,
                 within_batch_ctr=None,
                 predictor=None,
                 no_neg=False,
                 hipp_head=None,
                 update_in_forward=True,
                 two_queues=False,
                 **kwargs):
        super(MOCO, self).__init__()
        self.remove_momentum_encoder = remove_momentum_encoder
        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.encoder_q[0]
        if not remove_momentum_encoder:
            self.encoder_k = nn.Sequential(
                builder.build_backbone(backbone), builder.build_neck(neck))
            for param in self.encoder_k.parameters():
                param.requires_grad = False
        self.head = builder.build_head(head)
        if sync_bn:
            sync_bn_convert = torch.nn.SyncBatchNorm.convert_sync_batchnorm
            self.encoder_q = sync_bn_convert(self.encoder_q)
            if not remove_momentum_encoder:
                self.encoder_k = sync_bn_convert(self.encoder_k)
            self.head = sync_bn_convert(self.head)
        self.predictor = None
        if predictor is not None:
            self.predictor = builder.build_neck(predictor)
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum
        self.nn_num = nn_num
        self.nn_num_mult = nn_num_mult
        self.uniform_nn = uniform_nn
        self.shared_bn = shared_bn
        self.within_batch_ctr = within_batch_ctr
        self.no_neg = no_neg
        self.update_in_forward = update_in_forward

        if uniform_nn:
            from openselfsup.utils import AliasMethod
            self.multinomial = AliasMethod(torch.ones(queue_len))
            self.multinomial.cuda()

        self.hipp_head = None
        if hipp_head is not None:
            self.hipp_head = builder.build_head(hipp_head)

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        if two_queues:
            # Here `pretrain_queue` is for the storage batch
            self.register_buffer("pretrain_queue", torch.randn(feat_dim, queue_len))
            self.pretrain_queue = nn.functional.normalize(self.pretrain_queue, dim=0)
            self.register_buffer("pretrain_ptr", torch.zeros(1, dtype=torch.long))
        
    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights(init_linear='kaiming')
        if not self.remove_momentum_encoder:
            for param_q, param_k in zip(self.encoder_q.parameters(),
                                        self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
        if self.predictor is not None:
            self.predictor.init_weights(init_linear='normal')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        if self.remove_momentum_encoder:
            return
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, separate_queue=False):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        assert self.queue_len % batch_size == 0, \
                "%i, %i" % (self.queue_len, batch_size)  # for simplicity
        
        # replace the keys at ptr (dequeue and enqueue)
        if separate_queue:    # update queue with rep from pre-trained dataset
            pretrain_ptr = int(self.pretrain_ptr)
            self.pretrain_queue[
                :, pretrain_ptr: pretrain_ptr + batch_size] = keys.transpose(0, 1)
            pretrain_ptr = (pretrain_ptr + batch_size) % self.queue_len
            self.pretrain_ptr[0] = pretrain_ptr
        else:
            ptr = int(self.queue_ptr)
            self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
            ptr = (ptr + batch_size) % self.queue_len
            self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

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
        """Undo batch shuffle.

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
    
    ''' Co-training forward with separate memory queues for 
    batch from each dataset '''
    def copy_pretrain_queue(self):
        self.pretrain_queue = self.queue.clone()
        self.pretrain_ptr = self.queue_ptr.clone()

    def get_embeddings(self, img):
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        if self.predictor is not None:
            q = self.predictor([q])[0]
        q = nn.functional.normalize(q, dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            
            if self.update_in_forward:
                self._momentum_update_key_encoder()  # update the key encoder
            
            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            
            if not self.remove_momentum_encoder:
                k = self.encoder_k(im_k)[0]  # keys: NxC
            else:
                k = self.encoder_q(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return q, k

    def get_embeddings_shared_bn(self, img):
        assert self.remove_momentum_encoder
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        q_k = self.encoder_q(torch.cat([im_q, im_k], dim=0))[0]
        q_k = nn.functional.normalize(q_k, dim=1)
        q, k = torch.split(q_k, [im_q.shape[0], im_k.shape[0]], dim=0)
        if self.predictor is not None:
            q = self.predictor([q])[0]
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        k = k.clone().detach()
        return q, k

    def get_neg(self, q, k, queue_mode):
        def _compute_l_neg(queue_ts):
            if self.nn_num is None:
                l_neg = torch.einsum('nc,ck->nk', [q, queue_ts])
            else:
                if not self.uniform_nn:
                    k_dot = torch.einsum('nc,ck->nk', [k, queue_ts])
                    _, _idx = torch.topk(
                            k_dot, self.nn_num * self.nn_num_mult, 
                            dim=-1)
                    _idx = _idx[:, ::self.nn_num_mult]
                else:
                    bs = k.shape[0]
                    _idx = self.multinomial.draw(bs * self.nn_num).view(bs, self.nn_num)
                seleced_queue = torch.index_select(queue_ts.transpose(0, 1), 0, _idx.view(-1))
                seleced_queue = seleced_queue.view(_idx.shape[0], _idx.shape[1], -1)
                l_neg = torch.einsum('njc,nkc->njk', [q.unsqueeze(1), seleced_queue])
                l_neg = l_neg.squeeze(1)
            return l_neg
        
        if queue_mode == 'normal':
            # negative logits: NxK
            l_neg = _compute_l_neg(self.queue.clone().detach())
            self._dequeue_and_enqueue(k)
        elif queue_mode == 'separate_queue':
            l_neg = _compute_l_neg(self.pretrain_queue.clone().detach())
            self._dequeue_and_enqueue(k, separate_queue=True)
        elif queue_mode == 'no_queue_update':
            l_neg = _compute_l_neg(self.queue.clone().detach())
        return l_neg

    def get_pos_neg(self, q, k, queue_mode):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = self.get_neg(q, k, queue_mode)
        return l_pos, l_neg
        
    def forward_train(
            self, img, queue_mode='normal', 
            within_batch_ctr=None, 
            more_metrics=False,
            add_noise=None,
            **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.
            queue_mode (str): normal - maintain and update one queue as normal,
                              separate_queue - maintain two sets of queues and ptrs,
                              no_queue_update - do not update the queue.
        
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """        
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        if not self.shared_bn:
            q, k = self.get_embeddings(img)
        else:
            q, k = self.get_embeddings_shared_bn(img)

        if add_noise is not None:
            noise = torch.randn(k.shape[0], k.shape[1]).cuda() * add_noise
            k += noise
            k = nn.functional.normalize(k, dim=1)

        if self.hipp_head is not None:
            hipp_loss = self.hipp_head(q, two_views=False)
        
        l_pos, l_neg = self.get_pos_neg(q, k, queue_mode)
        losses = self.head(l_pos, l_neg)
        within_batch_ctr = within_batch_ctr or self.within_batch_ctr
        if within_batch_ctr is not None:
            _t = self.head.temperature
            curr_loss = losses['loss']
            l_batch_ctr = torch.exp(torch.matmul(q, k.transpose(0, 1)) / _t)
            l_batch_ctr = torch.sum(l_batch_ctr, dim=1)
            l_batch_ctr = torch.mean(
                    torch.log(torch.exp(l_pos/_t) / l_batch_ctr))
            curr_loss -= within_batch_ctr * l_batch_ctr
            losses['loss'] = curr_loss
        if more_metrics:
            losses['mean_l_pos'] = torch.mean(l_pos).cpu().item()
            losses['mean_l_neg'] = torch.mean(l_neg).cpu().item()
        if self.no_neg:
            losses['loss'] = 2 - 2 * torch.mean(l_pos)
        if self.hipp_head is not None:
            losses['loss'] += hipp_loss.pop('loss')
            for key in hipp_loss:
                assert key not in losses
            losses.update(hipp_loss)
        return losses

    def forward_test(self, img, **kwargs):
        assert img.dim() == 4, \
            "Input must have 4 dims, got: {}".format(img.dim())
        feature = self.encoder_q(img)[0]
        feature = nn.functional.normalize(feature)  # BxC

        if not self.remove_momentum_encoder:
            feature_m = self.encoder_k(img)[0]
        else:
            feature_m = feature
        feature_m = nn.functional.normalize(feature_m)  # BxC
        return {'embd': feature.cpu(), 'embd_m': feature_m.cpu()}

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, queue_mode='normal', **kwargs)
        elif mode == 'train_no_queue_update':        
            return self.forward_train(img, queue_mode='no_queue_update', **kwargs)
        elif mode == 'train_separate_queue':
            return self.forward_train(img, queue_mode='separate_queue', **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


@MODELS.register_module
class MOCOSimCLR(MOCO):
    '''
    No momentum encoder, using within batch contrast in SimCLR way
    '''
    def __init__(self, concat_neg=False, *args, **kwargs):
        self.concat_neg = concat_neg
        kwargs.update(dict(
                remove_momentum_encoder=True,
                sync_bn=True,
                shared_bn=True,
                within_batch_ctr=None,
                predictor=None,
                no_neg=False,
                ))
        super().__init__(*args, **kwargs)
        
    def forward_train(
            self, img, simclr_mode='normal', 
            queue_mode='normal', **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.
            queue_mode (str): normal - maintain and update one queue as normal,
                              separate_queue - maintain two sets of queues and ptrs,
                              no_queue_update - do not update the queue.
        
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """        
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img = img.reshape(
            img.size(0) * 2, img.size(2), img.size(3), img.size(4))
        z = self.encoder_q(img)[0]  # (2n)xd
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        curr_z = z
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = SimCLR._create_buffer(N, mode=simclr_mode)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        z_k = curr_z.reshape(curr_z.size(0) // 2, 2, curr_z.size(1))
        z_k = torch.cat((z_k[:, 1:2, :], z_k[:, 0:1, :]), dim=1)
        z_k = z_k.reshape(-1, z_k.size(2))
        moco_nega = self.get_neg(z, z_k, queue_mode)

        if not self.concat_neg:
            simclr_losses = self.head(positive, negative)
            moco_losses = self.head(positive, moco_nega)
            losses = {'loss': moco_losses['loss'] + simclr_losses['loss']}
        else:
            losses = self.head(
                    positive, 
                    torch.cat([negative, moco_nega], dim=1))
        return losses


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
