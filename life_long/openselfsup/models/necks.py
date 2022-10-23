import torch
import torch.nn as nn
from packaging import version
from mmcv.cnn import kaiming_init, normal_init
import torch.nn.functional as F
import pdb

from .registry import NECKS
from .utils import build_norm_layer


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming', 'trivial'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            elif init_linear == 'trivial':
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@NECKS.register_module
class LinearNeck(nn.Module):
    """Linear neck: fc only.
    """

    def __init__(
            self, in_channels, out_channels, with_avg_pool=True,
            init_linear=None):
        super(LinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)
        self.init_linear = init_linear

    def init_weights(self, init_linear='normal'):
        init_linear = self.init_linear or init_linear
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.fc(x.view(x.size(0), -1))]


@NECKS.register_module
class RelativeLocNeck(nn.Module):
    """Relative patch location neck: fc-bn-relu-dropout.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(RelativeLocNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc = nn.Linear(in_channels * 2, out_channels)
        if sync_bn:
            _, self.bn = build_norm_layer(
                dict(type='SyncBN', momentum=0.003),
                out_channels)
        else:
            self.bn = nn.BatchNorm1d(
                out_channels, momentum=0.003)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear, std=0.005, bias=0.1)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn, x)
        else:
            x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return [x]


@NECKS.register_module
class NonLinearNeckV0(nn.Module):
    """The non-linear neck in ODC, fc-bn-relu-dropout-fc-relu.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(NonLinearNeckV0, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc0 = nn.Linear(in_channels, hid_channels)
        if sync_bn:
            _, self.bn0 = build_norm_layer(
                dict(type='SyncBN', momentum=0.001, affine=False),
                hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(
                hid_channels, momentum=0.001, affine=False)

        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]


@NECKS.register_module
class NonLinearNeckV1(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckV1FW(NonLinearNeckV1):
    def forward(self, x, fast_weight):
        ret_from_fw = torch.einsum('bpq, bq->bp', fast_weight, x[0])
        mlp_out = super().forward(x)[0]
        assert mlp_out.size(1) % ret_from_fw.size(1) == 0
        ret_from_fw = ret_from_fw.repeat(1, mlp_out.size(1) // ret_from_fw.size(1))
        mlp_out = mlp_out + ret_from_fw
        return [mlp_out]


@NECKS.register_module
class NonLinearNeckV2(nn.Module):
    """The non-linear neck in byol: fc-bn-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 with_bn=True):
        super(NonLinearNeckV2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if with_bn:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hid_channels),
                nn.BatchNorm1d(hid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hid_channels, out_channels))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckGNV2(nn.Module):
    '''The non-linear neck in byol: fc-bn-relu-fc
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_groups,
                 with_avg_pool=True):
        super(NonLinearNeckGNV2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.GroupNorm(
                num_groups=num_groups, 
                num_channels=hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckV3(nn.Module):
    '''The non-linear neck in siamese: fc-bn-relu-fc-bn-relu-fc
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV3, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckSimCLR(nn.Module):
    """SimCLR non-linear neck.

    Structure: fc(no_bias)-bn(has_bias)-[relu-fc(no_bias)-bn(no_bias)].
        The substructures in [] can be repeated. For the SimCLR default setting,
        the repeat time is 1.
    However, PyTorch does not support to specify (weight=True, bias=False).
        It only support \"affine\" including the weight and bias. Hence, the
        second BatchNorm has bias in this implementation. This is different from
        the official implementation of SimCLR.
    Since SyncBatchNorm in pytorch<1.4.0 does not support 2D input, the input is
        expanded to 4D with shape: (N,C,1,1). Not sure if this workaround
        has no bugs. See the pull request here:
        https://github.com/pytorch/pytorch/pull/29626.

    Args:
        num_layers (int): Number of fc layers, it is 2 in the SimCLR default setting.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 sync_bn=True,
                 with_bias=False,
                 with_last_bn=True,
                 with_avg_pool=True,
                 use_group_norm=None,
                 bn_settings=None,
                 res_inter=False):
        super(NonLinearNeckSimCLR, self).__init__()
        self.sync_bn = sync_bn
        self.with_last_bn = with_last_bn
        self.with_avg_pool = with_avg_pool
        self.use_group_norm = use_group_norm
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.res_inter = res_inter

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn0 = self._get_BN_layer(hid_channels)

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            self.add_module(
                "fc{}".format(i),
                nn.Linear(hid_channels, this_channels, bias=with_bias))
            self.fc_names.append("fc{}".format(i))
            add_bn = i != num_layers - 1 or self.with_last_bn
            if bn_settings is not None:
                add_bn = add_bn and bn_settings[i-1]
            if add_bn:
                self.add_module(
                    "bn{}".format(i),
                    self._get_BN_layer(this_channels))
                self.bn_names.append("bn{}".format(i))
            else:
                self.bn_names.append(None)

    def _get_BN_layer(self, hid_channels):
        if self.sync_bn:
            return build_norm_layer(
                    dict(type='SyncBN'), hid_channels)[1]
        if self.use_group_norm is not None:
            assert isinstance(self.use_group_norm, int)
            return nn.GroupNorm(
                    num_groups=self.use_group_norm,
                    num_channels=hid_channels)
        return nn.BatchNorm1d(hid_channels)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            if self.res_inter:
                prev_x = x
            x = fc(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                if self.sync_bn:
                    x = self._forward_syncbn(bn, x)
                else:
                    x = bn(x)
            if self.res_inter and prev_x.size(-1) == x.size(-1):
                x = x + prev_x
        return [x]


@NECKS.register_module
class NonLinearNeckFW(nn.Module):
    '''The non-linear neck in siamese: fc-bn-relu-fc-bn-relu-fc
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_avg_pool=True):
        super().__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ln1 = nn.Linear(in_channels, in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.ln2 = nn.Linear(in_channels, in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.ln3 = nn.Linear(in_channels, out_channels)
        self.layernorm = nn.LayerNorm(in_channels)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x, fastweight):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x + F.relu(self.bn1(self.ln1(x))) + F.relu(torch.bmm(fastweight[-1], x.unsqueeze(2)).squeeze(2))
        x = self.layernorm(x)
        x = x + F.relu(self.bn2(self.ln2(x))) + F.relu(torch.bmm(fastweight[-1], x.unsqueeze(2)).squeeze(2))
        x = self.layernorm(x)
        x = self.ln3(x)
        return [x]


@NECKS.register_module
class AvgPoolNeck(nn.Module):
    """Average pooling neck.
    """

    def __init__(self):
        super(AvgPoolNeck, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        return [self.avg_pool(x[0])]


@NECKS.register_module
class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()
        self.iden = nn.Identity()

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        return [self.iden(x[0])]


@NECKS.register_module
class Logit(nn.Module):

    def __init__(self):
        super().__init__()

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        return [torch.logit(x[0], eps=1e-6)]


@NECKS.register_module
class ScaleTopK(nn.Module):
    def __init__(self, sparsity, scale):
        self.sparsity = sparsity
        self.scale = scale
        super().__init__()

    def forward(self, x):
        mask = torch.zeros_like(x)
        _, indices = torch.topk(
                x, k=self.sparsity, dim=1, sorted=False)
        mask = mask.scatter(1, indices, 1)
        x = x.masked_fill(mask==0, 0)
        x = x * self.scale
        return x


@NECKS.register_module
class SmplSprsMsk(nn.Module):
    def __init__(self, t=0.1, zero_offset=False):
        self.t = t
        self.zero_offset = zero_offset
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(x / self.t)
        if self.zero_offset:
            x = (x - 0.5) * 2
        return [x]
