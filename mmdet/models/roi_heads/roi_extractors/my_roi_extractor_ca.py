import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import build_plugin_layer
from torch import nn

from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor
from .deform_conv import DeformConv2D


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        # min(max(features,0),6)，激活值限定在[0,6]
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x

        n, c, H, W = x.shape

        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)

        # 按第三个维度进行cat，因为第三个维度为H和W，不一样，其他维度一样n,c,(H+W),1
        x_cat = torch.cat([x_h, x_w], dim=2)

        out = self.act1(self.bn1(self.conv1(x_cat)))

        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))

        return short * out_w * out_h


@ROI_EXTRACTORS.register_module()
class MyRoIExtractorCA(BaseRoIExtractor):
    """Extract RoI features from all level feature maps levels.

    This is the implementation of `A novel Region of Interest Extraction Layer
    for Instance Segmentation <https://arxiv.org/abs/2004.13665>`_.

    Args:
        aggregation (str): The method to aggregate multiple feature maps.
            Options are 'sum', 'concat'. Default: 'sum'.
        pre_cfg (dict | None): Specify pre-processing modules. Default: None.
        post_cfg (dict | None): Specify post-processing modules. Default: None.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`BaseRoIExtractor`.
    """

    def __init__(self,
                 aggregation='sum',
                 pre_cfg=None,
                 post_cfg=None,
                 **kwargs):
        super(MyRoIExtractorCA, self).__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.finest_scale = 56
        self.aggregation = aggregation
        self.with_post = True
        self.with_pre = pre_cfg is not None
        # build pre/post processing modules
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]

        # 后处理模块变为DCN
        if self.with_post:
            # self.post_module = CoordAttention(256, 256)
            self.post_module = DeformConv2D(256, 256)
        self.offsets = nn.Conv2d(256, 18, kernel_size=3, padding=1)
        self.dcn_bn = nn.BatchNorm2d(256)

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""

        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        # ROI池化层将任何ROI转换为固定空间大小的特征图
        out_size = self.roi_layers[0].output_size

        # 需要用到的特征图层数 默认是4
        num_levels = len(feats)

        # 初始化一个ROI特征图 形状为（ROI数量，输出通道数，7x7） 都为零 方便累加
        # 使用feats[0]方便数据类型（cpu or cuda） 匹配
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)

        # 有时候没有ROI
        if roi_feats.shape[0] == 0:
            return roi_feats

        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            # 映射到第i层的ROI
            inds = target_lvls == i
            if inds.any():
                rois_level_i = rois[inds, :]
                for j in range(i + 1):
                    roi_feats_t = self.roi_layers[j](feats[j], rois_level_i)
                    if self.with_pre:
                        # apply pre-processing to a RoI extracted from each layer
                        roi_feats_t = self.pre_module(roi_feats_t)
                    if self.aggregation == 'sum':
                        # and sum them all
                        roi_feats[inds] += roi_feats_t
            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.

        if self.with_post:
            # apply post-processing before return the result
            # deformable convolution
            offsets = self.offsets(roi_feats)
            roi_feats = F.relu(self.post_module(roi_feats, offsets))
            roi_feats = self.dcn_bn(roi_feats)

        return roi_feats
