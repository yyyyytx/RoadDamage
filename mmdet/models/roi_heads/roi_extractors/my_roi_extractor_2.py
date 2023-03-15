import torch
from mmcv.cnn.bricks import build_plugin_layer

from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class MyRoIExtractor2(BaseRoIExtractor):
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
                 finest_scale=56,
                 **kwargs):
        super(MyRoIExtractor2, self).__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.finest_scale = finest_scale
        self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        # build pre/post processing modules
        if self.with_post:
            self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]

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
        target_lvls_scale = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls_scale = target_lvls_scale.clamp(min=0, max=num_levels - 1).long()

        ratio = (rois[:, 3] - rois[:, 1]) / ((rois[:, 4] - rois[:, 2]) + 1e-6)

        # 把所有大于1.0的宽高比变为倒数
        inds = ratio > 1.0
        ratio[inds] = 1.0 / ratio[inds]
        # 只有宽高比在[0.5,2.0]的锚框可以映射到最高层 其余的宽高比越离谱越靠向底层
        target_lvls_ratio = torch.ceil(num_levels - 1 + torch.log2(ratio / 1.0))
        target_lvls_ratio = target_lvls_ratio.clamp(min=0, max=num_levels - 1).long()

        return torch.min(target_lvls_scale, target_lvls_ratio)

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""

        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        # ROI池化层将任何ROI转换为固定空间大小的特征图
        out_size = self.roi_layers[0].output_size

        # 需要用到的特征图层数 默认是4
        num_levels = len(feats)

        # 初始化一个ROI特征图 形状为（ROI数量，输出通道数，7x7）
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
            roi_feats = self.post_module(roi_feats)

        return roi_feats
