import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS


@NECKS.register_module()
class ASFF(nn.Module):
    """
    源自论文：Learning Spatial Fusion for Single-Shot Object Detection
    代码参考：https://github.com/ruinmessi/ASFF/
    """
    def __init__(self, in_channels, out_channels, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        # self.dim = [512, 256, 256]
        self.dim = in_channels
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(self.dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            # self.expand = add_conv(self.inter_dim, 1024, 3, 1)
            self.expand = add_conv(self.inter_dim, out_channels, 3, 1)
        elif level == 1:
            # self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            # self.expand = add_conv(self.inter_dim, 512, 3, 1)
            self.expand = add_conv(self.inter_dim, out_channels, 3, 1)
        elif level == 2:
            # self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            if self.dim[1] != self.dim[2]:
                self.compress_level_1 = add_conv(self.dim[1], self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, out_channels, 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(
            compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def init_weights(self):
        pass

    def forward(self, inputs):
        if self.level == 0:
            level_0_resized = inputs[2]
            level_1_resized = self.stride_level_1(inputs[1])

            level_2_downsampled_inter = F.max_pool2d(
                inputs[0], 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(inputs[2])
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = inputs[1]
            level_2_resized = self.stride_level_2(inputs[0])
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(inputs[0])
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=0.5, mode='nearest')
            if self.dim[1] != self.dim[2]:
                level_1_compressed = self.compress_level_1(inputs[1])
                level_1_resized = level_1_compressed
                # level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            else:
                level_1_resized = inputs[1]
                # level_1_resized = F.interpolate(inputs[1], scale_factor=2, mode='nearest')
            level_2_resized = inputs[2]

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return tuple(out)


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage
