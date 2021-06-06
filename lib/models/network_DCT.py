# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
"""
Modified by Myung-Joon Kwon
mjkwon2021@gmail.com
Aug 22, 2020
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HalfDilatedBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(HalfDilatedBlock, self).__init__()
        conv_planes = planes // 4
        dilconv_planes = (planes // 4) * 3
        assert planes == conv_planes + dilconv_planes
        self.conv1 = nn.Conv2d(inplanes, conv_planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(conv_planes, momentum=BN_MOMENTUM)
        self.dil_conv1 = nn.Conv2d(inplanes, dilconv_planes, kernel_size=3, stride=stride, padding=8, bias=False, dilation=8)
        self.dil_bn1 = nn.BatchNorm2d(dilconv_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(inplanes, conv_planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(conv_planes, momentum=BN_MOMENTUM)
        self.dil_conv2 = nn.Conv2d(inplanes, dilconv_planes, kernel_size=3, stride=stride, padding=8, bias=False, dilation=8)
        self.dil_bn2 = nn.BatchNorm2d(dilconv_planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out2 = self.dil_conv1(x)
        out2 = self.dil_bn1(out2)
        out = torch.cat([out1, out2], dim=1)
        out = self.relu(out)

        out1 = self.conv2(out)
        out1 = self.bn2(out1)
        out2 = self.dil_conv2(out)
        out2 = self.dil_bn2(out2)
        out = torch.cat([out1, out2], dim=1)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
#
# class dct2d_Conv_Layer(nn.Module):
#     # This class is written by IJ Yu
#     def __init__(self, scale, start, num_filters):
#         super(dct2d_Conv_Layer, self).__init__()
#         self.scale = scale
#         self.start = start
#         self.num_filters = num_filters // 3
#         self.dct_base = self.load_DCT_basis()
#         self.conv1, self.conv2, self.conv3 = self.dct2d_Conv(), self.dct2d_Conv(), self.dct2d_Conv()
#
#         for conv in [self.conv1, self.conv2, self.conv3]:
#             conv.weight = nn.Parameter(torch.from_numpy(self.dct_base).float())
#             conv.weight.requires_grad = False
#         self.swap = []
#         for i in range(self.num_filters):
#             self.swap += [i, i + self.num_filters, i + self.num_filters * 2]
#
#     def cal_scale(self, p, q):
#         if p == 0:
#             ap = 1 / (np.sqrt(self.scale))
#         else:
#             ap = np.sqrt(2 / self.scale)
#         if q == 0:
#             aq = 1 / (np.sqrt(self.scale))
#         else:
#             aq = np.sqrt(2 / self.scale)
#
#         return ap, aq
#
#     def cal_basis(self, p, q):
#         basis = np.zeros((self.scale, self.scale))
#         ap, aq = self.cal_scale(p, q)
#         for m in range(0, self.scale):
#             for n in range(0, self.scale):
#                 basis[m, n] = ap * aq * np.cos(np.pi * (2 * m + 1) * p / (2 * self.scale)) * np.cos(
#                     np.pi * (2 * n + 1) * q / (2 * self.scale))
#         return basis
#
#     def load_DCT_basis(self):
#         basis_64 = np.zeros((self.num_filters, self.scale, self.scale))
#         idx = 0
#         for i in range(self.scale * 2 - 1):
#             cur = max(0, i - self.scale + 1)
#             for j in range(cur, i - cur + 1):
#                 if idx >= self.num_filters + self.start:
#                     return basis_64.reshape((self.num_filters, 1, self.scale, self.scale))
#                 if idx >= self.start:
#                     basis_64[idx - self.start, :, :] = self.cal_basis(j, i - j)
#                 idx = idx + 1
#                 if idx >= self.num_filters + self.start:
#                     return basis_64.reshape((self.num_filters, 1, self.scale, self.scale))
#
#     def dct2d_Conv(self):
#         return nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=self.scale, stride=self.scale,
#                           bias=False)
#
#     def forward(self, input):
#         # bs, _,_,_ = input.shape()
#         # input.view(bs*(128//self.scale)**2, 3, self.scale,self.scale)
#         dct_outs = torch.cat([self.conv1(input[:, 0:1, ...]), self.conv2(input[:, 1:2, ...]), self.conv3(input[:, 2:3, ...])], dim=1)
#         dct_reallocate = torch.cat([dct_outs[:, index:index + 1, ...] for index in self.swap], dim=1)
#         return dct_reallocate

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class SplicingDetectionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(SplicingDetectionNet, self).__init__()

        # # RGB branch
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        #
        # self.stage1_cfg = extra['STAGE1']
        # num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        # block = blocks_dict[self.stage1_cfg['BLOCK']]
        # num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        # self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        # stage1_out_channel = block.expansion * num_channels
        #
        # self.stage2_cfg = extra['STAGE2']
        # num_channels = self.stage2_cfg['NUM_CHANNELS']
        # block = blocks_dict[self.stage2_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))]
        # self.transition1 = self._make_transition_layer(
        #     [stage1_out_channel], num_channels)
        # self.stage2, pre_stage_channels = self._make_stage(
        #     self.stage2_cfg, num_channels)
        #
        # self.stage3_cfg = extra['STAGE3']
        # num_channels = self.stage3_cfg['NUM_CHANNELS']
        # block = blocks_dict[self.stage3_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))]
        # self.transition2 = self._make_transition_layer(
        #     pre_stage_channels, num_channels)
        # self.stage3, pre_stage_channels = self._make_stage(
        #     self.stage3_cfg, num_channels)
        #
        # self.stage4_cfg = extra['STAGE4']
        # num_channels = self.stage4_cfg['NUM_CHANNELS']
        # block = blocks_dict[self.stage4_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))]
        # self.transition3 = self._make_transition_layer(
        #     pre_stage_channels, num_channels)
        # self.stage4, RGB_final_channels = self._make_stage(
        #     self.stage4_cfg, num_channels, multi_scale_output=True)

        # DCT coefficient branch
        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(in_channels=21,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      dilation=8,
                      padding=8),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        # self.dc_layer1 = self._make_layer(HalfDilatedBlock, inplanes=32, planes=32, blocks=1, stride=1)
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.dc_layer2 = self._make_layer(BasicBlock, inplanes=4 * 64 * 2, planes=96, blocks=4, stride=1)

        self.dc_stage3_cfg = extra['DC_STAGE3']
        num_channels = self.dc_stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.dc_stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.dc_transition2 = self._make_transition_layer(
            [96], num_channels)
        self.dc_stage3, pre_stage_channels = self._make_stage(
            self.dc_stage3_cfg, num_channels)

        self.dc_stage4_cfg = extra['DC_STAGE4']
        num_channels = self.dc_stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.dc_stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.dc_transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.dc_stage4, DC_final_stage_channels = self._make_stage(
            self.dc_stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(DC_final_stage_channels))  # 672

        # DC_final_stage_channels.insert(0, 0)  # to match # branches


        # # stage 5
        # self.stage5_cfg = extra['STAGE5']
        # num_channels = self.stage5_cfg['NUM_CHANNELS']
        # block = blocks_dict[self.stage5_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))]
        # self.transition4 = self._make_transition_layer(
        #     [i+j for (i, j) in zip(RGB_final_channels, DC_final_stage_channels)], num_channels)
        # self.stage5, pre_stage_channels = self._make_stage(
        #     self.stage5_cfg, num_channels)
        #
        # last_inp_channels = sum(pre_stage_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, qtable):
        # RGB, DCTcoef = x[:, :3, :, :], x[:, 3:, :, :]
        DCTcoef = x

        # # RGB branch
        # x = self.conv1(RGB)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.layer1(x)
        #
        # x_list = []
        # for i in range(self.stage2_cfg['NUM_BRANCHES']):
        #     if self.transition1[i] is not None:
        #         x_list.append(self.transition1[i](x))
        #     else:
        #         x_list.append(x)
        # y_list = self.stage2(x_list)
        #
        # x_list = []
        # for i in range(self.stage3_cfg['NUM_BRANCHES']):
        #     if self.transition2[i] is not None:
        #         x_list.append(self.transition2[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # y_list = self.stage3(x_list)
        #
        # x_list = []
        # for i in range(self.stage4_cfg['NUM_BRANCHES']):
        #     if self.transition3[i] is not None:
        #         x_list.append(self.transition3[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # RGB_list = self.stage4(x_list)

        # DC coefficient branch
        x = self.dc_layer0_dil(DCTcoef)
        # x = self.dc_layer1(x)
        x = self.dc_layer1_tail(x)
        B, C, H, W = x.shape
        x0 = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8,
                                                                                     W // 8)  # [B, 256, 32, 32]
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)  # [B, C, 8, 8, 32, 32]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, 32, 32]
        x1 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x = torch.cat([x0, x1], dim=1)
        x = self.dc_layer2(x)  # x.shape = torch.Size([1, 96, 64, 64])

        x_list = []
        for i in range(self.dc_stage3_cfg['NUM_BRANCHES']):
            if self.dc_transition2[i] is not None:
                x_list.append(self.dc_transition2[i](x))
            else:
                x_list.append(x)
        y_list = self.dc_stage3(x_list)

        x_list = []
        for i in range(self.dc_stage4_cfg['NUM_BRANCHES']):
            if self.dc_transition3[i] is not None:
                x_list.append(self.dc_transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.dc_stage4(x_list)

        # # stage 5
        # x = [torch.cat([RGB_list[i+1], DC_list[i]], 1) for i in range(self.stage5_cfg['NUM_BRANCHES']-1)]
        # x.insert(0, RGB_list[0])
        # x_list = []
        # for i in range(self.stage5_cfg['NUM_BRANCHES']):
        #     if self.transition4[i] is not None:
        #         x_list.append(self.transition4[i](x[i]))
        #     else:
        #         x_list.append(x[i])
        # x = self.stage5(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        # x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2], 1)

        x = self.last_layer(x)

        return x

    def init_weights(self, pretrained_rgb='', pretrained_dct='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # if m.kernel_size==(8,8):
                #     continue
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # if os.path.isfile(pretrained_rgb):
        #     # loaded_dict = torch.load(pretrained_rgb)['state_dict']
        #     loaded_dict = torch.load(pretrained_rgb)
        #     model_dict = self.state_dict()
        #     # loaded_dict = {k.replace('dc_', 'RGB_'):v for k, v in loaded_dict.items()}
        #     loaded_dict = {k: v for k, v in loaded_dict.items()
        #                        if k in model_dict.keys() and not k.startswith('lost_layer.')}  # RGB weight
        #     logger.info('=> (RGB) loading pretrained model {} ({})'.format(pretrained_rgb, len(loaded_dict)))
        #     model_dict.update(loaded_dict)
        #     self.load_state_dict(model_dict)
        #     #for param in self.named_parameters():
        #     #    if param[0] in loaded_dict.keys():
        #     #        param[1].requires_grad = True  # freeze RGB part
        # else:
        #     logger.warning('=> Cannot load pretrained RGB')
        if os.path.isfile(pretrained_dct):
            loaded_dict = torch.load(pretrained_dct)['state_dict']
            model_dict = self.state_dict()
            loaded_dict = {k: v for k, v in loaded_dict.items()
                               if k in model_dict.keys()}  # DC weight
            loaded_dict = {k:v for k,v in loaded_dict.items()
                           if not k.startswith('last_layer')}
            logger.info('=> (DCT) loading pretrained model {} ({})'.format(pretrained_dct, len(loaded_dict)))
            model_dict.update(loaded_dict)
            self.load_state_dict(model_dict)
            #for param in self.named_parameters():
            #    if param[0] in loaded_dict.keys():
            #       param[1].requires_grad = True  # False = freeze DCT part
        else:
            logger.warning('=> Cannot load pretrained DCT')


def get_seg_model(cfg, **kwargs):
    model = SplicingDetectionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED_RGB, cfg.MODEL.PRETRAINED_DCT)

    return model
