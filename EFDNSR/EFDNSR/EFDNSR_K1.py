# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
import torch.nn as nn
import block as block


def make_model(args, parent=False):
    model = EFDNSR_K1()

    return model

class EFDNSR_K1(nn.Module):


    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=52,
                 upscale=4):
        super(EFDNSR, self).__init__()

        self.conv_1 = block.conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = block.RLFB(feature_channels)
        self.block_2 = block.RLFB(feature_channels)
        self.block_3 = block.RLFB(feature_channels)
        self.block_4 = block.RLFB(feature_channels)
        self.block_5 = block.RLFB(feature_channels)
        self.block_6 = block.RLFB(feature_channels)

        self.conv_2 = block.conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = block.pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x, y):

        out_feature_early = self.conv_1(x) + self.conv_1(y)

        out_b1 = self.block_1(out_feature_early)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b6) + out_feature_early
        output = self.upsampler(out_low_resolution)

        return output
