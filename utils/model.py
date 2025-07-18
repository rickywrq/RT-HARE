"""
RT-HARE

Copyright (c) 2024, Ruiqi Wang and Peiqi Gao,
Washington University in St. Louis.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Portions of this code are adapted from torchvision (https://github.com/pytorch/vision/tree/main), which is licensed under the BSD 3-Clause License. The BSD 3-Clause 
licensed portions are governed by the terms below:

    Copyright (c) Soumith Chintala 2016, 
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following
    conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer
       in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
       derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Description:
This module defines the model architecture for the IMFE model, which is based on the RAFT model with additional
components for feature extraction and distillation. It is designed to extract features from video frames and
optical flow for downstream tasks such as video analysis and behavior recognition.

References:
- The RAFT-related code is adapted from: https://pytorch.org/vision/main/_modules/torchvision/models/optical_flow/raft.html
"""


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation

from .raft_torch_vision import FeatureEncoder
from .flow_feature_extractor_custom import ResNetFeatureHead, Bottleneck

import numpy as np

class CustomMotionEncoderNoFlow(nn.Module):
    """
    The motion encoder, part of the update block.
    Takes the current predicted flow and the correlation features as input and returns an encoded version of these.
    """

    def __init__(self, *, in_channels_corr, corr_layers=(256, 192), flow_layers=(128, 64), out_channels=128):
        super().__init__()

        if len(flow_layers) != 2:
            raise ValueError(f"The expected number of flow_layers is 2, instead got {len(flow_layers)}")
        if len(corr_layers) not in (1, 2):
            raise ValueError(f"The number of corr_layers should be 1 or 2, instead got {len(corr_layers)}")

        self.convcorr1 = Conv2dNormActivation(in_channels_corr, corr_layers[0], norm_layer=None, kernel_size=1)
        if len(corr_layers) == 2:
            self.convcorr2 = Conv2dNormActivation(corr_layers[0], corr_layers[1], norm_layer=None, kernel_size=3)
        else:
            self.convcorr2 = nn.Identity()

        self.convflow1 = Conv2dNormActivation(2, flow_layers[0], norm_layer=None, kernel_size=7)
        self.convflow2 = Conv2dNormActivation(flow_layers[0], flow_layers[1], norm_layer=None, kernel_size=3)

        # out_channels - 2 because we cat the flow (2 channels) at the end
        self.conv = Conv2dNormActivation(
            corr_layers[-1] ,
            out_channels ,
            norm_layer=None,
            kernel_size=3
        )

        self.out_channels = out_channels

    def forward(self, corr_features):
        corr = self.convcorr1(corr_features)
        corr = self.convcorr2(corr)

        corr_flow = self.conv(corr)
        return corr_flow

class CustomCorrBlock(nn.Module):
    """The correlation block.

    Creates a correlation pyramid with ``num_levels`` levels from the outputs of the feature encoder,
    and then indexes from this pyramid to create correlation features.
    The "indexing" of a given centroid pixel x' is done by concatenating its surrounding neighbors that
    are within a ``radius``, according to the infinity norm (see paper section 3.2).
    Note: typo in the paper, it should be infinity norm, not 1-norm.
    """

    def __init__(self, *, num_levels: int = 1, radius: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius

        self.corr_pyramid: List[Tensor] = [torch.tensor(0)]  # useless, but torchscript is otherwise confused :')

        # The neighborhood of a centroid pixel x' is {x' + delta, ||delta||_inf <= radius}
        # so it's a square surrounding x', and its sides have a length of 2 * radius + 1
        # The paper claims that it's ||.||_1 instead of ||.||_inf but it's a typo:
        # https://github.com/princeton-vl/RAFT/issues/122
        self.out_channels = num_levels * (2 * radius + 1) ** 2
        # self.skip_add = nn.quantized.FloatFunctional()    # Modified CPSL


    def build_pyramid_correlation(self, fmap1, fmap2):
        corr_volume = self._compute_corr_volume(fmap1, fmap2)

        batch_size, h, w, _, _, _ = corr_volume.shape  # _, _ = h, w
        corr_volume = corr_volume.reshape(batch_size, 1, h * w, h, w).transpose(0,1)
        return corr_volume

    def _compute_corr_volume(self, fmap1, fmap2):
        batch_size, num_channels, h, w = fmap1.detach().shape
        fmap1 = fmap1.view(batch_size, num_channels, h * w)
        fmap2 = fmap2.view(batch_size, num_channels, h * w)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        # corr = self.skip_add.matmul(fmap1.transpose(1, 2), fmap2)    # Modified CPSL
        corr = corr.view(batch_size, h, w, 1, h, w)
        # return corr / torch.sqrt(torch.tensor(num_channels))    # Modified CPSL
        return corr / np.sqrt(num_channels)

class DISTILLATION_RAFT_MODEL(nn.Module):
    def __init__(self, *, mask_predictor=None, resnet_weights=None):
        super().__init__()

        self.feature_encoder = FeatureEncoder()
        self.corr_block = CustomCorrBlock()
        self.motion_encoder = CustomMotionEncoderNoFlow(in_channels_corr=128*5)
        self.compress_conv = nn.Conv2d(1376,128,3,padding=1)
        self.mid_conv = nn.Conv2d(128,512,1)
        self.mask_predictor = mask_predictor
        self.resnet = ResNetFeatureHead(Bottleneck, [3,4,6,3], num_classes=1000, start_point_downscale=8)

        ### Load pre-trained weights
        PATH = resnet_weights
        if resnet_weights and os.path.isfile(PATH):
            print("=> loading checkpoint '{}'".format(PATH))
            checkpoint = torch.load(PATH)
            self.resnet.load_state_dict(checkpoint)
        ###


    def forward(self, batched_images):

        batch_size, N, C, h, w = batched_images.shape # 4,6,3,344,256

        x = batched_images.view(-1, C, h, w) # torch.Size([24, 3, 344, 256])

        fmaps = self.feature_encoder(x) # [24, 256, 43, 32]

        # if fmaps.shape[-2:] != (h // 8, w // 8):
        #     raise ValueError("The feature encoder should downsample H and W by 8")

        fmaps = fmaps.view(batch_size, N, self.feature_encoder.output_dim, *fmaps.shape[-2:]).transpose(0,1) # [6,4,256,32,43]

        corr_features = []
        for i in range(N-1):
            c_f = self.corr_block.build_pyramid_correlation(fmaps[i,...], fmaps[i+1,...])
            corr_features.append(c_f)
        corr_features = torch.cat(corr_features, dim=0)
        # corr_features -> (N-1), B, hw, h, w

        # As in the original paper, the actual output of the context encoder is split in 2 parts:
        # - one part is used to initialize the hidden state of the recurent units of the update block
        # - the rest is the "actual" context.
        # coords1 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
        # corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)
        corr_features = corr_features.transpose(0,1) # (N-1),B,hw,h,w -> B,(N-1),hw,h,w
        corr_features = corr_features.reshape(-1, *corr_features.shape[-3:])
        corr_features = self.compress_conv(corr_features)

        corr_features = corr_features.reshape(batch_size, -1, *corr_features.shape[-2:])
        motion_features = self.motion_encoder(corr_features)
        raft_out = self.mid_conv(motion_features)
        out = self.resnet(raft_out)
        return out

if __name__ == "__main__":
    model = DISTILLATION_RAFT_MODEL()
    model(torch.rand(4,6,3,344,256))
