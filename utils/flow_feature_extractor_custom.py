'''
Original code is under the PyTorch license, which is a BSD 3-Clause License.

    BSD 3-Clause License

    Copyright (c) Soumith Chintala 2016, 
    All rights reserved.

Source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

Modifications by Ruiqi Wang for the RT-HARE project are licensed under the Apache License, Version 2.0.

Copyright (c) 2024, Ruiqi Wang,
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

'''
import torch
import torch.nn as  nn
import torch.nn.functional as F
from torch import Tensor

from .resnet import ResNet, Bottleneck


class ResNetFeatureHead(ResNet):
    '''
    see utils/resnet.py:(137)class ResNet(nn.Module): for details.
    '''
    def __init__(self, *args, **kwargs):
        self.start_point_downscale = kwargs.pop('start_point_downscale', -1)
        assert self.start_point_downscale in [1,2,4,8]
        ResNet.__init__(self, *args, **kwargs)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # 3,224，224
        if self.start_point_downscale == 1:
            x = self.conv1(x)   
            # ResNet50: 3,224，224 -> 64,112,112 (1/2)
        if self.start_point_downscale <= 2:
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x) 
            # ResNet50: 64,112,112 -> 64,56,56 (1/4)
        if self.start_point_downscale <= 4:

            x = self.layer1(x)  
            # ResNet50: 64,56,56 -> 256,56,56 (1/4)
            x = self.layer2(x) 
            # ResNet50: 256,56,56 -> 512,28,28 (1/8)
        if self.start_point_downscale <= 8:
            x = self.layer3(x)  
            # ResNet50: 512,28,28 -> 1024,14,14 (1/16)
        if self.start_point_downscale <= 16:
            x = self.layer4(x)  # ResNet50: 1024,4,14 -> 2048,7,7

            x = self.avgpool(x) # ResNet50: 2048,7,7 -> 2048,1,1
            x = torch.flatten(x, 1)
            # x = self.fc(x)

        return x
        
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def ResNet50FeatureHead(start_point_downscale=8, num_classes=1000):
    '''
    @input_param
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    '''
    # return ResNetFeatureHead(start_point_downscale, Bottleneck, [3,4,6,3], num_classes, )
    return ResNetFeatureHead(Bottleneck, [3,4,6,3], num_classes=num_classes, start_point_downscale=start_point_downscale)