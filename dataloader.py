"""
IMFE Data Loader Utilities

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

Description:
This module defines dataset classes for loading and processing RGB frames and optical flow features used in distillation
and feature extraction tasks. The `DistillationData` class supports loading both RGB frames and optical flow features for 
training, validation, and inference, while the `FeatureExtractionData` class is optimized for loading RGB frames for 
inference only.

Each video’s RGB frames should be organized in subdirectories, and optical flow features should be stored as `.npy` files 
with corresponding names for seamless data association.

Classes:
- `DistillationData`: Loads RGB frames and optical flow features for distillation tasks, using a specified frame 
  interval and clip length.
- `FeatureExtractionData`: Loads only RGB frames for feature extraction in inference mode.
"""


import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import torchvision
from torchvision.transforms import v2

__all__ = ("DistillationData",)


rgb_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False
)


class DistillationData(Dataset):
    """
    Dataset for loading RGB frames and Optical Flow features for distillation.
    The dataset will load 6 RGB frames and corresponding optical flow features.
    """
    def __init__(
        self, rgb_folder, flow_folder, mode="train", image_h_w=(256, 344), debug=False
    ):
        """
        rgb_folder: Path to rgb
        flow_folder: Path to flow
        """
        self.mode = mode
        self.frame_interval = 6
        self.clip_len = 6
        self.image_h_w = image_h_w
        self.rgb_folder = rgb_folder
        self.flow_folder = flow_folder

        if mode == "train":
            if debug:
                self.file_list = sorted(os.listdir(self.rgb_folder))[:1]
            else:
                self.file_list = sorted(os.listdir(self.rgb_folder))[:-300]
        elif mode == "val":
            if debug:
                self.file_list = sorted(os.listdir(self.rgb_folder))[-1:]
            else:
                self.file_list = sorted(os.listdir(self.rgb_folder))[-300:]

        elif mode == "inference":
            self.file_list = sorted(os.listdir(self.rgb_folder))

        else:
            raise ValueError(f"expecting mode to be train or val. got {mode}")

        self.data = []
        for file in self.file_list:

            rgb_temp_list = os.listdir(str(self.rgb_folder) + "/" + str(file))
            rgb_temp_list = sorted(rgb_temp_list, key=lambda x: x[4:9])
            flow_feat_path = str(self.flow_folder) + "/" + str(file) + ".npy"

            # Create frame indices e.g., [1,2,3,4,5] [7,8,9,10,11]
            # so that the indeces are consistent with the configuration
            # in mmaction2.
            total_frames = len(rgb_temp_list) - 1
            clip_centers = np.arange(
                self.frame_interval // 2, total_frames, self.frame_interval
            )
            frame_inds = (
                clip_centers[:, None]
                + np.arange(
                    -(self.clip_len // 2), self.clip_len - (self.clip_len // 2)
                )[None, :]
                + 1
            )
            frame_inds = np.clip(frame_inds, 0, total_frames - 1)

            # store the data cache as [folder_name, [rgb_names], of_feature]
            temp_data = []
            for idx, f_id in enumerate(frame_inds):
                temp_data.append(
                    [file, [rgb_temp_list[i] for i in f_id], flow_feat_path, idx]
                )
            self.data += temp_data

        self.transforms = v2.Compose(
            [
                v2.Resize(size=image_h_w, antialias=True),
                v2.ToDtype(torch.float32),
                v2.Normalize(mean=rgb_norm_cfg["mean"], std=rgb_norm_cfg["std"]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid_name, rgb_name_list, feature_path, feature_idx = self.data[idx]

        feature = np.load(feature_path)[feature_idx]

        vid_path = os.path.join(self.rgb_folder, vid_name)
        img_path = [os.path.join(vid_path, img_name) for img_name in rgb_name_list]

        img_batch = torch.empty((6, 3, *self.image_h_w), dtype=torch.float32)
        for i, img_p in enumerate(img_path):
            img_batch[i] = self.transforms(torchvision.io.read_image(img_p))

        if self.mode == "inference":
            return img_batch, feature, vid_name
        return img_batch, feature



class FeatureExtractionData(Dataset):
    """
    Dataset for loading RGB frames for feature extraction in inference mode.
    The dataset loads 6 RGB frames per sample.
    """
    def __init__(self, rgb_folder, image_h_w=(256, 344), debug=False):
        """
        rgb_folder: Path to RGB frames
        """
        self.frame_interval = 6
        self.clip_len = 6
        self.image_h_w = image_h_w
        self.rgb_folder = rgb_folder

        # Define file list for inference mode
        self.file_list = sorted(os.listdir(self.rgb_folder))[:1] if debug else sorted(os.listdir(self.rgb_folder))

        # Prepare data entries
        self.data = []
        for file in self.file_list:
            rgb_temp_list = sorted(os.listdir(os.path.join(self.rgb_folder, file)), key=lambda x: x[4:9])

            # Generate frame indices
            total_frames = len(rgb_temp_list) - 1
            clip_centers = np.arange(self.frame_interval // 2, total_frames, self.frame_interval)
            frame_inds = (
                clip_centers[:, None] +
                np.arange(-(self.clip_len // 2), self.clip_len - (self.clip_len // 2))[None, :] + 1
            )
            frame_inds = np.clip(frame_inds, 0, total_frames - 1)

            # Store data as [folder_name, [rgb_names]]
            temp_data = [[file, [rgb_temp_list[i] for i in f_id]] for f_id in frame_inds]
            self.data += temp_data

        # Define transformations
        rgb_norm_cfg = {
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375]
        }
        self.transforms = v2.Compose(
            [
                v2.Resize(size=image_h_w, antialias=True),
                v2.ToDtype(torch.float32),
                v2.Normalize(mean=rgb_norm_cfg["mean"], std=rgb_norm_cfg["std"]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid_name, rgb_name_list = self.data[idx]
        vid_path = os.path.join(self.rgb_folder, vid_name)
        img_paths = [os.path.join(vid_path, img_name) for img_name in rgb_name_list]

        # Load and transform RGB frames
        img_batch = torch.empty((self.clip_len, 3, *self.image_h_w), dtype=torch.float32)
        for i, img_p in enumerate(img_paths):
            img_batch[i] = self.transforms(torchvision.io.read_image(img_p))

        return img_batch, vid_name



if __name__ == "__main__":
    dataloader = DistillationData("anet/v1-2/rawframes_rgb", "anet/v1-2/features/flow")
    print(next(iter(dataloader)))
