"""
IMFE Feature Extraction Script

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

Description:

This script performs feature extraction on video RGB frames using a pre-trained 
deep learning model. The code is designed for extracting and saving features in 
a structured format suitable for downstream tasks like video analysis and 
behavior recognition. Key components of the script include dataset handling, 
model loading, and feature saving functionality.

- **Input Data Structure**: The RGB frames for each video should be stored in a 
directory as follows:
  
  ```
  rawframes_rgb/
  ├── video_0001/
  │   ├── img_00001.jpg
  │   ├── img_00002.jpg
  │   └── ...
  ├── video_0002/
  │   ├── img_00001.jpg
  │   ├── img_00002.jpg
  │   └── ...
  ```

- **Feature Extraction**: 
  Using the `DISTILLATION_RAFT_MODEL`, each batch of frames is passed through 
  the model in evaluation mode. The extracted features are stored temporarily in
  a dictionary, then saved in `.npy` files for each video in the following 
  format:

  ```
  features/
  ├── video_0001.npy
  ├── video_0002.npy
  └── ...
  ```
"""

import os
import numpy as np
import torch
from torchvision import transforms as T
from dataloader import FeatureExtractionData
from tqdm import tqdm
from utils.model import DISTILLATION_RAFT_MODEL
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser(description="IMFE Feature Extraction")
parser.add_argument(
    "--data_dir",
    type=str,
    default="data_feature_extraction/50salads",
    help="Path to the data directory",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="imfe_checkpoint/2024-02-04-04-13-03-checkpoint-8.pt",
    help="Path to the checkpoint file",
)
args = parser.parse_args()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model and load checkpoint
model = DISTILLATION_RAFT_MODEL().to(device)
checkpoint_path = args.checkpoint_path
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model"])
print("=> Checkpoint loaded")

# Set model to evaluation mode
model.eval()

# Define dataset and data loader for inference
data_dir = args.data_dir
frame_dir = os.path.join(data_dir, "rawframes_rgb")
feature_dir = os.path.join(data_dir, "features")
inference_set = FeatureExtractionData(frame_dir, image_h_w=(256, 344))
inference_loader = torch.utils.data.DataLoader(
    inference_set, batch_size=16, shuffle=False, num_workers=8
)

# Prepare for feature extraction
features = {}
output_dir = feature_dir
os.makedirs(output_dir, exist_ok=True)
pbar = tqdm(enumerate(inference_loader), total=len(inference_loader))

# Perform inference
with torch.no_grad():
    for idx, (input_data, names) in pbar:
        input_data = input_data.to(device)
        output = model(input_data).detach().cpu().numpy()

        # Store features for each video
        for i, name in enumerate(names):
            if name not in features:
                features[name] = []
            features[name].append(output[i])

        # Update progress bar
        pbar.set_postfix({"Current Video": names[0]})

# Save extracted features
for name, feature_array in features.items():
    np.save(os.path.join(output_dir, f"{name}.npy"), feature_array)
    print(f"Features saved for video: {name}")
