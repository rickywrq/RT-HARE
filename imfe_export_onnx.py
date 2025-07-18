"""
RT-HARE IMFE Model Export to ONNX Format

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
This script exports the IMFE model to ONNX format for distributed deployment.

"""

import torch
from utils.model import DISTILLATION_RAFT_MODEL
import os

if __name__ == "__main__":

    # Initialize model and load checkpoint
    model = DISTILLATION_RAFT_MODEL(resnet_weights=None)
    checkpoint_path = "imfe_checkpoint/2024-02-04-04-13-03-checkpoint-8.pt"
    print(f"=> Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    # Set model to CUDA
    model.cuda()

    # Define random input tensor with shape matching the model's expected input
    # Adjust the dimensions as per model requirements
    random_input = torch.randn([1, 6, 3, 256, 344]).cuda()  # Example shape (batch_size, channels, height, width)

    # Create folder if it doesn't exist
    folder_path = "./onnx_model"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Export to ONNX
    torch.onnx.export(
        model,
        random_input,
        "./onnx_model/imfe.onnx",
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        verbose=True
    )
    print("Model exported to ONNX format.")
