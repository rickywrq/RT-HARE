"""
RT-HARE

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
This module contains the training script for the IMFE model in a distributed 
setting.

References:
This code was developed with reference to PyTorch's official example for 
distributed data parallel (DDP) training, 
which demonstrates best practices for multi-GPU training. For more details, see:

    PyTorch DDP Tutorial Series - Multi-GPU Training:
    https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py

"""


# Standard library imports
import os
import argparse
import datetime
from pathlib import Path
from pprint import pprint, pformat

# Third-party library imports
import numpy as np
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

# torchvision imports
import torchvision
from torchvision.transforms import v2

# Custom imports
from dataloader import DistillationData
from utils.model import DISTILLATION_RAFT_MODEL
from utils.logger import setup_logger, AverageMeter, Summary, ProgressMeter

# deterministic
random_seed = 0  # or any of your favorite number
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# data paths
rgb_path, feat_path = "data/anet/rawframes_rgb", "data/anet/flow"


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        gpu_id: int,
        cfg,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data

        for key in cfg:
            setattr(self, key, cfg[key])
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    def _run_val(self, epoch):
        if self.gpu_id == 0:
            self.logger.info(
                f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
            )
            self.logger.info(f"[GPU{self.gpu_id}] START validation ")
            self.logger.info(
                f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
            )
        losses = AverageMeter("Loss", ":.6f")
        progress = ProgressMeter(
            len(self.val_data),
            [losses],
            prefix="[GPU{}] Epoch: [{}]".format(self.gpu_id, epoch),
        )

        b_sz = len(next(iter(self.val_data))[0])
        if self.gpu_id == 0:
            self.logger.info(
                f"[GPU{self.gpu_id}] VALIDATION Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.val_data)}"
            )
        self.val_data.sampler.set_epoch(epoch)

        self.model.eval()
        pbar = tqdm(enumerate(self.val_data), total=len(self.val_data))
        with torch.no_grad():
            for i, (source, targets) in pbar:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)

                output = self.model(source)
                loss = F.mse_loss(output, targets)
                losses.update(loss.item())
        losses.all_reduce()
        if self.gpu_id == 0:
            progress.display(i + 1, self.logger)

        losses.reset()
        self.model.train()

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        PATH = (
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            + "-"
            + "checkpoint-"
            + str(epoch)
            + ".pt"
        )
        PATH = os.path.join(self.save_path, PATH)
        torch.save(checkpoint, PATH)
        self.logger.info(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        if self.args.amp:
            with torch.autocast(device_type="cuda", enabled=True):
                output = self.model(source)
                loss = F.mse_loss(output, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(source)
            loss = F.mse_loss(output, targets)
            loss.backward()
            self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        losses = AverageMeter("Loss", ":.6f")
        progress = ProgressMeter(
            len(self.train_data),
            [losses],
            prefix="[GPU{}] Epoch: [{}]".format(self.gpu_id, epoch),
        )

        b_sz = len(next(iter(self.train_data))[0])
        num_steps = len(self.train_data)
        self.logger.info(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {num_steps}"
        )
        self.train_data.sampler.set_epoch(epoch)

        self.model.train()
        pbar = tqdm(enumerate(self.train_data), total=len(self.train_data))
        for i, (source, targets) in pbar:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            losses.update(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            if num_steps // 20 == 0 or i % (num_steps // 20) == 0:
                losses.all_reduce()
                progress.display(i + 1, self.logger)
            if (i + 1) % (num_steps // 5) == 0:
                self.scheduler.step()

        losses.reset()

    def train(self, max_epochs: int):
        cur_epoch = self.epoch + 1

        for epoch in range(cur_epoch, max_epochs):
            if self.gpu_id == 0:
                self.logger.info(
                    f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                )

            self.epoch = epoch
            self._run_epoch(epoch)

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

            self._run_val(epoch)

            if self.gpu_id == 0:
                self.logger.info(
                    f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
                )
                self.logger.info(f"[GPU{self.gpu_id}] END of Epoch {epoch} ")
                self.logger.info(
                    f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                )
            torch.distributed.barrier()


def load_train_objs(debug):
    """
    Load the training objects.

    Args:
        debug (bool): Flag indicating whether to run in debug mode.

    Returns:
        train_set (DistillationData): The training dataset.
        model (DISTILLATION_RAFT_MODEL): The model for distillation.
        optimizer (torch.optim.AdamW): The optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler.ExponentialLR): The learning rate scheduler.

    """
    train_set = DistillationData(
        rgb_path, feat_path, mode="train", debug=debug
    )  # load your dataset
    model = DISTILLATION_RAFT_MODEL(
        resnet_weights='imfe_checkpoint/resnet50-11ad3fa6.pth'
    )  # load your model
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    return train_set, model, optimizer, scheduler


def prepare_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle=True,
    pin_memory=False,
    drop_last=False,
):
    """
    Prepare a data loader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): The batch size for the data loader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        pin_memory (bool, optional): Whether to pin memory for faster data transfer. Defaults to False.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.

    Returns:
        DataLoader: The prepared data loader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=False,  # this has to be false, shuffle will be handled by sampler.
        sampler=DistributedSampler(dataset, shuffle=shuffle),
        drop_last=drop_last,
        num_workers=1,
    )


def main(
    rank: int,
    world_size: int,
    save_every: int,
    total_epochs: int,
    batch_size: int,
    save_path: str,
    debug,
    args,
):
    """
    Main function for training the IMFE model in a distributed setting.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        save_every (int): The frequency of saving checkpoints during training.
        total_epochs (int): The total number of epochs to train for.
        batch_size (int): The batch size for training.
        save_path (str): The path to save the trained model.
        debug: Debug flag.
        args: Additional arguments.

    Returns:
        None
    """

    ddp_setup(rank, world_size)
    logger = setup_logger(save_path)

    dataset, model, optimizer, scheduler = load_train_objs(debug)
    scaler = torch.cuda.amp.GradScaler()

    logger.info(f"{rank} Begin to prepare dataset.")
    train_data = prepare_dataloader(dataset, batch_size, shuffle=True)
    val_dataset = DistillationData(rgb_path,
                                   feat_path,
                                   mode="val",
                                   debug=debug)
    val_data = prepare_dataloader(val_dataset,
                                  batch_size,
                                  shuffle=False,
                                  pin_memory=False,
                                  drop_last=True)
    logger.info(f"{rank} Dataset ready.")
    logger.info("Training set has {} instances".format(len(dataset)))

    model.to(rank)

    if not args.ckpt:
        ckpt_epoch = -1
    elif args.finetune:
        checkpoint = torch.load(args.ckpt)
        ckpt_model = checkpoint["model"]
        ckpt_epoch = checkpoint["epoch"]
    else:
        print("=> loading previous training from '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt)
        ckpt_model = checkpoint["model"]
        ckpt_epoch = checkpoint["epoch"]
        ckpt_optimizer = checkpoint["optimizer"]
        ckpt_lr_sched = checkpoint["scheduler"]
        ckpt_scaler = checkpoint["scaler"]
        model.load_state_dict(ckpt_model)
        scaler.load_state_dict(ckpt_scaler)
        optimizer.load_state_dict(ckpt_optimizer)
        scheduler.load_state_dict(ckpt_lr_sched)

    training_cfg = dict(
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=ckpt_epoch,
        logger=logger,
        save_path=save_path,
        save_every=save_every,
        args=args,
    )
    if rank == 0:
        logger.info(pformat(args))
        logger.info(pformat(training_cfg))

    trainer = Trainer(model, train_data, val_data, rank, training_cfg)
    trainer.train(total_epochs)
    destroy_process_group()
    logger.info("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="simple distributed training job for the combined model."
    )
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    parser.add_argument(
        "--save_path", default=".", type=str, help="Place to save the ckpt and log."
    )
    parser.add_argument("--ckpt", default="", type=str, help="Place load the ckpt.")
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        help="Whether to run with the smaller dataset. Used for debugging the code.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        help="Whether to torch AUTOMATIC MIXED PRECISION.",
    )
    parser.add_argument(
        "--finetune", action=argparse.BooleanOptionalAction, help="Whether to finetune."
    )
    args = parser.parse_args()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(args.save_path)

    logger.info(" ")
    logger.info("In main thread, start")

    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(
            world_size,
            args.save_every,
            args.total_epochs,
            args.batch_size,
            args.save_path,
            args.debug,
            args,
        ),
        nprocs=world_size,
    )
