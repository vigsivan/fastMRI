"""
Trains End 2 End VarNet
"""
import argparse
import os
from pathlib import Path
import random

import numpy as np

import torch
import torch.distributed as dist
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils import tensorboard

import fastmri
from fastmri.data.mri_data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.models import VarNet

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train E2E VarNet")

    parser.add_argument(
        "--training-dir",
        type=(
            lambda x: x
            if Path(x).is_dir()
            else parser.error("Invalid training directory")
        ),
        required=True,
        help="Path to training directory",
    )

    parser.add_argument(
        "--validation-dir",
        type=(
            lambda x: x
            if Path(x).is_dir()
            else parser.error("Invalid validation directory")
        ),
        required=True,
        help="Path to validation directory",
    )

    parser.add_argument(
        "--challenge",
        type=str,
        choices=["singlecoil", "multicoil"],
        default="multicoil",
        help="One of singlecoil or multicoil",
    )

    parser.add_argument(
        "--mask-type",
        type=str,
        choices=["equispaced", "equispaced_fraction", "magic", "magic_fraction"],
        default="equispaced_fraction",
    )
    parser.add_argument(
        "--center-fractions",
        type=float,
        nargs="+",
        default=[0.08],
    )
    parser.add_argument(
        "--accelerations",
        type=int,
        nargs="+",
        default=[4],
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-step-size", type=int, default=40)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--init-method", default="tcp://127.0.0.1:3456", type=str, help=""
    )
    parser.add_argument("--dist-backend", default="gloo", type=str, help="")
    parser.add_argument("--world-size", default=1, type=int, help="")
    parser.add_argument("--distributed", action="store_true", help="")

    parser.add_argument("--logdir", type=str, default="./log", help="")
    parser.add_argument("--epochs-per-val", type=int, default=5)

    return parser


def init_ddp_process_group(args: argparse.Namespace) -> int:
    ngpus = torch.cuda.device_count()

    local_rank = os.environ.get("SLURM_LOCALID")
    node_id = os.environ.get("SLURM_NODEID")
    assert local_rank is not None and node_id is not None

    rank = int(node_id) * ngpus + int(local_rank)
    current_device = int(local_rank)
    torch.cuda.set_device(current_device)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=rank,
    )
    return current_device


def main():
    args = get_parser().parse_args()
    current_device = init_ddp_process_group(args)
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    writer = tensorboard.writer.SummaryWriter(log_dir=log_dir)

    for seed in (torch.manual_seed, np.random.seed, random.seed):
        seed(args.seed)

    mask = create_mask_for_mask_type(
        mask_type_str=args.mask_type,
        center_fractions=args.center_fractions,
        accelerations=args.accelerations,
    )

    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)

    train_dataset = SliceDataset(
        root=args.training_dir,
        challenge=args.challenge,
        transform=train_transform,
        use_dataset_cache=True,
    )

    val_dataset = SliceDataset(
        root=args.validation_dir,
        challenge=args.challenge,
        transform=val_transform,
        use_dataset_cache=True,
    )

    model = VarNet().cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[current_device]
    )

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler
    )

    criterion = fastmri.SSIMLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in args.epochs:
        epoch_loss = []
        for x in train_loader:
            reconstructed_image = model(x)
            loss = criterion(x, reconstructed_image)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        scheduler.step()
        writer.add_scalar("Train loss", np.mean(epoch_loss))

        if epoch % args.epochs_per_val == 0:
            model.eval()
            with torch.no_grad():
                val_loss = []
                for x in val_loader:
                    reconstructed_image = model(x)
                    loss = criterion(x, reconstructed_image)
                    loss.backward()
                    val_loss.append(loss.item())
                writer.add_scalar("Validation loss", np.mean(val_loss))
