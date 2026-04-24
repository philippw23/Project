"""Distributed training utilities.

Supports two launch modes:
  - torchrun (sets RANK, LOCAL_RANK, WORLD_SIZE env vars)
  - SLURM srun (sets SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS env vars)

Usage in pretrain.py:
    from biomedclip import distributed

    if args.distributed:
        distributed.enable()
    ...
    if distributed.is_main_process():
        wandb.init(...)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def enable() -> None:
    """Initialize the process group.

    Detects the launch mode from environment variables and calls
    init_process_group with nccl backend, then pins this process to its
    assigned GPU.
    """
    if dist.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank       = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank % torch.cuda.device_count()))
    else:
        return

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    dist.barrier()


def is_enabled() -> bool:
    """Return True when the process group has been initialized."""
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    """Return True only for rank 0 (or when not running distributed)."""
    return get_global_rank() == 0


def get_global_rank() -> int:
    return dist.get_rank() if is_enabled() else 0


def get_global_size() -> int:
    return dist.get_world_size() if is_enabled() else 1


def get_local_rank() -> int:
    """Return the local GPU index for this process."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    return 0


def synchronize() -> None:
    """Global barrier — all processes wait here before continuing."""
    if is_enabled():
        dist.barrier()
