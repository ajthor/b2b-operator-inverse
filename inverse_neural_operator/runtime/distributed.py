"""Small raw-DDP helpers used by new training entrypoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def setup_distributed(device_preference: str = "cuda") -> DistributedContext:
    """Initialize torch.distributed from torchrun environment variables if present."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if device_preference == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch.distributed.init_process_group(backend=backend)

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if not context.is_distributed:
        return value
    reduced = value.detach().clone()
    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    reduced = reduced / context.world_size
    return reduced


def barrier(context: DistributedContext) -> None:
    if context.is_distributed:
        torch.distributed.barrier()
