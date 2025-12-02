import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from utils.device import get_device


@dataclass
class DistributedConfig:
    """Lightweight container for distributed training state."""

    is_distributed: bool
    world_size: int
    rank: int
    local_rank: int
    device: str


class NullSummaryWriter:
    """No-op writer used on non-main ranks to avoid file collisions."""

    def add_scalar(self, *args, **kwargs):
        return None

    def add_scalars(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def init_distributed_mode(requested_device: Optional[str] = None) -> DistributedConfig:
    """
    Initialize torch.distributed (if needed) and return configuration metadata.

    If WORLD_SIZE is 1 (or unset) the function is a no-op.
    """
    if not dist.is_available():
        device = get_device(requested_device)
        return DistributedConfig(False, 1, 0, 0, device)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = get_device(requested_device)
        return DistributedConfig(False, 1, 0, 0, device)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if backend == "nccl":
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return DistributedConfig(True, world_size, rank, local_rank, device)


def cleanup_distributed():
    """Tear down distributed process group when training finishes."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True when running on rank zero."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier():
    """Convenience wrapper that performs a distributed barrier if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
