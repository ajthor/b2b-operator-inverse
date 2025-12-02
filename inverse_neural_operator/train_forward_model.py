import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

from utils.device import set_seed, dataset_on_cpu
from utils.params import save_params
from utils.checkpoints import setup_checkpoint_dir
from utils.args import load_defaults_from_yaml
from utils.imports import import_model_functions
from data.load_dataset import load_dataset
from b2b.load_model import load_function_encoder_params, load_function_encoders
from b2b.create_model import create_forward_model
from config.paths import get_runs_dir, get_shared_dir
from utils.distributed import (
    init_distributed_mode,
    cleanup_distributed,
    is_main_process,
    NullSummaryWriter,
)

torch.set_float32_matmul_precision("high")

# Parse command line args
parser = argparse.ArgumentParser()

# Dataset args
parser.add_argument("--dataset", type=str)

# Model args
parser.add_argument("--model", type=str)
parser.add_argument("--hidden_sizes", type=int, nargs="+")

# Training args
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--lambda_u", type=float)

# Path args
parser.add_argument(
    "--base_dir",
    type=str,
    default=None,
    help="Base directory for models/results/logs (overrides B2B_RESULTS_DIR / ./results fallback)",
)

# Device args
parser.add_argument("--device", type=str)

# Seed args
parser.add_argument("--seed", type=int)

# Dataset device args
parser.add_argument(
    "--dataset_device",
    type=str,
    help='Location for dataset tensors ("cpu", "cuda", or "same" to match training device)',
)

# Checkpoint args
parser.add_argument("--checkpoint_interval", type=int)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--resume", type=bool)

# DataLoader args
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument(
    "--prefetch_factor",
    type=int,
    default=None,
    help="Number of batches to prefetch per worker (requires num_workers > 0)",
)
parser.add_argument(
    "--grad_accumulation_steps",
    type=int,
    default=1,
    help="Mini-batches to accumulate before each optimizer step",
)

# Load defaults from YAML
defaults_path = os.path.join(
    os.path.dirname(__file__), "train_forward_model_defaults.yaml"
)
defaults = load_defaults_from_yaml(defaults_path)
parser.set_defaults(**defaults)

params = parser.parse_args()

dist_config = init_distributed_mode(params.device)
device = dist_config.device
use_cuda = isinstance(device, str) and device.startswith("cuda")
if is_main_process():
    print(
        f"Using device: {device} "
        f"(distributed={dist_config.is_distributed}, world_size={dist_config.world_size})"
    )
set_seed(params.seed + dist_config.rank)

dataset_device = params.dataset_device or "cpu"
if dataset_device in ("same", "match"):
    dataset_device = device

# Forward models use "shared" as the model name and save to shared directory
log_dir = str(
    get_runs_dir(
        params.dataset, "shared", params.seed, base_dir_override=params.base_dir
    )
)
shared_dir = str(
    get_shared_dir(params.dataset, params.seed, base_dir_override=params.base_dir)
)

# Create SummaryWriter
if is_main_process():
    writer = SummaryWriter(log_dir=log_dir)
else:
    writer = NullSummaryWriter()

# Save args to shared directory (where forward model is saved)
if is_main_process():
    os.makedirs(shared_dir, exist_ok=True)
    save_params(params, shared_dir)

# Create checkpoint directories
params.checkpoint_dir = setup_checkpoint_dir(params.checkpoint_dir, log_dir)

# Load dataset using utility
train_dataset = load_dataset(params.dataset, params, dataset_device, split="train")
test_dataset = load_dataset(params.dataset, params, dataset_device, split="test")
dataset_info = train_dataset.get_info()

train_pin_memory = use_cuda and dataset_on_cpu(train_dataset)
test_pin_memory = use_cuda and dataset_on_cpu(test_dataset)

effective_num_workers = params.num_workers
if not train_pin_memory and effective_num_workers > 0:
    if is_main_process():
        print(
            "Dataset tensors already on accelerator; forcing num_workers=0 to avoid CUDA access in worker processes."
        )
    effective_num_workers = 0
persistent_workers = effective_num_workers > 0
prefetch_factor = (
    params.prefetch_factor
    if effective_num_workers > 0 and params.prefetch_factor is not None
    else None
)
# Load function encoder parameters to get sizes for model creation
input_encoder_params, output_encoder_params = load_function_encoder_params(shared_dir)

# Get the appropriate train/save functions based on model type
train_model, save_model = import_model_functions(params.model, "train", "save")

# Create forward model and optimizer
model, optimizer = create_forward_model(
    params.model,
    params,
    input_encoder_params.n_basis,  # input size (alpha coefficients)
    output_encoder_params.n_basis,  # output size (beta coefficients)
    device,
)

if dist_config.is_distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank] if use_cuda else None,
        output_device=dist_config.local_rank if use_cuda else None,
        broadcast_buffers=False,
    )

# Load function encoders (forward models need them for training)
input_function_encoder, output_function_encoder = load_function_encoders(
    shared_dir, dataset_info, params, device
)

# Train forward model

train_sampler = (
    DistributedSampler(
        train_dataset,
        num_replicas=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=True,
    )
    if dist_config.is_distributed
    else None
)
test_sampler = (
    DistributedSampler(
        test_dataset,
        num_replicas=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=False,
    )
    if dist_config.is_distributed
    else None
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=params.batch_size,
    shuffle=train_sampler is None,
    sampler=train_sampler,
    num_workers=effective_num_workers,
    pin_memory=train_pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=params.batch_size,
    shuffle=False,
    sampler=test_sampler,
    num_workers=effective_num_workers,
    pin_memory=test_pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)

train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    n_epochs=params.epochs,
    summary_writer=writer,
    model_name=params.model,
    params=params,
    resume_from_checkpoint=params.resume,
    checkpoint_dir=params.checkpoint_dir,
    checkpoint_interval=params.checkpoint_interval,
    device=device,
)

# Save forward model with model-specific name in shared directory
if is_main_process():
    os.makedirs(shared_dir, exist_ok=True)
    model_to_save = model.module if dist_config.is_distributed else model
    save_model(
        model=model_to_save,
        path=os.path.join(shared_dir, f"forward_{params.model}.safetensors"),
    )

cleanup_distributed()
