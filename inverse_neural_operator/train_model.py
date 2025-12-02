import argparse

import torch
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

from b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)
from b2b.load_model import load_forward_model
from b2b.load_model import (
    load_function_encoders,
    load_function_encoder_params,
)

from data.load_dataset import load_dataset
from models.create_model import create_model

from utils.device import set_seed, dataset_on_cpu
from utils.params import save_params
from utils.checkpoints import setup_checkpoint_dir
from utils.args import load_defaults_from_yaml
from utils.imports import import_model_functions
from config.paths import get_runs_dir, get_model_dir, get_shared_dir
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
parser.add_argument(
    "--forward_model",
    type=str,
    help="Forward model name for re-simulation loss (e.g., b2b_nonlinear, b2b_linear)",
)

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
defaults_path = os.path.join(os.path.dirname(__file__), "train_model_defaults.yaml")
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

log_dir = str(
    get_runs_dir(
        params.dataset, params.model, params.seed, base_dir_override=params.base_dir
    )
)
model_dir = str(
    get_model_dir(
        params.dataset, params.model, params.seed, base_dir_override=params.base_dir
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

# Save args to model directory
if is_main_process():
    os.makedirs(model_dir, exist_ok=True)
    save_params(params, model_dir)

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

# Get the appropriate train/save functions for the model
train_model, save_model = import_model_functions(params.model, "train", "save")

# Load function encoder parameters to get sizes for model creation
input_encoder_params, output_encoder_params = load_function_encoder_params(shared_dir)

# Create model with correct sizes (pass sizes for all models, some will ignore them)
model, optimizer = create_model(
    params.model,
    params,
    dataset_info,
    device,
    input_encoder_params.n_basis,  # input size (alpha coefficients)
    output_encoder_params.n_basis,  # output size (beta coefficients)
)

# Wrap in DistributedDataParallel if needed
if dist_config.is_distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank] if use_cuda else None,
        output_device=dist_config.local_rank if use_cuda else None,
        broadcast_buffers=False,
    )

# Load function encoders (all models will receive them, some may not use them)
input_function_encoder, output_function_encoder = load_function_encoders(
    shared_dir, dataset_info, params, device
)

# Load forward model (all models will receive it, some may not use it)
try:
    forward_model = load_forward_model(
        shared_dir, device=device, forward_model_name=params.forward_model
    )
    print(f"Loaded forward model '{params.forward_model}' for re-simulation loss")
except Exception as e:
    print(f"Warning: Could not load forward model for re-simulation loss: {e}")
    forward_model = None

# Train model

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

prefetch_factor = (
    params.prefetch_factor
    if effective_num_workers > 0 and params.prefetch_factor is not None
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

# Single consistent training function call for all models
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
    forward_model=forward_model,
)

# Save model
if is_main_process():
    os.makedirs(model_dir, exist_ok=True)
    model_to_save = model.module if dist_config.is_distributed else model
    save_model(model=model_to_save, path=os.path.join(model_dir, "model.safetensors"))

cleanup_distributed()
