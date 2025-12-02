import argparse
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.process_data import (
    InputFunctionEncoderDataset,
    OutputFunctionEncoderDataset,
)

from b2b.function_encoder import (
    create_model as create_function_encoder,
    train as train_function_encoder,
    save as save_function_encoder,
    memory_efficient_inner_product,
)
from utils.device import set_seed, dataset_on_cpu
from utils.params import save_params
from utils.checkpoints import setup_checkpoint_dir
from utils.args import load_defaults_from_yaml
from data.load_dataset import load_dataset
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

# Encoder type arg
parser.add_argument("--encoder_type", type=str, choices=["input", "output"])

# Dataset args
parser.add_argument("--dataset", type=str)
parser.add_argument("--model", type=str)

# Function encoder args
parser.add_argument("--n_basis", type=int)
parser.add_argument("--hidden_sizes", type=int, nargs="+")
parser.add_argument("--regularization", type=float)

# Training args
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--learning_rate", type=float)

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
    help="Number of mini-batches to accumulate before stepping optimizer",
)

# Load defaults from YAML
defaults_path = os.path.join(
    os.path.dirname(__file__), "train_function_encoder_defaults.yaml"
)
defaults = load_defaults_from_yaml(defaults_path)
parser.set_defaults(**defaults)

params = parser.parse_args()

if params.encoder_type not in ["input", "output"]:
    raise ValueError(f"Unknown encoder type: {params.encoder_type}")

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

match params.encoder_type:
    case "input":
        model_name = "input_function_encoder"
    case "output":
        model_name = "output_function_encoder"
    case _:
        raise ValueError(f"Unknown encoder type: {params.encoder_type}")

# Function encoders use "shared" as the model name and save to shared directory
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

# Save args to shared directory (where function encoder is saved)
if is_main_process():
    os.makedirs(shared_dir, exist_ok=True)
    save_params(params, shared_dir, filename_prefix=f"{model_name}_params")

# Create checkpoint directories
params.checkpoint_dir = setup_checkpoint_dir(params.checkpoint_dir, log_dir)

# Load dataset using utility
model_train_dataset = load_dataset(
    params.dataset, params, dataset_device, split="train"
)
model_test_dataset = load_dataset(params.dataset, params, dataset_device, split="test")

dataset_info = model_train_dataset.get_info()

match params.encoder_type:
    case "input":
        train_dataset = InputFunctionEncoderDataset(model_train_dataset, device=device)
        test_dataset = InputFunctionEncoderDataset(model_test_dataset, device=device)
    case "output":
        train_dataset = OutputFunctionEncoderDataset(model_train_dataset, device=device)
        test_dataset = OutputFunctionEncoderDataset(model_test_dataset, device=device)
    case _:
        raise ValueError(f"Unknown encoder type: {params.encoder_type}")

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

match params.encoder_type:
    case "input":
        input_size = dataset_info["X_size"]
        output_size = dataset_info["u_size"]
    case "output":
        input_size = dataset_info["Y_size"]
        output_size = dataset_info["s_size"]
    case _:
        raise ValueError(f"Unknown encoder type: {params.encoder_type}")


function_encoder = create_function_encoder(
    input_size=input_size,
    hidden_sizes=params.hidden_sizes,
    output_size=output_size,
    n_basis=params.n_basis,
    inner_product=(
        memory_efficient_inner_product if params.dataset in ["fwi"] else None
    ),
    regularization=params.regularization,
)
# function_encoder = torch.compile(function_encoder)
function_encoder.to(device)
optimizer = torch.optim.Adam(function_encoder.parameters(), lr=params.learning_rate)

if dist_config.is_distributed:
    function_encoder = torch.nn.parallel.DistributedDataParallel(
        function_encoder,
        device_ids=[dist_config.local_rank] if use_cuda else None,
        output_device=dist_config.local_rank if use_cuda else None,
        broadcast_buffers=False,
    )

# Train the function encoder
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

train_function_encoder(
    model=function_encoder,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    n_epochs=params.epochs,
    summary_writer=writer,
    model_name=model_name,
    resume_from_checkpoint=params.resume,
    checkpoint_dir=params.checkpoint_dir,
    checkpoint_interval=params.checkpoint_interval,
    device=device,
    grad_accumulation_steps=params.grad_accumulation_steps,
)

# Save the function encoder
if is_main_process():
    os.makedirs(shared_dir, exist_ok=True)
    model_to_save = (
        function_encoder.module if dist_config.is_distributed else function_encoder
    )
    save_function_encoder(
        model=model_to_save,
        path=os.path.join(shared_dir, f"{model_name}.safetensors"),
    )

cleanup_distributed()
