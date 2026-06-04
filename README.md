# Inverse Neural Operator

Inverse neural operator models and utilities for solving inverse problems.

## System Requirements

### Software Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- NumPy
- SciPy
- Matplotlib
- datasets
- safetensors
- torch-tb-profiler
- scikit-image
- function-encoder (https://github.com/ajthor/function-encoder.git)

### Operating Systems

This software has been tested on:
- Linux (various distributions)
- macOS

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support is highly recommended for training and inference. The code can run on CPU but performance will be significantly reduced.
- **Tested on**: NVIDIA RTX A5000
- **Memory**: Sufficient RAM to load datasets (varies by dataset size)

## Installation Guide

### Using Dev Container (Recommended)

This repository provides a dev container configuration for Docker:

1. Install Docker on your system
2. Install VS Code with the Dev Containers extension
3. Open this repository in VS Code
4. When prompted, click "Reopen in Container" or run the command "Dev Containers: Reopen in Container"

The container will automatically set up the complete development environment.

### Manual Installation

Alternatively, install dependencies manually:

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

**Note**: Installation time for dependencies varies (5-30 minutes) depending on your internet connection and whether PyTorch/CUDA components need to be downloaded.

## Demo

### Current Overhaul Status

This branch is being migrated from bash-script orchestration to config-driven stage
entrypoints. The function encoder stage is the first migrated stage and supports
single-GPU and raw PyTorch DDP execution through `python -m torch.distributed.run`.

Plan function encoder jobs without launching training:

```bash
python -m inverse_neural_operator.experiments.plan \
  configs/experiments/fwi_function_encoders.yaml
```

Run the validated two-GPU smoke test inside the dev container:

```bash
python -m torch.distributed.run --nproc_per_node 2 \
  -m inverse_neural_operator.function_encoders.train \
  --config configs/experiments/fwi_function_encoders_ddp_smoke.yaml \
  --encoder-type input \
  --seed 1 \
  --models-dir /tmp/b2b-ddp-smoke-models \
  --results-dir /tmp/b2b-ddp-smoke-results \
  --execute
```

Run the matching output encoder by changing `--encoder-type output`.

### Expected Runtime

- **Demo**: < 5 minutes (using pre-trained models and small dataset samples)
- **Full training**: Several hours for most datasets on NVIDIA RTX A5000
- **FWI dataset training**: Several days on NVIDIA RTX A5000

## Instructions for Use

### Output Directory Configuration

The overhaul separates uploadable model artifacts from lightweight run outputs:

- `B2B_MODELS_DIR` is required for commands that write model weights. This should
  point at store-backed storage intended for Hugging Face upload.
- `B2B_RESULTS_DIR` is optional and defaults to `./results` for TensorBoard logs,
  latest recovery checkpoints, metrics, and plots.

Function encoder model artifacts use:

```text
<B2B_MODELS_DIR>/models/<dataset>/function_encoders/<artifact>/seed_<seed>/
```

Run outputs use:

```text
<B2B_RESULTS_DIR>/<dataset>/function_encoders/<artifact>/seed_<seed>/<encoder_type>/
```

### Running on Your Own Data

1. Prepare your dataset in the appropriate format (see dataset documentation)
2. Add or update a YAML config under `configs/experiments/`
3. Use `python -m inverse_neural_operator.experiments.plan <config>` to inspect
   planned jobs and artifact paths before launching
4. Launch the relevant stage entrypoint with `--execute`

### Available Stage Entrypoints

- `python -m inverse_neural_operator.experiments.plan` - Plan jobs and inspect
  artifact paths without launching training
- `python -m inverse_neural_operator.experiments.status` - Check which planned
  artifacts are missing or complete
- `python -m inverse_neural_operator.function_encoders.train` - Train migrated
  function encoders

## Reproduction Instructions

This overhaul branch is not currently a manuscript reproduction branch. The old
bash orchestration, training scripts, evaluation scripts, and plotting modules
were removed from this worktree so the new config-driven pipeline can become the
source of truth. The `main` worktree remains the reference for old behavior while
stages are ported.

To reproduce results on this branch after migration:

1. Download or stream the configured datasets
2. Train each migrated stage from YAML experiment configs
3. Generate evaluation metrics with the new evaluation stage once ported
4. Generate figures with the new plotting/reporting path once ported

## License

This software is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite our paper:

[Citation information to be added upon publication]

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the corresponding author(s).
