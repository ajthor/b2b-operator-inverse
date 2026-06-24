"""Typed configuration helpers for overhaul experiment YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class DatasetConfig:
    name: str
    source: Optional[str] = None
    sample_limit: Optional[int] = None


@dataclass
class RuntimeConfig:
    dry_run: bool = True
    launcher: str = "torchrun"
    nproc_per_node: int = 1
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    env: Dict[str, str] = field(default_factory=dict)
    tensorboard_dir: Optional[str] = None


@dataclass
class MatrixConfig:
    seeds: List[int] = field(default_factory=lambda: [1])
    stages: List[str] = field(default_factory=list)


@dataclass
class BasisConfig:
    kind: str = "mlp"
    n_basis: int = 100
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 512, 512])
    activation: str = "relu"
    omega_0: float = 30.0
    omega_start: Optional[float] = None
    omega_end: Optional[float] = None
    omega_warmup_steps: int = 0
    winner_samples: int = 0
    winner_noise_scale: float = 1.0
    winner_first_layer_scale: float = 1.0
    winner_hidden_layer_scale: float = 1.0


@dataclass
class FunctionEncoderConfig:
    artifact: str = "default"
    encoder_types: List[str] = field(default_factory=lambda: ["input", "output"])
    basis: BasisConfig = field(default_factory=BasisConfig)
    basis_chunk_size: Optional[int] = None
    coefficient_grad: bool = True
    regularization: float = 1e-3
    orthonormality_loss_weight: float = 0.0
    ssim_loss_weight: float = 0.0
    ssim_max_points: Optional[int] = None
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    sample_with_replacement: bool = False
    epochs: int = 1
    max_steps: Optional[int] = None
    eval_interval: int = 250
    eval_batches: Optional[int] = None
    log_interval: int = 10
    learning_rate: float = 1e-4
    checkpoint_interval: int = 1


@dataclass
class ForwardModelsConfig:
    models: List[str] = field(default_factory=lambda: ["b2b_nonlinear"])
    function_encoder_artifact: str = "default"
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    coefficient_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 1.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    sample_with_replacement: bool = False
    epochs: int = 1
    max_steps: Optional[int] = None
    eval_interval: int = 250
    eval_batches: Optional[int] = None
    log_interval: int = 10
    learning_rate: float = 1e-4
    checkpoint_interval: int = 1
    linear_regularization: float = 1e-6


@dataclass
class IFNOConfig:
    modes: int = 16
    width: int = 64
    n_layers: int = 3
    beta: float = 2.0
    padding: int = 20
    vae_latent_dim: int = 24
    intermediate_dim: int = 64
    epochs_vae: int = 0
    epochs_ifno: int = 0
    lr_vae: float = 1e-4
    lr_ifno: float = 5e-3
    lr_forward: float = 1e-4
    lr_backward: Optional[float] = None


@dataclass
class InvertibleDeepONetConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    trunk_hidden_sizes: Optional[List[int]] = None
    n_coupling_layers: int = 6
    regularization: float = 1e-3
    forward_loss_weight: float = 1.0
    inverse_loss_weight: float = 1.0
    loss: str = "relative_l2"


@dataclass
class NIOConfig:
    branch_hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    trunk_hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    n_basis: int = 64
    lifting_channels: int = 32
    modes: int = 16
    n_fourier_layers: int = 3
    measurement_points: Optional[int] = None
    loss: str = "l1"


@dataclass
class BaselinesConfig:
    models: List[str] = field(default_factory=lambda: ["ifno"])
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    sample_with_replacement: bool = False
    epochs: int = 1
    max_steps: Optional[int] = None
    eval_interval: int = 250
    eval_batches: Optional[int] = None
    log_interval: int = 10
    learning_rate: float = 1e-4
    checkpoint_interval: int = 1
    ifno: IFNOConfig = field(default_factory=IFNOConfig)
    invertible_deeponet: InvertibleDeepONetConfig = field(
        default_factory=InvertibleDeepONetConfig
    )
    nio: NIOConfig = field(default_factory=NIOConfig)


@dataclass
class InverseModelsConfig:
    models: List[str] = field(default_factory=lambda: ["nonlinear"])
    function_encoder_artifact: str = "default"
    forward_model: Optional[str] = None
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    latent_size: int = 128
    n_coupling_layers: int = 6
    n_components: int = 5
    prediction_loss_weight: float = 1.0
    coefficient_loss_weight: float = 0.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    sample_with_replacement: bool = False
    epochs: int = 1
    max_steps: Optional[int] = None
    eval_interval: int = 250
    eval_batches: Optional[int] = None
    log_interval: int = 10
    learning_rate: float = 1e-4
    checkpoint_interval: int = 1
    linear_regularization: float = 1e-6


@dataclass
class PDEValidationConfig:
    enabled: bool = False
    backend: str = "external"
    command: Optional[str] = None
    max_samples: int = 32
    metrics: List[str] = field(
        default_factory=lambda: ["mae", "mse", "rmse", "relative_l2", "ssim"]
    )
    overwrite: bool = False


@dataclass
class ExperimentConfig:
    experiment: str
    description: str = ""
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(name="fwi"))
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    function_encoders: FunctionEncoderConfig = field(
        default_factory=FunctionEncoderConfig
    )
    forward_models: ForwardModelsConfig = field(default_factory=ForwardModelsConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    inverse_models: InverseModelsConfig = field(default_factory=InverseModelsConfig)
    pde_validation: PDEValidationConfig = field(default_factory=PDEValidationConfig)
    raw: Dict[str, Any] = field(default_factory=dict)


def _require_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping.")
    return value


def _dataclass_from_mapping(cls, data: Dict[str, Any]):
    field_names = set(cls.__dataclass_fields__.keys())
    kwargs = {key: value for key, value in data.items() if key in field_names}
    return cls(**kwargs)


def _normalize_env(value: Any) -> Dict[str, str]:
    if value is None:
        return {}
    raw_env = _require_mapping(value, "runtime.env")
    return {str(key): str(item) for key, item in raw_env.items()}


def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load an experiment YAML file into typed config objects."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _require_mapping(raw, str(path))

    dataset = _dataclass_from_mapping(
        DatasetConfig, _require_mapping(raw.get("dataset", {}), "dataset")
    )
    runtime = _dataclass_from_mapping(
        RuntimeConfig, _require_mapping(raw.get("runtime", {}), "runtime")
    )
    runtime.env = _normalize_env(runtime.env)
    matrix = _dataclass_from_mapping(
        MatrixConfig, _require_mapping(raw.get("matrix", {}), "matrix")
    )

    fe_raw = _require_mapping(raw.get("function_encoders", {}), "function_encoders")
    basis = _dataclass_from_mapping(
        BasisConfig, _require_mapping(fe_raw.get("basis", {}), "function_encoders.basis")
    )
    fe_without_basis = {key: value for key, value in fe_raw.items() if key != "basis"}
    function_encoders = _dataclass_from_mapping(
        FunctionEncoderConfig, fe_without_basis
    )
    function_encoders.basis = basis
    forward_models = _dataclass_from_mapping(
        ForwardModelsConfig,
        _require_mapping(raw.get("forward_models", {}), "forward_models"),
    )
    baselines_raw = _require_mapping(raw.get("baselines", {}), "baselines")
    ifno = _dataclass_from_mapping(
        IFNOConfig,
        _require_mapping(baselines_raw.get("ifno", {}), "baselines.ifno"),
    )
    invertible_deeponet = _dataclass_from_mapping(
        InvertibleDeepONetConfig,
        _require_mapping(
            baselines_raw.get("invertible_deeponet", {}),
            "baselines.invertible_deeponet",
        ),
    )
    nio = _dataclass_from_mapping(
        NIOConfig,
        _require_mapping(baselines_raw.get("nio", {}), "baselines.nio"),
    )
    baselines_without_ifno = {
        key: value for key, value in baselines_raw.items() if key != "ifno"
    }
    baselines_without_nested = {
        key: value
        for key, value in baselines_without_ifno.items()
        if key not in {"invertible_deeponet", "nio"}
    }
    baselines = _dataclass_from_mapping(BaselinesConfig, baselines_without_nested)
    baselines.ifno = ifno
    baselines.invertible_deeponet = invertible_deeponet
    baselines.nio = nio
    inverse_models = _dataclass_from_mapping(
        InverseModelsConfig,
        _require_mapping(raw.get("inverse_models", {}), "inverse_models"),
    )
    pde_validation = _dataclass_from_mapping(
        PDEValidationConfig,
        _require_mapping(raw.get("pde_validation", {}), "pde_validation"),
    )

    experiment = raw.get("experiment")
    if not experiment:
        raise ValueError(f"{path} is missing required key: experiment")

    return ExperimentConfig(
        experiment=experiment,
        description=raw.get("description", ""),
        dataset=dataset,
        runtime=runtime,
        matrix=matrix,
        function_encoders=function_encoders,
        forward_models=forward_models,
        baselines=baselines,
        inverse_models=inverse_models,
        pde_validation=pde_validation,
        raw=raw,
    )
