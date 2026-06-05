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


@dataclass
class FunctionEncoderConfig:
    artifact: str = "default"
    encoder_types: List[str] = field(default_factory=lambda: ["input", "output"])
    basis: BasisConfig = field(default_factory=BasisConfig)
    regularization: float = 1e-3
    batch_size: int = 4
    epochs: int = 1
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
    epochs: int = 1
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
class BaselinesConfig:
    models: List[str] = field(default_factory=lambda: ["ifno"])
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 1e-4
    checkpoint_interval: int = 1
    ifno: IFNOConfig = field(default_factory=IFNOConfig)


@dataclass
class InverseModelsConfig:
    models: List[str] = field(default_factory=lambda: ["nonlinear"])
    function_encoder_artifact: str = "default"
    forward_model: Optional[str] = None
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    prediction_loss_weight: float = 1.0
    coefficient_loss_weight: float = 0.0
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 1e-4
    checkpoint_interval: int = 1
    linear_regularization: float = 1e-6


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
    baselines_without_ifno = {
        key: value for key, value in baselines_raw.items() if key != "ifno"
    }
    baselines = _dataclass_from_mapping(BaselinesConfig, baselines_without_ifno)
    baselines.ifno = ifno
    inverse_models = _dataclass_from_mapping(
        InverseModelsConfig,
        _require_mapping(raw.get("inverse_models", {}), "inverse_models"),
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
        raw=raw,
    )
