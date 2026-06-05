"""Dry-run experiment matrix expansion and artifact status."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Iterable, List, Optional

from inverse_neural_operator.config.schema import ExperimentConfig
from inverse_neural_operator.forward.artifacts import missing_forward_model_files
from inverse_neural_operator.function_encoders.artifacts import (
    missing_function_encoder_files,
)
from inverse_neural_operator.inverse.artifacts import missing_inverse_model_files
from inverse_neural_operator.runtime.paths import (
    model_artifact_dir,
    models_root,
    results_root,
    run_artifact_dir,
)


@dataclass
class PlannedJob:
    stage: str
    name: str
    encoder_type: Optional[str]
    dataset: str
    seed: int
    model_dir: Optional[Path]
    run_dir: Path
    command: str
    complete: bool
    dependencies: List[str]
    missing_dependencies: List[str]

    @property
    def blocked(self) -> bool:
        return bool(self.missing_dependencies)


def _torchrun_prefix(config: ExperimentConfig) -> str:
    env_prefix = ""
    if config.runtime.env:
        assignments = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in sorted(config.runtime.env.items())
        )
        env_prefix = f"env {assignments} "
    if config.runtime.launcher == "torchrun":
        return env_prefix + (
            "python -m torch.distributed.run "
            f"--nproc_per_node {config.runtime.nproc_per_node}"
        )
    return env_prefix + "python3"


def _with_root_args(
    command: str,
    models_dir_override: Optional[str],
    results_dir_override: Optional[str],
    tensorboard_dir: Optional[str] = None,
) -> str:
    if models_dir_override:
        command += f" --models-dir {models_dir_override}"
    if results_dir_override:
        command += f" --results-dir {results_dir_override}"
    if tensorboard_dir:
        command += f" --tensorboard-dir {shlex.quote(tensorboard_dir)}"
    return command


def _dependency_label(path: Path, missing: List[str]) -> str:
    return f"{path} missing {', '.join(missing)}"


def _function_encoder_dependency(
    config: ExperimentConfig,
    models_dir: Optional[Path],
    seed: int,
) -> tuple[List[str], List[str]]:
    if models_dir is None:
        return [], ["B2B_MODELS_DIR unset; cannot check function encoder artifact"]
    path = model_artifact_dir(
        models_dir,
        config.dataset.name,
        "function_encoders",
        config.forward_models.function_encoder_artifact,
        seed,
    )
    missing = missing_function_encoder_files(path)
    dependencies = [str(path)]
    missing_dependencies = [_dependency_label(path, missing)] if missing else []
    return dependencies, missing_dependencies


def _inverse_function_encoder_dependency(
    config: ExperimentConfig,
    models_dir: Optional[Path],
    seed: int,
) -> tuple[List[str], List[str]]:
    if models_dir is None:
        return [], ["B2B_MODELS_DIR unset; cannot check function encoder artifact"]
    path = model_artifact_dir(
        models_dir,
        config.dataset.name,
        "function_encoders",
        config.inverse_models.function_encoder_artifact,
        seed,
    )
    missing = missing_function_encoder_files(path)
    dependencies = [str(path)]
    missing_dependencies = [_dependency_label(path, missing)] if missing else []
    return dependencies, missing_dependencies


def _forward_dependency(
    config: ExperimentConfig,
    models_dir: Optional[Path],
    seed: int,
) -> tuple[List[str], List[str]]:
    if not config.inverse_models.forward_model:
        return [], []
    if models_dir is None:
        return [], ["B2B_MODELS_DIR unset; cannot check forward model artifact"]
    path = model_artifact_dir(
        models_dir,
        config.dataset.name,
        "forward_models",
        config.inverse_models.forward_model,
        seed,
    )
    missing = missing_forward_model_files(path)
    dependencies = [str(path)]
    missing_dependencies = [_dependency_label(path, missing)] if missing else []
    return dependencies, missing_dependencies


def _function_encoder_jobs(
    config: ExperimentConfig,
    models_dir: Optional[Path],
    results_dir: Path,
    config_path: str,
    models_dir_override: Optional[str],
    results_dir_override: Optional[str],
) -> Iterable[PlannedJob]:
    fe = config.function_encoders
    dataset = config.dataset.name
    for seed in config.matrix.seeds:
        model_dir = (
            model_artifact_dir(
                models_dir,
                dataset,
                "function_encoders",
                fe.artifact,
                seed,
            )
            if models_dir is not None
            else None
        )
        for encoder_type in fe.encoder_types:
            run_dir = run_artifact_dir(
                results_dir,
                dataset,
                "function_encoders",
                fe.artifact,
                seed,
            ) / encoder_type
            expected_name = f"{encoder_type}_encoder.safetensors"
            complete = bool(model_dir and (model_dir / expected_name).exists())
            prefix = _torchrun_prefix(config)
            command = (
                f"{prefix} -m inverse_neural_operator.function_encoders.train "
                f"--config {config_path} --encoder-type {encoder_type} "
                f"--seed {seed} --execute"
            )
            command = _with_root_args(
                command,
                models_dir_override,
                results_dir_override,
                config.runtime.tensorboard_dir,
            )
            yield PlannedJob(
                stage="function_encoders",
                name=fe.artifact,
                encoder_type=encoder_type,
                dataset=dataset,
                seed=seed,
                model_dir=model_dir,
                run_dir=run_dir,
                command=command,
                complete=complete,
                dependencies=[],
                missing_dependencies=[],
            )


def _forward_model_jobs(
    config: ExperimentConfig,
    models_dir: Optional[Path],
    results_dir: Path,
    config_path: str,
    models_dir_override: Optional[str],
    results_dir_override: Optional[str],
) -> Iterable[PlannedJob]:
    forward_config = config.forward_models
    dataset = config.dataset.name
    for seed in config.matrix.seeds:
        for model_name in forward_config.models:
            model_dir = (
                model_artifact_dir(
                    models_dir,
                    dataset,
                    "forward_models",
                    model_name,
                    seed,
                )
                if models_dir is not None
                else None
            )
            run_dir = run_artifact_dir(
                results_dir,
                dataset,
                "forward_models",
                model_name,
                seed,
            )
            complete = bool(model_dir and not missing_forward_model_files(model_dir))
            dependencies, missing_dependencies = _function_encoder_dependency(
                config,
                models_dir,
                seed,
            )
            prefix = _torchrun_prefix(config)
            command = (
                f"{prefix} -m inverse_neural_operator.forward.train "
                f"--config {config_path} --model {model_name} "
                f"--seed {seed} --execute"
            )
            command = _with_root_args(command, models_dir_override, results_dir_override)
            yield PlannedJob(
                stage="forward_models",
                name=model_name,
                encoder_type=None,
                dataset=dataset,
                seed=seed,
                model_dir=model_dir,
                run_dir=run_dir,
                command=command,
                complete=complete,
                dependencies=dependencies,
                missing_dependencies=missing_dependencies,
            )


def _baseline_jobs(
    config: ExperimentConfig,
    models_dir: Optional[Path],
    results_dir: Path,
    config_path: str,
    models_dir_override: Optional[str],
    results_dir_override: Optional[str],
) -> Iterable[PlannedJob]:
    dataset = config.dataset.name
    for seed in config.matrix.seeds:
        for model_name in config.baselines.models:
            model_dir = (
                model_artifact_dir(
                    models_dir,
                    dataset,
                    "baselines",
                    model_name,
                    seed,
                )
                if models_dir is not None
                else None
            )
            run_dir = run_artifact_dir(
                results_dir,
                dataset,
                "baselines",
                model_name,
                seed,
            )
            complete = bool(model_dir and (model_dir / "model.safetensors").exists())
            prefix = _torchrun_prefix(config)
            command = (
                f"{prefix} -m inverse_neural_operator.baselines.train "
                f"--config {config_path} --model {model_name} "
                f"--seed {seed} --execute"
            )
            command = _with_root_args(command, models_dir_override, results_dir_override)
            yield PlannedJob(
                stage="baselines",
                name=model_name,
                encoder_type=None,
                dataset=dataset,
                seed=seed,
                model_dir=model_dir,
                run_dir=run_dir,
                command=command,
                complete=complete,
                dependencies=[],
                missing_dependencies=[],
            )


def _inverse_model_jobs(
    config: ExperimentConfig,
    models_dir: Optional[Path],
    results_dir: Path,
    config_path: str,
    models_dir_override: Optional[str],
    results_dir_override: Optional[str],
) -> Iterable[PlannedJob]:
    dataset = config.dataset.name
    for seed in config.matrix.seeds:
        for model_name in config.inverse_models.models:
            model_dir = (
                model_artifact_dir(
                    models_dir,
                    dataset,
                    "inverse_models",
                    model_name,
                    seed,
                )
                if models_dir is not None
                else None
            )
            run_dir = run_artifact_dir(
                results_dir,
                dataset,
                "inverse_models",
                model_name,
                seed,
            )
            complete = bool(model_dir and not missing_inverse_model_files(model_dir))
            fe_dependencies, fe_missing = _inverse_function_encoder_dependency(
                config,
                models_dir,
                seed,
            )
            forward_dependencies, forward_missing = _forward_dependency(
                config,
                models_dir,
                seed,
            )
            prefix = _torchrun_prefix(config)
            command = (
                f"{prefix} -m inverse_neural_operator.inverse.train "
                f"--config {config_path} --model {model_name} "
                f"--seed {seed} --execute"
            )
            command = _with_root_args(command, models_dir_override, results_dir_override)
            yield PlannedJob(
                stage="inverse_models",
                name=model_name,
                encoder_type=None,
                dataset=dataset,
                seed=seed,
                model_dir=model_dir,
                run_dir=run_dir,
                command=command,
                complete=complete,
                dependencies=fe_dependencies + forward_dependencies,
                missing_dependencies=fe_missing + forward_missing,
            )


def _evaluation_jobs(
    config: ExperimentConfig,
    models_dir: Optional[Path],
    results_dir: Path,
    config_path: str,
    models_dir_override: Optional[str],
    results_dir_override: Optional[str],
) -> Iterable[PlannedJob]:
    dataset = config.dataset.name
    for seed in config.matrix.seeds:
        for model_name in config.inverse_models.models:
            model_dir = (
                model_artifact_dir(
                    models_dir,
                    dataset,
                    "inverse_models",
                    model_name,
                    seed,
                )
                if models_dir is not None
                else None
            )
            run_dir = (
                run_artifact_dir(
                    results_dir,
                    dataset,
                    "evaluation",
                    "inverse_models",
                    seed,
                )
                / model_name
                / "test"
            )
            complete = (run_dir / "metrics.json").exists()
            dependencies: List[str] = []
            missing_dependencies: List[str] = []
            fe_dependencies, fe_missing = _inverse_function_encoder_dependency(
                config,
                models_dir,
                seed,
            )
            forward_dependencies, forward_missing = _forward_dependency(
                config,
                models_dir,
                seed,
            )
            dependencies.extend(fe_dependencies)
            dependencies.extend(forward_dependencies)
            missing_dependencies.extend(fe_missing)
            missing_dependencies.extend(forward_missing)
            if model_dir is None:
                missing_dependencies.append(
                    "B2B_MODELS_DIR unset; cannot check inverse model artifact"
                )
            else:
                dependencies.append(str(model_dir))
                missing = missing_inverse_model_files(model_dir)
                if missing:
                    missing_dependencies.append(_dependency_label(model_dir, missing))
            prefix = _torchrun_prefix(config)
            command = (
                f"{prefix} -m inverse_neural_operator.evaluation.inverse "
                f"--config {config_path} --model {model_name} "
                f"--seed {seed} --split test --execute"
            )
            command = _with_root_args(command, models_dir_override, results_dir_override)
            yield PlannedJob(
                stage="evaluation",
                name=model_name,
                encoder_type=None,
                dataset=dataset,
                seed=seed,
                model_dir=model_dir,
                run_dir=run_dir,
                command=command,
                complete=complete,
                dependencies=dependencies,
                missing_dependencies=missing_dependencies,
            )


def plan_jobs(
    config: ExperimentConfig,
    *,
    config_path: str = "<config>",
    models_dir_override: Optional[str] = None,
    results_dir_override: Optional[str] = None,
) -> List[PlannedJob]:
    models_dir = models_root(required=False, override=models_dir_override)
    results_dir = results_root(results_dir_override)

    jobs: List[PlannedJob] = []
    for stage in config.matrix.stages:
        if stage == "function_encoders":
            jobs.extend(
                _function_encoder_jobs(
                    config,
                    models_dir,
                    results_dir,
                    config_path,
                    models_dir_override,
                    results_dir_override,
                )
            )
        elif stage == "forward_models":
            jobs.extend(
                _forward_model_jobs(
                    config,
                    models_dir,
                    results_dir,
                    config_path,
                    models_dir_override,
                    results_dir_override,
                )
            )
        elif stage == "baselines":
            jobs.extend(
                _baseline_jobs(
                    config,
                    models_dir,
                    results_dir,
                    config_path,
                    models_dir_override,
                    results_dir_override,
                )
            )
        elif stage == "inverse_models":
            jobs.extend(
                _inverse_model_jobs(
                    config,
                    models_dir,
                    results_dir,
                    config_path,
                    models_dir_override,
                    results_dir_override,
                )
            )
        elif stage == "evaluation":
            jobs.extend(
                _evaluation_jobs(
                    config,
                    models_dir,
                    results_dir,
                    config_path,
                    models_dir_override,
                    results_dir_override,
                )
            )
        else:
            raise ValueError(f"Unsupported stage in planner: {stage}")
    return jobs
