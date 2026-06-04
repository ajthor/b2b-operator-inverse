"""Dry-run experiment matrix expansion and artifact status."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from inverse_neural_operator.config.schema import ExperimentConfig
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


def _function_encoder_jobs(
    config: ExperimentConfig,
    models_dir: Optional[Path],
    results_dir: Path,
    config_path: str,
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
            launcher = config.runtime.launcher
            if launcher == "torchrun":
                prefix = (
                    "python -m torch.distributed.run "
                    f"--nproc_per_node {config.runtime.nproc_per_node}"
                )
            else:
                prefix = "python3"
            command = (
                f"{prefix} -m inverse_neural_operator.function_encoders.train "
                f"--config {config_path} --encoder-type {encoder_type} "
                f"--seed {seed} --execute"
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
                _function_encoder_jobs(config, models_dir, results_dir, config_path)
            )
        else:
            raise ValueError(f"Unsupported stage in planner: {stage}")
    return jobs
