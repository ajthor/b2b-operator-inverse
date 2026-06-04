"""Safe sequential runner for planned overhaul experiment jobs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from typing import Iterable, List, Optional

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.experiments.planner import PlannedJob, plan_jobs


STAGES = [
    "function_encoders",
    "forward_models",
    "inverse_models",
    "baselines",
    "evaluation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run planned overhaul experiment jobs.")
    parser.add_argument("config", help="Path to experiment YAML.")
    parser.add_argument("--models-dir", default=None, help="Override B2B_MODELS_DIR.")
    parser.add_argument("--results-dir", default=None, help="Override B2B_RESULTS_DIR.")
    parser.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help="Limit to one stage. May be provided more than once.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run ready jobs. Without this flag, only prints the plan.",
    )
    return parser.parse_args()


def _status(job: PlannedJob) -> str:
    if job.complete:
        return "complete"
    if job.blocked:
        return "blocked"
    return "ready"


def _label(job: PlannedJob) -> str:
    label = f"{job.stage}/{job.name}"
    if job.encoder_type:
        label = f"{label}/{job.encoder_type}"
    return f"{label} seed={job.seed}"


def _filter_jobs(
    jobs: Iterable[PlannedJob],
    stages: Optional[List[str]],
) -> List[PlannedJob]:
    if not stages:
        return list(jobs)
    selected = set(stages)
    return [job for job in jobs if job.stage in selected]


def _print_jobs(jobs: List[PlannedJob]) -> None:
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}] {_label(job)}", flush=True)
        print(f"    status:    {_status(job)}", flush=True)
        print(f"    model_dir: {job.model_dir or '<B2B_MODELS_DIR unset>'}", flush=True)
        print(f"    run_dir:   {job.run_dir}", flush=True)
        for dependency in job.dependencies:
            print(f"    needs:     {dependency}", flush=True)
        for dependency in job.missing_dependencies:
            print(f"    blocked:   {dependency}", flush=True)
        print(f"    command:   {job.command}", flush=True)


def _plan(args: argparse.Namespace) -> List[PlannedJob]:
    config = load_experiment_config(args.config)
    jobs = plan_jobs(
        config,
        config_path=args.config,
        models_dir_override=args.models_dir,
        results_dir_override=args.results_dir,
    )
    return _filter_jobs(jobs, args.stage)


def _run_command(job: PlannedJob) -> int:
    print("", flush=True)
    print(f"Running {_label(job)}", flush=True)
    print(job.command, flush=True)
    return subprocess.run(shlex.split(job.command), check=False).returncode


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    initial_jobs = _plan(args)

    print(f"Experiment: {config.experiment}", flush=True)
    print(f"Dataset:    {config.dataset.name}", flush=True)
    print(f"Execute:    {args.execute}", flush=True)
    print(f"Jobs:       {len(initial_jobs)}", flush=True)
    print("", flush=True)
    _print_jobs(initial_jobs)

    if not args.execute:
        print("", flush=True)
        print("Dry run only. Re-run with --execute to launch ready jobs.", flush=True)
        return

    ran = 0
    skipped = len([job for job in initial_jobs if job.complete])
    while True:
        jobs = _plan(args)
        ready = [job for job in jobs if not job.complete and not job.blocked]
        blocked = [job for job in jobs if not job.complete and job.blocked]

        if not ready:
            if blocked:
                print("", flush=True)
                print("Stopped because remaining jobs are blocked:", flush=True)
                _print_jobs(blocked)
                raise SystemExit(1)
            print("", flush=True)
            print(
                f"Done. Ran {ran} job(s); skipped {skipped} complete job(s).",
                flush=True,
            )
            return

        job = ready[0]
        returncode = _run_command(job)
        if returncode != 0:
            print("", flush=True)
            print(f"Job failed with exit code {returncode}: {_label(job)}", flush=True)
            raise SystemExit(returncode)
        ran += 1


if __name__ == "__main__":
    main()
