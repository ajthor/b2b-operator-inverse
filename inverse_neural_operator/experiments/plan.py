"""CLI for dry-run planning and artifact status."""

from __future__ import annotations

import argparse
import json

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.experiments.planner import plan_jobs
from inverse_neural_operator.runtime.paths import to_plain_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan overhaul experiment jobs.")
    parser.add_argument("config", help="Path to experiment YAML.")
    parser.add_argument("--models-dir", default=None, help="Override B2B_MODELS_DIR.")
    parser.add_argument("--results-dir", default=None, help="Override B2B_RESULTS_DIR.")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    jobs = plan_jobs(
        config,
        config_path=args.config,
        models_dir_override=args.models_dir,
        results_dir_override=args.results_dir,
    )

    if args.format == "json":
        print(json.dumps([to_plain_data(job) for job in jobs], indent=2))
        return

    print(f"Experiment: {config.experiment}")
    print(f"Dataset:    {config.dataset.name}")
    print(f"Dry run:    {config.runtime.dry_run}")
    print(f"Jobs:       {len(jobs)}")
    print("")
    for index, job in enumerate(jobs, start=1):
        status = "complete" if job.complete else "missing"
        model_dir = str(job.model_dir) if job.model_dir else "<B2B_MODELS_DIR unset>"
        print(f"[{index}] {job.stage}/{job.name}/{job.encoder_type} seed={job.seed}")
        print(f"    status:    {status}")
        print(f"    model_dir: {model_dir}")
        print(f"    run_dir:   {job.run_dir}")
        print(f"    command:   {job.command}")


if __name__ == "__main__":
    main()
