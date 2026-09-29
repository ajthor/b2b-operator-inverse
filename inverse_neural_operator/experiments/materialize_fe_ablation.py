"""Materialize matched B2B function-encoder ablation experiment configs."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


VARIANTS = {
    "relu": {"kind": "mlp", "activation": "relu", "winner_samples": 0},
    "silu": {"kind": "mlp", "activation": "silu", "winner_samples": 0},
    "siren": {"kind": "siren", "activation": "relu", "winner_samples": 0},
    "siren2": {
        "kind": "siren_square",
        "activation": "relu",
        "winner_samples": 64,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", nargs=4, default=["0", "1", "2", "3"])
    parser.add_argument("--port-start", type=int, default=29710)
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = Path(args.base_config)
    output_dir = Path(args.output_dir)
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = base["dataset"]["name"]
    for index, (label, settings) in enumerate(VARIANTS.items()):
        config = copy.deepcopy(base)
        config["experiment"] = f"{dataset}_b2b_fe_{label}_full"
        config["description"] = (
            f"Full {dataset} B2B nonlinear run with the {label} function-encoder basis."
        )
        config["matrix"]["stages"] = [
            "function_encoders",
            "inverse_models",
            "evaluation",
        ]
        config["runtime"]["env"]["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[index])
        config["runtime"]["env"]["MASTER_PORT"] = str(args.port_start + index)
        artifact = f"fe_{label}_nbasis50_full"
        config["function_encoders"]["artifact"] = artifact
        config["function_encoders"]["basis"].update(settings)
        config["inverse_models"]["artifact"] = f"b2b_nonlinear_fe_{label}_full"
        config["inverse_models"]["function_encoder_artifact"] = artifact
        config["baselines"]["models"] = []
        output_path = output_dir / f"{dataset}_b2b_fe_{label}_full.yaml"
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        print(output_path)


if __name__ == "__main__":
    main()
