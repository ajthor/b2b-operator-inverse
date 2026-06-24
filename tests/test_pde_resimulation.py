from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.evaluation.metrics import mae, mse, relative_l2, rmse
from inverse_neural_operator.evaluation.pde_resimulation import (
    _chladni_original_path,
    _chladni_resimulate,
    _load_solver_output,
    _metric_payload,
    _write_solver_input,
)


def test_basic_metric_helpers():
    prediction = torch.tensor([[1.0, 3.0]])
    target = torch.tensor([[1.0, 1.0]])

    assert mae(prediction, target).item() == pytest.approx(1.0)
    assert mse(prediction, target).item() == pytest.approx(2.0)
    assert rmse(prediction, target).item() == pytest.approx(2.0**0.5)
    assert relative_l2(prediction, target).item() == pytest.approx(2.0 / (2.0**0.5))


def test_metric_payload_for_1d_has_no_ssim():
    prediction = np.array([[1.0], [2.0]], dtype=np.float32)
    target = np.array([[1.0], [4.0]], dtype=np.float32)

    payload = _metric_payload(prediction, target, spatial_dims=(2,))

    assert payload["mae"] == pytest.approx(1.0)
    assert payload["mse"] == pytest.approx(2.0)
    assert payload["ssim"] is None


def test_solver_input_and_sample_id_validation(tmp_path: Path):
    input_path = tmp_path / "inputs" / "sample_000000.npz"
    arrays = {
        "X": np.zeros((2, 1), dtype=np.float32),
        "Y": np.zeros((2, 1), dtype=np.float32),
        "u_pred": np.ones((2, 1), dtype=np.float32),
        "u_true": np.zeros((2, 1), dtype=np.float32),
        "s_true": np.zeros((2, 1), dtype=np.float32),
    }
    _write_solver_input(
        input_path,
        sample_id=0,
        arrays=arrays,
        normalization_stats={"u_mean": 0.0, "u_std": 1.0},
        overwrite=False,
    )

    data = np.load(input_path, allow_pickle=False)
    assert int(data["sample_id"].item()) == 0
    assert json.loads(str(data["normalization_stats"].item()))["u_std"] == 1.0

    output_path = tmp_path / "solver_outputs" / "sample_000000.npz"
    output_path.parent.mkdir(parents=True)
    np.savez_compressed(
        output_path,
        sample_id=np.asarray(1, dtype=np.int64),
        Y=arrays["Y"],
        s_resim=arrays["s_true"],
    )
    with pytest.raises(ValueError, match="sample_id"):
        _load_solver_output(output_path, sample_id=0)


def test_chladni_analytic_resimulation_shape():
    if not _chladni_original_path().exists():
        pytest.skip("Chladni original data file is not available.")

    result = _chladni_resimulate(np.zeros((625, 1), dtype=np.float32))

    assert result.shape == (625, 1)
    assert np.isfinite(result).all()


def test_pde_validation_config_parses(tmp_path: Path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
experiment: pde_validation_parse
dataset:
  name: burgers
pde_validation:
  enabled: true
  backend: external
  command: docker run solver --manifest {manifest}
  max_samples: 7
  metrics: [mae, relative_l2]
  overwrite: true
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.pde_validation.enabled is True
    assert config.pde_validation.command == "docker run solver --manifest {manifest}"
    assert config.pde_validation.max_samples == 7
    assert config.pde_validation.metrics == ["mae", "relative_l2"]
    assert config.pde_validation.overwrite is True
