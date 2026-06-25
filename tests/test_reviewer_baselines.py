"""CPU-only correctness checks for the reviewer baselines.

These intentionally avoid any GPU use (forced via CUDA_VISIBLE_DEVICES="" in the
runner) and only test architecture/wiring correctness against the papers:
  - Invertible DeepONet (RealNVP branch + trunk basis + LS inverse), arXiv:2209.02772
  - Neural Inverse Operators (DeepONet -> FNO), arXiv:2301.11167
"""

from __future__ import annotations

import types

import torch

from inverse_neural_operator.models.reviewer_baselines import (
    InvertibleDeepONetBaseline,
    NIOBaseline,
    RealNVPBranch,
    create_invertible_deeponet,
    create_nio,
)


def _cfg():
    """Minimal config object exposing .baselines.invertible_deeponet / .nio."""
    from inverse_neural_operator.config.schema import (
        InvertibleDeepONetConfig,
        NIOConfig,
    )

    baselines = types.SimpleNamespace(
        invertible_deeponet=InvertibleDeepONetConfig(
            hidden_sizes=[32, 32],
            trunk_hidden_sizes=[32, 32],
            n_coupling_layers=4,
            regularization=1e-6,
        ),
        nio=NIOConfig(
            branch_hidden_sizes=[32, 32],
            trunk_hidden_sizes=[32, 32],
            n_basis=16,
            lifting_channels=8,
            modes=8,
            n_fourier_layers=2,
            measurement_points=None,
        ),
    )
    return types.SimpleNamespace(baselines=baselines)


def _darcy_like_info(n_points=64, n_samples=200):
    """dataset_info exactly as the 1D datasets' get_info() produces it."""
    return {
        "X_size": 1,            # coordinate dim of x grid (shape[-1])
        "u_size": 1,            # channel dim of u (shape[-1])
        "Y_size": 1,            # coordinate dim of y grid (shape[-1])
        "s_size": 1,
        "u_len": n_samples,     # NOTE: this is shape[0] == number of SAMPLES
        "input_function_channels": 1,
        "output_function_channels": 1,
        "coordinate_dim": 1,
        "input_spatial_dims": (n_points,),
        "_n_points": n_points,  # not a real key; for the test's own use
    }


def _heat_like_info(height=16, width=16, n_samples=200):
    """dataset_info as the 2D datasets' get_info() produces it (e.g. heat)."""
    return {
        "X_size": 2,            # 2D coordinates
        "Y_size": 2,
        "u_len": n_samples,
        "input_function_channels": 1,
        "output_function_channels": 1,
        "coordinate_dim": 2,
        "input_spatial_dims": (height, width),
        "_n_points": height * width,
    }


def test_realnvp_is_invertible():
    torch.manual_seed(0)
    branch = RealNVPBranch(size=8, hidden_sizes=[16, 16], n_coupling_layers=4)
    x = torch.randn(5, 8)
    z, log_det = branch(x)
    x_rt = branch.inverse(z)
    assert torch.allclose(x, x_rt, atol=1e-5), (x - x_rt).abs().max()
    assert log_det.shape == (5,)


def test_invertible_deeponet_forward_inverse_roundtrip():
    """Verify the paper's forward (s = Y b) / least-squares inverse algebra
    round-trips the input function, AND that the result is correct shape.

    NOTE: the trunk is an MLP of a scalar coordinate, so at random init its
    columns are strongly correlated and Y is numerically near-singular -- a
    plain round-trip explodes purely from conditioning, not from a logic error.
    To isolate the *inverse algebra* (normal-equation solve + RealNVP inverse)
    we monkeypatch trunk_matrix to return a well-conditioned orthonormal basis.
    """
    torch.manual_seed(0)
    n_points = 48
    model = InvertibleDeepONetBaseline(
        input_points=n_points,
        input_channels=1,
        output_channels=1,
        coordinate_dim=1,
        hidden_sizes=[32, 32],
        trunk_hidden_sizes=[32, 32],
        n_coupling_layers=4,
        regularization=1e-8,
    )
    model.eval()
    batch = 3
    u = torch.randn(batch, n_points, 1)
    y = torch.linspace(0, 1, n_points).view(1, n_points, 1).repeat(batch, 1, 1)

    # Well-conditioned (orthonormal) trunk basis Y, shared across the batch.
    q, _ = torch.linalg.qr(torch.randn(n_points, n_points))
    fixed_trunk = q.unsqueeze(0).repeat(batch, 1, 1)
    model.trunk_matrix = lambda _y, _t=fixed_trunk: _t

    s = model.forward_from_input(u, y)
    assert s.shape == (batch, n_points, 1)
    u_hat = model.inverse_from_output(s, y)
    assert u_hat.shape == (batch, n_points, 1)
    err = (u - u_hat).abs().max().item()
    assert err < 1e-3, f"forward/inverse did not round-trip input; max err={err}"


def test_create_invertible_deeponet_wiring():
    """Reproduce exactly what training does: build from dataset_info and run a
    forward pass on a darcy-like batch. After the fix, input_points is the number
    of input spatial points (prod(input_spatial_dims)), not u_len (n_samples)."""
    info = _darcy_like_info(n_points=64, n_samples=200)
    model = create_invertible_deeponet(info, _cfg())
    batch = 2
    n_points = info["_n_points"]
    assert model.input_dim == n_points * info["input_function_channels"]
    u = torch.randn(batch, n_points, 1)
    y = torch.linspace(0, 1, n_points).view(1, n_points, 1).repeat(batch, 1, 1)
    s = model.forward_from_input(u, y)
    assert s.shape == (batch, n_points, 1)
    u_hat = model.inverse_from_output(s, y)
    assert u_hat.shape == (batch, n_points, 1)


def test_nio_forward_shapes():
    torch.manual_seed(0)
    model = NIOBaseline(
        output_coordinate_dim=1,
        output_channels=1,
        input_coordinate_dim=1,
        input_channels=1,
        branch_hidden_sizes=[32, 32],
        trunk_hidden_sizes=[32, 32],
        n_basis=16,
        lifting_channels=8,
        modes=8,
        n_fourier_layers=2,
        measurement_points=None,
        input_spatial_dims=(64,),
    )
    model.eval()
    batch, n_x, n_meas = 3, 64, 50
    x = torch.linspace(0, 1, n_x).view(1, n_x, 1).repeat(batch, 1, 1)
    y = torch.linspace(0, 1, n_meas).view(1, n_meas, 1).repeat(batch, 1, 1)
    s = torch.randn(batch, n_meas, 1)
    pred_u = model.predict_inverse(x, y, s)
    assert pred_u.shape == (batch, n_x, 1)


def test_nio_2d_forward_shapes():
    """2D FNO path: flattened H*W grid is reshaped, run through 2D FNO blocks,
    and projected back to the flattened input function."""
    torch.manual_seed(0)
    height, width = 12, 10
    n_x = height * width
    model = NIOBaseline(
        output_coordinate_dim=2,
        output_channels=1,
        input_coordinate_dim=2,
        input_channels=1,
        branch_hidden_sizes=[32, 32],
        trunk_hidden_sizes=[32, 32],
        n_basis=16,
        lifting_channels=8,
        modes=6,
        n_fourier_layers=2,
        measurement_points=None,
        input_spatial_dims=(height, width),
    )
    model.eval()
    batch, n_meas = 3, 40
    x = torch.rand(batch, n_x, 2)
    y = torch.rand(batch, n_meas, 2)
    s = torch.randn(batch, n_meas, 1)
    pred_u = model.predict_inverse(x, y, s)
    assert pred_u.shape == (batch, n_x, 1)


def test_create_nio_wiring():
    info = _darcy_like_info(n_points=64, n_samples=200)
    model = create_nio(info, _cfg())
    batch, n_x, n_meas = 2, info["_n_points"], 50
    x = torch.linspace(0, 1, n_x).view(1, n_x, 1).repeat(batch, 1, 1)
    y = torch.linspace(0, 1, n_meas).view(1, n_meas, 1).repeat(batch, 1, 1)
    s = torch.randn(batch, n_meas, 1)
    pred_u = model.predict_inverse(x, y, s)
    assert pred_u.shape == (batch, n_x, 1)


def test_create_nio_wiring_2d():
    info = _heat_like_info(height=16, width=16, n_samples=200)
    model = create_nio(info, _cfg())
    assert model.spatial_ndim == 2
    batch, n_x, n_meas = 2, info["_n_points"], 40
    x = torch.rand(batch, n_x, 2)
    y = torch.rand(batch, n_meas, 2)
    s = torch.randn(batch, n_meas, 1)
    pred_u = model.predict_inverse(x, y, s)
    assert pred_u.shape == (batch, n_x, 1)
