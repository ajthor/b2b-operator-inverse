"""CPU-only correctness checks for the iFNO baseline (arXiv:2402.11722).

Run on CPU (CUDA_VISIBLE_DEVICES="") so GPU jobs are never disturbed.

What these check:
  - The invertible multiplicative-coupling core is an exact bijection, i.e.
    coupling_inverse(coupling_forward(z)) == z. This is the paper's Eq. (4)/(6):
    forward  v1 = u1 * S(L(u2));  v2 = u2 * S(L(v1))
    inverse  u2 = v2 / S(L(v1));  u1 = v1 / S(L(u2))    (reverse layer order)
    We isolate it by setting the P/Q lift/project layers to identity.
  - The full create_model build path runs forward + inverse on a 1D symmetric
    (Darcy-like) batch with the right shapes.
  - The auxiliary loss helpers execute.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from inverse_neural_operator.models.ifno import (
    IFNO,
    create_model,
    ifno_forward_loss,
    ifno_backward_loss,
    ifno_vae_loss,
)


def test_coupling_core_is_exactly_invertible():
    """inverse(forward(x)) == x once P/Q projections are identities, isolating
    the multiplicative-coupling bijection that the paper guarantees."""
    torch.manual_seed(0)
    length = 16
    width = 4  # half_width = 2
    # input_channels = input_function_channels + coordinate_dim must equal width
    # so the identity projections line up: 3 + 1 == 4.
    model = IFNO(
        modes1=4,
        modes2=4,
        width=width,
        beta=2.0,
        n_layers=3,
        padding=2,
        vae_latent_dim=4,
        input_spatial_dims=(length,),
        output_spatial_dims=(length,),
        input_function_channels=3,
        output_function_channels=3,
        coordinate_dim=1,
        intermediate_dim=8,
    )
    model.eval()
    assert model.is_symmetric
    assert model.input_channels == width and model.output_channels == width

    # Replace lift/project with identities so forward/inverse reduce to the
    # coupling core (which shares self.convs/mlps/ws between both directions).
    model.p1 = nn.Identity()
    model.p2 = nn.Identity()
    model.q1 = nn.Identity()
    model.q2 = nn.Identity()

    x = torch.randn(2, length, width)
    y_pred, _ = model.forward(x)
    x_rt, _ = model.inverse(y_pred)
    err = (x - x_rt).abs().max().item()
    assert err < 1e-4, f"coupling core not invertible; max err={err}"


def _darcy_like_ifno():
    return create_model(
        input_size=None,
        modes1=12,
        modes2=12,
        width=16,
        beta=2.0,
        n_layers=2,
        padding=2,
        vae_latent_dim=4,
        input_spatial_dims=(32,),
        output_spatial_dims=(32,),
        input_function_channels=1,
        output_function_channels=1,
        coordinate_dim=1,
        intermediate_dim=8,
    )


def test_ifno_build_forward_inverse_shapes():
    torch.manual_seed(0)
    model = _darcy_like_ifno()
    model.eval()
    batch, length = 3, 32
    x = torch.linspace(0, 1, length).view(1, length, 1).repeat(batch, 1, 1)
    u = torch.randn(batch, length, 1)
    y = x.clone()
    s = torch.randn(batch, length, 1)

    # Forward: cat([coords, func]) -> (pred_s, recon_loss)
    pred_s, fwd_recon = model.forward(torch.cat([x, u], dim=-1))
    assert pred_s.shape == (batch, length, model.output_channels)
    assert torch.isfinite(pred_s).all() and torch.isfinite(torch.as_tensor(fwd_recon)).all()

    # Inverse: cat([coords, func]) -> (pred_u, recon_loss)
    pred_u, inv_recon = model.inverse(torch.cat([y, s], dim=-1))
    assert pred_u.shape == (batch, length, model.input_channels)
    assert torch.isfinite(pred_u).all() and torch.isfinite(torch.as_tensor(inv_recon)).all()


def test_ifno_loss_helpers_run():
    torch.manual_seed(0)
    model = _darcy_like_ifno()
    batch, length = 2, 32
    x = torch.linspace(0, 1, length).view(1, length, 1).repeat(batch, 1, 1)
    u = torch.randn(batch, length, 1)
    y = x.clone()
    s = torch.randn(batch, length, 1)
    batch_tuple = (x, u, y, s)
    for loss_fn in (ifno_forward_loss, ifno_backward_loss, ifno_vae_loss):
        value = loss_fn(model, batch_tuple)
        assert torch.isfinite(value).all(), f"{loss_fn.__name__} produced non-finite loss"


def test_ifno_vae_inverse_methods():
    """refine_inverse (paper inverse inference) and inverse_vae_loss (joint
    backward objective) produce correct shapes / finite values."""
    torch.manual_seed(0)
    model = _darcy_like_ifno()
    model.eval()
    batch, length = 3, 32
    y = torch.linspace(0, 1, length).view(1, length, 1).repeat(batch, 1, 1)
    s = torch.randn(batch, length, 1)
    u = torch.randn(batch, length, 1)

    pred_u_full, _ = model.inverse(torch.cat([y, s], dim=-1))
    refined = model.refine_inverse(pred_u_full)
    # VAE-native shape: (batch, in_function_channels, length)
    assert refined.shape == (batch, model.input_function_channels, length)
    assert torch.isfinite(refined).all()

    loss = model.inverse_vae_loss(pred_u_full, u, kl_weight=0.01)
    assert loss.dim() == 0 and torch.isfinite(loss)


# ---------------------------------------------------------------------------
# End-to-end faithful three-phase training (CPU). Validates that the training
# procedure ported from BayesianAIGroup/iFNO actually optimizes the model.
# ---------------------------------------------------------------------------

import types

from inverse_neural_operator.baselines.train import _run_ifno_training, _evaluate_ifno


def _fake_context():
    return types.SimpleNamespace(
        is_distributed=False,
        is_rank_zero=True,
        device=torch.device("cpu"),
        world_size=1,
        rank=0,
    )


def _fast_ifno(length=24):
    """Small, fast iFNO (shallow VAE) for the end-to-end training test."""
    return create_model(
        input_size=None,
        modes1=6,
        modes2=6,
        width=8,
        beta=2.0,
        n_layers=2,
        padding=2,
        vae_latent_dim=4,
        input_spatial_dims=(length,),
        output_spatial_dims=(length,),
        input_function_channels=1,
        output_function_channels=1,
        coordinate_dim=1,
        intermediate_dim=8,
        vae_hidden_dims=[16, 32],
    )


def _ifno_config(epochs_ifno=12, epochs_vae=3, epochs_joint=2, lr=3e-3):
    from inverse_neural_operator.config.schema import IFNOConfig

    ifno = IFNOConfig(
        epochs_vae=epochs_vae,
        epochs_ifno=epochs_ifno,
        lr_vae=lr,
        lr_ifno=lr,
        lr_forward=lr,
        lr_backward=lr,
    )
    baselines = types.SimpleNamespace(
        ifno=ifno, epochs=epochs_joint, log_interval=10_000, eval_batches=None
    )
    return types.SimpleNamespace(baselines=baselines)


def _synthetic_loaders(n=24, length=24, batch_size=6, seed=0):
    """Learnable task: forward operator is identity (s == u), so both forward
    and inverse should be easy to fit; the VAE learns the input manifold."""
    from torch.utils.data import DataLoader, TensorDataset

    g = torch.Generator().manual_seed(seed)
    coords = torch.linspace(0, 1, length).view(1, length, 1).repeat(n, 1, 1)
    grid = torch.linspace(0, 1, length).view(1, length)
    # Smooth random fields so the (downsampling) VAE can represent them.
    u = torch.zeros(n, length)
    for k in range(1, 4):
        amp = torch.randn(n, 1, generator=g)
        phase = torch.rand(n, 1, generator=g) * 6.28318
        u = u + amp * torch.sin(k * 3.14159 * grid + phase)
    u = (u / 3.0).unsqueeze(-1)
    x = coords
    y = coords.clone()
    s = u.clone()  # identity forward operator
    ds = TensorDataset(x, u, y, s)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), DataLoader(
        ds, batch_size=batch_size, shuffle=False
    )


def test_ifno_pretrain_phase_reduces_forward_and_inverse_error():
    """Phase 1 (pre-train invertible blocks with forward + backward + recon)
    directly supervises both directions; it should clearly reduce forward and
    raw (no-VAE) inverse error. VAE/joint phases disabled to isolate it."""
    torch.manual_seed(0)
    model = _fast_ifno()
    context = _fake_context()
    config = _ifno_config(epochs_ifno=12, epochs_vae=0, epochs_joint=0)
    train_loader, test_loader = _synthetic_loaders(n=18)

    before = _evaluate_ifno(model, test_loader, context, config, 1, model.output_function_channels)
    metrics, best_test, steps = _run_ifno_training(
        model, train_loader, test_loader, context, config, writer=None
    )
    assert steps > 0 and best_test == best_test
    for key, value in metrics.items():
        assert value == value, f"metric {key} is NaN"
    # A cold-start >10% reduction in BOTH directions shows the ported pretrain
    # objective (forward + backward + reconstruction) is wired and optimizing.
    assert metrics["forward_loss"] < 0.9 * before["forward_loss"], (
        before["forward_loss"],
        metrics["forward_loss"],
    )
    assert metrics["input_loss_raw"] < 0.9 * before["input_loss_raw"], (
        before["input_loss_raw"],
        metrics["input_loss_raw"],
    )


def test_ifno_full_pipeline_runs_and_vae_refines():
    """The full faithful 3-phase pipeline runs end-to-end with finite metrics,
    and the VAE-refined inverse (paper inference path) improves over the
    untrained model once the VAE has been pre-trained on the input functions."""
    torch.manual_seed(0)
    model = _fast_ifno()
    context = _fake_context()
    config = _ifno_config(epochs_ifno=4, epochs_vae=6, epochs_joint=1)
    train_loader, test_loader = _synthetic_loaders(n=18)

    before = _evaluate_ifno(model, test_loader, context, config, 1, model.output_function_channels)
    metrics, best_test, steps = _run_ifno_training(
        model, train_loader, test_loader, context, config, writer=None
    )
    assert steps > 0 and best_test == best_test
    expected_keys = {"loss", "forward_loss", "input_loss", "input_loss_raw"}
    assert expected_keys.issubset(metrics)
    for key, value in metrics.items():
        assert value == value, f"metric {key} is NaN"
    # VAE-refined inverse should beat the random-init baseline after VAE pretrain.
    assert metrics["input_loss"] < before["input_loss"], (
        before["input_loss"],
        metrics["input_loss"],
    )
