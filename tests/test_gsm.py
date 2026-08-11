"""The GSM stages: augmentations, windows, and the composed feature extractor."""

from __future__ import annotations

import pytest
import torch

from gsm_alpha.signature import gsm as gsm_mod
from gsm_alpha.signature import logsignature_channels
from gsm_alpha.signature.gsm import GSM, build_augmentation, window_slices


def test_time_augmentation_adds_a_normalised_time_channel():
    x = torch.randn(2, 7, 3)
    out = gsm_mod.TimeAugmentation()(x)
    assert out.shape == (2, 7, 4)
    assert torch.allclose(out[..., 0], torch.linspace(0, 1, 7).expand(2, 7))
    assert torch.allclose(out[..., 1:], x)


def test_basepoint_prepends_the_origin():
    x = torch.randn(2, 7, 3)
    out = gsm_mod.BasepointAugmentation()(x)
    assert out.shape == (2, 8, 3)
    assert torch.all(out[:, 0] == 0)


def test_basepoint_makes_the_signature_translation_sensitive():
    """Section 2.2.1's whole point: without it a shifted price path looks identical."""
    from gsm_alpha.signature import torch_backend as tb

    torch.manual_seed(0)
    x = torch.randn(1, 9, 2, dtype=torch.float64).cumsum(dim=1)
    shifted = x + 5.0
    assert torch.allclose(tb.signature(x, 3), tb.signature(shifted, 3), atol=1e-10)

    basepoint = gsm_mod.BasepointAugmentation()
    assert not torch.allclose(
        tb.signature(basepoint(x), 3), tb.signature(basepoint(shifted), 3), atol=1e-6
    )


def test_coordinate_projection_with_pairs_enumerates_channel_combinations():
    x = torch.randn(2, 7, 5)
    projection = gsm_mod.CoordinateProjection(size=2, ordered=False)
    out = projection(x)
    assert projection.out_streams(5) == 10  # C(5, 2)
    assert out.shape == (2, 10, 7, 2)
    assert torch.allclose(out[:, 0, :, 0], x[..., 0])
    assert torch.allclose(out[:, 0, :, 1], x[..., 1])
    assert gsm_mod.CoordinateProjection(size=2, ordered=True).out_streams(5) == 20


def test_lead_lag_doubles_channels_and_staggers_them():
    x = torch.arange(6.0).reshape(1, 3, 2)
    out = gsm_mod.LeadLagAugmentation()(x)
    assert out.shape == (1, 5, 4)
    assert torch.allclose(out[0, :, :2], torch.tensor([[0.0, 1], [2, 3], [2, 3], [4, 5], [4, 5]]))
    assert torch.allclose(out[0, :, 2:], torch.tensor([[0.0, 1], [0, 1], [2, 3], [2, 3], [4, 5]]))


def test_composed_augmentation_matches_the_report_setting():
    """Coordinate pairs, then time, then basepoint: 10 streams of (t, x_i, x_j)."""
    augment = build_augmentation(["coordinate_projection", "time", "basepoint"], 5)
    out = augment(torch.randn(3, 20, 5))
    assert augment.out_streams(5) == 10
    assert augment.out_channels(5) == 3
    assert out.shape == (3, 10, 21, 3)
    assert torch.all(out[:, :, 0, :] == 0)


@pytest.mark.parametrize(
    "kind,kwargs,expected",
    [
        ("global", {}, 1),
        ("sliding", {"size": 4, "step": 2}, 5),
        ("expanding", {"size": 4, "step": 3}, 4),  # 4, 7, 10, then the full stream
        ("dyadic", {"depth": 3}, 7),
    ],
)
def test_window_counts(kind, kwargs, expected):
    assert len(window_slices(kind, 12, **kwargs)) == expected


def test_dyadic_window_covers_the_stream_at_every_scale():
    slices = window_slices("dyadic", 16, depth=3)
    assert slices[0] == (0, 16)
    assert slices[1:3] == [(0, 8), (8, 16)]
    assert slices[3:7] == [(0, 4), (4, 8), (8, 12), (12, 16)]


def test_gsm_output_width_is_the_advertised_one():
    """10 pair-streams x depth-5 log-signature over 3 channels = 10 x 80."""
    model = GSM(in_channels=5, depth=5, backend="torch")
    assert model.channels_per_window == logsignature_channels(3, 5) == 80
    assert model.out_features(30) == 800
    assert model(torch.randn(4, 30, 5)).shape == (4, 800)
    assert not model.is_learnable


def test_gsm_windowing_multiplies_the_feature_count():
    model = GSM(in_channels=5, depth=3, window="dyadic", window_depth=2, backend="torch")
    assert model.n_windows(20) == 3
    assert model(torch.randn(2, 20, 5)).shape == (2, model.out_features(20))


def test_learnable_augmentation_is_flagged_and_trains():
    model = GSM(
        in_channels=5,
        depth=3,
        augmentations=["multi_headed_stream_preserving", "time", "basepoint"],
        n_projections=4,
        projection_out=2,
        backend="torch",
    )
    assert model.is_learnable
    out = model(torch.randn(2, 12, 5))
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_post_rescaling_scales_levels_by_factorial():
    plain = GSM(in_channels=3, depth=3, augmentations=["none"], backend="torch")
    scaled = GSM(in_channels=3, depth=3, augmentations=["none"], rescaling="post", backend="torch")
    x = torch.randn(2, 8, 3, dtype=torch.float32)
    ratio = scaled(x) / plain(x)
    # Depth-3 Lyndon words over 3 letters: 3 of length 1, 3 of length 2, 8 of length 3.
    assert torch.allclose(ratio[:, :3], torch.ones(2, 3), atol=1e-4)
    assert torch.allclose(ratio[:, 3:6], torch.full((2, 3), 2.0), atol=1e-4)
    assert torch.allclose(ratio[:, 6:], torch.full((2, 8), 6.0), atol=1e-3)
