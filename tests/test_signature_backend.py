"""Pin the pure-PyTorch signature backend to signatory's output.

The fallback exists so the project still runs where signatory could not be
built.  That is only worth anything if the two agree, so when signatory *is*
importable these tests compare them directly; when it is not, the
self-consistency properties (Chen's identity, the log/exp round trip) still
check the fallback on its own.
"""

from __future__ import annotations

import pytest
import torch

from gsm_alpha.signature import lyndon
from gsm_alpha.signature import torch_backend as tb
from gsm_alpha.signature.backend import SIGNATORY_AVAILABLE

if SIGNATORY_AVAILABLE:
    import signatory

SHAPES = [(2, 3, 6), (3, 5, 9), (4, 4, 12), (5, 3, 7), (3, 5, 49), (2, 4, 101)]


@pytest.mark.skipif(not SIGNATORY_AVAILABLE, reason="signatory is not installed")
@pytest.mark.parametrize("d,depth", [(2, 3), (3, 5), (4, 4), (5, 3)])
def test_lyndon_word_order_matches_signatory(d, depth):
    assert [list(w) for w in lyndon.lyndon_words(d, depth)] == signatory.lyndon_words(d, depth)


@pytest.mark.skipif(not SIGNATORY_AVAILABLE, reason="signatory is not installed")
@pytest.mark.parametrize("d,depth,length", SHAPES)
def test_signature_matches_signatory(d, depth, length):
    torch.manual_seed(0)
    path = torch.randn(3, length, d, dtype=torch.float64)
    assert torch.allclose(signatory.signature(path, depth), tb.signature(path, depth), atol=1e-10)


@pytest.mark.skipif(not SIGNATORY_AVAILABLE, reason="signatory is not installed")
@pytest.mark.parametrize("d,depth,length", SHAPES)
def test_logsignature_matches_signatory(d, depth, length):
    torch.manual_seed(0)
    path = torch.randn(3, length, d, dtype=torch.float64)
    reference = signatory.logsignature(path, depth, mode="words")
    assert torch.allclose(reference, tb.logsignature(path, depth), atol=1e-10)


@pytest.mark.parametrize("d,depth", [(2, 3), (3, 4)])
def test_chen_identity(d, depth):
    """S(path) = S(first half) * S(second half) — the property the fold relies on."""
    torch.manual_seed(1)
    path = torch.randn(2, 11, d, dtype=torch.float64)
    whole = tb.signature_terms(path, depth)
    combined = tb._chen(tb.signature_terms(path[:, :6], depth), tb.signature_terms(path[:, 5:], depth))
    for a, b in zip(whole, combined):
        assert torch.allclose(a, b, atol=1e-10)


def test_signature_of_a_single_segment_is_the_tensor_exponential():
    """One straight segment: level k is exactly dx^{ok} / k!."""
    path = torch.tensor([[[0.0, 0.0], [2.0, 3.0]]], dtype=torch.float64)
    levels = tb.signature_terms(path, 3)
    dx = torch.tensor([2.0, 3.0], dtype=torch.float64)
    assert torch.allclose(levels[0][0], dx)
    assert torch.allclose(levels[1][0], torch.outer(dx, dx).reshape(-1) / 2)
    third = torch.einsum("i,j,k->ijk", dx, dx, dx).reshape(-1) / 6
    assert torch.allclose(levels[2][0], third)


def test_signature_is_invariant_to_reparametrisation():
    """Without a time channel the signature cannot see sampling density."""
    torch.manual_seed(2)
    coarse = torch.randn(1, 5, 3, dtype=torch.float64).cumsum(dim=1)
    # Insert midpoints: same geometric path, twice the sample rate.
    fine = torch.cat([coarse, (coarse[:, :-1] + coarse[:, 1:]) / 2], dim=1)
    fine = fine[:, torch.argsort(torch.tensor([0, 2, 4, 6, 8, 1, 3, 5, 7]))]
    assert torch.allclose(tb.signature(coarse, 3), tb.signature(fine, 3), atol=1e-10)


def test_logsignature_is_differentiable():
    path = torch.randn(2, 10, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: tb.logsignature(p, 4), (path,), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("d,depth,expected", [(3, 5, 80), (2, 4, 8), (5, 3, 55)])
def test_channel_counts(d, depth, expected):
    assert tb.logsignature_channels(d, depth) == expected
    assert tb.signature_channels(d, depth) == sum(d**k for k in range(1, depth + 1))
