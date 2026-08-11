"""The Generalized Signature Method (report section 2).

GSM factorises every signature-method variant in the literature into four
stages, composed as

    z_{i,j} = (rho_post . S_N . rho_pre . W^j . phi^i)(x)

* ``phi`` — augmentation, one input stream fanned out into ``p`` streams
  (section 2.2): sensitivity-introducing (time, basepoint, invisibility-reset),
  dimension-reducing (coordinate projections, random/learnt projections,
  stream-preserving networks) and information-extracting (lead-lag).
* ``W`` — windowing, each stream cut into ``w`` sub-streams (section 2.3):
  global, sliding, expanding, hierarchical dyadic.
* ``S_N`` — the signature or log-signature truncated at depth ``N`` (section 2.4).
* ``rho`` — pre- or post-signature rescaling (section 2.5).

The report's own choice for GSM-Alpha (section 3.1) is
``coordinate projection with pairs`` + ``time`` + ``basepoint``, a global
window, a depth-5 log-signature and no rescaling; that is what
``configs/default.yaml`` sets.

Every stage is a plain ``nn.Module``, so a learnable augmentation
(``multi_headed_stream_preserving``) trains end to end, while the fixed
augmentations carry no parameters and can therefore be precomputed and cached
(see :mod:`gsm_alpha.data.cache`).
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import List, Sequence

import torch
import torch.nn as nn

from .backend import logsignature, logsignature_channels, signature, signature_channels

# --------------------------------------------------------------------------
# Augmentations (section 2.2)
# --------------------------------------------------------------------------


class Augmentation(nn.Module):
    """Base class: map one ``(B, L, d)`` stream to ``p`` streams ``(B, p, L', e)``."""

    def out_channels(self, in_channels: int) -> int:
        raise NotImplementedError

    def out_streams(self, in_channels: int) -> int:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Identity(Augmentation):
    """``None`` in table 1: a single stream, unchanged."""

    def out_channels(self, in_channels: int) -> int:
        return in_channels

    def out_streams(self, in_channels: int) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)


class TimeAugmentation(Augmentation):
    """Section 2.2.1: prepend a time channel, ``x -> ((t_1, x_1), ..., (t_n, x_n))``.

    Guarantees uniqueness of the signature map and makes it sensitive to time
    reparametrisation.  Time runs over ``[0, 1]`` so the channel is on the same
    scale as the z-scored inputs.
    """

    def out_channels(self, in_channels: int) -> int:
        return in_channels + 1

    def out_streams(self, in_channels: int) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length = x.shape[0], x.shape[-2]
        t = torch.linspace(0.0, 1.0, length, device=x.device, dtype=x.dtype)
        t = t.view(*([1] * (x.dim() - 2)), length, 1).expand(*x.shape[:-1], 1)
        return torch.cat([t, x], dim=-1)


class BasepointAugmentation(Augmentation):
    """Section 2.2.1: prepend the origin, ``x -> (0, x_1, ..., x_n)``.

    Introduces sensitivity to translation without adding a channel.
    """

    def out_channels(self, in_channels: int) -> int:
        return in_channels

    def out_streams(self, in_channels: int) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(x[..., :1, :])
        return torch.cat([zero, x], dim=-2)


class InvisibilityResetAugmentation(Augmentation):
    """Section 2.2.1: translation sensitivity that preserves the original signature.

    ``x -> ((1, x_1), ..., (1, x_n), (0, x_n), (0, 0))``, at the cost of one
    extra channel and two extra points.
    """

    def out_channels(self, in_channels: int) -> int:
        return in_channels + 1

    def out_streams(self, in_channels: int) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(x[..., :1])
        visible = torch.cat([ones, x], dim=-1)
        last = visible[..., -1:, :].clone()
        last[..., 0] = 0.0
        zero = torch.zeros_like(last)
        return torch.cat([visible, last, zero], dim=-2)


class LeadLagAugmentation(Augmentation):
    """Section 2.2.3: ``x -> ((x_1, x_1), (x_2, x_1), (x_2, x_2), ...)``.

    Doubles the channels and exposes the quadratic variation of the path
    explicitly to the signature.
    """

    def out_channels(self, in_channels: int) -> int:
        return 2 * in_channels

    def out_streams(self, in_channels: int) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = torch.repeat_interleave(x, 2, dim=-2)[..., 1:, :]
        lag = torch.repeat_interleave(x, 2, dim=-2)[..., :-1, :]
        return torch.cat([lead, lag], dim=-1)


class CoordinateProjection(Augmentation):
    """Section 2.2.2: signatures of channel subsets instead of the whole stream.

    The only way to shrink the ``O(d^N)`` feature count at a fixed depth is to
    shrink ``d``, and this does it while staying interpretable — each stream
    keeps the interaction *within* one subset and drops the interactions across
    subsets.  ``size=2`` (pairs) is what GSM-Alpha uses.

    Args:
        size: 1, 2 or 3 — singletons, pairs or triplets.
        ordered: If true emit ordered tuples (``d(d-1)`` streams for pairs, the
            count printed in table 1); if false emit combinations
            (``d(d-1)/2``), which is what the formula in section 2.2.2 lists.
            Both carry the same information for pairs, since swapping the two
            channels just transposes the signature levels; ``False`` is the
            cheaper default.
    """

    def __init__(self, size: int = 2, ordered: bool = False) -> None:
        super().__init__()
        if size not in (1, 2, 3):
            raise ValueError(f"coordinate projection supports size 1, 2 or 3, got {size}")
        self.size = size
        self.ordered = ordered

    def _subsets(self, in_channels: int) -> List[Sequence[int]]:
        if self.size > in_channels:
            raise ValueError(
                f"cannot take {self.size}-channel subsets of a {in_channels}-channel path"
            )
        if self.size == 1:
            return [(i,) for i in range(in_channels)]
        gen = permutations if self.ordered else combinations
        return [tuple(s) for s in gen(range(in_channels), self.size)]

    def out_channels(self, in_channels: int) -> int:
        return self.size

    def out_streams(self, in_channels: int) -> int:
        return len(self._subsets(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(
            [i for subset in self._subsets(x.shape[-1]) for i in subset],
            device=x.device,
            dtype=torch.long,
        )
        picked = x.index_select(-1, idx)  # (B, L, n_subsets * size)
        b, length, _ = picked.shape
        return picked.reshape(b, length, -1, self.size).transpose(1, 2)


class RandomProjection(Augmentation):
    """Section 2.2.2: ``p`` fixed random affine maps ``R^d -> R^e``.

    Registered as buffers, so the projection is part of the checkpoint and the
    features stay reproducible across runs.
    """

    def __init__(self, in_channels: int, out_channels: int, n_projections: int) -> None:
        super().__init__()
        self._out = out_channels
        self._p = n_projections
        weight = torch.randn(n_projections, in_channels, out_channels) / in_channels**0.5
        self.register_buffer("weight", weight)
        self.register_buffer("bias", torch.zeros(n_projections, out_channels))

    def out_channels(self, in_channels: int) -> int:
        return self._out

    def out_streams(self, in_channels: int) -> int:
        return self._p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bld,pde->bple", x, self.weight) + self.bias.view(1, self._p, 1, -1)


class MultiHeadedStreamPreserving(Augmentation):
    """Section 2.2.2: ``p`` learnable sequence-to-sequence maps ``R^d -> R^e``.

    The report's second GSM setting (figure 3) keeps this deliberately simple —
    "只从变量维数的角度考虑，使用多个 MLP 将高维序列映射到低维" — so each head is a
    pointwise MLP, which preserves the stream length exactly.

    Unlike every other augmentation here this one has parameters, so its
    signature features depend on the weights and cannot be precomputed; a
    config using it must run with ``data.cache.enabled: false``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self._out = out_channels
        self._p = n_heads
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_channels),
            )
            for _ in range(n_heads)
        )

    def out_channels(self, in_channels: int) -> int:
        return self._out

    def out_streams(self, in_channels: int) -> int:
        return self._p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(x) for head in self.heads], dim=1)


class ComposeAugmentation(Augmentation):
    """Run augmentations in sequence, fanning streams out multiplicatively.

    Section 2.2.3 notes the stages combine, and GSM-Alpha uses exactly that:
    coordinate projection, then time, then basepoint.
    """

    def __init__(self, stages: Sequence[Augmentation]) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def out_channels(self, in_channels: int) -> int:
        channels = in_channels
        for stage in self.stages:
            channels = stage.out_channels(channels)
        return channels

    def out_streams(self, in_channels: int) -> int:
        streams, channels = 1, in_channels
        for stage in self.stages:
            streams *= stage.out_streams(channels)
            channels = stage.out_channels(channels)
        return streams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.unsqueeze(1)  # (B, streams, L, C)
        for stage in self.stages:
            b, p, length, c = out.shape
            new = stage(out.reshape(b * p, length, c))
            if new.dim() == 3:  # stage kept one stream per input stream
                new = new.unsqueeze(1)
            _, q, new_len, new_c = new.shape
            out = new.reshape(b, p * q, new_len, new_c)
        return out


_SENSITIVITY = {
    "time": TimeAugmentation,
    "basepoint": BasepointAugmentation,
    "invisibility_reset": InvisibilityResetAugmentation,
    "lead_lag": LeadLagAugmentation,
}


def build_augmentation(spec: Sequence[str], in_channels: int, **kwargs) -> Augmentation:
    """Assemble the augmentation pipeline named in the config.

    Args:
        spec: Stage names in application order, e.g.
            ``["coordinate_projection", "time", "basepoint"]``.  Recognised
            names are ``none``, ``time``, ``basepoint``, ``invisibility_reset``,
            ``lead_lag``, ``coordinate_projection``, ``random_projection`` and
            ``multi_headed_stream_preserving``.
        in_channels: Channel count of the raw input stream.
        **kwargs: ``projection_size`` and ``projection_ordered`` for coordinate
            projection; ``projection_out``, ``n_projections`` and
            ``mhsp_hidden`` for the projection-style augmentations.

    Returns:
        The composed augmentation.
    """
    stages: List[Augmentation] = []
    channels = in_channels
    for name in spec:
        if name == "none":
            stage: Augmentation = Identity()
        elif name in _SENSITIVITY:
            stage = _SENSITIVITY[name]()
        elif name == "coordinate_projection":
            stage = CoordinateProjection(
                size=kwargs.get("projection_size", 2),
                ordered=kwargs.get("projection_ordered", False),
            )
        elif name == "random_projection":
            stage = RandomProjection(
                channels, kwargs.get("projection_out", 3), kwargs.get("n_projections", 10)
            )
        elif name == "multi_headed_stream_preserving":
            stage = MultiHeadedStreamPreserving(
                channels,
                kwargs.get("projection_out", 3),
                kwargs.get("n_projections", 10),
                hidden=kwargs.get("mhsp_hidden", 32),
            )
        else:
            raise ValueError(f"unknown augmentation {name!r}")
        stages.append(stage)
        channels = stage.out_channels(channels)
    if not stages:
        stages = [Identity()]
    return ComposeAugmentation(stages)


# --------------------------------------------------------------------------
# Windows (section 2.3)
# --------------------------------------------------------------------------


def window_slices(kind: str, length: int, *, size: int = 0, step: int = 0, depth: int = 3):
    """Sub-stream index ranges for one of the four window settings.

    Args:
        kind: ``global``, ``sliding``, ``expanding`` or ``dyadic``.
        length: Number of points in the (already augmented) stream.
        size: Initial/rolling window length for sliding and expanding windows.
        step: Stride for sliding and expanding windows.
        depth: ``q`` for the dyadic window — the stream itself plus its halves,
            quarters, ... down to ``2**(q-1)`` pieces, ``2**q - 1`` windows.

    Returns:
        List of ``(start, stop)`` half-open ranges, each at least 2 points long.
    """
    if kind == "global":
        return [(0, length)]
    if kind == "sliding":
        if size <= 0 or step <= 0:
            raise ValueError("sliding window needs size > 0 and step > 0")
        return [(s, s + size) for s in range(0, length - size + 1, step)]
    if kind == "expanding":
        if size <= 0 or step <= 0:
            raise ValueError("expanding window needs size > 0 and step > 0")
        stops = list(range(size, length + 1, step))
        if stops and stops[-1] != length:
            stops.append(length)
        return [(0, stop) for stop in stops]
    if kind == "dyadic":
        out = []
        for level in range(depth):
            pieces = 2**level
            edges = [round(i * length / pieces) for i in range(pieces + 1)]
            for i in range(pieces):
                if edges[i + 1] - edges[i] >= 2:
                    out.append((edges[i], edges[i + 1]))
        return out
    raise ValueError(f"unknown window kind {kind!r}")


# --------------------------------------------------------------------------
# The full GSM module
# --------------------------------------------------------------------------


class GSM(nn.Module):
    """Generalized Signature Method feature extractor.

    Turns a ``(batch, length, channels)`` multivariate stream into a flat
    cross-sectional feature vector, the concatenation over every
    (augmented stream, window) pair of that sub-stream's signature.

    Args:
        in_channels: Channels of the raw input stream (5 for OHLCV).
        depth: Signature truncation depth ``N``.  Section 2.6 finds 4-5 best
            across 26 datasets; GSM-Alpha uses 5.
        augmentations: Stage names for :func:`build_augmentation`.
        transform: ``"logsignature"`` or ``"signature"``.  The log-signature
            carries the same information in fewer channels but loses the
            universal-nonlinearity (linear-estimation) property.
        window: ``global``, ``sliding``, ``expanding`` or ``dyadic``.
        window_size: Window length for sliding/expanding.
        window_step: Stride for sliding/expanding.
        window_depth: ``q`` for the dyadic window.
        rescaling: ``none``, ``pre`` or ``post`` (section 2.5).  Table 7 finds
            ``none`` best, which is what GSM-Alpha uses.
        backend: ``auto``, ``signatory`` or ``torch``.
        **aug_kwargs: Forwarded to :func:`build_augmentation`.
    """

    def __init__(
        self,
        in_channels: int,
        depth: int = 5,
        augmentations: Sequence[str] = ("coordinate_projection", "time", "basepoint"),
        transform: str = "logsignature",
        window: str = "global",
        window_size: int = 0,
        window_step: int = 0,
        window_depth: int = 3,
        rescaling: str = "none",
        backend: str = "auto",
        **aug_kwargs,
    ) -> None:
        super().__init__()
        if transform not in ("logsignature", "signature"):
            raise ValueError(f"unknown transform {transform!r}")
        if rescaling not in ("none", "pre", "post"):
            raise ValueError(f"unknown rescaling {rescaling!r}")

        self.in_channels = in_channels
        self.depth = depth
        self.transform = transform
        self.window = window
        self.window_size = window_size
        self.window_step = window_step
        self.window_depth = window_depth
        self.rescaling = rescaling
        self.backend = backend

        self.augment = build_augmentation(augmentations, in_channels, **aug_kwargs)
        self.stream_channels = self.augment.out_channels(in_channels)
        self.n_streams = self.augment.out_streams(in_channels)

        per_window = (
            logsignature_channels(self.stream_channels, depth)
            if transform == "logsignature"
            else signature_channels(self.stream_channels, depth)
        )
        self.channels_per_window = per_window
        self._pre_scale = float(_factorial(depth) ** (1.0 / depth)) if rescaling == "pre" else 1.0
        self._post_scale = _post_scale_vector(self.stream_channels, depth, transform)

    @property
    def is_learnable(self) -> bool:
        """Whether the features depend on trained parameters (i.e. cannot be cached)."""
        return any(p.requires_grad for p in self.parameters())

    def n_windows(self, length: int) -> int:
        """Number of sub-streams the window setting produces for this input length."""
        augmented = self.augment(torch.zeros(1, length, self.in_channels))
        return len(
            window_slices(
                self.window,
                augmented.shape[-2],
                size=self.window_size,
                step=self.window_step,
                depth=self.window_depth,
            )
        )

    def out_features(self, length: int) -> int:
        """Total feature count for an input stream of ``length`` points."""
        return self.n_streams * self.n_windows(length) * self.channels_per_window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract GSM features.

        Args:
            x: ``(batch, length, in_channels)``.

        Returns:
            ``(batch, out_features(length))``.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (batch, length, channels), got {tuple(x.shape)}")
        streams = self.augment(x)  # (B, p, L', e)
        b, p, length, e = streams.shape
        if self.rescaling == "pre":
            streams = streams * self._pre_scale

        slices = window_slices(
            self.window, length, size=self.window_size, step=self.window_step, depth=self.window_depth
        )
        flat = streams.reshape(b * p, length, e)
        pieces = []
        for start, stop in slices:
            sub = flat[:, start:stop]
            if self.transform == "logsignature":
                feat = logsignature(sub, self.depth, backend=self.backend)
            else:
                feat = signature(sub, self.depth, backend=self.backend)
            if self.rescaling == "post":
                feat = feat * self._post_scale.to(feat.device, feat.dtype)
            pieces.append(feat.reshape(b, p, -1))
        return torch.cat(pieces, dim=-1).reshape(b, -1)


def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def _post_scale_vector(channels: int, depth: int, transform: str) -> torch.Tensor:
    """Per-channel ``k!`` multipliers for post-signature rescaling (section 2.5).

    Level ``k`` of a signature is ``O(1/k!)``; multiplying by ``k!`` brings every
    level to ``O(1)``.
    """
    if transform == "signature":
        lengths = [k for k in range(1, depth + 1) for _ in range(channels**k)]
    else:
        from .lyndon import lyndon_words

        lengths = [len(word) for word in lyndon_words(channels, depth)]
    return torch.tensor([float(_factorial(k)) for k in lengths])
