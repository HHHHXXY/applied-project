"""Selection between the signatory and pure-PyTorch signature backends.

``signatory`` is preferred whenever it imports (it is roughly two orders of
magnitude faster), and it only builds against PyTorch 1.x — see
``env/create_env.sh``.  On any interpreter where the import fails the pure
PyTorch implementation takes over with identical semantics, so the rest of the
codebase never branches on which one is live.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import torch_backend

try:  # pragma: no cover - exercised by whichever environment is in use
    import signatory as _signatory
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _signatory = None

SIGNATORY_AVAILABLE = _signatory is not None

_BACKENDS = ("auto", "signatory", "torch")


def resolve_backend(name: str = "auto") -> str:
    """Turn a backend request into the concrete backend that will be used.

    Args:
        name: ``"auto"``, ``"signatory"`` or ``"torch"``.

    Returns:
        Either ``"signatory"`` or ``"torch"``.

    Raises:
        ValueError: If ``name`` is not a known backend.
        RuntimeError: If signatory was demanded but is not importable.
    """
    if name not in _BACKENDS:
        raise ValueError(f"unknown signature backend {name!r}, expected one of {_BACKENDS}")
    if name == "signatory" and not SIGNATORY_AVAILABLE:
        raise RuntimeError(
            "signature.backend='signatory' was requested but signatory is not importable. "
            "It only builds against PyTorch 1.x; see env/create_env.sh, or use "
            "backend='torch'."
        )
    if name == "auto":
        return "signatory" if SIGNATORY_AVAILABLE else "torch"
    return name


def logsignature_channels(channels: int, depth: int) -> int:
    """Number of log-signature channels; identical for both backends."""
    return torch_backend.logsignature_channels(channels, depth)


def signature_channels(channels: int, depth: int) -> int:
    """Number of signature channels; identical for both backends."""
    return torch_backend.signature_channels(channels, depth)


def logsignature(path: torch.Tensor, depth: int, backend: str = "auto") -> torch.Tensor:
    """Log-signature in Lyndon-word coordinates.

    Args:
        path: ``(batch, length, channels)``.
        depth: Truncation depth.
        backend: See :func:`resolve_backend`.

    Returns:
        ``(batch, logsignature_channels(channels, depth))``.
    """
    if resolve_backend(backend) == "signatory":
        return _signatory.logsignature(path, depth, mode="words")
    return torch_backend.logsignature(path, depth, mode="words")


def signature(path: torch.Tensor, depth: int, backend: str = "auto") -> torch.Tensor:
    """Truncated signature.

    Args:
        path: ``(batch, length, channels)``.
        depth: Truncation depth.
        backend: See :func:`resolve_backend`.

    Returns:
        ``(batch, signature_channels(channels, depth))``.
    """
    if resolve_backend(backend) == "signatory":
        return _signatory.signature(path, depth)
    return torch_backend.signature(path, depth)


def backend_report(backend: str = "auto") -> str:
    """One-line human-readable description of the live backend, for logs."""
    resolved = resolve_backend(backend)
    if resolved == "signatory":
        version: Optional[str] = getattr(_signatory, "__version__", None)
        return f"signature backend: signatory {version} (torch {torch.__version__})"
    reason = "requested" if backend == "torch" else "signatory unavailable"
    return f"signature backend: pure-torch fallback ({reason}; torch {torch.__version__})"
