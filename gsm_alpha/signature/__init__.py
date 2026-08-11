"""Signature-method building blocks: Lyndon bookkeeping, backends and GSM."""

from .backend import (
    SIGNATORY_AVAILABLE,
    backend_report,
    logsignature,
    logsignature_channels,
    resolve_backend,
    signature,
    signature_channels,
)
from .gsm import GSM, build_augmentation, window_slices

__all__ = [
    "GSM",
    "SIGNATORY_AVAILABLE",
    "backend_report",
    "build_augmentation",
    "logsignature",
    "logsignature_channels",
    "resolve_backend",
    "signature",
    "signature_channels",
    "window_slices",
]
