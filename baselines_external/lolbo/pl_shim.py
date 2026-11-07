"""Minimal stand-in for pytorch_lightning so the vendored VAE imports unmodified.

The vendored SELFIES VAE (`vendor/lolbo/utils/mol_utils/selfies_vae/*.py`) is kept
byte-for-byte upstream so its SHA256 in PROVENANCE.md is verifiable. Those files
`import pytorch_lightning as pl` only to subclass `pl.LightningModule` /
`pl.LightningDataModule` and to reference two callbacks in training code we never
run. Installing Lightning just for that would add a large dependency (and a torch
version constraint) to the shared `trace_benchmark` environment, so this module
registers a stub package instead.

The only behaviour the inference path actually needs from LightningModule is
`self.device`, which the VAE uses when allocating start tokens and masks.
"""

from __future__ import annotations

import sys
import types

import torch
from torch import nn


class _LightningModule(nn.Module):
    """nn.Module plus the `.device` property LightningModule provides."""

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")


class _LightningDataModule:
    """Subclassed by SELFIESDataModule, which we never instantiate."""


class _Callback:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "pytorch_lightning callbacks are stubbed out; this package only runs "
            "VAE inference, never VAE training."
        )


def install() -> None:
    """Register the stub `pytorch_lightning` package (idempotent, real one wins)."""
    if "pytorch_lightning" in sys.modules:
        return
    try:  # if Lightning really is installed, prefer it
        import pytorch_lightning  # noqa: F401

        return
    except ImportError:
        pass

    pl = types.ModuleType("pytorch_lightning")
    pl.LightningModule = _LightningModule
    pl.LightningDataModule = _LightningDataModule
    pl.__version__ = "0.0.0+trace-shim"

    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    callbacks.ModelCheckpoint = _Callback
    callbacks.RichProgressBar = _Callback
    pl.callbacks = callbacks

    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.callbacks"] = callbacks
