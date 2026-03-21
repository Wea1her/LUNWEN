"""Device and seed helpers for train/eval scripts."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(device_arg: str) -> torch.device:
    """Resolve device string and validate CUDA availability when requested."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested, but torch.cuda.is_available() is False. "
                "Please install a CUDA-enabled PyTorch build."
            )
        return torch.device(device_arg)

    if device_arg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device_arg}")


def seed_everything(seed: int):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
