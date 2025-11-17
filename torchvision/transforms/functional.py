"""
Subset of torchvision.transforms.functional used by Transformers.

Only implements ``normalize`` which is required when running the HF
image pipelines. The goal is to avoid pulling in the heavy torchvision
wheel on systems where it is not available.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def _to_tensor(value: torch.Tensor | Sequence[float] | Iterable[float], *, ref: torch.Tensor) -> torch.Tensor:
    """
    Convert ``value`` to a tensor on the same device/dtype as ``ref``.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(value, device=ref.device, dtype=ref.dtype)


def normalize(
    tensor: torch.Tensor,
    mean: torch.Tensor | Sequence[float] | Iterable[float],
    std: torch.Tensor | Sequence[float] | Iterable[float],
    inplace: bool = False,
) -> torch.Tensor:
    """
    Minimal port of ``torchvision.transforms.functional.normalize``.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    if tensor.ndim not in (3, 4):
        raise ValueError("normalize expects a 3D (C, H, W) or 4D (N, C, H, W) tensor")

    if not inplace:
        tensor = tensor.clone()

    tensor = tensor.to(torch.float32 if not tensor.is_floating_point() else tensor.dtype)
    mean_tensor = _to_tensor(mean, ref=tensor)
    std_tensor = _to_tensor(std, ref=tensor)

    if mean_tensor.ndim != 1 or std_tensor.ndim != 1:
        raise ValueError("mean and std must be 1D sequences")

    if mean_tensor.numel() != std_tensor.numel():
        raise ValueError("mean and std must have the same number of elements")

    channel_dim = 0 if tensor.ndim == 3 else 1
    if tensor.shape[channel_dim] != mean_tensor.numel():
        raise ValueError(
            f"Expected {tensor.shape[channel_dim]} mean/std entries, got {mean_tensor.numel()}"
        )

    # Expand mean/std to broadcast over spatial dims
    view_shape = [1] * tensor.ndim
    view_shape[channel_dim] = mean_tensor.numel()
    mean_view = mean_tensor.view(*view_shape)
    std_view = std_tensor.view(*view_shape)

    tensor.sub_(mean_view).div_(std_view)
    return tensor


__all__ = ["normalize"]
