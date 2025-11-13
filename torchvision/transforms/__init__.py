from __future__ import annotations

from enum import Enum


class InterpolationMode(str, Enum):
    """
    Lightweight copy of torchvision's `InterpolationMode` enum covering the values
    referenced by Transformers preprocessing utilities.
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BOX = "box"
    BILINEAR = "bilinear"
    HAMMING = "hamming"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


__all__ = ["InterpolationMode"]
