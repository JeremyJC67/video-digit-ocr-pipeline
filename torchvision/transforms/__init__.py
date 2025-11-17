from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Sequence


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


class Compose:
    """
    Simple substitute for torchvision.transforms.Compose.
    """

    def __init__(self, transforms: Sequence):
        self.transforms: List = list(transforms)

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self) -> str:
        parts = ",\n    ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}([\n    {parts}\n])"


from . import functional

__all__ = ["InterpolationMode", "Compose", "functional"]
