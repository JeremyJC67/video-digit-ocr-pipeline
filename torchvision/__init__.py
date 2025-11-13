"""
Minimal torchvision stub to satisfy Transformers optional dependency checks.

This module only exposes the pieces required by `transformers` for image
pre-processing (currently `torchvision.transforms.InterpolationMode`).
If you later install the real torchvision wheel, it will automatically take
precedence on the import path and this stub will no longer be used.
"""

from . import transforms

__all__ = ["transforms"]
