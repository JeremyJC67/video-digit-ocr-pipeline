from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps

from .. import InterpolationMode


TensorLike = Union[torch.Tensor, np.ndarray]
PaddingType = Union[int, Tuple[int, int], Tuple[int, int, int, int], Sequence[int]]


class _FunctionalNamespace:
    def pil_to_tensor(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                return image.clone()
            raise ValueError("Unsupported tensor shape for pil_to_tensor.")
        if isinstance(image, np.ndarray):
            array = image
        else:
            array = np.array(image, copy=True)
        if array.ndim == 2:
            array = array[:, :, None]
        tensor = torch.from_numpy(array)
        tensor = tensor.permute(2, 0, 1).contiguous()
        return tensor

    def to_tensor(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        tensor = self.pil_to_tensor(image).to(torch.float32)
        return tensor / 255.0

    def pad(
        self,
        image: Union[Image.Image, torch.Tensor],
        padding: PaddingType,
        fill: Union[int, tuple[int, int, int]] = 0,
        padding_mode: str = "constant",
    ) -> Union[Image.Image, torch.Tensor]:
        padding = self._expand_padding(padding)
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            return torch.stack([self.pad(sample, padding, fill, padding_mode) for sample in image])
        if isinstance(image, Image.Image):
            return ImageOps.expand(image, border=padding, fill=fill)
        if not isinstance(image, torch.Tensor):
            image = self.pil_to_tensor(image)
        pad_left, pad_top, pad_right, pad_bottom = padding
        if padding_mode != "constant":
            raise ValueError("Only constant padding is supported in this stub.")
        pad_dims = (pad_left, pad_right, pad_top, pad_bottom)
        return torch.nn.functional.pad(image, pad_dims, mode="constant", value=fill if isinstance(fill, int) else 0)

    def crop(
        self,
        image: Union[Image.Image, torch.Tensor],
        top: int,
        left: int,
        height: int,
        width: int,
    ) -> Union[Image.Image, torch.Tensor]:
        if isinstance(image, Image.Image):
            return image.crop((left, top, left + width, top + height))
        if not isinstance(image, torch.Tensor):
            image = self.pil_to_tensor(image)
        return image[..., top : top + height, left : left + width]

    def resize(
        self,
        image: Union[Image.Image, torch.Tensor],
        size: tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ) -> Union[Image.Image, torch.Tensor]:
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            return torch.stack([self.resize(sample, size, interpolation, antialias) for sample in image])
        pil_image = image if isinstance(image, Image.Image) else self.to_pil_image(image)
        pil_resample = self._pil_resample(interpolation)
        resized = pil_image.resize(size[::-1], resample=pil_resample, reducing_gap=antialias)
        if isinstance(image, Image.Image):
            return resized
        return self.pil_to_tensor(resized)

    def normalize(self, tensor: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            tensor = self.to_tensor(tensor)
        if tensor.ndim == 4:
            mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)[None, :, None, None]
            std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)[None, :, None, None]
        else:
            mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)[:, None, None]
            std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)[:, None, None]
        return (tensor - mean_tensor) / std_tensor

    def _expand_padding(self, padding: PaddingType) -> tuple[int, int, int, int]:
        if isinstance(padding, int):
            return (padding, padding, padding, padding)
        if len(padding) == 2:
            horizontal, vertical = padding
            return (horizontal, vertical, horizontal, vertical)
        if len(padding) == 4:
            return tuple(padding)
        raise ValueError("Padding must be an int, tuple of 2 ints, or tuple of 4 ints.")

    def to_pil_image(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError("to_pil_image expects a 3D tensor.")
            array = image.detach().cpu().permute(1, 2, 0).numpy()
        else:
            array = image
        array = np.clip(array, 0, 1) if array.dtype.kind == "f" else array
        array = (array * 255).astype(np.uint8) if array.dtype.kind == "f" else array.astype(np.uint8)
        return Image.fromarray(array)

    def _pil_resample(self, interpolation: InterpolationMode) -> int:
        if interpolation == InterpolationMode.NEAREST:
            return Image.Resampling.NEAREST
        if interpolation == InterpolationMode.NEAREST_EXACT:
            return Image.Resampling.NEAREST
        if interpolation == InterpolationMode.BOX:
            return Image.Resampling.BOX
        if interpolation == InterpolationMode.BILINEAR:
            return Image.Resampling.BILINEAR
        if interpolation == InterpolationMode.HAMMING:
            return Image.Resampling.HAMMING
        if interpolation == InterpolationMode.BICUBIC:
            return Image.Resampling.BICUBIC
        if interpolation == InterpolationMode.LANCZOS:
            return Image.Resampling.LANCZOS
        return Image.Resampling.BILINEAR


functional = _FunctionalNamespace()

__all__ = ["functional"]
