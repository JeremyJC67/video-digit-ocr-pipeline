"""
Minimal stub for `torchvision.io` so that Transformers video utilities can import it.
The actual video decoding features are not implemented; attempting to instantiate
`VideoReader` will raise a RuntimeError.
"""


class VideoReader:  # pragma: no cover - runtime guard
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torchvision.io.VideoReader is unavailable in this stub build.")


__all__ = ["VideoReader"]
