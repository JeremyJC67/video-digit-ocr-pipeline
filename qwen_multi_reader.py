#!/usr/bin/env python3
"""
Frame-by-frame multi-metric reader powered by Qwen/Qwen3-VL-2B-Thinking.

Extracts two values per sampled frame:
1. Measured temperature on the green controller (°C).
2. BC/O pressure (mbar) from the top monitor table.

Results are saved to CSV and JSONL files for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import types
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

if hasattr(torch, "compile"):
    try:
        torch.compile(lambda x: x)
    except RuntimeError as compile_exc:  # pragma: no cover
        if "not supported" in str(compile_exc).lower():

            def _noop_compile(fn=None, *args, **kwargs):
                if callable(fn):
                    return fn

                def decorator(func):
                    return func

                return decorator

            torch.compile = _noop_compile  # type: ignore[assignment]

_original_metadata_version = importlib_metadata.version


def _patched_metadata_version(pkg_name: str) -> str:
    try:
        return _original_metadata_version(pkg_name)
    except importlib_metadata.PackageNotFoundError:
        if pkg_name == "torchvision":
            return "0.0.0"
        raise


importlib_metadata.version = _patched_metadata_version  # type: ignore[misc]

try:
    import torch.distributed.tensor  # noqa: F401
except Exception as dist_exc:  # pragma: no cover
    if os.environ.get("QWEN_DEBUG_STUB"):
        print(f"[qwen_multi_reader] torch.distributed.tensor unavailable: {dist_exc}", file=sys.stderr)

    class _TorchDistTensorStub(types.ModuleType):
        def __getattr__(self, name):  # pragma: no cover
            def _missing(*args, **kwargs):
                raise RuntimeError(
                    f"torch.distributed.tensor attribute '{name}' unavailable because import failed: {dist_exc}"
                )

            return _missing

    dist_stub = types.ModuleType("torch.distributed")
    tensor_stub = _TorchDistTensorStub("torch.distributed.tensor")
    dist_stub.tensor = tensor_stub
    dist_stub.__spec__ = ModuleSpec("torch.distributed", loader=None)
    tensor_stub.__spec__ = ModuleSpec("torch.distributed.tensor", loader=None)
    sys.modules["torch.distributed"] = dist_stub
    sys.modules["torch.distributed.tensor"] = tensor_stub

try:
    import sklearn  # noqa: F401
except Exception as exc:  # pragma: no cover
    if os.environ.get("QWEN_DEBUG_STUB"):
        print(f"[qwen_multi_reader] sklearn unavailable: {exc}", file=sys.stderr)
    for name in list(sys.modules):
        if name.startswith("sklearn"):
            sys.modules.pop(name, None)

    class _SklearnMetricsStub(types.ModuleType):
        def __getattr__(self, name):  # pragma: no cover
            def _missing(*args, **kwargs):
                raise RuntimeError(f"scikit-learn metric '{name}' unavailable because import failed: {exc}")

            return _missing

    stub = types.ModuleType("sklearn")
    metrics_stub = _SklearnMetricsStub("sklearn.metrics")
    stub.metrics = metrics_stub
    stub.__spec__ = ModuleSpec("sklearn", loader=None)
    metrics_stub.__spec__ = ModuleSpec("sklearn.metrics", loader=None)
    sys.modules["sklearn"] = stub
    sys.modules["sklearn.metrics"] = metrics_stub
    if os.environ.get("QWEN_DEBUG_STUB"):
        print(f"[qwen_multi_reader] stub modules: {sys.modules['sklearn']}, {sys.modules['sklearn.metrics']}", file=sys.stderr)

from transformers import pipeline

PROMPT_MULTI = (
    "Return ONLY one JSON object on a single line. The VERY FIRST character of your reply must be '{'. "
    "Do not explain, do not describe your reasoning, do not output any other text.\n\n"
    "You are reading two numeric values from this lab image:\n"
    "1) The measured temperature on the GREEN controller at the bottom. It is on the line starting with "
    "'Measured Temp:' and ending with '°C'. Read the number in degrees Celsius (it can be negative and can have decimals).\n"
    "2) The chamber pressure labeled 'BC/O' on the TOP monitor in units of mbar. Find the row whose label is exactly "
    "'BC/O' and read its numeric value. The value can be in scientific notation (for example 4.33e-004). "
    "In the table, 'BC/O' is one specific row; DO NOT use the values from 'IGPc1' or 'IGPco'.\n\n"
    "Output format (JSON on one line):\n"
    '{"measured_temp_c": -132.1, "bc_o_mbar": 4.33e-4}\n'
    "If a value is completely unreadable (for example, covered or missing), use null instead of a number. "
    "Otherwise, always output your best numeric guess.\n"
)



JSON_PATTERN_MULTI = re.compile(r'\{[^{}]*"measured_temp_c"[^{}]*"bc_o_mbar"[^{}]*\}')
TEMP_RE = re.compile(r"Measured Temp[^-\d]*(-?\d+(?:\.\d+)?)\s*°C", re.IGNORECASE)
BCO_RE = re.compile(
    r"^[ \t]*BC/O[^0-9+\-]*([+\-]?\d+(?:\.\d+)?(?:e[+\-]?\d+)?)\s*mbar",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class FrameResult:
    frame_idx: int
    timestamp_sec: float
    measured_temp_c: Optional[float]
    bc_o_mbar: Optional[float]
    raw_text: str


def parse_roi(arg: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not arg:
        return None
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be formatted as x,y,w,h")
    roi = tuple(int(v) for v in parts)
    x, y, w, h = roi
    if min(w, h) <= 0:
        raise ValueError("ROI width and height must be positive")
    return x, y, w, h


def crop_roi(image: Image.Image, roi: Tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = roi
    width, height = image.size
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x1 >= x2 or y1 >= y2:
        return image
    return image.crop((x1, y1, x2, y2))


def seconds_to_timestamp(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def resolve_dtype(dtype_str: str) -> Optional[torch.dtype]:
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_str}")
    return mapping[dtype_str]


def _device_index_from_str(device_str: str) -> int:
    if ":" in device_str:
        return int(device_str.split(":")[1])
    return 0


def _device_map_from_device_str(device_str: str) -> object:
    normalized = device_str.lower()
    if normalized == "auto":
        return "auto"
    if normalized == "cpu":
        return {"": "cpu"}
    if normalized == "mps":
        return {"": "mps"}
    if normalized.startswith("cuda"):
        idx = _device_index_from_str(normalized)
        return {"": idx}
    return "auto"


def resolve_device_options(
    device_str: str,
    force_device_map: bool,
) -> Tuple[Optional[torch.device], Optional[object]]:
    normalized = (device_str or "auto").lower()
    if force_device_map:
        return None, _device_map_from_device_str(normalized)
    if normalized == "auto":
        return None, "auto"
    if normalized == "cpu":
        return torch.device("cpu"), None
    if normalized == "mps":
        return torch.device("mps"), None
    if normalized.startswith("cuda"):
        idx = _device_index_from_str(normalized)
        return torch.device("cuda", idx), None
    return None, "auto"


def build_qwen_pipeline(
    model_name: str,
    device: str,
    dtype_str: str,
    load_in_4bit: bool,
) -> object:
    torch_dtype = resolve_dtype(dtype_str)
    model_kwargs = {}
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        if torch_dtype is not None:
            model_kwargs["bnb_4bit_compute_dtype"] = torch_dtype

    resolved_device, device_map = resolve_device_options(device, force_device_map=load_in_4bit)

    pipeline_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
    }
    if model_kwargs:
        pipeline_kwargs["model_kwargs"] = model_kwargs
    if torch_dtype is not None and not load_in_4bit:
        pipeline_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        pipeline_kwargs["device_map"] = device_map
    elif resolved_device is not None:
        pipeline_kwargs["device"] = resolved_device

    return pipeline("image-text-to-text", **pipeline_kwargs)


def run_inference(
    gen_pipe,
    image: Image.Image,
    max_new_tokens: int,
) -> str:
    cleaned_image = image.convert("RGB")
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_MULTI},
            ],
        }
    ]
    outputs = gen_pipe(
        images=cleaned_image,
        text=chat,
        return_full_text=False,
        generate_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
    )
    if not outputs:
        return ""
    text = outputs[0].get("generated_text", "")
    return " ".join(text.strip().split())


def parse_multi_metrics(raw_text: str) -> Tuple[Optional[float], Optional[float], str]:
    candidate = raw_text.strip()
    match = JSON_PATTERN_MULTI.search(candidate)

    measured_temp: Optional[float] = None
    bc_o: Optional[float] = None
    clean_snippet = candidate

    if match:
        candidate = match.group(0)
        try:
            payload = json.loads(candidate)
            mt = payload.get("measured_temp_c")
            if isinstance(mt, (int, float)):
                measured_temp = float(mt)
            elif isinstance(mt, str):
                measured_temp = float(mt)

            bc = payload.get("bc_o_mbar")
            if isinstance(bc, (int, float)):
                bc_o = float(bc)
            elif isinstance(bc, str):
                bc_o = float(bc)
        except Exception:
            measured_temp = None
            bc_o = None

    if measured_temp is None:
        temp_match = TEMP_RE.search(raw_text)
        if temp_match:
            try:
                measured_temp = float(temp_match.group(1))
            except ValueError:
                measured_temp = None

    if bc_o is None:
        bc_match = BCO_RE.search(raw_text)
        if bc_match:
            try:
                bc_o = float(bc_match.group(1))
            except ValueError:
                bc_o = None

    if measured_temp is not None and not (-300.0 <= measured_temp <= 300.0):
        measured_temp = None
    if bc_o is not None:
        if not (1e-6 <= bc_o <= 1e-2):
            bc_o = None

    try:
        clean_snippet = json.dumps(
            {
                "measured_temp_c": measured_temp,
                "bc_o_mbar": bc_o,
            },
            ensure_ascii=False,
        )
    except Exception:
        clean_snippet = candidate

    return measured_temp, bc_o, clean_snippet


def sample_frames(
    video_path: str,
    extract_fps: float,
    limit: Optional[int],
) -> List[Tuple[int, float, Image.Image, np.ndarray]]:
    if extract_fps <= 0:
        raise ValueError("--extract_fps must be > 0")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_index = max(total_frames - 1, 0)

    results: List[Tuple[int, float, Image.Image, np.ndarray]] = []
    step_sec = 1.0 / extract_fps
    timestamp = 0.0
    taken = 0

    try:
        while True:
            if limit is not None and taken >= limit:
                break
            frame_idx = int(round(timestamp * native_fps))
            frame_idx = min(max(frame_idx, 0), max_index)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            results.append((frame_idx, timestamp, pil_image, frame.copy()))

            taken += 1
            timestamp += step_sec

            if total_frames and frame_idx >= max_index:
                break
    finally:
        cap.release()

    return results


def infer_video_multi(
    video_path: str,
    gen_pipe,
    extract_fps: float,
    limit: Optional[int],
    roi: Optional[Tuple[int, int, int, int]],
    max_new_tokens: int,
) -> List[FrameResult]:
    frame_batches = sample_frames(video_path, extract_fps, limit)
    results: List[FrameResult] = []
    progress = tqdm(total=len(frame_batches), desc="Processing frames", unit="frame")

    for idx, (frame_idx, timestamp_sec, pil_image, _) in enumerate(frame_batches):
        roi_image = crop_roi(pil_image, roi) if roi else pil_image
        text = run_inference(gen_pipe, roi_image, max_new_tokens)
        measured_temp, bc_o, parsed_summary = parse_multi_metrics(text)
        results.append(
            FrameResult(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                measured_temp_c=measured_temp,
                bc_o_mbar=bc_o,
                raw_text=text,
            )
        )
        print(
            f"[{idx + 1:04d}] frame={frame_idx:06d} "
            f"time={seconds_to_timestamp(timestamp_sec)} "
            f"measured_temp_c={measured_temp if measured_temp is not None else 'null'} "
            f"bc_o_mbar={bc_o if bc_o is not None else 'null'} "
            f"raw={parsed_summary}"
        )
        progress.update(1)
    progress.close()
    return results


def save_results_multi(
    data: List[FrameResult],
    out_dir: Path,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results_multi.csv"
    jsonl_path = out_dir / "results_multi.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "video_timestamp", "measured_temp_c", "bc_o_mbar"])
        for item in data:
            writer.writerow(
                [
                    item.frame_idx,
                    seconds_to_timestamp(item.timestamp_sec),
                    "" if item.measured_temp_c is None else f"{item.measured_temp_c:.2f}",
                    "" if item.bc_o_mbar is None else f"{item.bc_o_mbar:.6g}",
                ]
            )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in data:
            payload = {
                "frame": item.frame_idx,
                "video_timestamp": seconds_to_timestamp(item.timestamp_sec),
                "measured_temp_c": item.measured_temp_c,
                "bc_o_mbar": item.bc_o_mbar,
                "raw": item.raw_text,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def smooth_bc_o(results: List[FrameResult]) -> None:
    last_valid: Optional[float] = None
    for item in results:
        if item.bc_o_mbar is not None:
            last_valid = item.bc_o_mbar
        elif last_valid is not None:
            item.bc_o_mbar = last_valid


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frame-wise multi-metric reader using Qwen and Transformers pipeline."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--extract_fps", type=float, default=1.0, help="Sampling FPS (default 1)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sampled frames")
    parser.add_argument("--roi", type=str, default=None, help="Crop ROI formatted as x,y,w,h")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Thinking",
        help="Transformers model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device hint: auto|cpu|mps|cuda[:id]",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Desired torch dtype",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the model in 4-bit (requires bitsandbytes)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens for generation (default 64)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_multi",
        help="Directory for CSV/JSONL outputs",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    roi = parse_roi(args.roi)
    out_dir = Path(args.out_dir)

    print(
        f"Loading model {args.model} "
        f"(device={args.device}, dtype={args.dtype}, load_in_4bit={args.load_in_4bit})"
    )
    gen_pipe = build_qwen_pipeline(args.model, args.device, args.dtype, args.load_in_4bit)

    print(f"Extracting frames from {args.video} at {args.extract_fps} fps")
    if roi:
        print(f"Applying ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    results = infer_video_multi(
        video_path=args.video,
        gen_pipe=gen_pipe,
        extract_fps=args.extract_fps,
        limit=args.limit,
        roi=roi,
        max_new_tokens=args.max_new_tokens,
    )

    smooth_bc_o(results)
    csv_path, jsonl_path = save_results_multi(results, out_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
