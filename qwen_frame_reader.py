#!/usr/bin/env python3
"""
Frame-by-frame temperature reader using classic OCR (OpenCV + Tesseract).

This script dynamically finds the blue LCD screen, rectifies it, locates the °C
anchor, and extracts the numeric temperature line with adaptive preprocessing.
All readings are constrained to stay within a monotonic non-increasing curve,
matching the physical behaviour of the experiment (≈22.9 °C down to ≈-133.0 °C).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ocr_utils import find_degree_anchor_strict, run_ocr, set_temp_bounds


@dataclass
class TempResult:
    frame_idx: int
    timestamp_sec: float
    value: Optional[float]
    confidence: float
    raw_text: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic OCR temperature reader (no heavy LLM).")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--extract_fps", type=float, default=1.0, help="Sample FPS (default 1)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N frames")
    parser.add_argument("--start_time", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--max_duration", type=float, default=None, help="Max seconds to analyse (from start)")
    parser.add_argument("--roi", type=str, default=None, help="Optional ROI override: x,y,w,h")
    parser.add_argument("--out_dir", type=str, default="outputs_ocr", help="Folder for CSV/JSON outputs")
    parser.add_argument("--temp_min", type=float, default=-200.0, help="Minimum valid temperature")
    parser.add_argument("--temp_max", type=float, default=200.0, help="Maximum valid temperature")
    parser.add_argument(
        "--no-monotonic",
        action="store_false",
        dest="monotonic",
        help="Disable non-increasing enforcement",
    )
    parser.set_defaults(monotonic=True)
    parser.add_argument(
        "--monotonic_slack",
        type=float,
        default=0.2,
        help="Allowed upward slack before clamping (°C)",
    )
    return parser.parse_args()


def parse_roi(arg: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not arg:
        return None
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be formatted as x,y,w,h")
    x, y, w, h = map(int, parts)
    if min(w, h) <= 0:
        raise ValueError("ROI width/height must be > 0")
    return x, y, w, h


def seconds_to_timestamp(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def blue_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 40, 40], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_screen(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]], warp_size=(900, 600)) -> np.ndarray:
    if roi:
        x, y, w, h = roi
        sub = frame[y : y + h, x : x + w]
        if sub.size == 0:
            sub = frame
    else:
        sub = frame

    mask = blue_mask(sub)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(sub, warp_size, interpolation=cv2.INTER_LINEAR)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.01 * (sub.shape[0] * sub.shape[1]):
        return cv2.resize(sub, warp_size, interpolation=cv2.INTER_LINEAR)

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) < 4:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        approx = np.int0(box)

    if len(approx) >= 4:
        hull = cv2.convexHull(approx)
        if len(hull) < 4:
            hull = approx
        if len(hull) > 4:
            hull = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(hull) != 4:
            hull = cv2.boxPoints(cv2.minAreaRect(cnt))
        pts = order_points(np.float32(hull.reshape(-1, 2)))
        dst = np.array(
            [
                [0, 0],
                [warp_size[0] - 1, 0],
                [warp_size[0] - 1, warp_size[1] - 1],
                [0, warp_size[1] - 1],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(sub, M, warp_size)
        return warped

    x, y, w, h = cv2.boundingRect(cnt)
    cropped = sub[y : y + h, x : x + w]
    if cropped.size == 0:
        cropped = sub
    return cv2.resize(cropped, warp_size, interpolation=cv2.INTER_LINEAR)


def extract_digit_band(screen_bgr: np.ndarray) -> np.ndarray:
    H = screen_bgr.shape[0]
    anchor = find_degree_anchor_strict(screen_bgr)
    if anchor:
        _, ay, _, ah = anchor
        cy = ay + ah // 2
        band_h = max(int(ah * 2.2), int(H * 0.28))
        top = max(0, cy - band_h // 2)
        bottom = min(H, cy + band_h // 2)
    else:
        top = int(0.25 * H)
        bottom = int(0.65 * H)
    if bottom <= top:
        top, bottom = 0, H
    band = screen_bgr[top:bottom, :]
    return band if band.size else screen_bgr


def enforce_monotonic(results: List[TempResult], slack: float = 0.2) -> None:
    previous: Optional[float] = None
    for r in results:
        if r.value is None:
            continue
        if previous is None:
            previous = r.value
            continue
        if r.value > previous + slack:
            r.value = previous
        else:
            previous = r.value


def sample_video(
    video_path: str,
    extract_fps: float,
    start_sec: float,
    limit: Optional[int],
    max_duration: Optional[float],
    roi: Optional[Tuple[int, int, int, int]],
) -> List[TempResult]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or extract_fps or 30.0
    results: List[TempResult] = []
    step = 1.0 / max(extract_fps, 1e-6)
    timestamp = max(0.0, start_sec)
    processed = 0
    max_ts = timestamp + max_duration if max_duration is not None else None

    while True:
        if limit is not None and processed >= limit:
            break
        if max_ts is not None and timestamp > max_ts + 1e-6:
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        screen = detect_screen(frame, roi)
        digit_band = extract_digit_band(screen)
        value_str, conf, _ = run_ocr(digit_band, strict_unit=True)

        value: Optional[float]
        try:
            value = float(value_str) if value_str is not None else None
        except (TypeError, ValueError):
            value = None

        frame_idx = int(round(timestamp * native_fps))
        results.append(
            TempResult(
                frame_idx=frame_idx,
                timestamp_sec=timestamp,
                value=value,
                confidence=float(conf),
                raw_text=value_str,
            )
        )

        processed += 1
        timestamp += step

    cap.release()
    return results


def save_outputs(results: List[TempResult], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    jsonl_path = out_dir / "results.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame", "timestamp", "temp_c", "confidence", "raw"])
        for r in results:
            writer.writerow(
                [
                    r.frame_idx,
                    seconds_to_timestamp(r.timestamp_sec),
                    "" if r.value is None else f"{r.value:.1f}",
                    f"{r.confidence:.3f}",
                    r.raw_text or "",
                ]
            )

    with jsonl_path.open("w", encoding="utf-8") as f_json:
        for r in results:
            obj = {
                "frame": r.frame_idx,
                "timestamp": seconds_to_timestamp(r.timestamp_sec),
                "temp_c": r.value,
                "confidence": r.confidence,
                "raw": r.raw_text,
            }
            f_json.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def print_stats(results: List[TempResult]) -> None:
    vals = [r.value for r in results if r.value is not None]
    total = len(results)
    valid = len(vals)
    print(f"\nTotal frames: {total} | Valid readings: {valid} | Null: {total - valid}")
    if not vals:
        return
    print(f"min/mean/max: {min(vals):.1f} / {sum(vals) / len(vals):.1f} / {max(vals):.1f} °C")


def main() -> None:
    args = parse_args()

    roi = parse_roi(args.roi)
    set_temp_bounds(args.temp_min, args.temp_max)

    results = sample_video(
        video_path=args.video,
        extract_fps=args.extract_fps,
        start_sec=args.start_time,
        limit=args.limit,
        max_duration=args.max_duration,
        roi=roi,
    )

    if args.monotonic:
        enforce_monotonic(results, slack=max(0.0, float(args.monotonic_slack)))

    out_dir = Path(args.out_dir)
    csv_path, jsonl_path = save_outputs(results, out_dir)
    print_stats(results)
    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
