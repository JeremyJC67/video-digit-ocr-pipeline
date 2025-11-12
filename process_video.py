"""CLI pipeline for sampling video frames and extracting temperature readings."""
from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ocr_utils import (
    OCRBox,
    find_degree_anchor_strict,
    run_ocr,
    set_temp_bounds,
    get_temp_bounds,
)

LOGGER = logging.getLogger("process_video")


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a video at fixed FPS and extract LCD digits")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out_csv", default="results.csv", help="Output CSV path")
    parser.add_argument("--annotated_mp4", help="Optional annotated MP4 output")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling rate (frames per second)")
    parser.add_argument("--max_duration", type=float, help="Only process the first N seconds of the video")
    parser.add_argument(
        "--max_percent",
        type=float,
        help="Only process the first X percent of the video duration (0-100)",
    )
    parser.add_argument("--start_time", type=float, default=0.0, help="Start sampling after N seconds")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Pre-defined ROI to skip GUI picker")
    parser.add_argument("--frames_dir", default="frames", help="Directory to store sampled frames for review UI")
    parser.add_argument("--engine_easy_first", action="store_true", help="Try EasyOCR before Tesseract")
    parser.add_argument("--strict_unit", action="store_true", help="Require detected text to include °C/C units")
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="Median smoothing window (set <=1 to disable)",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=0,
        help="Log progress every N processed frames (0 disables)",
    )
    parser.add_argument(
        "--temp_min",
        type=float,
        default=-200.0,
        help="Minimum valid temperature (inclusive)",
    )
    parser.add_argument(
        "--temp_max",
        type=float,
        default=200.0,
        help="Maximum valid temperature (inclusive)",
    )
    return parser.parse_args()


def select_roi_interactive(frame) -> ROI:
    r = cv2.selectROI("Select ROI (screen)", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI (screen)")
    x, y, w, h = map(int, r)
    if w == 0 or h == 0:
        raise RuntimeError("ROI selection cancelled or invalid")
    return ROI(x, y, w, h)


def format_hms(total_seconds: float) -> str:
    seconds_int = max(0, int(round(total_seconds)))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    seconds = seconds_int % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def annotate_frame(frame, roi: ROI, boxes: Sequence[OCRBox], label: str) -> None:
    for box in boxes:
        x, y, w, h = box.bbox
        cv2.rectangle(frame, (roi.x + x, roi.y + y), (roi.x + x + w, roi.y + y + h), (0, 255, 0), 2)
    if label:
        cv2.putText(frame, label, (roi.x, max(30, roi.y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def compute_video_duration_ms(cap) -> Optional[float]:
    """Best-effort video duration estimation (milliseconds)."""
    duration_prop = getattr(cv2, "CAP_PROP_DURATION", None)
    if duration_prop is not None:
        duration = cap.get(duration_prop)
        if duration and duration > 0:
            return float(duration)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_native = cap.get(cv2.CAP_PROP_FPS)
    if frame_count and fps_native and frame_count > 0 and fps_native > 0:
        return float(frame_count / fps_native * 1000.0)
    return None


def median_smooth_value(value: Optional[str], history: Optional[Deque[float]], window: int) -> Optional[str]:
    if history is None or window <= 1:
        return value
    numeric = parse_float_value(value)
    if numeric is None:
        return value
    history.append(numeric)
    med_val = float(median(history))
    temp_min, temp_max = get_temp_bounds()
    med_val = min(temp_max, max(temp_min, med_val))
    return f"{med_val:.1f}"


def parse_float_value(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except ValueError:
        return None
    temp_min, temp_max = get_temp_bounds()
    if temp_min <= numeric <= temp_max:
        return numeric
    mirrored = -numeric
    if temp_min <= mirrored <= temp_max:
        return mirrored
    return None


def normalize_value_str(value: Optional[str]) -> Optional[str]:
    numeric = parse_float_value(value)
    if numeric is None:
        return None
    return f"{numeric:.1f}"


def temporal_patch_rows(rows: List[dict]) -> None:
    if len(rows) < 3:
        return
    temp_min, temp_max = get_temp_bounds()

    for i in range(1, len(rows) - 1):
        prev_row, cur_row, next_row = rows[i - 1], rows[i], rows[i + 1]
        prev_val = parse_float_value(prev_row["detected_number"])
        cur_val = parse_float_value(cur_row["detected_number"])
        next_val = parse_float_value(next_row["detected_number"])

        prev_conf = float(prev_row["confidence"])
        cur_conf = float(cur_row["confidence"])
        next_conf = float(next_row["confidence"])

        def _fill(value: float, base_conf: float) -> None:
            cur_row["detected_number"] = f"{value:.1f}"
            cur_row["confidence"] = round(base_conf, 3)

        if cur_val is None and prev_val is not None and next_val is not None:
            if abs(prev_val - next_val) <= 2.0:
                fill_val = float(np.median([prev_val, next_val]))
                base_conf = max(0.0, min(prev_conf, next_conf) * 0.9)
                _fill(fill_val, base_conf)
                continue

        if cur_val is None:
            continue

        if not (temp_min <= cur_val <= temp_max):
            cur_row["detected_number"] = ""
            cur_row["confidence"] = 0.0
            continue

        if cur_val >= 0:
            if prev_val is not None and next_val is not None and prev_val < 0 and next_val < 0:
                med_val = float(np.median([prev_val, next_val]))
                base_conf = max(cur_conf, min(prev_conf, next_conf) * 0.8)
                _fill(med_val, base_conf)
                cur_val = med_val
                continue
            if prev_val is not None and prev_val < 0:
                _fill(prev_val, max(cur_conf, prev_conf * 0.8))
                cur_val = prev_val
                continue
            if next_val is not None and next_val < 0:
                _fill(next_val, max(cur_conf, next_conf * 0.8))
                cur_val = next_val
                continue

        if prev_val is not None and next_val is not None:
            if abs(prev_val - cur_val) > 5 and abs(next_val - cur_val) > 5 and abs(prev_val - next_val) <= 2:
                med_val = float(np.median([prev_val, next_val]))
                base_conf = max(cur_conf, min(prev_conf, next_conf) * 0.85)
                _fill(med_val, base_conf)



def focus_blue_region(img: np.ndarray) -> Tuple[np.ndarray, ROI]:
    if img.size == 0:
        return img, ROI(0, 0, 0, 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 40, 40], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = img.shape[:2]
        return img, ROI(0, 0, w, h)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img.shape[1], x + w)
    y1 = min(img.shape[0], y + h)
    cropped = img[y0:y1, x0:x1]
    return cropped if cropped.size else img, ROI(x0, y0, x1 - x0, y1 - y0)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    set_temp_bounds(args.temp_min, args.temp_max)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {args.video}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError("Failed to read first frame from video")

    if args.roi:
        roi = ROI(*args.roi)
    else:
        roi = select_roi_interactive(first_frame)

    duration_ms = compute_video_duration_ms(cap)
    limit_ms: Optional[float] = None
    if args.max_duration is not None:
        limit_ms = max(0.0, float(args.max_duration) * 1000.0)
    if args.max_percent is not None:
        percent = max(0.0, min(float(args.max_percent), 100.0))
        if duration_ms is not None:
            limit_from_percent = duration_ms * (percent / 100.0)
            limit_ms = min(limit_ms, limit_from_percent) if limit_ms is not None else limit_from_percent
        else:
            LOGGER.warning("Video duration unavailable; ignoring --max_percent")
    if limit_ms is not None and duration_ms is not None:
        limit_ms = min(limit_ms, duration_ms)
    if limit_ms is not None:
        LOGGER.info("Processing limited to first %.2f seconds", limit_ms / 1000.0)

    start_ms = max(0.0, float(args.start_time) * 1000.0)
    if limit_ms is not None and start_ms >= limit_ms:
        LOGGER.warning("start_time exceeds configured limit; nothing to process")
        return
    if duration_ms is not None and start_ms >= duration_ms:
        LOGGER.warning("start_time is beyond video duration; nothing to process")
        return
    if start_ms > 0:
        LOGGER.info("Starting at %.2f seconds", start_ms / 1000.0)

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if args.annotated_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width = first_frame.shape[:2]
        video_writer = cv2.VideoWriter(args.annotated_mp4, fourcc, args.fps, (width, height))

    rows: List[dict] = []
    smooth_history: Optional[Deque[float]] = None
    if args.smooth_window and args.smooth_window > 1:
        smooth_history = deque(maxlen=args.smooth_window)
    debug_save_enabled = os.environ.get("DEBUG_SAVE_CROPS", "0") == "1"
    debug_save_limit = int(os.environ.get("DEBUG_SAVE_LIMIT", "5"))
    debug_saves = 0
    last_value_float: Optional[float] = None
    last_conf = 0.0
    progress_every = max(0, int(args.progress_every))
    processed = 0

    step_ms = 1000.0 / max(args.fps, 1e-3)
    timestamp_ms = start_ms

    while True:
        if limit_ms is not None and timestamp_ms > limit_ms + 1e-6:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ok, frame = cap.read()
        if not ok:
            break

        base_x, base_y, base_w, base_h = roi.as_tuple()
        roi_img = frame[base_y : base_y + base_h, base_x : base_x + base_w]
        roi_focused, focus_local = focus_blue_region(roi_img)
        focus_roi_global = ROI(base_x + focus_local.x, base_y + focus_local.y, focus_local.w, focus_local.h)
        if roi_focused.size == 0:
            roi_focused = roi_img
            focus_roi_global = roi

        h_focus = roi_focused.shape[0]
        anchor_local = find_degree_anchor_strict(roi_focused)
        if anchor_local:
            _, ay, _, ah = anchor_local
            band_half = max(int(1.2 * max(ah, 1)), max(ah, 4))
            cy = ay + ah // 2
            band_top = max(0, cy - band_half)
            band_bottom = min(h_focus, cy + band_half)
        else:
            band_top = int(0.25 * h_focus)
            band_bottom = int(0.65 * h_focus)
        if band_bottom <= band_top or band_bottom > h_focus:
            band_top, band_bottom = 0, h_focus
        roi_band = roi_focused[band_top:band_bottom, :]
        if roi_band.size == 0:
            roi_band = roi_focused
            band_top = 0
        digits_roi = ROI(
            focus_roi_global.x,
            focus_roi_global.y + band_top,
            roi_band.shape[1],
            roi_band.shape[0],
        )

        timestamp_sec = timestamp_ms / 1000.0
        ts_str = format_hms(timestamp_sec)
        capture_debug = debug_save_enabled and debug_saves < debug_save_limit
        if capture_debug:
            dbg_band = frames_dir / f"DBG_band_{ts_str.replace(':', '-')}.jpg"
            cv2.imwrite(str(dbg_band), roi_band)
            debug_saves += 1

        value, conf, boxes = run_ocr(
            roi_band,
            use_easyocr_first=args.engine_easy_first,
            strict_unit=args.strict_unit,
            debug_id=ts_str if capture_debug else None,
            debug_dir=frames_dir if capture_debug else None,
        )
        value = normalize_value_str(value)
        smoothed_value = median_smooth_value(value, smooth_history, args.smooth_window)
        label_value = smoothed_value if smoothed_value not in (None, "") else (value or "")

        numeric_label: Optional[float] = None
        if label_value:
            try:
                numeric_label = float(label_value)
            except ValueError:
                numeric_label = None
        if not label_value and last_value_float is not None:
            label_value = f"{last_value_float:.1f}"
            conf = round(last_conf * 0.8, 3)
            numeric_label = last_value_float
        if numeric_label is not None:
            last_value_float = numeric_label
            last_conf = conf

        frame_filename = f"{ts_str.replace(':', '-')}" + ".jpg"
        frame_path = frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        line_ratio = digits_roi.w / max(1.0, digits_roi.h) if digits_roi.h else 0.0
        rows.append(
            {
                "video_time": ts_str,
                "detected_number": label_value,
                "confidence": round(conf, 3),
                "frame_file": frame_filename,
                "line_ratio": line_ratio,
            }
        )
        processed += 1
        if progress_every and processed % progress_every == 0:
            LOGGER.info("Processed %d frames (last %s)", processed, ts_str)

        if video_writer is not None:
            annotated = frame.copy()
            annotate_frame(annotated, digits_roi, boxes, f"{ts_str} -> {label_value or '?'}")
            video_writer.write(annotated)

        timestamp_ms += step_ms

    cap.release()
    if video_writer is not None:
        video_writer.release()

    temporal_patch_rows(rows)

    with open(args.out_csv, "w", newline="") as f:
        fieldnames = ["video_time", "detected_number", "confidence", "frame_file"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for row in rows:
            writer_csv.writerow({k: row.get(k, "") for k in fieldnames})

    LOGGER.info("Saved CSV to %s (frames in %s)", args.out_csv, frames_dir)
    if args.annotated_mp4:
        LOGGER.info("Annotated video written to %s", args.annotated_mp4)


if __name__ == "__main__":
    main()
