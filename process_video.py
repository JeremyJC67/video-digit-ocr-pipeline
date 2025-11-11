"""CLI pipeline for sampling video frames and extracting temperature readings."""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2

from ocr_utils import OCRBox, run_ocr

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
    parser.add_argument(
        "--max_duration",
        type=float,
        help="Only process the first N seconds of the video",
    )
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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

    writer = None
    if args.annotated_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(args.annotated_mp4, fourcc, args.fps, (width, height))

    rows: List[dict] = []
    step_ms = 1000.0 / max(args.fps, 1e-3)
    timestamp_ms = start_ms

    while True:
        if limit_ms is not None and timestamp_ms > limit_ms + 1e-6:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ok, frame = cap.read()
        if not ok:
            break
        x, y, w, h = roi.as_tuple()
        roi_img = frame[y : y + h, x : x + w]
        value, conf, boxes = run_ocr(
            roi_img,
            use_easyocr_first=args.engine_easy_first,
            strict_unit=args.strict_unit,
        )
        label_value = value if value is not None else ""
        timestamp_sec = timestamp_ms / 1000.0
        ts_str = format_hms(timestamp_sec)
        frame_filename = f"{ts_str.replace(':', '-')}.jpg"
        frame_path = frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        rows.append(
            {
                "video_time": ts_str,
                "detected_number": label_value,
                "confidence": round(conf, 3),
                "frame_file": frame_filename,
            }
        )

        if writer is not None:
            annotated = frame.copy()
            annotate_frame(annotated, roi, boxes, f"{ts_str} -> {label_value or '?'}")
            writer.write(annotated)

        timestamp_ms += step_ms

    cap.release()
    if writer is not None:
        writer.release()

    with open(args.out_csv, "w", newline="") as f:
        fieldnames = ["video_time", "detected_number", "confidence", "frame_file"]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for row in rows:
            writer_csv.writerow(row)

    LOGGER.info("Saved CSV to %s (frames in %s)", args.out_csv, frames_dir)
    if args.annotated_mp4:
        LOGGER.info("Annotated video written to %s", args.annotated_mp4)


if __name__ == "__main__":
    main()
