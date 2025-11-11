"""OCR utilities for extracting numeric temperatures from LCD / LED displays."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover - dependency missing at runtime
    pytesseract = None
    Output = None  # type: ignore

try:  # pragma: no cover - optional fallback dependency
    import easyocr  # type: ignore

    _easyocr_reader = easyocr.Reader(["en"], gpu=False)
except Exception:  # pragma: no cover - dependency missing at runtime
    _easyocr_reader = None

LOGGER = logging.getLogger(__name__)
DIGIT_WHITELIST = "0123456789.,°C"
VALUE_RE = re.compile(r"\d{1,3}(?:[.,]\d)?")
UNIT_RE = re.compile(r"(?:°\s?C|\bC\b)", re.IGNORECASE)


@dataclass
class OCRBox:
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float


def preprocess(img: np.ndarray) -> np.ndarray:
    """Apply denoising and binarization tuned for emissive screens."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Bilateral filter preserves edges while softening glare.
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)
    # A mild CLAHE improves contrast for dim displays.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, bw = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _aggregate_conf(conf_values: Sequence[float]) -> float:
    vals = [c for c in conf_values if c >= 0]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _tesseract_boxes(data: dict) -> List[OCRBox]:
    boxes: List[OCRBox] = []
    if not data:
        return boxes
    texts = data.get("text", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])
    confs = data.get("conf", [])
    for idx, text in enumerate(texts):
        if not text:
            continue
        if not any(ch in DIGIT_WHITELIST for ch in text):
            continue
        bbox = (
            int(lefts[idx]),
            int(tops[idx]),
            int(widths[idx]),
            int(heights[idx]),
        )
        try:
            conf = float(confs[idx]) / 100.0
        except Exception:
            conf = 0.0
        boxes.append(OCRBox(bbox=bbox, text=text.strip(), confidence=conf))
    return boxes


def ocr_tesseract(img: np.ndarray) -> Tuple[str, float, List[OCRBox]]:
    if pytesseract is None or Output is None:
        LOGGER.warning("pytesseract is unavailable; skipping primary OCR")
        return "", 0.0, []
    config = "--oem 1 --psm 7 -c tessedit_char_whitelist=" + DIGIT_WHITELIST
    try:
        data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    except Exception as exc:  # pragma: no cover - passthrough logging
        LOGGER.error("Tesseract OCR failure: %s", exc)
        return "", 0.0, []
    text = " ".join([t for t in data.get("text", []) if t.strip()])
    boxes = _tesseract_boxes(data)
    conf = _aggregate_conf([b.confidence for b in boxes])
    return text, conf, boxes


def ocr_easyocr(img: np.ndarray) -> Tuple[str, float, List[OCRBox]]:
    if _easyocr_reader is None:
        LOGGER.info("EasyOCR not installed; skipping fallback")
        return "", 0.0, []
    try:
        results = _easyocr_reader.readtext(img)
    except Exception as exc:  # pragma: no cover - passthrough logging
        LOGGER.error("EasyOCR failure: %s", exc)
        return "", 0.0, []
    boxes: List[OCRBox] = []
    texts: List[str] = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)
        boxes.append(
            OCRBox(
                bbox=(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                text=text.strip(),
                confidence=float(conf),
            )
        )
        if text.strip():
            texts.append(text.strip())
    text_combined = " ".join(texts)
    conf = _aggregate_conf([b.confidence for b in boxes])
    return text_combined, conf, boxes


def parse_temperature(text: str, *, require_unit: bool = False) -> Optional[str]:
    if not text:
        return None
    normalized = text.replace(",", ".")
    if require_unit and not UNIT_RE.search(normalized):
        return None
    match = VALUE_RE.search(normalized)
    if not match:
        return None
    value = match.group(0)
    return value


EngineFunc = Callable[[np.ndarray], Tuple[str, float, List[OCRBox]]]


def run_ocr(
    roi: np.ndarray,
    *,
    use_easyocr_first: bool = False,
    strict_unit: bool = False,
) -> Tuple[Optional[str], float, List[OCRBox]]:
    """Run OCR with retries.

    Returns (parsed_value, agg_confidence, boxes)
    """

    bw = preprocess(roi)
    engines: List[EngineFunc] = [ocr_tesseract, ocr_easyocr]
    if use_easyocr_first:
        engines.reverse()

    best_result: Tuple[Optional[str], float, List[OCRBox]] = (None, 0.0, [])

    for engine in engines:
        target_img = roi if engine is ocr_easyocr else bw
        text, conf, boxes = engine(target_img)
        parsed = parse_temperature(text, require_unit=strict_unit)
        if parsed:
            return parsed, conf, boxes
        if conf > best_result[1]:
            best_result = (parsed, conf, boxes)

    return best_result
