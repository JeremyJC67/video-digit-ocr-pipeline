"""Robust OCR utilities for extracting numeric temperatures from LCD / LED displays."""
from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
DEBUG_OCR = os.environ.get("DEBUG_OCR", "0") == "1"

DIGIT_WHITELIST = "0123456789.,°C:%+-"
VALUE_RE = re.compile(r"\d+(?:[.,]\d+)?")
UNIT_RE = re.compile(r"(?:°\s?C|\bC\b)", re.IGNORECASE)

try:
    import pytesseract
    from pytesseract import Output
    _tess_path = shutil.which("tesseract")
    if not _tess_path:
        for candidate in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"):
            if Path(candidate).exists():
                _tess_path = candidate
                break
    if _tess_path:
        pytesseract.pytesseract.tesseract_cmd = _tess_path
except Exception:  # pragma: no cover
    pytesseract, Output = None, None  # type: ignore

try:
    import easyocr  # type: ignore

    _easyocr_reader = easyocr.Reader(["en"], gpu=False)
except Exception:  # pragma: no cover
    _easyocr_reader = None


@dataclass
class OCRBox:
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float


def _aggregate_conf(vals: Sequence[float]) -> float:
    vals = [float(v) for v in vals if v is not None and v >= 0]
    return float(np.mean(vals)) if vals else 0.0


def _morph_close(bw: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)


def _preprocess_dual(gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    outs = [("otsu", _morph_close(bw, 3))]
    mean_val = float(bw.mean())
    if mean_val > 235 or mean_val < 20:
        bw_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        outs.append(("otsu_inv", _morph_close(bw_inv, 3)))
    return outs


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
        if not text or not any(ch in DIGIT_WHITELIST for ch in text):
            continue
        try:
            conf = float(confs[idx]) / 100.0
        except Exception:
            conf = 0.0
        boxes.append(
            OCRBox(
                bbox=(
                    int(lefts[idx]),
                    int(tops[idx]),
                    int(widths[idx]),
                    int(heights[idx]),
                ),
                text=text.strip(),
                confidence=conf,
            )
        )
    return boxes


def _tess_text(img: np.ndarray, config: str) -> Tuple[str, List[OCRBox], Optional[dict]]:
    if pytesseract is None or Output is None:
        return "", [], None
    try:
        data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Tesseract call failed: %s", exc)
        return "", [], None
    text = " ".join([t for t in data.get("text", []) if t.strip()])
    boxes = _tesseract_boxes(data)
    return text, boxes, data


# ==== 强化工具：白字掩膜 / 锚点查找 / 回退行定位 ====

def white_ink_mask(bgr: np.ndarray) -> np.ndarray:
    """Segment low-saturation, high-value ink for blue LCDs."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask_sat = cv2.inRange(s, 0, 80)
    mask_val = cv2.inRange(v, 170, 255)
    mask = cv2.bitwise_and(mask_sat, mask_val)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def find_degree_anchor_strict(roi_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Aggressively locate °/C anchor near the right edge."""
    H, W = roi_bgr.shape[:2]
    xR = int(W * 0.55)
    roi_right = roi_bgr[:, xR:]
    if roi_right.size == 0:
        roi_right = roi_bgr
        xR = 0

    mask = white_ink_mask(roi_right)
    gray = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.bitwise_and(gray, mask)
    gray_big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    best = None
    best_area = -1
    for cfg in (
        "--oem 1 --psm 10 -c tessedit_char_whitelist=°C",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=°C",
    ):
        _, boxes, _ = _tess_text(gray_big, cfg)
        for b in boxes:
            x, y, w, h = b.bbox
            area = w * h
            if area > best_area:
                best_area = area
                best = (x, y, w, h)
    if best is None:
        return None
    x, y, w, h = best
    x = x // 2 + xR
    y = y // 2
    w = max(1, w // 2)
    h = max(1, h // 2)
    return (x, y, w, h)


def extract_main_line_by_anchor(roi_bgr: np.ndarray, anchor: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    H, W = roi_bgr.shape[:2]
    ax, ay, aw, ah = anchor
    if ax <= 0:
        return None
    cy = ay + ah // 2
    band_h = max(ah * 22 // 10, ah * 2)
    y0 = max(0, cy - band_h // 2)
    y1 = min(H, cy + band_h // 2)
    band = roi_bgr[y0:y1, :]
    band_left = band[:, :max(0, ax)]
    if band_left.size == 0:
        return None

    mask = white_ink_mask(band_left)
    mask = _morph_close(mask, 3)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w < 0.15 * band_left.shape[1]:
            continue
        aspect = w / max(1.0, float(h))
        if aspect < 2.5:
            continue
        candidates.append((area, (x, y, w, h)))
    if candidates:
        _, (x, y, w, h) = max(candidates, key=lambda t: t[0])
        pad = max(2, h // 6)
        x0 = max(0, x - pad)
        y0_local = max(0, y - pad)
        x1 = min(band_left.shape[1], x + w + pad)
        y1_local = min(band_left.shape[0], y + h + pad)
        line = band_left[y0_local:y1_local, x0:x1]
        if line.size:
            return line

    width = min(ax, max(int(ah * 8), aw * 4, int(W * 0.4)))
    x0 = max(0, ax - width)
    fallback = roi_bgr[y0:y1, x0:ax]
    return fallback if fallback.size else None


def fallback_main_line_from_band(roi_bgr: np.ndarray) -> Optional[np.ndarray]:
    H, W = roi_bgr.shape[:2]
    if H <= 160:
        y0, y1 = 0, H
    else:
        y0, y1 = int(H * 0.26), int(H * 0.62)
    band = roi_bgr[y0:y1, :]
    if band.size == 0:
        return None
    mask = white_ink_mask(band)
    kernel_w = max(9, band.shape[1] // 18)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 5))
    fused = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    num, _, stats, _ = cv2.connectedComponentsWithStats(fused, connectivity=8)
    band_h = band.shape[0]
    candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if w < 0.25 * band.shape[1]:
            continue
        if not (0.18 * band_h <= h <= 0.55 * band_h):
            continue
        aspect = w / max(1.0, float(h))
        if aspect < 3.0:
            continue
        cy = y + h / 2.0
        if cy > 0.65 * band_h:
            continue
        candidates.append((w, (x, y, w, h)))
    if not candidates:
        return None
    _, (x, y, w, h) = max(candidates, key=lambda t: t[0])
    pad = max(2, h // 8)
    x0 = max(0, x - pad)
    y0b = max(0, y - pad)
    x1 = min(band.shape[1], x + w + pad)
    y1b = min(band.shape[0], y + h + pad)
    line = band[y0b:y1b, x0:x1]
    return line if line.size else None


def _has_decimal_dot(bw: np.ndarray) -> bool:
    H, W = bw.shape
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area <= 0.02 * (H * H) and h <= 0.35 * H and 0.25 * W < (x + w / 2) < 0.85 * W:
            return True
    return False


def _ocr_data_and_conf(bw: np.ndarray, psm: int, decimal_hint: bool = False) -> Tuple[Optional[str], float]:
    if pytesseract is None or Output is None:
        return None, 0.0

    cfg = (
        f"--oem 1 --psm {psm} "
        "-c classify_bln_numeric_mode=1 "
        "-c tessedit_char_whitelist=0123456789.,"
    )
    try:
        data = pytesseract.image_to_data(bw, config=cfg, output_type=Output.DICT)
    except Exception as exc:
        LOGGER.error("Tesseract data call failed: %s", exc)
        return None, 0.0

    texts = data.get("text", [])
    levels = data.get("level", [])
    confs = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    words = []
    for i, raw in enumerate(texts):
        level = int(levels[i]) if i < len(levels) else 0
        if level != 5:
            continue
        token = raw.strip()
        if not token:
            continue
        try:
            conf = float(confs[i])
        except Exception:
            conf = -1.0
        x = int(lefts[i]) if i < len(lefts) else 0
        y = int(tops[i]) if i < len(tops) else 0
        w = int(widths[i]) if i < len(widths) else 0
        h = int(heights[i]) if i < len(heights) else 0
        words.append((token, conf, x, y, w, h))

    text_source = (
        " ".join(token for token, *_ in words)
        if words
        else " ".join(t.strip() for t in texts if t and t.strip())
    )
    tokens = [s.replace(",", ".") for s in VALUE_RE.findall(text_source)]
    if not tokens:
        return None, 0.0

    tokens.sort(key=lambda s: ("." not in s, -len(s)))
    best = tokens[0]

    if "." not in best and decimal_hint and len(best) >= 3:
        best = f"{best[:-1]}.{best[-1]}"

    agg: List[float] = []
    if words:
        idx = 0
        for token, conf, *_ in words:
            norm = token.replace(",", ".")
            if best.startswith(norm, idx):
                if conf > 0:
                    agg.append(conf / 100.0)
                idx += len(norm)
            if idx >= len(best):
                break

    if agg:
        return best, float(np.mean(agg))

    pixel_cov = float(np.count_nonzero(bw)) / float(max(1, bw.size))
    proxy = min(0.95, max(0.05, 1.0 - abs(0.45 - pixel_cov) * 2.0))
    return best, proxy
def _save_debug_crop(img: np.ndarray, debug_dir: Optional[Path], tag: str) -> None:
    if img is None or img.size == 0 or debug_dir is None:
        return
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{tag}.jpg"), img)
    except Exception as exc:
        LOGGER.debug("Failed to save debug crop %s: %s", tag, exc)


def ocr_single_line(line_bgr: np.ndarray) -> Tuple[Optional[str], float]:
    zoom = cv2.resize(line_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=5)

    def _prep(bw: np.ndarray) -> np.ndarray:
        bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        return bw

    bw_pos_raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bw_inv_raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bw_pos = _prep(bw_pos_raw)
    bw_inv = _prep(bw_inv_raw)

    has_dot_pos = _has_decimal_dot(bw_pos_raw)
    has_dot_inv = _has_decimal_dot(bw_inv_raw)
    dot_hint_any = has_dot_pos or has_dot_inv

    prefer_bw = bw_pos
    prefer_hint = has_dot_pos
    if has_dot_inv and not has_dot_pos:
        prefer_bw = bw_inv
        prefer_hint = has_dot_inv

    other_bw = bw_inv if prefer_bw is bw_pos else bw_pos
    other_hint = has_dot_inv if prefer_bw is bw_pos else has_dot_pos

    order = [
        ("prefer", prefer_bw, prefer_hint),
        ("alternate", other_bw, other_hint),
        ("gray", gray, dot_hint_any),
    ]

    def _attempt(image: np.ndarray, hint: bool) -> Tuple[Optional[str], float]:
        val, conf = _ocr_data_and_conf(image, 7, decimal_hint=hint)
        if val is not None:
            return val, conf
        return _ocr_data_and_conf(image, 13, decimal_hint=hint)

    best_val: Optional[str] = None
    best_score = -1.0
    best_conf = 0.0

    for label, img, hint in order:
        if img is None or img.size == 0:
            continue
        value_str, conf = _attempt(img, hint)
        if value_str is None:
            continue
        try:
            numeric = float(value_str)
        except ValueError:
            continue
        if not 0 <= numeric <= 120:
            continue
        score = conf + (0.08 if "." in value_str else 0.0)
        if score > best_score:
            best_score = score
            best_val = value_str
            best_conf = conf
        if label != "gray" and "." in value_str and conf >= 0.4:
            break

    if best_val is None:
        return None, 0.0
    return f"{float(best_val):.1f}", float(best_conf)
def ocr_tesseract(img: np.ndarray) -> Tuple[str, float, List[OCRBox]]:
    if pytesseract is None or Output is None:
        LOGGER.warning("pytesseract unavailable; skipping")
        return "", 0.0, []
    config = "--oem 1 --psm 6 -c tessedit_char_whitelist=" + DIGIT_WHITELIST
    try:
        data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    except Exception as exc:
        LOGGER.error("Tesseract OCR failure: %s", exc)
        return "", 0.0, []
    text = " ".join(t for t in data.get("text", []) if t.strip())
    boxes = _tesseract_boxes(data)
    conf = _aggregate_conf([b.confidence for b in boxes])
    if DEBUG_OCR:
        print(f"[DBG][tess] text='{text}' conf={conf:.3f}")
    return text, conf, boxes


def ocr_easyocr(img: np.ndarray) -> Tuple[str, float, List[OCRBox]]:
    if _easyocr_reader is None:
        return "", 0.0, []
    try:
        results = _easyocr_reader.readtext(img)
    except Exception as exc:
        LOGGER.error("EasyOCR failure: %s", exc)
        return "", 0.0, []
    boxes: List[OCRBox] = []
    texts: List[str] = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, y0 = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))
        text = text.strip()
        if text:
            boxes.append(OCRBox((x0, y0, w, h), text, float(conf)))
            texts.append(text)
    combined = " ".join(texts)
    conf = _aggregate_conf([b.confidence for b in boxes])
    if DEBUG_OCR:
        print(f"[DBG][easy] text='{combined}' conf={conf:.3f}")
    return combined, conf, boxes


def _near_unit(box: OCRBox, all_boxes: List[OCRBox]) -> bool:
    bx, by, bw, bh = box.bbox
    cx, cy = bx + bw / 2, by + bh / 2
    for other in all_boxes:
        if ("C" not in other.text) and ("°" not in other.text):
            continue
        x, y, w, h = other.bbox
        cx2, cy2 = x + w / 2, y + h / 2
        dist = ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5
        if dist < max(bw, bh) * 2.5:
            return True
    return False


def _score_candidate(val: str, box: OCRBox, roi_h: int, near_unit: bool) -> float:
    has_decimal = 1.0 if ("." in val or "," in val) else 0.0
    _, y, _, h = box.bbox
    height_score = h / max(1.0, roi_h)
    pos_score = 1.0 - abs((y + h / 2) / max(roi_h, 1) - 0.4)
    unit_bonus = 0.4 if near_unit else 0.0
    return 1.5 * has_decimal + 2.0 * height_score + 0.8 * pos_score + unit_bonus


def _pick_main_value(text: str, boxes: List[OCRBox], roi_h: int) -> Optional[str]:
    candidates: List[Tuple[str, OCRBox]] = []
    norm_text = text.replace(",", ".")
    for box in boxes:
        for match in VALUE_RE.findall(box.text.replace(",", ".")):
            normalized = _normalize_candidate_value(match)
            if normalized:
                candidates.append((normalized, box))
    if not candidates and norm_text:
        for match in VALUE_RE.findall(norm_text):
            normalized = _normalize_candidate_value(match)
            if normalized:
                candidates.append((normalized, OCRBox((0, 0, 0, 0), normalized, 0.0)))
    if not candidates:
        return None
    best_val, best_score = None, -1e9
    for val, box in candidates:
        score = _score_candidate(val, box, roi_h, _near_unit(box, boxes))
        if score > best_score:
            best_val, best_score = val, score
    return best_val


def _normalize_candidate_value(token: str) -> Optional[str]:
    cleaned = token.strip().replace(",", ".")
    digits_only = "".join(ch for ch in cleaned if ch.isdigit())
    if not digits_only:
        return None
    try:
        numeric = float(cleaned)
    except ValueError:
        numeric = None
    if numeric is not None and 0 <= numeric <= 120:
        return cleaned
    if "." not in cleaned and len(digits_only) >= 3:
        adjusted = digits_only[:-1] + "." + digits_only[-1]
        try:
            numeric_adj = float(adjusted)
        except ValueError:
            return None
        if 0 <= numeric_adj <= 120:
            return adjusted
    return None


def _anchor_result_needs_retry(value: Optional[str], conf: float) -> bool:
    if not value:
        return True
    digits = sum(ch.isdigit() for ch in value)
    has_decimal = "." in value
    if has_decimal:
        if digits >= 3:
            return False
        return True
    return digits < 2 or conf < 0.6


EngineFunc = Callable[[np.ndarray], Tuple[str, float, List[OCRBox]]]


def run_ocr(
    roi_bgr: np.ndarray,
    *,
    use_easyocr_first: bool = False,
    strict_unit: bool = False,
    debug_id: Optional[str] = None,
    debug_dir: Optional[Path] = None,
) -> Tuple[Optional[str], float, List[OCRBox]]:
    anchor_best: Optional[Tuple[str, float]] = None
    anchor = find_degree_anchor_strict(roi_bgr)
    if anchor is not None:
        line = extract_main_line_by_anchor(roi_bgr, anchor)
        if line is not None:
            value, conf = ocr_single_line(line)
            if DEBUG_OCR:
                print(f"[DBG][anchor-line] val={value} conf={conf:.3f}")
            if value is not None:
                if debug_id and debug_dir:
                    _save_debug_crop(line, debug_dir, f"DBG_line_anchor_{debug_id}")
                if _anchor_result_needs_retry(value, conf):
                    anchor_best = (value, conf)
                else:
                    return value, conf, []

    line_fb = fallback_main_line_from_band(roi_bgr)
    if line_fb is not None:
        value, conf = ocr_single_line(line_fb)
        if DEBUG_OCR:
            print(f"[DBG][band-fallback] val={value} conf={conf:.3f}")
        if value is not None:
            if debug_id and debug_dir:
                _save_debug_crop(line_fb, debug_dir, f"DBG_line_band_{debug_id}")
            return value, conf, []

    roi_scaled = cv2.resize(roi_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2GRAY)
    variants = [("gray", gray)] + _preprocess_dual(gray)

    engines: List[EngineFunc] = [ocr_tesseract, ocr_easyocr]
    if use_easyocr_first:
        engines.reverse()

    height = roi_scaled.shape[0]
    best: Tuple[Optional[str], float, List[OCRBox]] = (None, 0.0, [])

    for vname, vimg in variants:
        for engine in engines:
            img_input = roi_scaled if engine is ocr_easyocr else vimg
            text, conf, boxes = engine(img_input)
            if strict_unit and not UNIT_RE.search(text.replace(" ", "")):
                parsed = None
            else:
                parsed = _pick_main_value(text, boxes, height)
            if DEBUG_OCR:
                print(f"[DBG][{vname}+{engine.__name__}] text='{text}' -> parsed={parsed} conf={conf:.3f}")
            if parsed:
                return parsed, conf, boxes
            if conf > best[1]:
                best = (parsed, conf, boxes)

    if best[0] is None and anchor_best is not None:
        return anchor_best[0], anchor_best[1], []
    return best
