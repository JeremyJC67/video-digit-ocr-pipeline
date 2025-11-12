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


def _tess_text(img: np.ndarray, config: str) -> Tuple[str, List[OCRBox]]:
    if pytesseract is None or Output is None:
        return "", []
    try:
        data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Tesseract call failed: %s", exc)
        return "", []
    text = " ".join([t for t in data.get("text", []) if t.strip()])
    boxes = _tesseract_boxes(data)
    return text, boxes


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
        _, boxes = _tess_text(gray_big, cfg)
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
    y0, y1 = int(H * 0.22), int(H * 0.68)
    band = roi_bgr[y0:y1, :]
    if band.size == 0:
        return None
    mask = white_ink_mask(band)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if w < 0.25 * band.shape[1]:
            continue
        if h < 0.10 * band.shape[0] or h > 0.65 * band.shape[0]:
            continue
        if (w / max(1.0, float(h))) < 2.5:
            continue
        candidates.append((area, (x, y, w, h)))
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


def _save_debug_crop(img: np.ndarray, debug_dir: Optional[Path], tag: str) -> None:
    if img is None or img.size == 0 or debug_dir is None:
        return
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{tag}.jpg"), img)
    except Exception as exc:
        LOGGER.debug("Failed to save debug crop %s: %s", tag, exc)


def _ocr_data_and_conf(bw: np.ndarray, psm: int) -> Tuple[Optional[str], float]:
    if pytesseract is None or Output is None:
        return None, 0.0
    cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789.,"
    data = pytesseract.image_to_data(bw, config=cfg, output_type=Output.DICT)
    full_txt = " ".join([t for t in data.get("text", []) if t.strip()])
    tokens = [t.replace(",", ".") for t in VALUE_RE.findall(full_txt)]
    if not tokens:
        return None, 0.0
    tokens.sort(key=lambda s: ("." not in s, -len(s)))
    best = tokens[0]

    confs: List[float] = []
    idx = 0
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        norm = txt.strip().replace(",", ".")
        if not norm:
            continue
        if best.startswith(norm, idx):
            idx += len(norm)
            try:
                cval = float(conf)
                if cval >= 0:
                    confs.append(cval / 100.0)
            except Exception:
                pass
        if idx >= len(best):
            break
    return best, float(np.mean(confs)) if confs else 0.0


def ocr_single_line(line_bgr: np.ndarray) -> Tuple[Optional[str], float]:
    zoom = cv2.resize(line_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=5)

    def _prep(bw: np.ndarray) -> np.ndarray:
        bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        return bw

    bw_pos = _prep(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    bw_inv = _prep(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])

    best_val: Optional[str] = None
    best_score = -1.0
    best_conf = 0.0

    for bw in (bw_pos, bw_inv):
        for psm in (7, 13):
            value_str, conf = _ocr_data_and_conf(bw, psm)
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
            candidates.append((match.replace(",", "."), box))
    if not candidates and norm_text:
        for match in VALUE_RE.findall(norm_text):
            candidates.append((match, OCRBox((0, 0, 0, 0), match, 0.0)))
    if not candidates:
        return None
    filtered: List[Tuple[str, OCRBox]] = []
    for val, box in candidates:
        try:
            fval = float(val)
        except ValueError:
            continue
        if 0 <= fval <= 120:
            filtered.append((val, box))
    candidates = filtered or candidates
    best_val, best_score = None, -1e9
    for val, box in candidates:
        score = _score_candidate(val, box, roi_h, _near_unit(box, boxes))
        if score > best_score:
            best_val, best_score = val, score
    return best_val


EngineFunc = Callable[[np.ndarray], Tuple[str, float, List[OCRBox]]]


def run_ocr(
    roi_bgr: np.ndarray,
    *,
    use_easyocr_first: bool = False,
    strict_unit: bool = False,
    debug_id: Optional[str] = None,
    debug_dir: Optional[Path] = None,
) -> Tuple[Optional[str], float, List[OCRBox]]:
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
    variants = _preprocess_dual(gray)

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

    return best
