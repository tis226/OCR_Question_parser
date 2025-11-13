#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QA parser using EasyOCR for text extraction.

This script mirrors the flow-based chunking approach of
``pdfplumber_QA_parsing_Final.py`` but replaces PDF text extraction with
EasyOCR. Subject detection is removed – every detected chunk inside the
column bounding boxes is parsed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # EasyOCR is only needed at runtime; allow import for tooling/tests without it.
    import easyocr  # type: ignore
except Exception:  # pragma: no cover - handled lazily when constructing the reader.
    easyocr = None  # type: ignore[assignment]
import numpy as np
import pdfplumber
from PIL import Image, ImageDraw


logger = logging.getLogger(__name__)

# Default column window fractions (relative to page width/height)
DEFAULT_TOP_FRAC = 0.10
DEFAULT_BOTTOM_FRAC = 0.90
DEFAULT_GUTTER_FRAC = 0.005

DEFAULT_SEARCHABLE_FONT_CANDIDATES: Tuple[str, ...] = (
    "HYGoThic-Medium",
    "HYGothic-Medium",
    "HYSMyeongJo-Medium",
    "HeiseiKakuGo-W5",
    "HeiseiMin-W3",
    "STSong-Light",
)

# =========================
# Helpers
# =========================

def list_pdfs(folder: str) -> List[str]:
    try:
        items = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder, f))
            ]
        )
    except Exception:
        items = []
    return items


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _normalize_visible_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[\u00A0\u2000-\u200B]", " ", s)
    s = (
        s.replace("ㆍ", "·")
        .replace("∙", "·")
        .replace("・", "·")
        .replace("•", "·")
    )
    return s


# =========================
# Logging helpers
# =========================


def _parse_log_level(value: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    try:
        level = getattr(logging, value.upper())
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"Unknown log level: {value}") from exc
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"Unknown log level: {value}")
    return level


class HeartbeatLogger:
    """Periodically emit log messages while work is ongoing."""

    def __init__(self, interval: float = 30.0, message: str = "Working..."):
        self.interval = max(1.0, float(interval))
        self.message = message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is not None:
            return
        logger.debug("Starting heartbeat logger every %.1f seconds", self.interval)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.wait(self.interval):
            logger.info(self.message)

    def stop(self):
        if self._thread is None:
            return
        logger.debug("Stopping heartbeat logger")
        self._stop_event.set()
        self._thread.join()
        self._thread = None


# =========================
# Option / question regex helpers
# =========================
OPTION_RANGES = [
    (0x2460, 0x2473),  # ①-⑳
    (0x2474, 0x2487),  # ⑴-⒇
    (0x2488, 0x249B),  # ⒈-⒛
    (0x24F5, 0x24FE),  # ⓵-⓾
]
OPTION_EXTRA = {0x24EA, 0x24FF, 0x24DB}  # ⓪, ⓿, ⓛ
OPTION_SET = {
    chr(cp)
    for start, end in OPTION_RANGES
    for cp in range(start, end + 1)
}
OPTION_SET.update(chr(cp) for cp in OPTION_EXTRA)
OPTION_CLASS = "".join(sorted(OPTION_SET))
QUESTION_CIRCLED_RANGE = f"{OPTION_CLASS}{chr(0x3250)}-{chr(0x32FF)}"

ASCII_OPTION_RE = re.compile(r"(?<!\d)(?:\(|\[)?(1[0-9]|20|[1-9])\s*[).:]")
ASCII_OPTION_PREFIX_RE = re.compile(
    r"(?m)(\n)\s*(?:\(|\[)?(1[0-9]|20|[1-9])\s+(?=\S)"
)
DIGIT_TO_CIRCLED = {
    str(i): chr(0x2460 + i - 1) if 1 <= i <= 20 else str(i)
    for i in range(1, 21)
}

OPT_SPLIT_RE = re.compile(rf"(?=([{OPTION_CLASS}]))")
CIRCLED_STRIP_RE = re.compile(rf"^[{OPTION_CLASS}]\s*")
QUESTION_START_LINE_RE = re.compile(
    rf"^\s*(?:[{QUESTION_CIRCLED_RANGE}]|[0-9]{{1,3}}[.)]|제\s*[0-9]{{1,3}}\s*문)",
    re.MULTILINE,
)
QUESTION_NUM_RE = re.compile(
    r"^\s*(?:\(\s*(\d{1,3})\s*\)|(\d{1,3})\s*번|(\d{1,3}))\s*[.)]?\s*"
)

DISPUTE_RE = re.compile(
    r"\(?\s*다툼이\s*(?:있는\s*경우|있으면)\s*(?P<site>[^)\n]*?)\s*(?:판례|결정)\s*에\s*의함\)?",
    re.IGNORECASE,
)

LEADING_HEADER_STRIP = re.compile(
    r"^\s*(?:[【\[]\s*[^】\]]+\s*[】\]])\s*(?:\([^)]*\))?\s*"
)


def _strip_header_garbage(text: str) -> str:
    return norm_space(LEADING_HEADER_STRIP.sub("", text or ""))


def _normalize_option_markers(text: str) -> str:
    def repl(match: re.Match) -> str:
        if match.start() == 0:
            return match.group(0)
        prior = text[: match.start()]
        if prior.strip() == "":
            return match.group(0)
        num = match.group(1)
        circled = DIGIT_TO_CIRCLED.get(num)
        if not circled or circled == num:
            return match.group(0)
        trailing = " " if not match.group(0).endswith(" ") else ""
        return f"{circled}{trailing}"

    text = ASCII_OPTION_RE.sub(repl, text)

    def prefix_repl(match: re.Match) -> str:
        lead, num = match.group(1), match.group(2)
        prior = text[: match.start(2)]
        if prior.strip() == "":
            return match.group(0)
        circled = DIGIT_TO_CIRCLED.get(num)
        if not circled or circled == num:
            return match.group(0)
        return f"{lead}{circled} "

    return ASCII_OPTION_PREFIX_RE.sub(prefix_repl, text)


def infer_year_from_filename(path: str) -> Optional[int]:
    fname = os.path.basename(path)
    m = re.search(r"(\d{2})년", fname)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r"(20\d{2}|19\d{2})", fname)
    if m:
        return int(m.group(1))
    return None


# =========================
# EasyOCR extraction
# =========================
@dataclass
class OCRSettings:
    dpi: int = 800
    languages: Sequence[str] = ("ko", "en")
    gpu: bool = False
    column_pad_x: float = 2.0
    column_pad_top: float = 2.0
    column_pad_bottom: float = 60.0
    column_filter_top: float = 4.0
    column_filter_bottom: float = 60.0


class EasyOCRTextExtractor:
    """Render PDF pages to images and run EasyOCR within bboxes."""

    def __init__(self, pdf: pdfplumber.pdf.PDF, settings: OCRSettings):
        if easyocr is None:
            raise ImportError(
                "easyocr is not installed. Install it with 'pip install easyocr' to use this script."
            )
        self.pdf = pdf
        self.settings = settings
        self.reader = easyocr.Reader(list(settings.languages), gpu=settings.gpu)
        self._image_cache: Dict[int, Image.Image] = {}
        self._scale = settings.dpi / 72.0
        self.page_word_boxes: Dict[int, List[Dict[str, object]]] = {}
        self.crop_reports: Dict[int, List[Dict[str, Any]]] = {}
        logger.debug(
            "Initialized EasyOCRTextExtractor: dpi=%s languages=%s gpu=%s pads(x=%.1f top=%.1f bottom=%.1f) filters(top=%.1f bottom=%.1f)",
            settings.dpi,
            ",".join(settings.languages),
            settings.gpu,
            settings.column_pad_x,
            settings.column_pad_top,
            settings.column_pad_bottom,
            settings.column_filter_top,
            settings.column_filter_bottom,
        )

    def _page_image(self, page_index: int) -> Image.Image:
        if page_index not in self._image_cache:
            logger.debug("Rendering page %d at %d DPI", page_index + 1, self.settings.dpi)
            pil = (
                self.pdf.pages[page_index]
                .to_image(resolution=int(self.settings.dpi))
                .original.convert("RGB")
            )
            self._image_cache[page_index] = pil
        return self._image_cache[page_index]

    def _record_crop_sample(
        self,
        page_index: int,
        column_tag: Optional[str],
        orig_bbox: Tuple[float, float, float, float],
        padded_bbox: Tuple[float, float, float, float],
        filter_top: float,
        filter_bottom: float,
        pad_x: float,
        pad_top: float,
        pad_bottom: float,
        y_cut: Optional[float],
        drop_zone: Optional[Tuple[float, float, float, float]],
        raw_word_count: int,
        kept_word_count: int,
        line_count: int,
    ) -> None:
        drop_tuple: Optional[Tuple[float, float, float, float]] = None
        if drop_zone is not None:
            drop_tuple = tuple(float(v) for v in drop_zone)
        entry: Dict[str, Any] = {
            "column": column_tag or "?",
            "orig_bbox": tuple(float(v) for v in orig_bbox),
            "padded_bbox": tuple(float(v) for v in padded_bbox),
            "filter_band": (float(filter_top), float(filter_bottom)),
            "pad": {
                "x": float(pad_x),
                "top": float(pad_top),
                "bottom": float(pad_bottom),
            },
            "y_cut": float(y_cut) if y_cut is not None else None,
            "drop_zone": drop_tuple,
            "raw_word_count": int(raw_word_count),
            "kept_word_count": int(kept_word_count),
            "line_count": int(line_count),
        }
        bucket = self.crop_reports.setdefault(page_index, [])
        bucket.append(entry)

    def extract_lines(
        self,
        page_index: int,
        bbox: Tuple[float, float, float, float],
        y_tol: float = 3.0,
        y_cut: Optional[float] = None,
        drop_zone: Optional[Tuple[float, float, float, float]] = None,
        column_tag: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        page = self.pdf.pages[page_index]
        page_width = float(page.width)
        page_height = float(page.height)
        x0, y0, x1, y1 = bbox
        if y_cut is not None:
            y0 = max(y0, y_cut)
        if x1 <= x0 or y1 <= y0:
            return []

        pad_x = max(0.0, float(getattr(self.settings, "column_pad_x", 0.0)))
        pad_top = max(0.0, float(getattr(self.settings, "column_pad_top", 0.0)))
        pad_bottom = max(0.0, float(getattr(self.settings, "column_pad_bottom", 0.0)))
        orig_bbox = (x0, y0, x1, y1)
        padded_bbox = (x0, y0, x1, y1)
        if pad_x or pad_top or pad_bottom:
            padded_bbox = (
                max(0.0, x0 - pad_x),
                max(0.0, y0 - pad_top),
                min(page_width, x1 + pad_x),
                min(page_height, y1 + pad_bottom),
            )
            x0, y0, x1, y1 = padded_bbox
        filter_top = max(
            0.0,
            orig_bbox[1]
            - max(float(getattr(self.settings, "column_filter_top", 0.0)), pad_top),
        )
        filter_bottom = min(
            page_height,
            orig_bbox[3]
            + max(float(getattr(self.settings, "column_filter_bottom", 0.0)), pad_bottom),
        )

        scale = self._scale
        im = self._page_image(page_index)
        logger.debug(
            "Extracting lines from page %d bbox=(%.1f, %.1f, %.1f, %.1f) scale=%.3f (padded to %.1f, %.1f, %.1f, %.1f)",
            page_index + 1,
            orig_bbox[0],
            orig_bbox[1],
            orig_bbox[2],
            orig_bbox[3],
            scale,
            x0,
            y0,
            x1,
            y1,
        )
        crop_box = (
            int(round(x0 * scale)),
            int(round(y0 * scale)),
            int(round(x1 * scale)),
            int(round(y1 * scale)),
        )
        if crop_box[2] - crop_box[0] <= 0 or crop_box[3] - crop_box[1] <= 0:
            self._record_crop_sample(
                page_index,
                column_tag,
                orig_bbox,
                padded_bbox,
                filter_top,
                filter_bottom,
                pad_x,
                pad_top,
                pad_bottom,
                y_cut,
                drop_zone,
                raw_word_count=0,
                kept_word_count=0,
                line_count=0,
            )
            return []

        crop = im.crop(crop_box)
        np_img = np.array(crop)
        results = self.reader.readtext(np_img)

        entries = []
        word_entries: List[Dict[str, object]] = []
        for pts, text, conf in results:
            if not text:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            px0, px1 = min(xs), max(xs)
            py0, py1 = min(ys), max(ys)
            abs_x0 = x0 + px0 / scale
            abs_x1 = x0 + px1 / scale
            abs_y0 = y0 + py0 / scale
            abs_y1 = y0 + py1 / scale
            mid_y = 0.5 * (abs_y0 + abs_y1)
            if mid_y < filter_top or mid_y > filter_bottom:
                logger.debug(
                    "Skipping OCR word outside filter band: page=%d mid_y=%.1f filter=(%.1f, %.1f)",
                    page_index + 1,
                    mid_y,
                    filter_top,
                    filter_bottom,
                )
                continue
            if drop_zone and _rects_intersect(
                (abs_x0, abs_y0, abs_x1, abs_y1), drop_zone
            ):
                continue
            normalized_text = _normalize_visible_text(text)
            entries.append(
                {
                    "x0": float(abs_x0),
                    "x1": float(abs_x1),
                    "top": float(abs_y0),
                    "bottom": float(abs_y1),
                    "text": normalized_text,
                }
            )
            poly = [
                {
                    "x": float(x0 + px / scale),
                    "y": float(y0 + py / scale),
                }
                for px, py in pts
            ]
            word_entries.append(
                {
                    "x0": float(abs_x0),
                    "x1": float(abs_x1),
                    "top": float(abs_y0),
                    "bottom": float(abs_y1),
                    "text": normalized_text,
                    "confidence": float(conf) if conf is not None else None,
                    "column": column_tag,
                    "poly": poly,
                }
            )

        if word_entries:
            bucket = self.page_word_boxes.setdefault(page_index, [])
            bucket.extend(word_entries)

        entries.sort(key=lambda r: (r["top"], r["x0"]))
        grouped: List[List[Dict[str, float]]] = []
        cur: List[Dict[str, float]] = []
        cur_top: Optional[float] = None
        for item in entries:
            top = item["top"]
            if cur_top is None or abs(top - cur_top) <= y_tol:
                cur.append(item)
                cur_top = top if cur_top is None else cur_top
            else:
                grouped.append(cur)
                cur = [item]
                cur_top = top
        if cur:
            grouped.append(cur)

        lines: List[Dict[str, float]] = []
        for group in grouped:
            gx0 = min(it["x0"] for it in group)
            gx1 = max(it["x1"] for it in group)
            gy0 = min(it["top"] for it in group)
            gy1 = max(it["bottom"] for it in group)
            gtext = " ".join(it["text"] for it in group)
            lines.append(
                {
                    "x0": gx0,
                    "x1": gx1,
                    "top": gy0,
                    "bottom": gy1,
                    "y": 0.5 * (gy0 + gy1),
                    "text": gtext,
                }
            )

        lines.sort(key=lambda ln: (ln["top"], ln["x0"]))
        self._record_crop_sample(
            page_index,
            column_tag,
            orig_bbox,
            padded_bbox,
            filter_top,
            filter_bottom,
            pad_x,
            pad_top,
            pad_bottom,
            y_cut,
            drop_zone,
            raw_word_count=len(results),
            kept_word_count=len(word_entries),
            line_count=len(lines),
        )
        logger.debug(
            "Page %d OCR produced %d text lines in column window",
            page_index + 1,
            len(lines),
        )
        return lines


# =========================
# Layout helpers
# =========================

def two_col_bboxes(
    page,
    top_frac: float = DEFAULT_TOP_FRAC,
    bottom_frac: float = DEFAULT_BOTTOM_FRAC,
    gutter_frac: float = DEFAULT_GUTTER_FRAC,
):
    w, h = float(page.width), float(page.height)
    top = h * top_frac
    bottom = h * bottom_frac
    gutter = w * gutter_frac
    mid = w * 0.5
    return (0.0, top, mid - gutter, bottom), (mid + gutter, top, w, bottom)


def _rects_intersect(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


# =========================
# Question detection
# =========================

def _extract_qnum_from_text(text: str) -> Optional[int]:
    m = QUESTION_NUM_RE.match(text.strip())
    if not m:
        return None
    raw = next((g for g in m.groups() if g), None)
    if raw is None:
        return None
    try:
        num = int(raw)
    except Exception:
        return None
    if num >= 1000:
        return None
    return num


def detect_question_starts(
    lines: Sequence[Dict[str, float]],
    margin_abs: Optional[float],
    col_left: float,
    tol: float = 1.0,
    last_qnum: Optional[int] = None,
) -> Tuple[List[int], Optional[int]]:
    starts: List[int] = []
    target_rel = None if margin_abs is None else (margin_abs - col_left)
    current_last = last_qnum
    for i, ln in enumerate(lines):
        raw_text = ln.get("text") or ""
        stripped = raw_text.lstrip()
        if not stripped:
            continue
        if stripped[0] in OPTION_SET:
            continue
        text = stripped.rstrip()
        rel = ln["x0"] - col_left
        left_ok = True if target_rel is None else abs(rel - target_rel) <= tol
        text_ok = bool(QUESTION_START_LINE_RE.match(text))
        if not text_ok:
            continue
        qnum = _extract_qnum_from_text(text)
        seq_ok = qnum is not None and (
            current_last is None or qnum == current_last + 1
        )
        if left_ok or seq_ok:
            starts.append(i)
            if qnum is not None:
                current_last = qnum
    return starts, current_last


# =========================
# Chunk building
# =========================

def build_flow_segments(
    pdf: pdfplumber.pdf.PDF,
    extractor: EasyOCRTextExtractor,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
    y_tol: float,
    clip_mode: str,
    ycut_map: Dict[int, Optional[float]],
    band_map: Dict[int, Optional[Tuple[float, float, float, float]]],
):
    column_order = {"L": 0, "R": 1}
    segs = []
    for i, page in enumerate(pdf.pages):
        logger.debug("Building flow segments for page %d", i + 1)
        L, R = two_col_bboxes(page, top_frac, bottom_frac, gutter_frac)
        ycut = ycut_map.get(i + 1) if clip_mode == "ycut" else None
        band = band_map.get(i + 1) if clip_mode == "band" else None
        L_lines = extractor.extract_lines(
            i, L, y_tol=y_tol, y_cut=ycut, drop_zone=band, column_tag="L"
        )
        R_lines = extractor.extract_lines(
            i, R, y_tol=y_tol, y_cut=ycut, drop_zone=band, column_tag="R"
        )
        segs.append((i, "L", L, L_lines))
        segs.append((i, "R", R, R_lines))

    segs.sort(key=lambda entry: (entry[0], column_order.get(entry[1], 99)))
    return segs


def flow_chunk_all_pages(
    pdf: pdfplumber.pdf.PDF,
    extractor: EasyOCRTextExtractor,
    L_rel_offset: Optional[float],
    R_rel_offset: Optional[float],
    y_tol: float,
    tol: float,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
    clip_mode: str,
    ycut_map: Dict[int, Optional[float]],
    band_map: Dict[int, Optional[Tuple[float, float, float, float]]],
):
    segs = build_flow_segments(
        pdf,
        extractor,
        top_frac,
        bottom_frac,
        gutter_frac,
        y_tol,
        clip_mode,
        ycut_map,
        band_map,
    )
    logger.debug("Built %d column segments", len(segs))

    seg_meta = []
    page_text_map: Dict[int, List[Dict[str, object]]] = {
        i: [] for i in range(len(pdf.pages))
    }
    for (pi, col, bbox, lines) in segs:
        L, R = two_col_bboxes(pdf.pages[pi], top_frac, bottom_frac, gutter_frac)
        if col == "L":
            margin_abs = None if L_rel_offset is None else (L[0] + L_rel_offset)
            col_left = L[0]
        else:
            margin_abs = None if R_rel_offset is None else (R[0] + R_rel_offset)
            col_left = R[0]
        if logger.isEnabledFor(logging.DEBUG):
            rel_offset = L_rel_offset if col == "L" else R_rel_offset
            logger.debug(
                "Page %d column %s bbox=%s margin_abs=%s rel_offset=%s",
                pi + 1,
                col,
                tuple(round(v, 2) for v in bbox),
                None if margin_abs is None else round(margin_abs, 2),
                None if rel_offset is None else round(rel_offset, 2),
            )
        seg_meta.append((pi, col, bbox, lines, margin_abs, col_left))
        if lines:
            col_lines = page_text_map.setdefault(pi, [])
            for line in lines:
                entry = dict(line)
                entry["column"] = col
                col_lines.append(entry)

    seg_starts = []
    last_detected_qnum = None
    for (pi, col, bbox, lines, m_abs, col_left) in seg_meta:
        starts, last_detected_qnum = detect_question_starts(
            lines, m_abs, col_left, tol=tol, last_qnum=last_detected_qnum
        )
        seg_starts.append(starts)
        logger.debug(
            "Detected %d candidate question starts on page %d column %s",
            len(starts),
            pi + 1,
            col,
        )

    chunks = []
    current = None
    for seg_idx, (pi, col, bbox, lines, m_abs, col_left) in enumerate(seg_meta):
        starts = set(seg_starts[seg_idx])
        i = 0
        while i < len(lines):
            if i in starts:
                if current is not None:
                    chunks.append(current)
                current = {
                    "pieces": [],
                    "start": {"page": pi, "col": col, "line_idx": i},
                }
            if current is not None:
                next_mark = min((j for j in starts if j > i), default=None)
                end_idx = (next_mark - 1) if next_mark is not None else (len(lines) - 1)
                block = lines[i : end_idx + 1]
                if block:
                    x0 = min(l["x0"] for l in block)
                    x1 = max(l["x1"] for l in block)
                    top = min(l["top"] for l in block) - 2.0
                    bot = max(l["bottom"] for l in block) + 2.0
                    text = "\n".join(l["text"] for l in block)
                    current["pieces"].append(
                        {
                            "page": pi,
                            "col": col,
                            "box": {"x0": x0, "x1": x1, "top": top, "bottom": bot},
                            "start_line": i,
                            "end_line": end_idx,
                            "text": text,
                        }
                    )
                i = end_idx + 1
            else:
                i += 1
    if current is not None:
        chunks.append(current)
    logger.debug("Assembled %d candidate chunks", len(chunks))

    per_page_boxes = {i: [] for i in range(len(pdf.pages))}
    for ch_id, ch in enumerate(chunks, start=1):
        for p in ch.get("pieces", []):
            b = p["box"].copy()
            b["chunk_id"] = ch_id
            b["col"] = p["col"]
            per_page_boxes[p["page"]].append(b)

    return chunks, per_page_boxes, page_text_map


# =========================
# QA extraction helpers
# =========================

def parse_dispute(stem: str, keep_text: bool = True):
    if not stem:
        return False, None, stem
    m = DISPUTE_RE.search(stem)
    if not m:
        return False, None, norm_space(stem)
    site = norm_space(m.group("site") or "")
    if keep_text:
        return True, (site or None), norm_space(stem)
    new_stem = norm_space(DISPUTE_RE.sub("", stem))
    return True, (site or None), new_stem


def extract_leading_qnum_and_clean(stem: str) -> Tuple[Optional[int], str]:
    if not stem:
        return None, stem
    m = QUESTION_NUM_RE.match(stem)
    if not m:
        return None, stem
    digits = next((g for g in m.groups() if g), None)
    try:
        qnum = int(digits) if digits is not None else None
    except Exception:
        qnum = None
    return qnum, stem[m.end() :].lstrip()


def _trim_to_first_question(text: str) -> Tuple[str, Optional[int]]:
    if not text:
        return text, None
    m = QUESTION_START_LINE_RE.search(text)
    if not m:
        return text, None
    trimmed = text[m.start() :]
    digits = None
    dm = QUESTION_NUM_RE.match(trimmed)
    if dm:
        raw = next((g for g in dm.groups() if g), None)
        try:
            digits = int(raw) if raw is not None else None
        except Exception:
            digits = None
    return trimmed, digits


def sanitize_chunk_text(text: str, expected_next_qnum: Optional[int]) -> str:
    if not text:
        return text

    text = _normalize_visible_text(text)
    trimmed, current_qnum = _trim_to_first_question(text)
    text = trimmed

    if current_qnum is not None:
        target_next = current_qnum + 1
    else:
        target_next = expected_next_qnum

    if target_next is None:
        return text

    for match in QUESTION_START_LINE_RE.finditer(text):
        if match.start() == 0:
            continue
        candidate_slice = text[match.start() :]
        dm = QUESTION_NUM_RE.match(candidate_slice)
        if not dm:
            continue
        raw = next((g for g in dm.groups() if g), None)
        if raw is None:
            continue
        try:
            num = int(raw)
        except Exception:
            continue
        if num >= 1000:
            continue
        if num == target_next:
            return text[: match.start()].rstrip()

    return text


def extract_qa_from_chunk_text(text: str):
    if not text:
        return None, None, False, None, None

    text = _normalize_option_markers(text)
    text = _strip_header_garbage(text)

    first_match = re.search(rf"[{OPTION_CLASS}]", text)
    if not first_match:
        return None, None, False, None, None

    first = first_match.start()
    stem, opts_blob = text[:first], text[first:]

    dispute, dispute_site, stem = parse_dispute(stem, keep_text=True)
    stem = norm_space(stem)

    detected_qnum, stem = extract_leading_qnum_and_clean(stem)
    stem = norm_space(stem)

    parts = [p for p in OPT_SPLIT_RE.split(opts_blob) if p]
    options = []
    i = 0
    while i < len(parts):
        sym = parts[i].strip()
        if sym and sym[0] in OPTION_SET:
            raw_txt = parts[i + 1] if (i + 1) < len(parts) else ""
            clean_txt = norm_space(CIRCLED_STRIP_RE.sub("", raw_txt))
            options.append({"index": sym[0], "text": clean_txt})
            i += 2
        else:
            i += 1
    options = [o for o in options if o["index"] in OPTION_SET]
    if not options:
        return None, None, dispute, dispute_site, detected_qnum

    return stem, options, dispute, dispute_site, detected_qnum


def format_qa_items_as_text(qa_items: Sequence[Dict[str, object]]) -> str:
    lines: List[str] = []
    for item in qa_items or []:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, dict):
            continue

        qnum = content.get("question_number")
        stem = content.get("question_text") or ""
        dispute = bool(content.get("dispute_bool"))
        dispute_site = content.get("dispute_site") or ""
        options = content.get("options") or []

        stem_lines = (stem.splitlines() if stem else [])
        if qnum is not None:
            if stem_lines:
                lines.append(f"{qnum}. {stem_lines[0]}".rstrip())
                lines.extend(stem_lines[1:])
            else:
                lines.append(f"{qnum}.")
        else:
            lines.extend(stem_lines)

        if dispute:
            tag = "[Dispute]" if not dispute_site else f"[Dispute] {dispute_site}"
            lines.append(tag.strip())

        for opt in options:
            if not isinstance(opt, dict):
                continue
            idx = (opt.get("index") or "").strip()
            text = opt.get("text") or ""
            opt_lines = text.splitlines() if text else [""]
            if opt_lines:
                first_line = opt_lines[0]
                prefix = f"{idx} " if idx else ""
                lines.append(f"{prefix}{first_line}".rstrip())
                lines.extend(opt_lines[1:])

        lines.append("")

    if lines and lines[-1] != "":
        lines.append("")

    return "\n".join(lines)


@dataclass
class QAValidationResult:
    total_items: int
    numbered_items: int
    first_number: Optional[int]
    last_number: Optional[int]
    missing_numbers: List[int]
    duplicate_numbers: List[int]
    out_of_order_pairs: List[Tuple[int, int]]
    expected_count: Optional[int] = None
    start_number: Optional[int] = None

    def is_complete(self) -> bool:
        baseline_ok = self.total_items > 0 or (self.expected_count == 0)
        return (
            baseline_ok
            and not self.missing_numbers
            and not self.duplicate_numbers
            and not self.out_of_order_pairs
            and not self.count_mismatch()
        )

    def count_mismatch(self) -> bool:
        return self.expected_count is not None and self.total_items != self.expected_count


def validate_qa_sequence(
    qa_items: Sequence[Dict[str, object]],
    expected_count: Optional[int] = None,
    start_number: Optional[int] = None,
) -> QAValidationResult:
    items = list(qa_items or [])
    numbers: List[int] = []
    counts: Dict[int, int] = {}
    out_of_order: List[Tuple[int, int]] = []
    last_seen: Optional[int] = None

    for item in items:
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, dict):
            continue
        qnum = content.get("question_number")
        if not isinstance(qnum, int):
            continue
        numbers.append(qnum)
        counts[qnum] = counts.get(qnum, 0) + 1
        if last_seen is not None and qnum < last_seen:
            out_of_order.append((last_seen, qnum))
        last_seen = qnum

    duplicates = sorted({num for num, count in counts.items() if count > 1})

    missing: List[int] = []
    if numbers:
        numbers_sorted = sorted(set(numbers))
        first = numbers_sorted[0]
        last = numbers_sorted[-1]
        missing.extend(n for n in range(first, last + 1) if counts.get(n, 0) == 0)
    else:
        first = start_number
        last = None

    if expected_count is not None and start_number is not None:
        expected_last = start_number + max(expected_count - 1, 0)
        missing.extend(
            n
            for n in range(start_number, expected_last + 1)
            if counts.get(n, 0) == 0 and n not in missing
        )
    missing.sort()

    return QAValidationResult(
        total_items=len(items),
        numbered_items=len(numbers),
        first_number=first,
        last_number=last,
        missing_numbers=missing,
        duplicate_numbers=duplicates,
        out_of_order_pairs=out_of_order,
        expected_count=expected_count,
        start_number=start_number,
    )


def log_validation_summary(result: QAValidationResult, label: str) -> None:
    base_msg = (
        f"Validation summary for {label}: {result.total_items} total items, "
        f"{result.numbered_items} numbered"
    )
    range_msg = ""
    if result.first_number is not None and result.last_number is not None:
        range_msg = f", range {result.first_number}-{result.last_number}"
    logger.info("%s%s", base_msg, range_msg)

    if result.total_items == 0:
        logger.warning("No QA items detected for %s", label)

    if result.count_mismatch():
        logger.warning(
            "Expected %d items but produced %d", result.expected_count, result.total_items
        )

    if result.missing_numbers:
        logger.warning("Missing question numbers: %s", ", ".join(map(str, result.missing_numbers)))

    if result.duplicate_numbers:
        logger.warning("Duplicate question numbers detected: %s", ", ".join(map(str, result.duplicate_numbers)))

    if result.out_of_order_pairs:
        formatted = ", ".join(f"{a}->{b}" for a, b in result.out_of_order_pairs)
        logger.warning("Out-of-order question numbers: %s", formatted)


# =========================
# Chunk preview images
# =========================

def save_chunk_preview(
    page,
    bbox,
    preview_dir,
    page_index,
    column_tag,
    chunk_idx_in_column,
    global_idx,
    dpi=220,
    pad=2.0,
):
    if not bbox or not preview_dir:
        return None
    abs_dir = os.path.abspath(os.path.expanduser(preview_dir))
    os.makedirs(abs_dir, exist_ok=True)
    width, height = float(page.width), float(page.height)
    x0, top, x1, bottom = map(float, bbox)
    pad = max(0.0, float(pad))
    padded = (
        max(0.0, x0 - pad),
        max(0.0, top - pad),
        min(width, x1 + pad),
        min(height, bottom + pad),
    )
    cropped = page.within_bbox(padded)
    img = cropped.to_image(resolution=int(dpi))
    pil = img.original.convert("RGB")
    del img
    fn = f"p{page_index:03d}_{column_tag}{chunk_idx_in_column:02d}_{global_idx:04d}.jpg"
    out_path = os.path.join(abs_dir, fn)
    pil.save(out_path, format="JPEG", quality=90)
    pil.close()
    return os.path.abspath(out_path)


def save_rasterized_pdf(pdf_path: str, out_path: str, dpi: int) -> None:
    abs_out = os.path.abspath(os.path.expanduser(out_path))
    ensure_dir(os.path.dirname(abs_out))
    images: List[Image.Image] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                logger.debug(
                    "Rasterizing page %d/%d at %d DPI for PDF export",
                    page_index,
                    len(pdf.pages),
                    dpi,
                )
                pil = page.to_image(resolution=int(dpi)).original.convert("RGB")
                images.append(pil)
    except Exception:
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        raise

    if not images:
        logger.warning("No pages rendered for raster PDF export of %s", pdf_path)
        return

    first, rest = images[0], images[1:]
    try:
        first.save(
            abs_out,
            format="PDF",
            save_all=True,
            append_images=rest,
            resolution=int(dpi),
        )
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass


def _resolve_searchable_font(
    font_spec: Optional[str],
    pdfmetrics,
):
    candidates: List[str] = []
    if font_spec:
        candidates.append(font_spec)
    candidates.extend(DEFAULT_SEARCHABLE_FONT_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(expanded):
            try:
                from reportlab.pdfbase.ttfonts import TTFont  # type: ignore

                alias = os.path.splitext(os.path.basename(expanded))[0]
                pdfmetrics.registerFont(TTFont(alias, expanded))
                logger.info(
                    "Registered TrueType font %s for searchable PDF output", alias
                )
                return alias
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to register TrueType font %s: %s", expanded, exc
                )
                continue
        try:
            pdfmetrics.getFont(candidate)
            return candidate
        except KeyError:
            try:
                from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # type: ignore

                pdfmetrics.registerFont(UnicodeCIDFont(candidate))
                logger.debug(
                    "Registered CID font %s for searchable PDF output", candidate
                )
                return candidate
            except KeyError:
                logger.debug(
                    "CID font %s not available in this ReportLab build", candidate
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to register CID font %s: %s", candidate, exc
                )
    try:
        pdfmetrics.getFont("Helvetica")
        logger.warning(
            "Falling back to Helvetica for searchable PDF text layer; CJK glyphs may be missing"
        )
        return "Helvetica"
    except KeyError:  # pragma: no cover - extremely unlikely
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # type: ignore

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        logger.warning(
            "Falling back to STSong-Light for searchable PDF text layer"
        )
        return "STSong-Light"


def save_searchable_pdf(
    pdf_path: str,
    out_path: str,
    dpi: int,
    page_text_map: Dict[int, List[Dict[str, object]]],
    font_spec: Optional[str] = None,
) -> None:
    try:
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Saving searchable PDFs requires the reportlab package"
        ) from exc

    abs_out = os.path.abspath(os.path.expanduser(out_path))
    ensure_dir(os.path.dirname(abs_out))

    font_name = _resolve_searchable_font(font_spec, pdfmetrics)

    with pdfplumber.open(pdf_path) as pdf:
        canv = canvas.Canvas(abs_out)
        for page_index, page in enumerate(pdf.pages):
            width, height = float(page.width), float(page.height)
            canv.setPageSize((width, height))
            pil = page.to_image(resolution=int(dpi)).original.convert("RGB")
            try:
                canv.drawImage(
                    ImageReader(pil),
                    0,
                    0,
                    width=width,
                    height=height,
                    mask=None,
                )
            finally:
                pil.close()

            lines = page_text_map.get(page_index) or []
            lines_sorted = sorted(lines, key=lambda ln: (ln.get("top", 0.0), ln.get("x0", 0.0)))
            for ln in lines_sorted:
                raw_text = ln.get("text")
                if not raw_text:
                    continue
                text = norm_space(str(raw_text))
                if not text:
                    continue
                try:
                    x0 = float(ln.get("x0", 0.0))
                    top = float(ln.get("top", 0.0))
                    bottom = float(ln.get("bottom", top))
                except (TypeError, ValueError):
                    continue
                line_height = max(bottom - top, 6.0)
                font_size = min(max(line_height * 0.95, 6.0), 36.0)
                baseline = height - top - (line_height * 0.2)
                text_obj = canv.beginText()
                text_obj.setTextRenderMode(3)  # invisible but searchable text layer
                text_obj.setFont(font_name, font_size)
                text_obj.setTextOrigin(x0, baseline)
                text_obj.textLine(text)
                canv.drawText(text_obj)

        canv.showPage()

    canv.save()


def save_bbox_overlay_pdf(
    pdf_path: str,
    out_path: str,
    dpi: int,
    word_map: Dict[int, List[Dict[str, object]]],
    line_map: Optional[Dict[int, List[Dict[str, object]]]] = None,
    word_color: Tuple[int, int, int] = (255, 0, 0),
    line_color: Tuple[int, int, int] = (0, 128, 255),
) -> None:
    abs_out = os.path.abspath(os.path.expanduser(out_path))
    ensure_dir(os.path.dirname(abs_out))

    images: List[Image.Image] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            scale = float(dpi) / 72.0
            page_total = len(pdf.pages)
            for page_index, page in enumerate(pdf.pages):
                logger.debug(
                    "Rendering OCR overlay for page %d/%d at %d DPI",
                    page_index + 1,
                    page_total,
                    dpi,
                )
                base_img = page.to_image(resolution=int(dpi)).original.convert("RGB")
                draw = ImageDraw.Draw(base_img)
                stroke_words = max(1, int(round(dpi / 300)))
                stroke_lines = max(1, int(round(dpi / 360)))

                words = word_map.get(page_index) or []
                for word in words:
                    try:
                        x0 = float(word.get("x0")) * scale
                        x1 = float(word.get("x1")) * scale
                        top = float(word.get("top")) * scale
                        bottom = float(word.get("bottom")) * scale
                    except (TypeError, ValueError):
                        continue
                    draw.rectangle((x0, top, x1, bottom), outline=word_color, width=stroke_words)

                if line_map:
                    lines = line_map.get(page_index) or []
                    for line in lines:
                        try:
                            x0 = float(line.get("x0")) * scale
                            x1 = float(line.get("x1")) * scale
                            top = float(line.get("top")) * scale
                            bottom = float(line.get("bottom", top)) * scale
                        except (TypeError, ValueError):
                            continue
                        draw.rectangle((x0, top, x1, bottom), outline=line_color, width=stroke_lines)

                del draw
                images.append(base_img)
    except Exception:
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        raise

    if not images:
        logger.warning("No pages rendered for OCR overlay PDF export of %s", pdf_path)
        return

    first, rest = images[0], images[1:]
    try:
        first.save(
            abs_out,
            format="PDF",
            save_all=True,
            append_images=rest,
            resolution=int(dpi),
        )
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass


def save_crop_report(
    pdf_path: str,
    out_path: str,
    crop_report: Dict[int, List[Dict[str, Any]]],
) -> None:
    abs_out = os.path.abspath(os.path.expanduser(out_path))
    ensure_dir(os.path.dirname(abs_out))

    def fmt_bbox(bbox: Sequence[float]) -> str:
        return "(%.2f, %.2f, %.2f, %.2f)" % tuple(float(v) for v in bbox)

    def fmt_band(band: Sequence[float]) -> str:
        return "(%.2f – %.2f)" % (float(band[0]), float(band[1]))

    lines: List[str] = []
    lines.append(f"Crop report for {os.path.basename(pdf_path)}")
    lines.append(f"Generated {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for page_index in sorted(crop_report.keys()):
        entries = crop_report.get(page_index) or []
        lines.append(f"Page {page_index + 1}")
        if not entries:
            lines.append("  (no OCR segments)")
            lines.append("")
            continue
        ordered = sorted(
            entries,
            key=lambda e: (
                0 if str(e.get("column", "")).upper() == "L" else 1,
                (e.get("orig_bbox") or (0.0, 0.0, 0.0, 0.0))[1],
            ),
        )
        for entry in ordered:
            col = str(entry.get("column") or "?").upper()
            orig = entry.get("orig_bbox") or (0.0, 0.0, 0.0, 0.0)
            padded = entry.get("padded_bbox") or orig
            band = entry.get("filter_band") or (0.0, 0.0)
            pad = entry.get("pad") or {}
            drop = entry.get("drop_zone")
            lines.append(f"  Column {col}:")
            lines.append(f"    original bbox : {fmt_bbox(orig)}")
            lines.append(f"    padded bbox   : {fmt_bbox(padded)}")
            lines.append(f"    filter band   : {fmt_band(band)}")
            lines.append(
                "    padding       : x=%s top=%s bottom=%s"
                % (
                    f"{float(pad.get('x', 0.0)):.2f}",
                    f"{float(pad.get('top', 0.0)):.2f}",
                    f"{float(pad.get('bottom', 0.0)):.2f}",
                )
            )
            if entry.get("y_cut") is not None:
                lines.append(f"    y-cut         : {float(entry['y_cut']):.2f}")
            if drop:
                lines.append(f"    drop zone     : {fmt_bbox(drop)}")
            lines.append(
                "    OCR words     : kept %d / raw %d"
                % (int(entry.get("kept_word_count", 0)), int(entry.get("raw_word_count", 0)))
            )
            lines.append(f"    line count    : {int(entry.get('line_count', 0))}")
        lines.append("")

    with open(abs_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


# =========================
# Top-level parse
# =========================

def pdf_to_qa_flow_chunks(
    pdf_path: str,
    year: int,
    start_num: int,
    L_rel: Optional[float],
    R_rel: Optional[float],
    tol: float,
    top_frac: float = DEFAULT_TOP_FRAC,
    bottom_frac: float = DEFAULT_BOTTOM_FRAC,
    gutter_frac: float = DEFAULT_GUTTER_FRAC,
    y_tol: float = 3.0,
    clip_mode: str = "none",
    chunk_preview_dir: Optional[str] = None,
    chunk_preview_dpi: int = 220,
    chunk_preview_pad: float = 2.0,
    ocr_settings: Optional[OCRSettings] = None,
    chunk_debug_dir: Optional[str] = None,
    failed_chunk_log_chars: int = 240,
):
    if ocr_settings is None:
        ocr_settings = OCRSettings()

    out: List[Dict[str, object]] = []
    last_assigned_qno = start_num - 1
    global_idx = 0
    preview_dir = (
        os.path.abspath(os.path.expanduser(chunk_preview_dir))
        if chunk_preview_dir
        else None
    )
    debug_dir = (
        os.path.abspath(os.path.expanduser(chunk_debug_dir))
        if chunk_debug_dir
        else None
    )
    if debug_dir:
        ensure_dir(debug_dir)

    page_text_map: Dict[int, List[Dict[str, object]]] = {}

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(
            "Opened %s with %d pages (year=%s, start=%s)",
            os.path.basename(pdf_path),
            len(pdf.pages),
            year,
            start_num,
        )
        extractor = EasyOCRTextExtractor(pdf, ocr_settings)
        ycut_map: Dict[int, Optional[float]] = {}
        band_map: Dict[int, Optional[Tuple[float, float, float, float]]] = {}

        chunks, _, page_text_map = flow_chunk_all_pages(
            pdf,
            extractor,
            L_rel,
            R_rel,
            y_tol,
            tol,
            top_frac,
            bottom_frac,
            gutter_frac,
            clip_mode=clip_mode,
            ycut_map=ycut_map,
            band_map=band_map,
        )
        logger.info("Detected %d raw chunks before QA filtering", len(chunks))

        for idx, ch in enumerate(chunks, start=1):
            pieces = ch.get("pieces") or []
            if not pieces:
                continue
            p1 = pieces[0]["page"] + 1

            expected_next = (
                last_assigned_qno + 1 if last_assigned_qno is not None else None
            )
            raw_text = "\n".join(p["text"] for p in pieces if p.get("text"))
            text = sanitize_chunk_text(raw_text, expected_next)
            debug_basename = f"p{p1:03d}_{pieces[0]['col']}_{idx:04d}"
            if debug_dir:
                raw_path = os.path.join(debug_dir, f"{debug_basename}_raw.txt")
                clean_path = os.path.join(debug_dir, f"{debug_basename}_clean.txt")
                with open(raw_path, "w", encoding="utf-8") as fh:
                    fh.write(raw_text)
                with open(clean_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            stem, options, dispute, dispute_site, detected_qnum = (
                extract_qa_from_chunk_text(text)
            )
            if stem is None or not options:
                snippet = text.strip().replace("\n", " ")
                if len(snippet) > failed_chunk_log_chars:
                    snippet = snippet[: failed_chunk_log_chars].rstrip() + "…"
                logger.warning(
                    "Skipping chunk starting on page %d column %s: no QA detected. Sample: %s",
                    pieces[0]["page"] + 1,
                    pieces[0]["col"],
                    snippet or "<empty>",
                )
                continue

            if detected_qnum is not None:
                qno = detected_qnum
            elif expected_next is not None:
                qno = expected_next
            else:
                qno = start_num
            global_idx += 1

            preview_path = None
            if preview_dir:
                b = pieces[0]["box"]
                bbox = (b["x0"], b["top"], b["x1"], b["bottom"])
                page = pdf.pages[pieces[0]["page"]]
                preview_path = save_chunk_preview(
                    page,
                    bbox,
                    preview_dir,
                    p1,
                    pieces[0]["col"],
                    1,
                    global_idx,
                    dpi=chunk_preview_dpi,
                    pad=chunk_preview_pad,
                )

            out.append(
                {
                    "year": year,
                    "content": {
                        "question_number": qno,
                        "question_text": stem,
                        "dispute_bool": bool(dispute),
                        "dispute_site": dispute_site,
                        "options": options,
                        "source": {"pieces": pieces, "start_page": p1},
                        **({"preview_image": preview_path} if preview_path else {}),
                    },
                }
            )

            last_assigned_qno = qno
            logger.debug(
                "Accepted chunk %d => question %d (%d options)",
                global_idx,
                qno,
                len(options),
            )

        logger.info("Accepted %d QA items after filtering", len(out))

        page_word_map: Dict[int, List[Dict[str, object]]] = {
            idx: extractor.page_word_boxes.get(idx, [])
            for idx in range(len(pdf.pages))
        }
        crop_report_map: Dict[int, List[Dict[str, Any]]] = {
            idx: extractor.crop_reports.get(idx, [])
            for idx in range(len(pdf.pages))
        }

    return out, page_text_map, page_word_map, crop_report_map


# =========================
# CLI helpers
# =========================

def _auto_detect_margins_for_pdf(
    pdf_path: str, top_frac: float, bottom_frac: float, gutter_frac: float
) -> Tuple[Optional[float], Optional[float]]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pg in pdf.pages:
                Lb, Rb = two_col_bboxes(pg, top_frac, bottom_frac, gutter_frac)

                def first_x(bbox):
                    sub = pg.within_bbox(bbox)
                    xs = [c["x0"] for c in (sub.chars or []) if c.get("x0") is not None]
                    if not xs:
                        words = sub.extract_words(
                            x_tolerance=3, y_tolerance=3, keep_blank_chars=False
                        )
                        xs = [w["x0"] for w in words if w.get("x0") is not None]
                    return min(xs) if xs else None

                lx = first_x(Lb)
                rx = first_x(Rb)
                if lx is not None and rx is not None:
                    return float(lx - Lb[0]), float(rx - Rb[0])
    except Exception:
        pass
    return None, None


def interactive_margin_selection(
    pdf_path: str,
    page_index: int,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
    render_dpi: int = 200,
    initial_left: Optional[float] = None,
    initial_right: Optional[float] = None,
) -> Tuple[float, float]:
    """Interactive GUI to pick left/right margins on a preview image."""

    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:  # pragma: no cover - Tkinter availability depends on env
        raise RuntimeError("Tkinter is required for the margin selector") from exc

    try:
        from PIL import ImageTk
    except Exception as exc:  # pragma: no cover - Pillow build specific
        raise RuntimeError("Pillow ImageTk is required for the margin selector") from exc

    if page_index < 0:
        raise ValueError("page_index must be >= 0")

    pdf = pdfplumber.open(pdf_path)
    try:
        page_count = len(pdf.pages)
        if page_index >= page_count:
            raise ValueError("page_index exceeds total page count")
    except Exception:
        pdf.close()
        raise

    state = {
        "zoom": 1.0,
        "page_index": page_index,
        "left_offset": initial_left,
        "right_offset": initial_right,
        "left_pdf_x": None,
        "right_pdf_x": None,
        "base_img": None,
        "base_w": None,
        "base_h": None,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "L_bbox": None,
        "R_bbox": None,
        "result": None,
    }

    root = tk.Tk()
    root.title("Manual Margin Selector")

    main_frame = ttk.Frame(root, padding=8)
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    canvas = tk.Canvas(main_frame, background="#222", highlightthickness=0)
    hbar = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
    vbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    hbar.grid(row=1, column=0, sticky="ew")
    vbar.grid(row=0, column=1, sticky="ns")
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    controls = ttk.Frame(main_frame)
    controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    controls.columnconfigure(6, weight=1)

    zoom_var = tk.DoubleVar(value=100.0)
    zoom_display_var = tk.StringVar(value="100%")
    selection_var = tk.StringVar(value="L")
    status_var = tk.StringVar(value="Click inside a column to set a margin line.")

    photo_cache = {"image": None}

    def pdf_to_canvas_coords(x: float, y: float) -> Tuple[float, float]:
        zoom = state["zoom"]
        return (
            x * state["scale_x"] * zoom,
            y * state["scale_y"] * zoom,
        )

    def refresh_image(*_):
        base_img = state.get("base_img")
        if base_img is None:
            return
        zoom = zoom_var.get() / 100.0
        if zoom <= 0:
            zoom = 0.1
        state["zoom"] = zoom
        base_w = state.get("base_w") or base_img.width
        base_h = state.get("base_h") or base_img.height
        disp_w = max(1, int(round(base_w * zoom)))
        disp_h = max(1, int(round(base_h * zoom)))
        resized = base_img.resize((disp_w, disp_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        photo_cache["image"] = photo
        if "image_id" not in photo_cache:
            photo_cache["image_id"] = canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            canvas.itemconfigure(photo_cache["image_id"], image=photo)
        canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
        zoom_display_var.set(f"{zoom * 100:.0f}%")
        draw_overlays()

    def draw_overlays():
        canvas.delete("overlay")
        zoom = state["zoom"]
        L_bbox = state.get("L_bbox")
        R_bbox = state.get("R_bbox")
        if L_bbox is None or R_bbox is None:
            return
        for tag, bbox, color in (
            ("L", L_bbox, "#33aaff"),
            ("R", R_bbox, "#ff9933"),
        ):
            x0, y0 = pdf_to_canvas_coords(bbox[0], bbox[1])
            x1, y1 = pdf_to_canvas_coords(bbox[2], bbox[3])
            canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline=color,
                width=max(1, int(2 * zoom)),
                tags="overlay",
            )
        if state["left_pdf_x"] is not None:
            x = pdf_to_canvas_coords(state["left_pdf_x"], L_bbox[1])[0]
            y0 = pdf_to_canvas_coords(0, L_bbox[1])[1]
            y1 = pdf_to_canvas_coords(0, L_bbox[3])[1]
            canvas.create_line(
                x,
                y0,
                x,
                y1,
                fill="#00ffff",
                width=max(1, int(3 * zoom)),
                tags="overlay",
            )
            canvas.create_text(
                x + 6 * zoom,
                y0 + 6 * zoom,
                text=f"L: {state['left_pdf_x']:.1f} pt",
                anchor="nw",
                fill="#00ffff",
                font=("TkDefaultFont", max(8, int(9 * zoom))),
                tags="overlay",
            )
        if state["right_pdf_x"] is not None:
            x = pdf_to_canvas_coords(state["right_pdf_x"], R_bbox[1])[0]
            y0 = pdf_to_canvas_coords(0, R_bbox[1])[1]
            y1 = pdf_to_canvas_coords(0, R_bbox[3])[1]
            canvas.create_line(
                x,
                y0,
                x,
                y1,
                fill="#ff66aa",
                width=max(1, int(3 * zoom)),
                tags="overlay",
            )
            canvas.create_text(
                x + 6 * zoom,
                y0 + 6 * zoom,
                text=f"R: {state['right_pdf_x']:.1f} pt",
                anchor="nw",
                fill="#ff66aa",
                font=("TkDefaultFont", max(8, int(9 * zoom))),
                tags="overlay",
            )

    def on_click(event):
        canvas.focus_set()
        canvas_x = canvas.canvasx(event.x)
        canvas_y = canvas.canvasy(event.y)
        zoom = state["zoom"]
        pdf_x = canvas_x / (state["scale_x"] * zoom)
        pdf_y = canvas_y / (state["scale_y"] * zoom)
        L_bbox = state.get("L_bbox")
        R_bbox = state.get("R_bbox")
        if L_bbox is None or R_bbox is None:
            return
        side = selection_var.get()
        if side == "L":
            if not (L_bbox[0] <= pdf_x <= L_bbox[2]):
                status_var.set("Click inside the highlighted left column.")
                return
            state["left_pdf_x"] = pdf_x
            state["left_offset"] = pdf_x - L_bbox[0]
            status_var.set(f"Left margin set at x={pdf_x:.2f} pt (y={pdf_y:.2f})")
        else:
            if not (R_bbox[0] <= pdf_x <= R_bbox[2]):
                status_var.set("Click inside the highlighted right column.")
                return
            state["right_pdf_x"] = pdf_x
            state["right_offset"] = pdf_x - R_bbox[0]
            status_var.set(f"Right margin set at x={pdf_x:.2f} pt (y={pdf_y:.2f})")
        draw_overlays()

    def on_key(event):
        if event.char.lower() == "l":
            selection_var.set("L")
        elif event.char.lower() == "r":
            selection_var.set("R")

    def confirm():
        if state["left_offset"] is None or state["right_offset"] is None:
            messagebox.showwarning(
                "Incomplete", "Set both left and right margins before continuing."
            )
            return
        state["result"] = (state["left_offset"], state["right_offset"])
        root.destroy()

    def cancel():
        state["result"] = None
        root.destroy()

    canvas.bind("<Button-1>", on_click)
    canvas.bind("<Key>", on_key)
    canvas.focus_set()
    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", cancel)

    ttk.Label(controls, text="Zoom:").grid(row=0, column=0, sticky="w")
    zoom_scale = ttk.Scale(
        controls,
        from_=50,
        to=300,
        orient="horizontal",
        variable=zoom_var,
        command=lambda _evt: refresh_image(),
    )
    zoom_scale.grid(row=0, column=1, sticky="ew", padx=(4, 12))
    ttk.Label(controls, textvariable=zoom_display_var, width=6).grid(
        row=0, column=2, sticky="w"
    )

    ttk.Label(controls, text="Select margin:").grid(row=0, column=3, padx=(8, 0))
    ttk.Radiobutton(
        controls, text="Left (L)", value="L", variable=selection_var
    ).grid(row=0, column=4, padx=(4, 4))
    ttk.Radiobutton(
        controls, text="Right (R)", value="R", variable=selection_var
    ).grid(row=0, column=5, padx=(4, 4))

    ttk.Button(controls, text="Reset", command=lambda: reset_lines()).grid(
        row=0, column=6, sticky="w"
    )
    ttk.Button(controls, text="Cancel", command=cancel).grid(
        row=0, column=7, padx=(12, 4)
    )
    ttk.Button(controls, text="Apply", command=confirm).grid(row=0, column=8)

    page_info_var = tk.StringVar()

    def change_page(delta: int):
        new_index = state["page_index"] + delta
        if new_index < 0 or new_index >= page_count:
            return
        load_page(new_index)

    ttk.Separator(controls, orient="horizontal").grid(
        row=1, column=0, columnspan=9, sticky="ew", pady=(6, 6)
    )
    ttk.Label(controls, textvariable=page_info_var).grid(
        row=2, column=0, columnspan=3, sticky="w"
    )
    ttk.Button(controls, text="◀ Prev", command=lambda: change_page(-1)).grid(
        row=2, column=3, padx=(4, 4)
    )
    ttk.Button(controls, text="Next ▶", command=lambda: change_page(1)).grid(
        row=2, column=4, padx=(4, 12)
    )
    controls.columnconfigure(1, weight=1)
    controls.columnconfigure(6, weight=1)

    status = ttk.Label(main_frame, textvariable=status_var, anchor="w")
    status.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    def reset_lines():
        state["left_pdf_x"] = None
        state["right_pdf_x"] = None
        state["left_offset"] = None
        state["right_offset"] = None
        status_var.set("Margins cleared. Click to set new positions.")
        draw_overlays()

    def load_page(idx: int):
        if not (0 <= idx < page_count):
            return
        page = pdf.pages[idx]
        L_bbox, R_bbox = two_col_bboxes(page, top_frac, bottom_frac, gutter_frac)
        base_img = page.to_image(resolution=int(render_dpi)).original.convert("RGB")
        base_w, base_h = base_img.size
        pdf_w, pdf_h = float(page.width), float(page.height)
        scale_x = base_w / pdf_w if pdf_w else 1.0
        scale_y = base_h / pdf_h if pdf_h else 1.0

        state["page_index"] = idx
        state["base_img"] = base_img
        state["base_w"] = base_w
        state["base_h"] = base_h
        state["scale_x"] = scale_x
        state["scale_y"] = scale_y
        state["L_bbox"] = L_bbox
        state["R_bbox"] = R_bbox
        if state["left_offset"] is not None:
            state["left_pdf_x"] = L_bbox[0] + state["left_offset"]
        else:
            state["left_pdf_x"] = None
        if state["right_offset"] is not None:
            state["right_pdf_x"] = R_bbox[0] + state["right_offset"]
        else:
            state["right_pdf_x"] = None

        page_info_var.set(f"Page {idx + 1} / {page_count}")
        status_var.set(
            "Click inside a column to set a margin line."
            if state["left_offset"] is None or state["right_offset"] is None
            else "Margins loaded. Adjust as needed then click Apply."
        )
        refresh_image()

    try:
        load_page(state["page_index"])

        root.mainloop()

        if state["result"] is None:
            raise RuntimeError("Margin selection cancelled")
        return state["result"]
    finally:
        pdf.close()


def process_single_pdf(
    pdf_path: str,
    out_path: str,
    year: Optional[int],
    start_num: int,
    tol: float,
    top_frac: float,
    bottom_frac: float,
    gutter_frac: float,
    clip_mode: str,
    margin_left: Optional[float],
    margin_right: Optional[float],
    preview_dir: Optional[str],
    preview_dpi: int,
    preview_pad: float,
    ocr_settings: OCRSettings,
    margin_ui: bool = False,
    margin_ui_page: int = 1,
    margin_ui_dpi: int = 200,
    heartbeat_interval: float = 30.0,
    chunk_debug_dir: Optional[str] = None,
    failed_chunk_log_chars: int = 240,
    raster_pdf_path: Optional[str] = None,
    raster_pdf_dpi: Optional[int] = None,
    searchable_pdf_path: Optional[str] = None,
    searchable_pdf_dpi: Optional[int] = None,
    searchable_pdf_font: Optional[str] = None,
    bbox_overlay_pdf_path: Optional[str] = None,
    bbox_overlay_pdf_dpi: Optional[int] = None,
    text_out_path: Optional[str] = None,
    crop_report_path: Optional[str] = None,
    expected_question_count: Optional[int] = None,
    fail_on_incomplete: bool = False,
):
    if year is None:
        year = infer_year_from_filename(pdf_path) or datetime.now().year

    if margin_ui:
        try:
            margin_left, margin_right = interactive_margin_selection(
                pdf_path=pdf_path,
                page_index=max(0, margin_ui_page - 1),
                top_frac=top_frac,
                bottom_frac=bottom_frac,
                gutter_frac=gutter_frac,
                render_dpi=margin_ui_dpi,
                initial_left=margin_left,
                initial_right=margin_right,
            )
            logger.info(
                "Interactive margins selected: L=%.2f R=%.2f",
                margin_left,
                margin_right,
            )
        except RuntimeError as exc:
            logger.exception("Margin selector failed: %s", exc)
            sys.exit(3)
    elif margin_left is None or margin_right is None:
        auto_l, auto_r = _auto_detect_margins_for_pdf(
            pdf_path, top_frac, bottom_frac, gutter_frac
        )
        logger.debug(
            "Auto-detected margins for %s: L=%s R=%s",
            os.path.basename(pdf_path),
            "None" if auto_l is None else f"{auto_l:.2f}",
            "None" if auto_r is None else f"{auto_r:.2f}",
        )
        margin_left = margin_left if margin_left is not None else auto_l
        margin_right = margin_right if margin_right is not None else auto_r

    logger.info(
        "Processing %s with margins L=%s R=%s",
        os.path.basename(pdf_path),
        "auto" if margin_left is None else f"{margin_left:.2f}",
        "auto" if margin_right is None else f"{margin_right:.2f}",
    )

    heartbeat = HeartbeatLogger(
        interval=heartbeat_interval,
        message=f"Still processing {os.path.basename(pdf_path)}...",
    )
    heartbeat.start()

    crop_report_map: Dict[int, List[Dict[str, Any]]] = {}
    try:
        qa, page_text_map, page_word_map, crop_report_map = pdf_to_qa_flow_chunks(
            pdf_path=pdf_path,
            year=year,
            start_num=start_num,
            L_rel=margin_left,
            R_rel=margin_right,
            tol=tol,
            top_frac=top_frac,
            bottom_frac=bottom_frac,
            gutter_frac=gutter_frac,
            y_tol=3.0,
            clip_mode=clip_mode,
            chunk_preview_dir=preview_dir,
            chunk_preview_dpi=preview_dpi,
            chunk_preview_pad=preview_pad,
            ocr_settings=ocr_settings,
            chunk_debug_dir=chunk_debug_dir,
            failed_chunk_log_chars=failed_chunk_log_chars,
        )
    finally:
        heartbeat.stop()

    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d QA items to %s", len(qa), out_path)

    if text_out_path:
        ensure_dir(os.path.dirname(os.path.abspath(text_out_path)))
        with open(text_out_path, "w", encoding="utf-8") as fh:
            fh.write(format_qa_items_as_text(qa))
        logger.info("Wrote QA text export to %s", text_out_path)

    if crop_report_path:
        save_crop_report(pdf_path, crop_report_path, crop_report_map)
        logger.info("Wrote column crop report to %s", crop_report_path)

    if raster_pdf_path:
        dpi = raster_pdf_dpi or (ocr_settings.dpi if ocr_settings else 800)
        save_rasterized_pdf(pdf_path, raster_pdf_path, dpi)
        logger.info("Saved rasterized PDF preview to %s", raster_pdf_path)
    if searchable_pdf_path:
        dpi = searchable_pdf_dpi or (ocr_settings.dpi if ocr_settings else 800)
        save_searchable_pdf(
            pdf_path,
            searchable_pdf_path,
            dpi,
            page_text_map,
            font_spec=searchable_pdf_font,
        )
        logger.info("Saved searchable OCR PDF to %s", searchable_pdf_path)
    if bbox_overlay_pdf_path:
        dpi = bbox_overlay_pdf_dpi or (ocr_settings.dpi if ocr_settings else 800)
        save_bbox_overlay_pdf(
            pdf_path,
            bbox_overlay_pdf_path,
            dpi,
            page_word_map,
            page_text_map,
        )
        logger.info("Saved OCR bounding box overlay PDF to %s", bbox_overlay_pdf_path)
    validation = validate_qa_sequence(qa, expected_question_count, start_num)
    log_validation_summary(validation, os.path.basename(pdf_path))
    if fail_on_incomplete and not validation.is_complete():
        raise RuntimeError(
            "Validation failed: gaps, duplicates, or count mismatch detected in parsed QA items"
        )

    return qa, validation


# =========================
# Main CLI
# =========================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Parse QA pairs from exam PDFs using EasyOCR for text extraction."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Input single PDF file")
    g.add_argument("--pdf-dir", help="Folder containing PDFs (non-recursive)")

    ap.add_argument("--out", required=True, help="Output JSON path or folder")
    ap.add_argument("--year", type=int, help="Year of the exam; inferred from filename if omitted")
    ap.add_argument("--start-num", type=int, default=1, help="Starting question number")

    ap.add_argument("--tol", type=float, default=1.0, help="Margin match tolerance (pt)")
    ap.add_argument(
        "--top-frac",
        type=float,
        default=DEFAULT_TOP_FRAC,
        help="Top fraction for the column window (default: 0.10)",
    )
    ap.add_argument(
        "--bottom-frac",
        type=float,
        default=DEFAULT_BOTTOM_FRAC,
        help="Bottom fraction for the column window (default: 0.90)",
    )
    ap.add_argument(
        "--gutter-frac",
        type=float,
        default=DEFAULT_GUTTER_FRAC,
        help="Half-width of the gutter between columns as a fraction of page width",
    )

    ap.add_argument(
        "--clip-mode",
        choices=["none", "band", "ycut"],
        default="none",
        help="Optional header clipping mode",
    )
    ap.add_argument("--margin-left", type=float, help="Left column margin offset (pt)")
    ap.add_argument("--margin-right", type=float, help="Right column margin offset (pt)")

    ap.add_argument("--chunk-preview-dir", help="Save JPEG previews of detected chunks")
    ap.add_argument("--chunk-preview-dpi", type=int, default=220)
    ap.add_argument("--chunk-preview-pad", type=float, default=2.0)
    ap.add_argument(
        "--chunk-debug-dir",
        help="Directory to dump raw and sanitized chunk text for debugging",
    )
    ap.add_argument(
        "--failed-chunk-log-chars",
        type=int,
        default=240,
        help="Max characters of chunk text to include in skip warnings",
    )

    ap.add_argument(
        "--easyocr-dpi", type=int, default=800, help="Rendering DPI for EasyOCR"
    )
    ap.add_argument(
        "--easyocr-langs",
        default="ko,en",
        help="Comma-separated EasyOCR language codes",
    )
    ap.add_argument(
        "--easyocr-gpu",
        action="store_true",
        help="Enable GPU acceleration for EasyOCR if available",
    )
    ap.add_argument(
        "--easyocr-column-pad-x",
        type=float,
        default=2.0,
        help="Horizontal padding (pt) added to each column crop before OCR",
    )
    ap.add_argument(
        "--easyocr-column-pad-top",
        type=float,
        default=2.0,
        help="Top padding (pt) added to each column crop before OCR",
    )
    ap.add_argument(
        "--easyocr-column-pad-bottom",
        type=float,
        default=60.0,
        help="Bottom padding (pt) added to each column crop before OCR",
    )
    ap.add_argument(
        "--easyocr-column-filter-top",
        type=float,
        default=4.0,
        help="Extra tolerance (pt) above the column window when keeping OCR words",
    )
    ap.add_argument(
        "--easyocr-column-filter-bottom",
        type=float,
        default=60.0,
        help="Extra tolerance (pt) below the column window when keeping OCR words",
    )

    ap.add_argument(
        "--raster-pdf-out",
        help="Path or directory to save a rasterized copy of each processed PDF",
    )
    ap.add_argument(
        "--raster-pdf-dpi",
        type=int,
        help="DPI to use for raster PDF export (defaults to EasyOCR DPI)",
    )

    ap.add_argument(
        "--searchable-pdf-out",
        help="Path or directory to save a searchable OCR PDF for each input",
    )
    ap.add_argument(
        "--searchable-pdf-dpi",
        type=int,
        help="DPI to use when rasterizing pages for the searchable PDF layer",
    )
    ap.add_argument(
        "--searchable-pdf-font",
        help=(
            "Font name (CID) or path to a TTF/OTF file for the searchable PDF text layer"
        ),
    )
    ap.add_argument(
        "--bbox-overlay-pdf-out",
        help="Path or directory to save a PDF with OCR bounding boxes overlaid",
    )
    ap.add_argument(
        "--bbox-overlay-pdf-dpi",
        type=int,
        help="DPI to use for the bounding-box overlay PDF (defaults to EasyOCR DPI)",
    )

    ap.add_argument(
        "--text-out",
        help="Path or directory to save a combined plain-text export of parsed QAs",
    )

    ap.add_argument(
        "--crop-report-out",
        help="Path or directory to save a text report describing each column crop",
    )

    ap.add_argument(
        "--expected-question-count",
        type=int,
        help="Expected number of QA items; used for validation reporting",
    )
    ap.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help=(
            "Exit with an error if validation finds missing, duplicate, out-of-order, or mismatched question counts"
        ),
    )

    ap.add_argument(
        "--margin-ui",
        action="store_true",
        help="Launch an interactive preview to pick left/right margins",
    )
    ap.add_argument(
        "--margin-ui-page",
        type=int,
        default=1,
        help="1-indexed page to preview when choosing margins",
    )
    ap.add_argument(
        "--margin-ui-dpi",
        type=int,
        default=200,
        help="Rendering DPI for the margin preview window",
    )

    ap.add_argument(
        "--log-level",
        default=_parse_log_level("INFO"),
        type=_parse_log_level,
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    ap.add_argument(
        "--heartbeat-secs",
        type=float,
        default=30.0,
        help="Seconds between progress heartbeat log messages",
    )

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info(
        "Log level set to %s",
        logging.getLevelName(args.log_level),
    )

    if args.pdf:
        pdfs = [args.pdf]
        out_paths = [args.out]
        batch_mode = False
        base_pdf = os.path.splitext(os.path.basename(args.pdf))[0]
        if args.raster_pdf_out:
            if os.path.isdir(args.raster_pdf_out) or args.raster_pdf_out.endswith(os.sep):
                target_dir = args.raster_pdf_out
                ensure_dir(target_dir)
                raster_paths = [
                    os.path.join(target_dir, f"{base_pdf}_raster.pdf")
                ]
            else:
                ensure_dir(os.path.dirname(os.path.abspath(args.raster_pdf_out)))
                raster_paths = [args.raster_pdf_out]
        else:
            raster_paths = [None]

        if args.searchable_pdf_out:
            if os.path.isdir(args.searchable_pdf_out) or args.searchable_pdf_out.endswith(os.sep):
                target_dir = args.searchable_pdf_out
                ensure_dir(target_dir)
                searchable_paths = [
                    os.path.join(target_dir, f"{base_pdf}_searchable.pdf")
                ]
            else:
                ensure_dir(
                    os.path.dirname(os.path.abspath(args.searchable_pdf_out))
                )
                searchable_paths = [args.searchable_pdf_out]
        else:
            searchable_paths = [None]

        if args.bbox_overlay_pdf_out:
            if os.path.isdir(args.bbox_overlay_pdf_out) or args.bbox_overlay_pdf_out.endswith(os.sep):
                target_dir = args.bbox_overlay_pdf_out
                ensure_dir(target_dir)
                bbox_overlay_paths = [
                    os.path.join(target_dir, f"{base_pdf}_ocr_overlay.pdf")
                ]
            else:
                ensure_dir(
                    os.path.dirname(os.path.abspath(args.bbox_overlay_pdf_out))
                )
                bbox_overlay_paths = [args.bbox_overlay_pdf_out]
        else:
            bbox_overlay_paths = [None]

        if args.text_out:
            if os.path.isdir(args.text_out) or args.text_out.endswith(os.sep):
                target_dir = args.text_out
                ensure_dir(target_dir)
                text_paths = [os.path.join(target_dir, f"{base_pdf}.txt")]
            else:
                ensure_dir(os.path.dirname(os.path.abspath(args.text_out)))
                text_paths = [args.text_out]
        else:
            text_paths = [None]

        if args.crop_report_out:
            if os.path.isdir(args.crop_report_out) or args.crop_report_out.endswith(os.sep):
                target_dir = args.crop_report_out
                ensure_dir(target_dir)
                crop_report_paths = [
                    os.path.join(target_dir, f"{base_pdf}_crop_report.txt")
                ]
            else:
                ensure_dir(os.path.dirname(os.path.abspath(args.crop_report_out)))
                crop_report_paths = [args.crop_report_out]
        else:
            crop_report_paths = [None]
    else:
        pdfs = list_pdfs(args.pdf_dir)
        if not pdfs:
            logger.error("No PDFs found in %s", args.pdf_dir)
            sys.exit(2)
        ensure_dir(args.out)
        out_paths = [
            os.path.join(
                args.out,
                os.path.splitext(os.path.basename(p))[0] + ".json",
            )
            for p in pdfs
        ]
        batch_mode = True
        logger.info("Processing %d PDFs from %s", len(pdfs), args.pdf_dir)
        if args.raster_pdf_out:
            ensure_dir(args.raster_pdf_out)
            raster_paths = [
                os.path.join(
                    args.raster_pdf_out,
                    os.path.splitext(os.path.basename(p))[0] + "_raster.pdf",
                )
                for p in pdfs
            ]
        else:
            raster_paths = [None] * len(pdfs)

        if args.searchable_pdf_out:
            ensure_dir(args.searchable_pdf_out)
            searchable_paths = [
                os.path.join(
                    args.searchable_pdf_out,
                    os.path.splitext(os.path.basename(p))[0] + "_searchable.pdf",
                )
                for p in pdfs
            ]
        else:
            searchable_paths = [None] * len(pdfs)

        if args.bbox_overlay_pdf_out:
            ensure_dir(args.bbox_overlay_pdf_out)
            bbox_overlay_paths = [
                os.path.join(
                    args.bbox_overlay_pdf_out,
                    os.path.splitext(os.path.basename(p))[0] + "_ocr_overlay.pdf",
                )
                for p in pdfs
            ]
        else:
            bbox_overlay_paths = [None] * len(pdfs)

        if args.text_out:
            ensure_dir(args.text_out)
            text_paths = [
                os.path.join(
                    args.text_out,
                    os.path.splitext(os.path.basename(p))[0] + ".txt",
                )
                for p in pdfs
            ]
        else:
            text_paths = [None] * len(pdfs)

        if args.crop_report_out:
            ensure_dir(args.crop_report_out)
            crop_report_paths = [
                os.path.join(
                    args.crop_report_out,
                    os.path.splitext(os.path.basename(p))[0]
                    + "_crop_report.txt",
                )
                for p in pdfs
            ]
        else:
            crop_report_paths = [None] * len(pdfs)

    ocr_settings = OCRSettings(
        dpi=args.easyocr_dpi,
        languages=[lang.strip() for lang in args.easyocr_langs.split(",") if lang.strip()],
        gpu=args.easyocr_gpu,
        column_pad_x=args.easyocr_column_pad_x,
        column_pad_top=args.easyocr_column_pad_top,
        column_pad_bottom=args.easyocr_column_pad_bottom,
        column_filter_top=args.easyocr_column_filter_top,
        column_filter_bottom=args.easyocr_column_filter_bottom,
    )

    for (
        pdf_path,
        out_path,
        raster_path,
        searchable_path,
        bbox_overlay_path,
        text_path,
        crop_report_path,
    ) in zip(
        pdfs,
        out_paths,
        raster_paths,
        searchable_paths,
        bbox_overlay_paths,
        text_paths,
        crop_report_paths,
    ):
        logger.info("Processing %s -> %s", os.path.basename(pdf_path), out_path)
        if raster_path:
            logger.info(
                "Rasterized PDF output will be saved to %s",
                raster_path,
            )
        if searchable_path:
            logger.info(
                "Searchable OCR PDF will be saved to %s",
                searchable_path,
            )
        if bbox_overlay_path:
            logger.info(
                "OCR bounding-box overlay will be saved to %s",
                bbox_overlay_path,
            )
        if text_path:
            logger.info("QA text export will be saved to %s", text_path)
        if crop_report_path:
            logger.info(
                "Column crop report will be saved to %s", crop_report_path
            )
        debug_dir = None
        if args.chunk_debug_dir:
            base_debug = os.path.abspath(os.path.expanduser(args.chunk_debug_dir))
            if batch_mode:
                ensure_dir(base_debug)
                debug_dir = os.path.join(
                    base_debug, os.path.splitext(os.path.basename(pdf_path))[0]
                )
            else:
                debug_dir = base_debug
            ensure_dir(debug_dir)
        try:
            qa, validation = process_single_pdf(
                pdf_path=pdf_path,
                out_path=out_path,
                year=args.year,
                start_num=args.start_num,
                tol=args.tol,
                top_frac=args.top_frac,
                bottom_frac=args.bottom_frac,
                gutter_frac=args.gutter_frac,
                clip_mode=args.clip_mode,
                margin_left=args.margin_left,
                margin_right=args.margin_right,
                preview_dir=args.chunk_preview_dir,
                preview_dpi=args.chunk_preview_dpi,
                preview_pad=args.chunk_preview_pad,
                ocr_settings=ocr_settings,
                margin_ui=args.margin_ui,
                margin_ui_page=args.margin_ui_page,
                margin_ui_dpi=args.margin_ui_dpi,
                heartbeat_interval=args.heartbeat_secs,
                chunk_debug_dir=debug_dir,
                failed_chunk_log_chars=args.failed_chunk_log_chars,
                raster_pdf_path=raster_path,
                raster_pdf_dpi=args.raster_pdf_dpi,
                searchable_pdf_path=searchable_path,
                searchable_pdf_dpi=args.searchable_pdf_dpi,
                searchable_pdf_font=args.searchable_pdf_font,
                bbox_overlay_pdf_path=bbox_overlay_path,
                bbox_overlay_pdf_dpi=args.bbox_overlay_pdf_dpi,
                text_out_path=text_path,
                crop_report_path=crop_report_path,
                expected_question_count=args.expected_question_count,
                fail_on_incomplete=args.fail_on_incomplete,
            )
        except RuntimeError as exc:
            logger.error("Validation failed for %s: %s", pdf_path, exc)
            if args.fail_on_incomplete:
                sys.exit(4)
            raise
        logger.info("Completed %s with %d QA items", os.path.basename(pdf_path), len(qa))
        if args.expected_question_count and validation.count_mismatch():
            logger.warning(
                "Expected %d QA items but parsed %d from %s",
                args.expected_question_count,
                len(qa),
                os.path.basename(pdf_path),
            )

    if batch_mode:
        logger.info("Batch finished")


if __name__ == "__main__":
    main()
