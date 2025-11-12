#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, json, argparse, tempfile, unicodedata, copy
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pdfplumber
from source_drop import process_data

# =========================
# GUI / HiDPI
# =========================
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

def set_win_dpi_awareness():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            try:
                ctypes.OleDLL('shcore').SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# =========================
# Year inference
# =========================
def infer_year_from_filename(path: str) -> Optional[int]:
    fname = os.path.basename(path)
    m = re.search(r"(\d{2})년", fname)
    if m:
        yy = int(m.group(1))
        return 2000 + yy
    m = re.search(r"(20\d{2}|19\d{2})", fname)
    if m:
        return int(m.group(1))
    return None

# =========================
# Helpers
# =========================
def list_pdfs(folder: str) -> List[str]:
    try:
        items = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder, f))]
        )
    except Exception:
        items = []
    return items

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _normalize_visible_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[\u00A0\u2000-\u200B]", " ", s)
    s = s.replace("ㆍ","·").replace("∙","·").replace("・","·").replace("•","·")
    return s

def _norm_header_token(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[\u00A0\u2000-\u200B]", " ", s)
    s = re.sub(r"[【】()\[\]{}·･•※〈〉《》『』—–\-:·•\s]+", "", s)
    return s

def _norm_collapse(s: str) -> str:
    s = _normalize_visible_text(s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[【】()\[\]{}·･•※〈〉《》『』—–\-:·•]", "", s)
    return s

# =========================
# Config / Regex
# =========================
SUBJECT_CANDIDATES = ["경찰학개론","헌법","형사법","경찰학","형법","형사소송법","경찰학개론"]
SUBJECT_CANDIDATES = list(dict.fromkeys(SUBJECT_CANDIDATES))
CANDIDATE_SET = {_norm_header_token(x): x for x in SUBJECT_CANDIDATES}
CANDIDATE_COLLAPSED = {_norm_collapse(x): x for x in SUBJECT_CANDIDATES}

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

OPT_SPLIT_RE = re.compile(rf"(?=([{OPTION_CLASS}]))")
CIRCLED_STRIP_RE = re.compile(rf"^[{OPTION_CLASS}]\s*")

QUESTION_START_LINE_RE = re.compile(
    rf"^\s*(?:[{QUESTION_CIRCLED_RANGE}]|[0-9]{{1,3}}[.)]|제\s*[0-9]{{1,3}}\s*문)",
    re.MULTILINE,
)
QUESTION_NUM_RE = re.compile(r"^\s*(?:\(\s*(\d{1,3})\s*\)|(\d{1,3})\s*번|(\d{1,3}))\s*[.)]?\s*")

DISPUTE_RE = re.compile(
    r"\(?\s*다툼이\s*(?:있는\s*경우|있으면)\s*(?P<site>[^)\n]*?)\s*(?:판례|결정)\s*에\s*의함\)?",
    re.IGNORECASE,
)

HEADER_PAT = re.compile(r"【\s*(?P<subject>[^】]+)\s*】")
TARGET_PAT = re.compile(r"\((?P<target>[^)]+)\)")

LEADING_HEADER_STRIP = re.compile(r"^\s*(?:[【\[]\s*[^】\]]+\s*[】\]])\s*(?:\([^)]*\))?\s*")

def _strip_header_garbage(text: str) -> str:
    return norm_space(LEADING_HEADER_STRIP.sub("", text or ""))

def _clean_target_text(raw: str) -> str:
    return norm_space(_normalize_visible_text(raw))

# =========================
# Dispute parsing
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

# =========================
# Header detection (anywhere) + bbox
# =========================
def _group_words_by_line_with_bbox(words, y_tol=2.0):
    if not words: return []
    ws = sorted(words, key=lambda w: (round(w.get("top", 0.0), 1), w.get("x0", 0.0)))
    lines, cur, prev_top = [], [], None
    for w in ws:
        t = float(w.get("top", 0.0))
        if prev_top is None or abs(t - prev_top) <= y_tol:
            cur.append(w)
        else:
            lines.append(cur); cur = [w]
        prev_top = t
    if cur: lines.append(cur)
    out = []
    for ln in lines:
        text = " ".join(w.get("text","") for w in ln)
        x0 = min(w["x0"] for w in ln); x1 = max(w["x1"] for w in ln)
        top = min(w["top"] for w in ln); bottom = max(w["bottom"] for w in ln)
        out.append({"text": text, "x0": float(x0), "x1": float(x1),
                    "top": float(top), "bottom": float(bottom), "words": ln})
    return out

def _bbox_union(a, b):
    if a is None: return b
    return {
        "x0": min(a["x0"], b["x0"]),
        "y0": min(a["y0"], b["y0"]),
        "y1": max(a["y1"], b["y1"]),
        "x1": max(a["x1"], b["x1"]),
    }

def detect_header_bbox_subject_target_anywhere(page: pdfplumber.page.Page, y_tol=2.0):
    working = page.rotate(-page.rotation) if getattr(page, "rotation", 0) else page
    words = working.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False) or []
    lines = _group_words_by_line_with_bbox(words, y_tol=y_tol)
    if not lines:
        return ("none", None, None, None)

    best = None
    for i, ln in enumerate(lines):
        raw = _normalize_visible_text(ln["text"])
        m = HEADER_PAT.search(raw)
        subj = None; kind = "none"; target = None; bbox = None

        if m:
            inside = m.group("subject").strip()
            normed = _norm_header_token(inside)
            if normed in CANDIDATE_SET:
                kind = "known"; subj = CANDIDATE_SET[normed]
            else:
                kind = "unknown"
            right = raw[m.end():]
            mt = TARGET_PAT.search(right)
            bbox = {"x0": ln["x0"], "y0": ln["top"], "x1": ln["x1"], "y1": ln["bottom"]}
            if mt:
                target = _clean_target_text(mt.group("target"))
            else:
                if i+1 < len(lines) and lines[i+1]["text"].lstrip().startswith("("):
                    mtn = re.search(r"\(([^)]*)\)", lines[i+1]["text"])
                    if mtn:
                        target = _clean_target_text(mtn.group(1))
                        bbox = _bbox_union(bbox, {"x0": lines[i+1]["x0"], "y0": lines[i+1]["top"],
                                                  "x1": lines[i+1]["x1"], "y1": lines[i+1]["bottom"]})
        else:
            collapsed = _norm_collapse(ln["text"])
            if collapsed in CANDIDATE_COLLAPSED:
                kind = "known"; subj = CANDIDATE_COLLAPSED[collapsed]
                bbox = {"x0": ln["x0"], "y0": ln["top"], "x1": ln["x1"], "y1": ln["bottom"]}
                if i+1 < len(lines) and lines[i+1]["text"].lstrip().startswith("("):
                    mtn = re.search(r"\(([^)]*)\)", lines[i+1]["text"])
                    if mtn:
                        target = _clean_target_text(mtn.group(1))
                        bbox = _bbox_union(bbox, {"x0": lines[i+1]["x0"], "y0": lines[i+1]["top"],
                                                  "x1": lines[i+1]["x1"], "y1": lines[i+1]["bottom"]})
        if kind != "none":
            cand = (ln["top"], kind, subj, target, bbox)
            if best is None or cand[0] < best[0]:
                best = cand

    if best is None:
        return ("none", None, None, None)
    _, kind, subj, target, bbox = best
    if kind == "known":
        return ("known", subj, target, bbox)
    else:
        return ("unknown", None, target, bbox)

# =========================
# Header clip helpers
# =========================
def header_clip_ycut(page: pdfplumber.page.Page, pad: float = 4.0) -> Optional[float]:
    kind, subj, target, bbox = detect_header_bbox_subject_target_anywhere(page)
    if bbox:
        return float(bbox["y1"] + pad)
    return None

def header_clip_band(page: pdfplumber.page.Page,
                     band_pt: float = 10.0,
                     xpad: float = 2.0) -> Optional[Tuple[float, float, float, float]]:
    """
    Signed band clip anchored to the header *bottom* (y1):
      - band_pt > 0  => mask downward from header bottom by band_pt
      - band_pt < 0  => mask upward   from header bottom by abs(band_pt)
    """
    kind, subj, target, bbox = detect_header_bbox_subject_target_anywhere(page)
    if not bbox:
        return None

    hx0 = float(bbox["x0"])
    hx1 = float(bbox["x1"])
    hbot = float(bbox["y1"])

    x0 = max(0.0, hx0 - max(0.0, xpad))
    x1 = min(float(page.width), hx1 + max(0.0, xpad))

    if band_pt >= 0:
        y0 = hbot
        y1 = min(float(page.height), hbot + band_pt)
    else:
        h = abs(band_pt)
        y0 = max(0.0, hbot - h)
        y1 = hbot
    return (x0, y0, x1, y1)

# =========================
# Subject/target inheritance & skipping
# =========================
def detect_subject_target_by_page(pdf: pdfplumber.PDF) -> Dict[int, dict]:
    hits = {}
    for i, pg in enumerate(pdf.pages, start=1):
        kind, subj, target, _ = detect_header_bbox_subject_target_anywhere(pg)
        hits[i] = {"kind": kind, "subject": subj, "target": target}
    return hits

def subject_inheritance_map(hits: Dict[int, dict]) -> Dict[int, Optional[str]]:
    cur = None; out = {}
    for p in sorted(hits.keys()):
        if hits[p]["kind"] == "known": cur = hits[p]["subject"]
        out[p] = cur
    return out

def target_inheritance_map(hits: Dict[int, dict]) -> Dict[int, Optional[str]]:
    cur = None; out = {}
    for p in sorted(hits.keys()):
        if hits[p]["kind"] == "known":
            cur = hits[p]["target"] if hits[p]["target"] else None
        out[p] = cur
    return out

def compute_skip_pages(hits: dict[int, dict]) -> set[int]:
    """
    Skip policy:
      - BEFORE first 'known': skip all pages (kind in {'none','unknown'}).
      - IN_SUBJECT (after a 'known'): keep 'none'; if 'unknown' appears,
        start SKIPPING_UNKNOWN (skip 'unknown' and subsequent 'none' pages)
        UNTIL the next 'known', then resume IN_SUBJECT.
    """
    SKIP_BEFORE_KNOWN = 0
    IN_SUBJECT        = 1
    SKIPPING_UNKNOWN  = 2

    state = SKIP_BEFORE_KNOWN
    skip: set[int] = set()

    for p in sorted(hits):
        kind = hits[p]["kind"]  # 'known' | 'unknown' | 'none'

        if state == SKIP_BEFORE_KNOWN:
            if kind == "known":
                state = IN_SUBJECT  # keep this page
            else:
                skip.add(p)         # skip 'none'/'unknown' before first known

        elif state == IN_SUBJECT:
            if kind == "known":
                # stay IN_SUBJECT; keep this page
                pass
            elif kind == "unknown":
                # start skipping until next known
                skip.add(p)
                state = SKIPPING_UNKNOWN
            else:  # kind == 'none'
                # keep 'none' while in-subject
                pass

        elif state == SKIPPING_UNKNOWN:
            if kind == "known":
                state = IN_SUBJECT  # resume keeping from this known page
            else:
                # skip both 'unknown' and 'none' while we're in the gap
                skip.add(p)

    return skip


# =========================
# Layout & chunking
# =========================
def two_col_bboxes(page, top_frac=0.04, bottom_frac=0.96, gutter_frac=0.005):
    w, h = float(page.width), float(page.height)
    top = h * top_frac
    bottom = h * bottom_frac
    gutter = w * gutter_frac
    mid = w * 0.5
    return (0.0, top,  mid - gutter, bottom), (mid + gutter, top, w, bottom)

def extract_words_in_bbox(page, bbox, x_tol=3, y_tol=3):
    sub = page.within_bbox(bbox)
    return sub.extract_words(x_tolerance=x_tol, y_tolerance=y_tol, keep_blank_chars=False, use_text_flow=True) or []

def _rects_intersect(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    return (ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0)

def group_words_into_lines(words: List[Dict[str, Any]], y_tol=3.0):
    if not words: return []
    ws = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines, cur, cur_top = [], [], None
    for w in ws:
        if cur_top is None or abs(w["top"] - cur_top) <= y_tol:
            cur.append(w); cur_top = w["top"] if cur_top is None else cur_top
        else:
            cur.sort(key=lambda x: x["x0"]); lines.append(cur)
            cur = [w]; cur_top = w["top"]
    if cur:
        cur.sort(key=lambda x: x["x0"]); lines.append(cur)
    return lines

def extract_lines_in_bbox(page, bbox, y_tol=3.0,
                          y_cut: Optional[float] = None,
                          drop_zone: Optional[Tuple[float,float,float,float]] = None):
    x0b, y0b, x1b, y1b = bbox
    if y_cut is not None:
        y0b = max(y0b, y_cut)
        bbox = (x0b, y0b, x1b, y1b)

    words = extract_words_in_bbox(page, bbox, x_tol=3, y_tol=3)
    words_slim = []
    for w in words:
        wx0, wx1, wt, wb, tx = w.get("x0"), w.get("x1"), w.get("top"), w.get("bottom"), w.get("text", "")
        if None in (wx0, wx1, wt, wb): continue
        if not (y0b <= wt <= y1b or y0b <= wb <= y1b): continue
        if drop_zone is not None and _rects_intersect((float(wx0), float(wt), float(wx1), float(wb)), drop_zone):
            continue
        words_slim.append({"x0": float(wx0), "x1": float(wx1), "top": float(wt), "bottom": float(wb), "text": tx})

    lines = group_words_into_lines(words_slim, y_tol=y_tol)
    out = []
    for ln in lines:
        lx0 = min(w["x0"] for w in ln); lx1 = max(w["x1"] for w in ln)
        top = min(w["top"] for w in ln); bot = max(w["bottom"] for w in ln)
        txt = " ".join(w["text"] for w in ln if w.get("text"))
        out.append({"x0": lx0, "x1": lx1, "top": top, "bottom": bot, "y": 0.5*(top+bot), "text": txt})
    return out

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


def detect_question_starts(lines, margin_abs: Optional[float], col_left: float,
                           tol=1.0, last_qnum: Optional[int] = None):
    starts = []
    target_rel = None if margin_abs is None else (margin_abs - col_left)
    current_last = last_qnum
    for i, ln in enumerate(lines):
        raw_text = (ln.get("text") or "")
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
        seq_ok = qnum is not None and (current_last is None or qnum == current_last + 1)
        if left_ok or seq_ok:
            starts.append(i)
            if qnum is not None:
                current_last = qnum
    return starts, current_last

def build_flow_segments(pdf, top_frac, bottom_frac, gutter_frac, y_tol,
                        clip_mode: str,
                        ycut_map: Dict[int, Optional[float]],
                        band_map: Dict[int, Optional[Tuple[float,float,float,float]]]):
    segs = []
    for i, page in enumerate(pdf.pages):
        L, R = two_col_bboxes(page, top_frac, bottom_frac, gutter_frac)
        ycut = ycut_map.get(i+1) if clip_mode == "ycut" else None
        band = band_map.get(i+1) if clip_mode == "band" else None
        L_lines = extract_lines_in_bbox(page, L, y_tol=y_tol, y_cut=ycut, drop_zone=band)
        R_lines = extract_lines_in_bbox(page, R, y_tol=y_tol, y_cut=ycut, drop_zone=band)
        segs.append((i, 'L', L, L_lines))
        segs.append((i, 'R', R, R_lines))
    return segs

def flow_chunk_all_pages(pdf, L_rel_offset, R_rel_offset, y_tol, tol, top_frac, bottom_frac, gutter_frac,
                         clip_mode: str,
                         ycut_map: Dict[int, Optional[float]],
                         band_map: Dict[int, Optional[Tuple[float,float,float,float]]]):
    segs = build_flow_segments(pdf, top_frac, bottom_frac, gutter_frac, y_tol, clip_mode, ycut_map, band_map)

    seg_meta = []
    for (pi, col, bbox, lines) in segs:
        L, R = two_col_bboxes(pdf.pages[pi], top_frac, bottom_frac, gutter_frac)
        if col == 'L':
            margin_abs = None if L_rel_offset is None else (L[0] + L_rel_offset)
            col_left = L[0]
        else:
            margin_abs = None if R_rel_offset is None else (R[0] + R_rel_offset)
            col_left = R[0]
        seg_meta.append((pi, col, bbox, lines, margin_abs, col_left))

    seg_starts = []
    last_detected_qnum = None
    for (pi, col, bbox, lines, m_abs, col_left) in seg_meta:
        starts, last_detected_qnum = detect_question_starts(
            lines, m_abs, col_left, tol=tol, last_qnum=last_detected_qnum
        )
        seg_starts.append(starts)

    chunks, current = [], None
    for seg_idx, (pi, col, bbox, lines, m_abs, col_left) in enumerate(seg_meta):
        starts = set(seg_starts[seg_idx]); i = 0
        while i < len(lines):
            if i in starts:
                if current is not None:
                    chunks.append(current)
                current = {"pieces": [], "start": {"page": pi, "col": col, "line_idx": i}}
            if current is not None:
                next_mark = min((j for j in starts if j > i), default=None)
                end_idx = (next_mark-1) if next_mark is not None else (len(lines)-1)
                block = lines[i:end_idx+1]
                if block:
                    x0 = min(l["x0"] for l in block); x1 = max(l["x1"] for l in block)
                    top = min(l["top"] for l in block) - 2.0; bot = max(l["bottom"] for l in block) + 2.0
                    text = "\n".join(l["text"] for l in block)
                    current["pieces"].append({
                        "page": pi, "col": col,
                        "box": {"x0": x0, "x1": x1, "top": top, "bottom": bot},
                        "start_line": i, "end_line": end_idx, "text": text
                    })
                i = end_idx + 1
            else:
                i += 1
    if current is not None:
        chunks.append(current)

    per_page_boxes = {i: [] for i in range(len(pdf.pages))}
    for ch_id, ch in enumerate(chunks, start=1):
        for p in ch.get("pieces", []):
            b = p["box"].copy()
            b["chunk_id"] = ch_id
            b["col"] = p["col"]
            per_page_boxes[p["page"]].append(b)

    return chunks, per_page_boxes

# =========================
# QA extraction
# =========================
def extract_leading_qnum_and_clean(stem: str):
    if not stem: return None, stem
    m = QUESTION_NUM_RE.match(stem)
    if not m: return None, stem
    digits = next((g for g in m.groups() if g), None)
    try: qnum = int(digits) if digits is not None else None
    except Exception: qnum = None
    return qnum, stem[m.end():].lstrip()

def _trim_to_first_question(text: str) -> Tuple[str, Optional[int]]:
    if not text:
        return text, None
    m = QUESTION_START_LINE_RE.search(text)
    if not m:
        return text, None
    trimmed = text[m.start():]
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
        candidate_slice = text[match.start():]
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
        if num >= 1000:  # likely a date/year, skip trimming
            continue
        if num == target_next:
            return text[:match.start()].rstrip()

    return text

def extract_qa_from_chunk_text(text: str):
    if not text: return None, None, False, None, None
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
            raw_txt = parts[i+1] if (i+1) < len(parts) else ""
            clean_txt = norm_space(CIRCLED_STRIP_RE.sub("", raw_txt))
            options.append({"index": sym[0], "text": clean_txt})
            i += 2
        else:
            i += 1
    options = [o for o in options if o["index"] in OPTION_SET]
    if not options:
        return None, None, dispute, dispute_site, detected_qnum

    return stem, options, dispute, dispute_site, detected_qnum

# =========================
# Chunk preview images
# =========================
def save_chunk_preview(page, bbox, preview_dir, page_index, column_tag, chunk_idx_in_column,
                       global_idx, dpi=220, pad=2.0):
    if not bbox or not preview_dir: return None
    abs_dir = os.path.abspath(os.path.expanduser(preview_dir))
    os.makedirs(abs_dir, exist_ok=True)
    width, height = float(page.width), float(page.height)
    x0, top, x1, bottom = map(float, bbox)
    pad = max(0.0, float(pad))
    padded = (max(0.0, x0 - pad), max(0.0, top - pad),
              min(width, x1 + pad), min(height, bottom + pad))
    cropped = page.within_bbox(padded)
    img = cropped.to_image(resolution=int(dpi))
    pil = img.original.convert("RGB")
    del img
    fn = f"p{page_index:03d}_{column_tag}{chunk_idx_in_column:02d}_{global_idx:04d}.jpg"
    out_path = os.path.join(abs_dir, fn)
    pil.save(out_path, format="JPEG", quality=90)
    pil.close()
    return os.path.abspath(out_path)

# =========================
# Top-level parse
# =========================
def pdf_to_qa_flow_chunks(pdf_path: str,
                          subject_default: str,
                          target_default: str,
                          year: int,
                          start_num: int,
                          L_rel: Optional[float],
                          R_rel: Optional[float],
                          tol: float,
                          top_frac=0.04,
                          bottom_frac=0.96,
                          gutter_frac=0.005,
                          y_tol=3.0,
                          clip_mode: str = "band",
                          header_y_pad: float = 4.0,
                          header_band_pt: float = 10.0,
                          header_band_xpad: float = 2.0,
                          chunk_preview_dir: Optional[str] = None,
                          chunk_preview_dpi: int = 220,
                          chunk_preview_pad: float = 2.0):
    out = []
    last_assigned_qno = start_num - 1
    global_idx = 0
    preview_dir = os.path.abspath(os.path.expanduser(chunk_preview_dir)) if chunk_preview_dir else None

    with pdfplumber.open(pdf_path) as pdf:
        hits = detect_subject_target_by_page(pdf)
        inh_subject = subject_inheritance_map(hits)
        inh_target  = target_inheritance_map(hits)
        skip_pages  = compute_skip_pages(hits)

        # build clip maps
        ycut_map = {}
        band_map = {}
        for i, pg in enumerate(pdf.pages, start=1):
            if clip_mode == "ycut":
                ycut_map[i] = header_clip_ycut(pg, pad=header_y_pad)
            elif clip_mode == "band":
                band_map[i] = header_clip_band(pg, band_pt=header_band_pt, xpad=header_band_xpad)

        chunks, _ = flow_chunk_all_pages(
            pdf, L_rel, R_rel, y_tol, tol,
            top_frac, bottom_frac, gutter_frac,
            clip_mode=clip_mode, ycut_map=ycut_map, band_map=band_map
        )

        for ch in chunks:
            pieces = ch.get("pieces") or []
            if not pieces: continue
            p1 = pieces[0]["page"] + 1
            if p1 in skip_pages:
                continue

            expected_next = last_assigned_qno + 1 if last_assigned_qno is not None else None
            text = "\n".join(p["text"] for p in pieces if p.get("text"))
            text = sanitize_chunk_text(text, expected_next)
            stem, options, dispute, dispute_site, detected_qnum = extract_qa_from_chunk_text(text)
            if stem is None or not options:
                continue

            subj = inh_subject.get(p1) or subject_default
            targ = inh_target.get(p1)  or target_default
            if detected_qnum is not None:
                qno = detected_qnum
            elif expected_next is not None:
                qno = expected_next
            else:
                qno = start_num
            global_idx += 1

            preview_path = None
            if preview_dir:
                b = pieces[0]["box"]; bbox = (b["x0"], b["top"], b["x1"], b["bottom"])
                page = pdf.pages[pieces[0]["page"]]
                preview_path = save_chunk_preview(page, bbox, preview_dir, p1,
                                                  pieces[0]["col"], 1, global_idx,
                                                  dpi=chunk_preview_dpi, pad=chunk_preview_pad)

            out.append({
                "subject": subj,
                "year": year,
                "target": targ,
                "content": {
                    "question_number": qno,
                    "question_text": stem,
                    "dispute_bool": bool(dispute),
                    "dispute_site": dispute_site,
                    "options": options,
                    "source": {"pieces": pieces, "start_page": p1},
                    **({"preview_image": preview_path} if preview_path else {})
                }
            })

            last_assigned_qno = qno

    return out

# =========================
# Auto per-file margin detection
# =========================
def _auto_detect_margins_for_pdf(pdf_path: str,
                                 top_frac: float,
                                 bottom_frac: float,
                                 gutter_frac: float) -> Tuple[Optional[float], Optional[float]]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pg in pdf.pages:
                Lb, Rb = two_col_bboxes(pg, top_frac, bottom_frac, gutter_frac)
                def first_x(b):
                    sub = pg.within_bbox(b)
                    xs = [c["x0"] for c in (sub.chars or []) if c.get("x0") is not None]
                    if not xs:
                        ws = extract_words_in_bbox(pg, b)
                        xs = [w["x0"] for w in ws if w.get("x0") is not None]
                    return min(xs) if xs else None
                lx, rx = first_x(Lb), first_x(Rb)
                if lx is not None and rx is not None:
                    return float(lx - Lb[0]), float(rx - Rb[0])
    except Exception:
        pass
    return None, None

# =========================
# GUI App (with Prev/Next FILE)
# =========================
class MarginPreviewApp(tk.Tk):
    def __init__(self, pdf_path: str, out_path: str, subject_default: str, target_default: str, year: int,
                 dpi=300, tol=1.0, top_frac=0.04, bottom_frac=0.96, gutter_frac=0.005,
                 clip_mode="band", header_y_pad=4.0, header_band_pt=10.0, header_band_xpad=2.0,
                 chunk_preview_dir: Optional[str] = None, chunk_preview_dpi: int = 220, chunk_preview_pad: float = 2.0,
                 pdf_dir: Optional[str] = None):
        super().__init__()
        self.title("QA Parser (auto per-file margins • anywhere-header • band/y-cut)")
        self.geometry("1280x920"); self.minsize(900, 700)

        self.pdf_path = pdf_path
        self.pdf_dir = pdf_dir  # enables batch + file navigation
        self.out_path = out_path
        self.year = year
        self.subject_default = subject_default
        self.target_default = target_default

        self.dpi_var = tk.IntVar(value=int(dpi))
        self.tol = tk.DoubleVar(value=float(tol))
        self.top_frac = tk.DoubleVar(value=float(top_frac))
        self.bottom_frac = tk.DoubleVar(value=float(bottom_frac))
        self.gutter_frac = tk.DoubleVar(value=float(gutter_frac))

        self.clip_mode = tk.StringVar(value=clip_mode)  # "band"|"ycut"|"none"
        self.header_y_pad = tk.DoubleVar(value=float(header_y_pad))
        self.header_band_pt = tk.DoubleVar(value=float(header_band_pt))
        self.header_band_xpad = tk.DoubleVar(value=float(header_band_xpad))

        # GUI-assigned global margins (fallback)
        self.L_rel = None; self.R_rel = None

        self.tk_img = None; self._tmp_png = None; self.current_idx = 0

        self.chunk_preview_dir = os.path.abspath(os.path.expanduser(chunk_preview_dir)) if chunk_preview_dir else None
        self.chunk_preview_dpi = int(chunk_preview_dpi)
        self.chunk_preview_pad = float(chunk_preview_pad)

        # ----- FILE LIST + index for navigation -----
        self.file_list = list_pdfs(self.pdf_dir) if self.pdf_dir else [self.pdf_path]
        self.file_idx = 0
        if self.pdf_dir:
            try:
                self.file_idx = self.file_list.index(self.pdf_path)
            except ValueError:
                self.file_idx = 0

        # Load initial PDF
        try:
            self.pdf = pdfplumber.open(self.pdf_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF:\n{e}"); self.destroy(); return

        self.num_pages = len(self.pdf.pages)

        self.hits = detect_subject_target_by_page(self.pdf)
        self.inh_subject = subject_inheritance_map(self.hits)
        self.inh_target  = target_inheritance_map(self.hits)
        self.skip_pages  = compute_skip_pages(self.hits)

        self.header_ycut = {i+1: header_clip_ycut(pg, pad=self.header_y_pad.get()) for i, pg in enumerate(self.pdf.pages)}
        self.header_band = {i+1: header_clip_band(pg, band_pt=self.header_band_pt.get(),
                                                  xpad=self.header_band_xpad.get())
                            for i, pg in enumerate(self.pdf.pages)}

        # ---- layout
        self.columnconfigure(0, weight=0); self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1); self.rowconfigure(1, weight=0)

        panel = ttk.Frame(self, padding=8); panel.grid(row=0, column=0, sticky="ns")
        ttk.Label(panel, text="Navigation").pack(anchor="w", pady=(0,4))
        nav = ttk.Frame(panel); nav.pack(anchor="w", pady=(0,8))
        ttk.Button(nav, text="◀ Prev page", command=self.prev_page, width=12).pack(side="left")
        ttk.Button(nav, text="Next page ▶", command=self.next_page, width=12).pack(side="left", padx=(6,0))
        # FILE NAV BUTTONS
        if self.pdf_dir:
            ttk.Button(nav, text="« Prev file", command=self.prev_file, width=12).pack(side="left", padx=(12,0))
            ttk.Button(nav, text="Next file »", command=self.next_file, width=12).pack(side="left", padx=(6,0))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(panel, text="Column crop (fractions)").pack(anchor="w", pady=(0,4))
        cf = ttk.Frame(panel); cf.pack(anchor="w", pady=(0,8))
        ttk.Label(cf, text="Top").grid(row=0, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.top_frac, width=7).grid(row=0, column=1, padx=(4,0))
        ttk.Label(cf, text="Bottom").grid(row=1, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.bottom_frac, width=7).grid(row=1, column=1, padx=(4,0))
        ttk.Label(cf, text="Gutter").grid(row=2, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.gutter_frac, width=7).grid(row=2, column=1, padx=(4,0))
        ttk.Button(cf, text="Apply crop", command=self.refresh).grid(row=3, column=0, columnspan=2, pady=(6,0), sticky="ew")

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(panel, text="Render DPI").pack(anchor="w", pady=(0,4))
        dpi_row = ttk.Frame(panel); dpi_row.pack(anchor="w", pady=(0,8))
        ttk.Spinbox(dpi_row, from_=120, to=600, increment=24, textvariable=self.dpi_var, width=7).pack(side="left")
        ttk.Button(dpi_row, text="Re-render", command=self.refresh).pack(side="left", padx=(6,0))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(panel, text="Header clip mode").pack(anchor="w", pady=(0,4))
        modef = ttk.Frame(panel); modef.pack(anchor="w")
        ttk.Radiobutton(modef, text="Band (safe)", variable=self.clip_mode, value="band", command=self.refresh).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(modef, text="Y-cut (aggressive)", variable=self.clip_mode, value="ycut", command=self.refresh).grid(row=0, column=1, sticky="w", padx=(8,0))
        ttk.Radiobutton(modef, text="None", variable=self.clip_mode, value="none", command=self.refresh).grid(row=0, column=2, sticky="w", padx=(8,0))

        bandf = ttk.Frame(panel); bandf.pack(anchor="w", pady=(4,0))
        ttk.Label(bandf, text="Band height (pt)").grid(row=0, column=0, sticky="w")
        ttk.Entry(bandf, textvariable=self.header_band_pt, width=7).grid(row=0, column=1, padx=(4,12))
        ttk.Label(bandf, text="Band x-pad (pt)").grid(row=0, column=2, sticky="w")
        ttk.Entry(bandf, textvariable=self.header_band_xpad, width=7).grid(row=0, column=3, padx=(4,0))
        ttk.Button(bandf, text="Recalc band", command=self.recalc_band).grid(row=0, column=4, padx=(8,0))

        ycutf = ttk.Frame(panel); ycutf.pack(anchor="w", pady=(6,8))
        ttk.Label(ycutf, text="Y-cut pad (pt)").grid(row=0, column=0, sticky="w")
        ttk.Entry(ycutf, textvariable=self.header_y_pad, width=7).grid(row=0, column=1, padx=(4,0))
        ttk.Button(ycutf, text="Recalc y-cut", command=self.recalc_ycut).grid(row=0, column=2, padx=(8,0))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)

        ttk.Label(panel, text="Margin tolerance (pt)").pack(anchor="w", pady=(0,4))
        ttk.Entry(panel, textvariable=self.tol, width=8).pack(anchor="w")
        self.marg_label = ttk.Label(panel, text="L: —   R: —"); self.marg_label.pack(anchor="w", pady=(8,2))
        ttk.Button(panel, text="Use this page’s AUTO margins for ALL pages",
                   command=self.assign_rel_margins, width=28).pack(anchor="w")
        ttk.Button(panel, text="Parse → QA JSON",
                   command=self.parse_all_single, width=28).pack(anchor="w", pady=(10,0))

        if self.pdf_dir:
            ttk.Button(panel, text="Parse Folder (auto per-file margins) → JSONs",
                       command=self.parse_folder_auto, width=28).pack(anchor="w", pady=(6,0))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(panel, text="Output").pack(anchor="w")
        self.out_path_label = ttk.Label(panel, text=os.path.abspath(out_path), wraplength=220)
        self.out_path_label.pack(anchor="w")

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(panel, text="Page detection").pack(anchor="w")
        self.page_subj_label = ttk.Label(panel, text=""); self.page_subj_label.pack(anchor="w")

        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda e: self.render_page())

        status = ttk.Frame(self, padding=8); status.grid(row=1, column=0, columnspan=2, sticky="ew")
        status.columnconfigure(0, weight=1)
        self.info = ttk.Label(status, text=""); self.info.grid(row=0, column=0, sticky="w")

        # Shortcuts
        self.bind("<Left>", lambda e: self.prev_page())
        self.bind("<Right>", lambda e: self.next_page())
        if self.pdf_dir:
            self.bind("<Control-Shift-Left>",  lambda e: self.prev_file())
            self.bind("<Control-Shift-Right>", lambda e: self.next_file())

        self.render_page()

    # --------- FILE loader & navigation ----------
    def _load_pdf(self, new_path: str):
        try:
            self.pdf.close()
        except Exception:
            pass
        self.pdf_path = new_path
        self.current_idx = 0

        try:
            self.pdf = pdfplumber.open(self.pdf_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF:\n{e}")
            return
        self.num_pages = len(self.pdf.pages)

        self.hits = detect_subject_target_by_page(self.pdf)
        self.inh_subject = subject_inheritance_map(self.hits)
        self.inh_target  = target_inheritance_map(self.hits)
        self.skip_pages  = compute_skip_pages(self.hits)
        self.header_ycut = {i+1: header_clip_ycut(pg, pad=self.header_y_pad.get()) for i, pg in enumerate(self.pdf.pages)}
        self.header_band = {i+1: header_clip_band(pg, band_pt=self.header_band_pt.get(),
                                                  xpad=self.header_band_xpad.get())
                            for i, pg in enumerate(self.pdf.pages)}
        self.render_page()

    def prev_file(self):
        if not self.file_list: return
        self.file_idx = (self.file_idx - 1) % len(self.file_list)
        self._load_pdf(self.file_list[self.file_idx])

    def next_file(self):
        if not self.file_list: return
        self.file_idx = (self.file_idx + 1) % len(self.file_list)
        self._load_pdf(self.file_list[self.file_idx])

    # --------- GUI helpers ----------
    def page_bboxes_and_margins(self):
        page = self.pdf.pages[self.current_idx]
        Lbbox, Rbbox = two_col_bboxes(page, self.top_frac.get(), self.bottom_frac.get(), self.gutter_frac.get())
        def auto(b):
            sub = page.within_bbox(b)
            xs = [c["x0"] for c in (sub.chars or []) if c.get("x0") is not None]
            if not xs:
                ws = extract_words_in_bbox(page, b)
                xs = [w["x0"] for w in ws if w.get("x0") is not None]
            return min(xs) if xs else None
        return page, Lbbox, Rbbox, auto(Lbbox), auto(Rbbox)

    def render_page(self):
        page, Lb, Rb, L_auto, R_auto = self.page_bboxes_and_margins()
        im = page.to_image(resolution=int(self.dpi_var.get()))

        mode = self.clip_mode.get()
        if mode == "ycut":
            ycut = self.header_ycut.get(self.current_idx + 1)
            if ycut is not None:
                im.draw_line([(0, ycut), (float(page.width), ycut)],
                             stroke="#ffbf00", stroke_width=4)
        elif mode == "band":
            band = self.header_band.get(self.current_idx + 1)
            if band is not None:
                x0, y0, x1, y1 = band
                im.draw_rect((x0, y0, x1, y1), stroke="#ffbf00", stroke_width=3)
                im.draw_rect((x0+1, y0+1, x1-1, y1-1), stroke="#ffdf66", stroke_width=2)

        for (x0, top, x1, bottom) in (Lb, Rb):
            im.draw_rect((x0, top, x1, bottom), stroke="#66B2FF", stroke_width=3)

        if L_auto is not None:
            im.draw_line([(Lb[0], Lb[1] + 8), (L_auto, Lb[1] + 8)], stroke="orange", stroke_width=6)
        if R_auto is not None:
            im.draw_line([(Rb[0], Rb[1] + 8), (R_auto, Rb[1] + 8)], stroke="orange", stroke_width=6)
        if self.L_rel is not None:
            im.draw_line([(Lb[0], Lb[1] + 20), (Lb[0] + self.L_rel, Lb[1] + 20)], stroke="red", stroke_width=5)
        if self.R_rel is not None:
            im.draw_line([(Rb[0], Rb[1] + 20), (Rb[0] + self.R_rel, Rb[1] + 20)], stroke="red", stroke_width=5)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False); path = tmp.name; tmp.close()
        im.save(path)
        with Image.open(path) as _pil:
            pil = _pil.copy()
        if getattr(self, "_tmp_png", None) and os.path.exists(self._tmp_png):
            try: os.remove(self._tmp_png)
            except PermissionError: pass
        self._tmp_png = path

        cw = max(100, self.canvas.winfo_width()); ch = max(100, self.canvas.winfo_height())
        scale = min(cw / pil.width, ch / pil.height, 1.0)
        tw = max(1, int(round(pil.width * scale))); th = max(1, int(round(pil.height * scale)))
        if (tw, th) != (pil.width, pil.height):
            pil = pil.resize((tw, th), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all"); self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img)

        p = self.current_idx + 1
        det = self.hits.get(p, {"kind": "none", "subject": None, "target": None})
        inh_s = self.inh_subject.get(p) or self.subject_default
        inh_t = self.inh_target.get(p)  or self.target_default
        skip = " [SKIP]" if p in self.skip_pages else ""
        hdr = f"{'known:'+det['subject'] if det['kind']=='known' else ('unknown' if det['kind']=='unknown' else '—')}"
        if det.get("target"): hdr += f" | target:{det['target']}"
        base = os.path.basename(self.pdf_path)
        self.page_subj_label.config(text=f"{base} | p{p}: detected={hdr} || inherited={inh_s}/{inh_t}{skip}")
        self.marg_label.config(text=f"L(rel): {'—' if self.L_rel is None else f'{self.L_rel:.2f}'} | "
                                    f"R(rel): {'—' if self.R_rel is None else f'{self.R_rel:.2f}'}")
        self.info.config(text=f"File {self.file_idx+1}/{len(self.file_list)} • Page {p}/{self.num_pages} | mode={self.clip_mode.get()} | "
                              f"band={self.header_band_pt.get():.1f}pt | DPI={self.dpi_var.get()}")

    def recalc_band(self):
        self.header_band = {i+1: header_clip_band(pg, band_pt=self.header_band_pt.get(),
                                                  xpad=self.header_band_xpad.get())
                            for i, pg in enumerate(self.pdf.pages)}
        self.render_page()

    def recalc_ycut(self):
        self.header_ycut = {i+1: header_clip_ycut(pg, pad=self.header_y_pad.get())
                            for i, pg in enumerate(self.pdf.pages)}
        self.render_page()

    def refresh(self):
        self.render_page()

    def prev_page(self):
        if self.current_idx > 0:
            self.current_idx -= 1; self.render_page()

    def next_page(self):
        if self.current_idx < self.num_pages - 1:
            self.current_idx += 1; self.render_page()

    def assign_rel_margins(self):
        page, Lb, Rb, L_auto, R_auto = self.page_bboxes_and_margins()
        if L_auto is None or R_auto is None:
            messagebox.showwarning("No text", "Couldn’t auto-detect both margins on this page.")
            return
        self.L_rel = float(L_auto - Lb[0]); self.R_rel = float(R_auto - Rb[0])
        messagebox.showinfo("Defaults set",
                            f"Applied REL margins to ALL pages (single-file mode):\nL={self.L_rel:.2f} pt  R={self.R_rel:.2f} pt")
        self.render_page()

    def _resolve_output_paths(self, single_pdf_path: str) -> Tuple[str, str, str, int, str]:
        base = os.path.splitext(os.path.basename(single_pdf_path))[0]
        yr = self.year if isinstance(self.year, int) else infer_year_from_filename(single_pdf_path) or datetime.now().year

        out_spec = os.path.abspath(os.path.expanduser(self.out_path))
        is_json_spec = out_spec.lower().endswith(".json")
        if is_json_spec:
            base_parent = os.path.dirname(out_spec) or "."
        else:
            base_parent = out_spec or "."
        ensure_dir(base_parent)

        base_dir = os.path.join(base_parent, base)
        ensure_dir(base_dir)

        preview_name = None
        if self.chunk_preview_dir:
            preview_name = os.path.basename(os.path.normpath(self.chunk_preview_dir))
        if not preview_name:
            preview_name = "previews"
        preview_dir = os.path.join(base_dir, preview_name)
        ensure_dir(preview_dir)

        json_with_source = os.path.join(base_dir, f"{base}_with_source.json")
        json_without_source = os.path.join(base_dir, f"{base}.json")

        return json_with_source, json_without_source, preview_dir, yr, base_dir

    def parse_all_single(self):
        """Single-PDF mode: uses GUI-assigned margins (if any),
           otherwise auto-detect per this file as a fallback."""
        json_with_source, json_without_source, prev_dir, yr, base_dir = self._resolve_output_paths(self.pdf_path)

        L_rel, R_rel = self.L_rel, self.R_rel
        if L_rel is None or R_rel is None:
            L_rel, R_rel = _auto_detect_margins_for_pdf(
                self.pdf_path, self.top_frac.get(), self.bottom_frac.get(), self.gutter_frac.get()
            )
            if L_rel is None or R_rel is None:
                messagebox.showwarning("Margins not found",
                                       "Couldn’t auto-detect margins for this PDF. "
                                       "Please assign margins on a page first.")
                return

        try:
            qa = pdf_to_qa_flow_chunks(
                pdf_path=self.pdf_path,
                subject_default=self.subject_default,
                target_default=self.target_default,
                year=yr,
                start_num=1,
                L_rel=L_rel,
                R_rel=R_rel,
                tol=float(self.tol.get()),
                top_frac=float(self.top_frac.get()),
                bottom_frac=float(self.bottom_frac.get()),
                gutter_frac=float(self.gutter_frac.get()),
                y_tol=3.0,
                clip_mode=self.clip_mode.get(),
                header_y_pad=float(self.header_y_pad.get()),
                header_band_pt=float(self.header_band_pt.get()),
                header_band_xpad=float(self.header_band_xpad.get()),
                chunk_preview_dir=prev_dir,
                chunk_preview_dpi=self.chunk_preview_dpi,
                chunk_preview_pad=self.chunk_preview_pad
            )
        except Exception as e:
            messagebox.showerror("Parse failed", str(e)); return

        try:
            with open(json_with_source, "w", encoding="utf-8") as f:
                json.dump(qa, f, ensure_ascii=False, indent=2)

            qa_without_source = copy.deepcopy(qa)
            process_data(qa_without_source, remove_source=True, truncate_text=0)
            with open(json_without_source, "w", encoding="utf-8") as f:
                json.dump(qa_without_source, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Save failed", str(e)); return

        preview_count = sum(1 for item in qa if item.get("content", {}).get("preview_image"))
        msg = [
            f"Wrote {len(qa)} QA items.",
            f"With source → {os.path.abspath(json_with_source)}",
            f"Without source → {os.path.abspath(json_without_source)}",
        ]
        if preview_count and prev_dir:
            msg += ["", f"Saved {preview_count} previews →", os.path.abspath(prev_dir)]
        messagebox.showinfo("Done", "\n".join(msg))

    def parse_folder_auto(self):
        if not self.pdf_dir:
            return

        pdfs = list_pdfs(self.pdf_dir)
        if not pdfs:
            messagebox.showwarning("No PDFs", f"No .pdf files found in:\n{self.pdf_dir}")
            return

        if not os.path.isdir(self.out_path):
            out_dir = filedialog.askdirectory(title="Choose output folder for JSON files")
            if not out_dir:
                return
            self.out_path = out_dir
            self.out_path_label.config(text=os.path.abspath(out_dir))

        if self.chunk_preview_dir and not os.path.isdir(self.chunk_preview_dir):
            prev_dir = filedialog.askdirectory(title="Choose preview base folder (optional)")
            if prev_dir:
                self.chunk_preview_dir = prev_dir

        ok, skipped, failed = 0, 0, 0
        rows = []
        for p in pdfs:
            L_rel, R_rel = _auto_detect_margins_for_pdf(
                p, self.top_frac.get(), self.bottom_frac.get(), self.gutter_frac.get()
            )
            used = "auto"
            if L_rel is None or R_rel is None:
                if self.L_rel is not None and self.R_rel is not None:
                    L_rel, R_rel = self.L_rel, self.R_rel
                    used = "fallback-global"
                else:
                    skipped += 1
                    rows.append((os.path.basename(p), "SKIP (no margins)", "—"))
                    continue

            json_with_source, json_without_source, prev_dir, yr, base_dir = self._resolve_output_paths(p)
            try:
                qa = pdf_to_qa_flow_chunks(
                    pdf_path=p,
                    subject_default=self.subject_default,
                    target_default=self.target_default,
                    year=yr,
                    start_num=1,
                    L_rel=L_rel,
                    R_rel=R_rel,
                    tol=float(self.tol.get()),
                    top_frac=float(self.top_frac.get()),
                    bottom_frac=float(self.bottom_frac.get()),
                    gutter_frac=float(self.gutter_frac.get()),
                    y_tol=3.0,
                    clip_mode=self.clip_mode.get(),
                    header_y_pad=float(self.header_y_pad.get()),
                    header_band_pt=float(self.header_band_pt.get()),
                    header_band_xpad=float(self.header_band_xpad.get()),
                    chunk_preview_dir=prev_dir,
                    chunk_preview_dpi=self.chunk_preview_dpi,
                    chunk_preview_pad=self.chunk_preview_pad
                )
                with open(json_with_source, "w", encoding="utf-8") as f:
                    json.dump(qa, f, ensure_ascii=False, indent=2)

                qa_without_source = copy.deepcopy(qa)
                process_data(qa_without_source, remove_source=True, truncate_text=0)
                with open(json_without_source, "w", encoding="utf-8") as f:
                    json.dump(qa_without_source, f, ensure_ascii=False, indent=2)
                ok += 1
                rows.append((os.path.basename(p), f"OK ({len(qa)} QA)", f"{used} → {os.path.abspath(base_dir)}"))
            except Exception as e:
                failed += 1
                rows.append((os.path.basename(p), f"FAILED: {e}", used))

        lines = [f"Folder: {self.pdf_dir}", f"OK: {ok}  |  Skipped: {skipped}  |  Failed: {failed}", ""]
        for name, status, used in rows[:30]:
            lines.append(f"- {name}: {status}  [{used}]")
        if len(rows) > 30:
            lines.append(f"... and {len(rows)-30} more.")
        messagebox.showinfo("Batch finished", "\n".join(lines))

    def destroy(self):
        try:
            if self._tmp_png and os.path.exists(self._tmp_png):
                os.remove(self._tmp_png)
        except PermissionError:
            pass
        try:
            self.pdf.close()
        except Exception:
            pass
        super().destroy()

# =========================
# CLI
# =========================
def main():
    set_win_dpi_awareness()

    ap = argparse.ArgumentParser(description="Preview/assign margins, anywhere-header detect, band/y-cut clip, chunk QAs, export JSON. Folder mode uses AUTO per-file margins.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Input single PDF")
    g.add_argument("--pdf-dir", help="Folder containing PDFs (non-recursive)")

    ap.add_argument("--out", required=True, help="Output JSON path (single) or output folder (batch recommended)")
    ap.add_argument("--year", type=int, help="Year of the exam; if omitted, inferred from each filename")
    ap.add_argument("--subject-default", default="SubjectError", dest="subject_default")
    ap.add_argument("--target-default", default="Default", dest="target_default")

    ap.add_argument("--dpi", type=int, default=300, help="Preview render DPI")
    ap.add_argument("--tol", type=float, default=1.0, help="Margin match tolerance (pt)")
    ap.add_argument("--top-frac", type=float, default=0.04)
    ap.add_argument("--bottom-frac", type=float, default=0.96)
    ap.add_argument("--gutter-frac", type=float, default=0.005)

    ap.add_argument("--clip-mode", choices=["band","ycut","none"], default="band")
    ap.add_argument("--header-y-pad", type=float, default=4.0, help="Extra pts below header when clip-mode=ycut")
    ap.add_argument("--header-band-pt", type=float, default=-26.0, help="Band height (pts); negative grows upward from header bottom")
    ap.add_argument("--header-band-xpad", type=float, default=7.0, help="X padding (pts) around header box when band-clipping")

    ap.add_argument("--chunk-preview-dir", help="Optional base folder to save chunk JPEG previews")
    ap.add_argument("--chunk-preview-dpi", type=int, default=220)
    ap.add_argument("--chunk-preview-pad", type=float, default=2.0)

    args = ap.parse_args()

    if args.pdf:
        first_pdf = args.pdf
        batch_dir = None
    else:
        pdfs = list_pdfs(args.pdf_dir)
        if not pdfs:
            print(f"[ERROR] No PDFs in folder: {args.pdf_dir}", file=sys.stderr)
            sys.exit(2)
        first_pdf = pdfs[0]
        batch_dir = args.pdf_dir
        print(f"[INFO] Folder mode: {len(pdfs)} PDFs found. Previewing first: {os.path.basename(first_pdf)}")

    year_for_gui = args.year if args.year is not None else (infer_year_from_filename(first_pdf) or datetime.now().year)

    app = MarginPreviewApp(
        pdf_path=first_pdf,
        pdf_dir=batch_dir,
        out_path=args.out,
        subject_default=args.subject_default,
        target_default=args.target_default,
        year=year_for_gui,
        dpi=args.dpi,
        tol=args.tol,
        top_frac=args.top_frac,
        bottom_frac=args.bottom_frac,
        gutter_frac=args.gutter_frac,
        clip_mode=args.clip_mode,
        header_y_pad=args.header_y_pad,
        header_band_pt=args.header_band_pt,
        header_band_xpad=args.header_band_xpad,
        chunk_preview_dir=args.chunk_preview_dir,
        chunk_preview_dpi=args.chunk_preview_dpi,
        chunk_preview_pad=args.chunk_preview_pad
    )
    app.mainloop()

if __name__ == "__main__":
    main()
