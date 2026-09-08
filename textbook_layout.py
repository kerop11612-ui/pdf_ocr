"""
Textbook Layout Preservation & Recognition Accuracy Enhancement Engine.
Features:
1. CLAHE Image Contrast Enhancement (prevent kana diacritic loss, remove scan bleed-through).
2. Contextual Homoglyph Corrector (ー vs 一, カ vs 力, ロ vs 口, ニ vs 二).
3. Sticky Number & Token Boundary Decoupler (14次の文 -> 14 次の文).
4. XY-Cut Multi-Column Detection & Ordering (no zigzag reading order).
5. Furigana (Ruby) Filtering/Handling (no sentence disruption).
6. Margin Crop (eliminate header/footer/isolated page number noise).
7. Punctuation-Aware Sentence Reflow (seamless Japanese paragraph restoration).
"""

import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

# --- 1. CLAHE Preprocessing ---
def enhance_image_for_ocr(img_np: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances small diacritics (濁点 ゛ / 半濁点 ゜) and thin kanji strokes,
    while suppressing scanned background gray bleed-through.
    """
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)


# --- 2. Homoglyphs & Boundary Corrections ---
RE_KATAKANA = r"[\u30A0-\u30FF]"
RE_KANJI = r"[\u4E00-\u9FFF]"
RE_HIRAGANA = r"[\u3040-\u309F]"
RE_CJK = r"[\u4E00-\u9FFF\u3040-\u30FF]"

def correct_homoglyphs_and_boundaries(text: str) -> Tuple[str, int]:
    """
    Context-constrained correction of common Japanese OCR confusion.
    Returns: (corrected_text, total_corrections_made)
    """
    if not text:
        return text, 0

    corrections = 0
    cur = text

    # 1. Katakana カ vs Kanji 力
    pat_ka = re.compile(rf"(?<={RE_KATAKANA})力(?={RE_KATAKANA})")
    cur, n1 = pat_ka.subn("カ", cur)
    corrections += n1

    # 2. Katakana ロ vs Kanji 口
    pat_ro = re.compile(rf"(?<=[\u30A0-\u30FF一])口(?={RE_KATAKANA})")
    cur, n2 = pat_ro.subn("ロ", cur)
    corrections += n2

    # 3. Katakana ニ vs Kanji 二
    pat_ni = re.compile(rf"(?<={RE_KATAKANA})二(?={RE_KATAKANA})")
    cur, n3 = pat_ni.subn("ニ", cur)
    corrections += n3

    # 4. Long vowel mark (ー) vs Kanji 一 / hyphen / dash
    # Any 一, —, or - immediately preceded by Katakana is the Chōonpu (ー)
    pat_long_vowel = re.compile(rf"(?<={RE_KATAKANA})[一—\-](?!{RE_KANJI})")
    cur, n4 = pat_long_vowel.subn("ー", cur)
    corrections += n4

    # 5. Sticky question numbers / index numbers: e.g. "14次の" -> "14 次の"
    pat_sticky_start = re.compile(rf"^(\d{{1,3}})({RE_CJK})")
    cur, n5 = pat_sticky_start.subn(r"\1 \2", cur)
    corrections += n5

    pat_sticky_inner = re.compile(rf"(\s\d{{1,3}})({RE_CJK})")
    cur, n6 = pat_sticky_inner.subn(r"\1 \2", cur)
    corrections += n6

    # 6. Normalize fullwidth grammar tilde: "~" -> "〜"
    if "~" in cur:
        n9 = cur.count("~")
        cur = cur.replace("~", "〜")
        corrections += n9

    return cur, corrections


# --- 3. Layout Engine: Margin Crop, Column Ordering & Sentence Reflow ---
SENTENCE_TERMINATORS = ("。", "！", "？", "!", "?", "：", "；", "…", "」", "』")
BULLET_PREFIXES = ("・", "①", "②", "③", "④", "⑤", "⑥", "1.", "2.", "3.", "4.", "5.", "(1)", "(2)", "(3)", "◆", "■", "▲", "Q.", "A.")
HEADING_KEYWORDS = ("第", "課", "文法", "ポイント", "意味", "接續", "接続", "例文", "解説", "注意", "問題", "練習")

def process_page_layout(
    txts: List[str],
    boxes: List[Any],
    page_width: float,
    page_height: float,
    enable_reflow: bool = True,
    enable_two_column: bool = True,
    enable_furigana_filter: bool = True,
    enable_homoglyphs: bool = True
) -> Dict[str, Any]:
    """
    Restores natural Japanese layout and reading order.
    Returns:
    {
        "clean_text": str,
        "raw_text": str,
        "corrections_count": int,
        "columns_detected": int,
        "furigana_filtered": int
    }
    """
    # Preserve engine order and whitespace before all cleanup and geometry filters.
    raw_text = "\n".join(txts)
    if len(txts) == 0 or len(boxes) == 0:
        return {
            "clean_text": "",
            "raw_text": raw_text,
            "corrections_count": 0,
            "columns_detected": 1,
            "furigana_filtered": 0
        }

    total_corrections = 0
    items = []

    # 1. Parse boxes & compute geometry
    for i, t in enumerate(txts):
        raw_t = t.strip()
        if not raw_t:
            continue
        corr_t, n_corr = correct_homoglyphs_and_boundaries(raw_t) if enable_homoglyphs else (raw_t, 0)
        total_corrections += n_corr

        b = boxes[i] if i < len(boxes) else [[0, 0], [0, 0], [0, 0], [0, 0]]
        x_min = min(pt[0] for pt in b)
        x_max = max(pt[0] for pt in b)
        y_min = min(pt[1] for pt in b)
        y_max = max(pt[1] for pt in b)
        w = max(1.0, x_max - x_min)
        h = max(1.0, y_max - y_min)

        items.append({
            "text": corr_t,
            "raw_text": raw_t,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "w": w,
            "h": h,
            "x_center": (x_min + x_max) / 2.0,
            "y_center": (y_min + y_max) / 2.0
        })

    if not items:
        return {
            "clean_text": "",
            "raw_text": raw_text,
            "corrections_count": 0,
            "columns_detected": 1,
            "furigana_filtered": 0
        }

    # 2. Margin Crop (Filter top 4% and bottom 4% noise)
    valid_items = []
    for it in items:
        # Header noise check
        if it["y_min"] < page_height * 0.04:
            # Keep only if explicitly a grammar header
            if not any(k in it["text"] for k in HEADING_KEYWORDS):
                continue
        # Footer noise check (isolated digits / publisher info)
        if it["y_max"] > page_height * 0.96:
            if re.match(r"^\d{1,3}$", it["text"]):
                continue
        valid_items.append(it)

    # 3. Furigana (Ruby) Filtering
    avg_h = np.median([it["h"] for it in valid_items]) if valid_items else 15.0
    furigana_count = 0
    non_furigana_items = []

    if enable_furigana_filter:
        for it in valid_items:
            # Check if this item is tiny and exclusively kana
            is_tiny = it["h"] < avg_h * 0.58
            is_all_kana = bool(re.match(r"^[\u3040-\u30FF\s]+$", it["text"]))
            if is_tiny and is_all_kana:
                furigana_count += 1
                continue  # Filter out ruby line to prevent sentence rupture
            non_furigana_items.append(it)
    else:
        non_furigana_items = valid_items

    # 4. Multi-Column (XY-Cut) Detection
    # Determine if text is formatted as 2 columns
    columns_detected = 1
    sorted_items = []

    if enable_two_column and len(non_furigana_items) >= 4:
        mid_x = page_width / 2.0
        left_col = []
        right_col = []
        spanning_headers = []

        for it in non_furigana_items:
            # Spans across both columns (e.g. major section title)
            if it["w"] > page_width * 0.65:
                spanning_headers.append(it)
            elif it["x_center"] < mid_x:
                left_col.append(it)
            else:
                right_col.append(it)

        # Only treat as two-column if both left and right have substantial text
        if len(left_col) >= 2 and len(right_col) >= 2:
            columns_detected = 2
            # Sort spanning headers first by Y
            spanning_headers.sort(key=lambda x: x["y_min"])
            # Sort left column top to bottom
            left_col.sort(key=lambda x: (round(x["y_min"] / 12) * 12, x["x_min"]))
            # Sort right column top to bottom
            right_col.sort(key=lambda x: (round(x["y_min"] / 12) * 12, x["x_min"]))
            
            # Place top headers, then left column, then right column
            sorted_items = spanning_headers + left_col + right_col
        else:
            sorted_items = sorted(non_furigana_items, key=lambda x: (round(x["y_min"] / 12) * 12, x["x_min"]))
    else:
        sorted_items = sorted(non_furigana_items, key=lambda x: (round(x["y_min"] / 12) * 12, x["x_min"]))

    # 5. Sentence Reflow Engine (自然段落與文句重組)

    if not enable_reflow:
        clean_text = "\n\n".join([it["text"] for it in sorted_items])
        return {
            "clean_text": clean_text,
            "raw_text": raw_text,
            "corrections_count": total_corrections,
            "columns_detected": columns_detected,
            "furigana_filtered": furigana_count
        }

    formatted_paragraphs = []
    current_para = ""

    for it in sorted_items:
        t = it["text"].strip()
        if not t:
            continue

        # Check if heading (short label or section delimiter)
        is_heading = (
            (len(t) < 20 and any(k in t for k in HEADING_KEYWORDS))
            or t.startswith("【")
            or t.startswith("第")
        )

        # Check if bullet item (e.g. list of examples or items)
        is_bullet = any(t.startswith(b) for b in BULLET_PREFIXES)

        if is_heading:
            if current_para:
                formatted_paragraphs.append(current_para)
                current_para = ""
            formatted_paragraphs.append(f"### {t}")
            continue

        if is_bullet:
            if current_para:
                formatted_paragraphs.append(current_para)
                current_para = ""
            current_para = f"- {t}"
            continue

        if not current_para:
            current_para = t
        else:
            # Check if previous sentence ended cleanly
            prev_ends_term = current_para.endswith(SENTENCE_TERMINATORS)
            if prev_ends_term:
                formatted_paragraphs.append(current_para)
                current_para = t
            else:
                # Merge smoothly without linebreak or space (Japanese natural flow)
                current_para += t

    if current_para:
        formatted_paragraphs.append(current_para)

    clean_text = "\n\n".join(formatted_paragraphs)

    return {
        "clean_text": clean_text,
        "raw_text": raw_text,
        "corrections_count": total_corrections,
        "columns_detected": columns_detected,
        "furigana_filtered": furigana_count
    }
