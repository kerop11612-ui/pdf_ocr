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


def _classify_page(items: List[Dict[str, Any]]) -> str:
    """Classify only the layout modes that need different reconstruction."""
    if not items:
        return "decorative_or_blank"
    texts = [it["text"] for it in items]
    combined = "\n".join(texts)
    choice_count = sum(bool(re.match(r"^\s*[1-4１-４](?:[.．、\s]|$)", text)) for text in texts)
    if choice_count >= 3 or any(token in combined for token in ("選びなさい", "えらびなさい", "隨堂測驗", "随堂測驗", "解答：")):
        return "questions"
    vocab_ids = sum(bool(re.search(r"\b[A-Z]?\d{3}-\d{2}\b", text)) for text in texts)
    if vocab_ids >= 2:
        return "vocabulary_cards"
    page_refs = sum(bool(re.search(r"(?:^|\s)p\.?\s*\d{1,3}\b", text, re.I)) for text in texts)
    unit_labels = sum(bool(re.search(r"\d{1,2}\s*(?:單元|单元)", text)) for text in texts)
    if page_refs >= 3 or (page_refs >= 2 and unit_labels >= 2):
        return "toc"
    return "prose"


def _line_review_reasons(text: str, confidence: Optional[float]) -> List[str]:
    reasons = []
    if confidence is not None and confidence < 0.70:
        reasons.append("low_confidence")
    if any(char in text for char in ("�", "□")):
        reasons.append("unknown_character")
    if re.search(r"([\u3040-\u30ff\u3400-\u9fff])\1{3,}", text):
        reasons.append("repeated_character")
    if re.search(r"(?<=[\u3040-\u30ff\u3400-\u9fff])<(?=[\u3040-\u30ff\u3400-\u9fff])", text):
        reasons.append("mixed_symbol")
    return reasons


def _structured_line(item: Dict[str, Any]) -> Dict[str, Any]:
    reasons = _line_review_reasons(item["text"], item.get("confidence"))
    line = {
        "text": item["text"],
        "raw_text": item["raw_text"],
        "bbox": [item["x_min"], item["y_min"], item["x_max"], item["y_max"]],
        "confidence": item.get("confidence"),
        "needs_review": bool(reasons),
        "review_reasons": reasons,
    }
    if "included_in_clean" in item:
        line["included_in_clean"] = item["included_in_clean"]
        line["filter_reason"] = item.get("filter_reason")
    return line


def _make_block(block_type: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": block_type,
        "bbox": [
            min(it["x_min"] for it in items), min(it["y_min"] for it in items),
            max(it["x_max"] for it in items), max(it["y_max"] for it in items),
        ],
        "lines": [_structured_line(it) for it in items],
    }


def _is_heading(item: Dict[str, Any], median_h: float) -> bool:
    text = item["text"]
    explicit = text.startswith("【") or bool(re.match(r"^第\s*\d+\s*[課章單元篇]", text))
    prominent = item["h"] >= median_h * 1.20 and len(text) < 40
    return explicit or prominent or (len(text) < 20 and text in HEADING_KEYWORDS)


def _format_prose_block(items: List[Dict[str, Any]], median_h: float, page_width: float) -> str:
    paragraphs: List[str] = []
    current = ""
    previous = None
    block_left = min(it["x_min"] for it in items)
    for item in items:
        text = item["text"].strip()
        if not text:
            continue
        if _is_heading(item, median_h):
            if current:
                paragraphs.append(current)
                current = ""
            paragraphs.append(f"### {text}")
            previous = item
            continue
        if any(text.startswith(prefix) for prefix in BULLET_PREFIXES):
            if current:
                paragraphs.append(current)
            current = f"- {text}"
            previous = item
            continue
        if not current:
            current = text
        else:
            gap = item["y_min"] - previous["y_max"] if previous else 0
            indent = item["x_min"] - block_left
            paragraph_break = gap > median_h * 1.15 or (
                current.endswith(SENTENCE_TERMINATORS) and indent > max(median_h * 1.0, page_width * 0.025)
            )
            if paragraph_break:
                paragraphs.append(current)
                current = text
            else:
                current += text
        previous = item
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)

def process_page_layout(
    txts: List[str],
    boxes: List[Any],
    page_width: float,
    page_height: float,
    enable_reflow: bool = True,
    enable_two_column: bool = True,
    enable_furigana_filter: bool = True,
    enable_homoglyphs: bool = True,
    scores: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Restores natural Japanese layout and reading order.
    Returns:
    {
        "clean_text": str,
        "raw_text": str,
        "corrections_count": int,
        "columns_detected": int,
        "furigana_filtered": int,
        "page_type": str,
        "blocks": list,
        "needs_review": bool,
        "review_reasons": list,
        "source_lines": list
    }
    """
    # Preserve engine order and whitespace before all cleanup and geometry filters.
    raw_text = "\n".join(txts)
    if len(txts) == 0 or len(boxes) == 0:
        source_lines = [{
            "text": text.strip(), "raw_text": text, "bbox": None,
            "confidence": float(scores[i]) if scores is not None and i < len(scores) else None,
            "needs_review": True, "review_reasons": ["missing_geometry"],
            "included_in_clean": False, "filter_reason": "missing_geometry",
        } for i, text in enumerate(txts)]
        return {
            "clean_text": "",
            "raw_text": raw_text,
            "corrections_count": 0,
            "columns_detected": 1,
            "furigana_filtered": 0,
            "page_type": "decorative_or_blank",
            "blocks": [],
            "needs_review": bool(source_lines),
            "review_reasons": ["missing_geometry"] if source_lines else [],
            "source_lines": source_lines,
        }

    total_corrections = 0
    items = []
    source_line_records: List[Optional[Dict[str, Any]]] = [None] * len(txts)

    # 1. Parse boxes & compute geometry
    for i, t in enumerate(txts):
        stripped_t = t.strip()
        corr_t, n_corr = correct_homoglyphs_and_boundaries(stripped_t) if enable_homoglyphs else (stripped_t, 0)
        total_corrections += n_corr

        if i >= len(boxes) or boxes[i] is None or len(boxes[i]) == 0:
            source_line_records[i] = {
                "text": corr_t, "raw_text": t, "bbox": None,
                "confidence": float(scores[i]) if scores is not None and i < len(scores) else None,
                "needs_review": True, "review_reasons": ["missing_geometry"],
                "included_in_clean": False, "filter_reason": "missing_geometry",
            }
            continue
        b = boxes[i]
        x_min = float(min(pt[0] for pt in b))
        x_max = float(max(pt[0] for pt in b))
        y_min = float(min(pt[1] for pt in b))
        y_max = float(max(pt[1] for pt in b))
        w = max(1.0, x_max - x_min)
        h = max(1.0, y_max - y_min)

        items.append({
            "text": corr_t,
            "raw_text": t,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "w": w,
            "h": h,
            "x_center": (x_min + x_max) / 2.0,
            "y_center": (y_min + y_max) / 2.0,
            "confidence": float(scores[i]) if scores is not None and i < len(scores) else None,
            "included_in_clean": bool(stripped_t),
            "filter_reason": None if stripped_t else "empty_text",
            "source_index": i,
        })

    if not items:
        source_lines = [line for line in source_line_records if line is not None]
        review_reasons = sorted({reason for line in source_lines for reason in line["review_reasons"]})
        return {
            "clean_text": "",
            "raw_text": raw_text,
            "corrections_count": 0,
            "columns_detected": 1,
            "furigana_filtered": 0,
            "page_type": "decorative_or_blank",
            "blocks": [],
            "needs_review": bool(review_reasons),
            "review_reasons": review_reasons,
            "source_lines": source_lines,
        }

    # 2. Margin Crop (Filter top 4% and bottom 4% noise)
    valid_items = []
    for it in items:
        if not it["included_in_clean"]:
            continue
        # Header noise check
        if it["y_min"] < page_height * 0.04:
            # Keep only if explicitly a grammar header
            if not any(k in it["text"] for k in HEADING_KEYWORDS):
                it["included_in_clean"] = False
                it["filter_reason"] = "top_margin"
                continue
        # Footer noise check (isolated digits / publisher info)
        if it["y_max"] > page_height * 0.96:
            if re.match(r"^\d{1,3}$", it["text"]):
                it["included_in_clean"] = False
                it["filter_reason"] = "footer_page_number"
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
                it["included_in_clean"] = False
                it["filter_reason"] = "furigana"
                continue  # Filter out ruby line to prevent sentence rupture
            non_furigana_items.append(it)
    else:
        non_furigana_items = valid_items

    # 4. Page type and coarse region blocks
    page_type = _classify_page(non_furigana_items)

    # 5. Multi-Column (XY-Cut) Detection
    # Determine if text is formatted as 2 columns
    columns_detected = 1
    sorted_items = []
    median_h = float(np.median([it["h"] for it in non_furigana_items])) if non_furigana_items else 15.0
    block_groups: List[Tuple[str, List[Dict[str, Any]]]] = []

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
            
            remaining_left = list(left_col)
            remaining_right = list(right_col)
            for spanning in spanning_headers:
                band_left = [it for it in remaining_left if it["y_center"] < spanning["y_min"]]
                band_right = [it for it in remaining_right if it["y_center"] < spanning["y_min"]]
                if band_left:
                    block_groups.append(("column", band_left))
                if band_right:
                    block_groups.append(("column", band_right))
                remaining_left = [it for it in remaining_left if it not in band_left]
                remaining_right = [it for it in remaining_right if it not in band_right]
                block_groups.append(("heading" if _is_heading(spanning, median_h) else "spanning", [spanning]))
            if remaining_left:
                block_groups.append(("column", remaining_left))
            if remaining_right:
                block_groups.append(("column", remaining_right))
            sorted_items = [item for _, group in block_groups for item in group]
        else:
            sorted_items = sorted(non_furigana_items, key=lambda x: (round(x["y_min"] / 12) * 12, x["x_min"]))
            if sorted_items:
                block_groups.append(("body", sorted_items))
    else:
        sorted_items = sorted(non_furigana_items, key=lambda x: (round(x["y_min"] / 12) * 12, x["x_min"]))
        if sorted_items:
            block_groups.append(("body", sorted_items))

    blocks = [_make_block(block_type, group) for block_type, group in block_groups]
    for item in items:
        source_line_records[item["source_index"]] = _structured_line(item)
    source_lines = [line for line in source_line_records if line is not None]

    review_reason_set = {reason for line in source_lines for reason in line["review_reasons"]}
    if page_type in {"toc", "vocabulary_cards", "questions"}:
        review_reason_set.add("complex_layout")
    review_reasons = sorted(review_reason_set)

    # 6. Page-type-aware rendering

    if not enable_reflow:
        clean_text = "\n\n".join([it["text"] for it in sorted_items])
        return {
            "clean_text": clean_text,
            "raw_text": raw_text,
            "corrections_count": total_corrections,
            "columns_detected": columns_detected,
            "furigana_filtered": furigana_count,
            "page_type": page_type,
            "blocks": blocks,
            "needs_review": bool(review_reasons),
            "review_reasons": review_reasons,
            "source_lines": source_lines,
        }

    if page_type == "prose":
        clean_parts = [_format_prose_block(group, median_h, page_width) for _, group in block_groups]
        clean_text = "\n\n".join(part for part in clean_parts if part)
    else:
        clean_text = "\n\n".join("\n".join(line["text"] for line in block["lines"]) for block in blocks)

    return {
        "clean_text": clean_text,
        "raw_text": raw_text,
        "corrections_count": total_corrections,
        "columns_detected": columns_detected,
        "furigana_filtered": furigana_count,
        "page_type": page_type,
        "blocks": blocks,
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "source_lines": source_lines,
    }
