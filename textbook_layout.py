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

    # 4b. Katakana read inside hiragana/kanji context -> hiragana look-alike
    #     (リゅう -> りゅう, あリがとう -> ありがとう, 学校ヘ -> 学校へ, 食ベる -> 食べる).
    #     A neighbouring katakana means a real katakana word, so it is left alone.
    kana_pairs = {"リ": "り", "ヘ": "へ", "ベ": "べ", "ペ": "ぺ"}
    pat_kana = re.compile(
        rf"(?<![゠-ヿ])[リヘベペ](?={RE_HIRAGANA})"
        rf"|(?<=[぀-ゟ一-鿿])[リヘベペ](?![゠-ヿ])"
    )
    cur, n7 = pat_kana.subn(lambda m: kana_pairs[m.group(0)], cur)
    corrections += n7

    # 4c. Katakana ロ right after a kanji and not followed by katakana -> kanji 口 (入ロ -> 入口, 人ロ -> 人口)
    pat_kuchi = re.compile(rf"(?<={RE_KANJI})ロ(?![゠-ヿ])")
    cur, n8 = pat_kuchi.subn("口", cur)
    corrections += n8

    # 4d. Circled question number glued to the next token: "②1 わかりました" -> "② 1 わかりました"
    pat_circled = re.compile(r"([①-⑳])(?=[0-9０-９぀-ヿ一-鿿])")
    cur, n10 = pat_circled.subn(r"\1 ", cur)
    corrections += n10

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
    choice_count = sum(bool(re.match(r"^\s*(?:[1-4１-４](?:[.．、\s]|$)|[①-⑳])", text)) for text in texts)
    question_tokens = ("選びなさい", "えらびなさい", "選んでください", "えらんでください", "隨堂測驗", "随堂測驗", "解答：", "スクリプト")
    if choice_count >= 3 or any(token in combined for token in question_tokens):
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
    reasons = list(dict.fromkeys(_line_review_reasons(item["text"], item.get("confidence")) + item.get("extra_review_reasons", [])))
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


def _question_rows(items: List[Dict[str, Any]], median_h: float) -> List[List[Dict[str, Any]]]:
    """Keep each horizontal question/choice row together before reading the next row."""
    rows: List[List[Dict[str, Any]]] = []
    for item in sorted(items, key=lambda it: (it["y_center"], it["x_min"])):
        if not rows or abs(item["y_center"] - np.median([it["y_center"] for it in rows[-1]])) > median_h * 0.6:
            rows.append([item])
        else:
            rows[-1].append(item)
    for row in rows:
        row.sort(key=lambda it: it["x_min"])
    return rows


def _is_ruby_of_neighbour(item: Dict[str, Any], items: List[Dict[str, Any]]) -> bool:
    """True when a kana box sits directly above or below a larger kanji box it overlaps horizontally."""
    for other in items:
        if other is item or not re.search(RE_KANJI, other["text"]) or item["h"] >= other["h"] * 0.8:
            continue
        overlap = min(item["x_max"], other["x_max"]) - max(item["x_min"], other["x_min"])
        if overlap < item["w"] * 0.5:
            continue
        # Ruby may overlap the kanji box (tight scans), so compare centres instead of requiring a gap.
        below = item["y_center"] > other["y_center"] + other["h"] * 0.3 and item["y_min"] - other["y_max"] <= other["h"] * 0.5
        above = item["y_center"] < other["y_center"] - other["h"] * 0.3 and other["y_min"] - item["y_max"] <= other["h"] * 0.5
        if below or above:
            return True
    return False


def _is_unreadable_noise(text: str, confidence: Optional[float]) -> bool:
    """Low-confidence lines that are mostly digits/symbols, i.e. text in a script the model can't read."""
    compact = re.sub(r"\s", "", text)
    if confidence is None or confidence >= 0.8 or len(compact) < 4:
        return False  # short items (answer digits, ①) are never treated as noise
    meaningful = len(re.findall(r"[A-Za-z぀-ヿ一-鿿]", compact))
    return meaningful / len(compact) < 0.4


CHOICE_RE = re.compile(r"^\s*([1-4１-４])(?:[.．、\s]|$)")


def _mark_choice_number_mismatch(row: List[Dict[str, Any]]) -> None:
    """A row of four choices should read 1-2-3-4; flag the box that breaks the sequence."""
    recognized = [CHOICE_RE.match(item["text"]) for item in row]
    if len(row) == 4 and sum(match is not None for match in recognized) >= 3:
        for expected, (item, match) in enumerate(zip(row, recognized), start=1):
            observed = match.group(1).translate(str.maketrans("１２３４", "1234")) if match else None
            if observed != str(expected):
                item.setdefault("extra_review_reasons", []).append("choice_number_mismatch")


def _find_gutter(items: List[Dict[str, Any]], page_width: float, median_h: float) -> Optional[Tuple[float, bool]]:
    """Find a vertical gutter no box crosses that separates a margin sidebar or a second column.

    Returns (x, is_narrow_sidebar) or None. A gutter must not split a row of answer choices,
    and the right side must be either a narrow note column or a column as substantial as the left,
    so table columns (answer digit | script) are never torn apart.
    """
    if len(items) < 8:
        return None
    total_chars = sum(len(it["text"]) for it in items) or 1
    rows = _question_rows(items, median_h)
    best: Optional[Tuple[float, bool, int]] = None
    for step in range(30, 86):
        x = page_width * step / 100
        crossing = sum(1 for it in items if it["x_min"] < x < it["x_max"])
        if crossing > max(1, len(items) * 0.03):
            continue
        left = [it for it in items if it["x_min"] < x]
        right = [it for it in items if it["x_min"] >= x]
        if len(left) < 4 or len(right) < 4:
            continue
        right_chars = sum(len(it["text"]) for it in right)
        left_chars = total_chars - right_chars
        narrow = (max(it["x_max"] for it in right) - x) < page_width * 0.3
        balanced = min(left_chars, right_chars) >= total_chars * 0.25
        if not ((narrow and right_chars >= total_chars * 0.06) or balanced):
            continue
        splits_choices = any(
            any(CHOICE_RE.match(it["text"]) and it["x_min"] < x for it in row)
            and any(CHOICE_RE.match(it["text"]) and it["x_min"] >= x for it in row)
            for row in rows
        )
        if splits_choices:
            continue
        if best is None or crossing < best[2]:
            best = (x, narrow, crossing)
    return (best[0], best[1]) if best else None


LINE_START_MARKER = re.compile(r"^\s*(?:CD|[男女][：:]|[A-ZＡ-Ｚ][：:]|[①-⑳]|[0-9０-９]|問題|例|[・◆■▲※*＊-])")


def _continues_wrapped_line(prev_row: List[Dict[str, Any]], row: List[Dict[str, Any]], prev_text: str, text: str, right_edge: float, median_h: float) -> bool:
    """A row that ran to the right margin mid-sentence continues on the next row (dialogue scripts)."""
    if not prev_text or prev_text.endswith(SENTENCE_TERMINATORS + ("）", ")")) or LINE_START_MARKER.match(text):
        return False
    prev_right = max(it["x_max"] for it in prev_row)
    gap = min(it["y_min"] for it in row) - max(it["y_max"] for it in prev_row)
    return prev_right >= right_edge - median_h * 1.5 and gap < median_h * 1.0 and min(it["x_min"] for it in row) >= min(it["x_min"] for it in prev_row) - median_h


def _sidebar_notes(rows: List[List[Dict[str, Any]]], median_h: float) -> List[Tuple[str, List[int]]]:
    """Join wrapped sidebar lines into notes; a vertical gap starts a new note."""
    notes: List[Tuple[str, List[int]]] = []
    previous_bottom = None
    for row in rows:
        text = " ".join(it["text"].strip() for it in row)
        sources = [it["source_index"] for it in row]
        top = min(it["y_min"] for it in row)
        if notes and previous_bottom is not None and top - previous_bottom < median_h * 0.9:
            notes[-1] = (notes[-1][0] + text, notes[-1][1] + sources)
        else:
            notes.append((text, sources))
        previous_bottom = max(it["y_max"] for it in row)
    return notes


def _is_heading(item: Dict[str, Any], median_h: float) -> bool:
    text = item["text"]
    explicit = text.startswith("【") or bool(re.match(r"^第\s*\d+\s*[課章單元篇]", text))
    prominent = item["h"] >= median_h * 1.20 and len(text) < 40
    return explicit or prominent or (len(text) < 20 and text in HEADING_KEYWORDS)


def _format_prose_block(items: List[Dict[str, Any]], median_h: float, page_width: float) -> Tuple[str, List[List[int]]]:
    paragraphs: List[Tuple[str, List[int]]] = []
    current = ""
    current_sources: List[int] = []
    previous = None
    block_left = min(it["x_min"] for it in items)
    for item in items:
        text = item["text"].strip()
        if not text:
            continue
        if _is_heading(item, median_h):
            if current:
                paragraphs.append((current, current_sources))
                current = ""
                current_sources = []
            paragraphs.append((f"### {text}", [item["source_index"]]))
            previous = item
            continue
        if any(text.startswith(prefix) for prefix in BULLET_PREFIXES):
            if current:
                paragraphs.append((current, current_sources))
            current = f"- {text}"
            current_sources = [item["source_index"]]
            previous = item
            continue
        if not current:
            current = text
            current_sources = [item["source_index"]]
        else:
            gap = item["y_min"] - previous["y_max"] if previous else 0
            indent = item["x_min"] - block_left
            paragraph_break = gap > median_h * 1.15 or (
                current.endswith(SENTENCE_TERMINATORS) and indent > max(median_h * 1.0, page_width * 0.025)
            )
            if paragraph_break:
                paragraphs.append((current, current_sources))
                current = text
                current_sources = [item["source_index"]]
            else:
                current += text
                current_sources.append(item["source_index"])
        previous = item
    if current:
        paragraphs.append((current, current_sources))
    mapping: List[List[int]] = []
    for _, sources in paragraphs:
        if mapping:
            mapping.append([])
        mapping.append(sources)
    return "\n\n".join(text for text, _ in paragraphs), mapping

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
    # Body-text height comes from kanji-bearing lines: ruby never contains kanji, so dense
    # furigana cannot drag the reference size down and hide itself.
    kanji_items = [it for it in valid_items if re.search(RE_KANJI, it["text"])]
    reference_items = kanji_items if len(kanji_items) >= 5 else valid_items
    avg_h = np.median([it["h"] for it in reference_items]) if reference_items else 15.0
    furigana_count = 0
    non_furigana_items = []

    if enable_furigana_filter:
        for it in valid_items:
            text = it["text"].strip()
            is_all_kana = bool(re.match(r"^[\u3040-\u30FF\s\u30FB]+$", text))
            reason = None
            if is_all_kana and it["h"] < avg_h * 0.7:
                reason = "furigana"
            elif is_all_kana and re.search(r"\S\s+\S", text) and it["h"] < avg_h * 0.9:
                # A whole ruby row read as one box: kana words separated by spaces.
                reason = "furigana"
            elif it["h"] < avg_h * 0.7 and len(text) <= 4 and (
                not re.search(RE_KANJI, text) or (it.get("confidence") is not None and it["confidence"] < 0.8)
            ):
                # Misread ruby or small icon digits (e.g. "=24", CD track "12", おとこ read as "主");
                # body-size choice numbers are kept.
                reason = "small_fragment"
            elif is_all_kana and len(text.replace(" ", "")) <= 16 and _is_ruby_of_neighbour(it, valid_items):
                # Ruby printed right above/below a (often bold, oversized) kanji: 番 / ばん.
                reason = "furigana"
            elif _is_unreadable_noise(text, it.get("confidence")):
                # Script the model cannot read (e.g. Korean) comes out as low-confidence digit/symbol soup.
                reason = "noise"
            if reason:
                furigana_count += 1
                it["included_in_clean"] = False
                it["filter_reason"] = reason
                continue  # Filter out ruby line to prevent sentence rupture
            non_furigana_items.append(it)
    else:
        non_furigana_items = valid_items

    # 3b. CD icons: the track number printed under "CD1" becomes "CD1-58" instead of a stray number.
    for cd in [it for it in non_furigana_items if re.fullmatch(r"CD\s*\d", it["text"].strip())]:
        for track in non_furigana_items:
            if track is cd or not re.fullmatch(r"\d{1,3}", track["text"].strip()):
                continue
            overlap = min(track["x_max"], cd["x_max"]) - max(track["x_min"], cd["x_min"])
            if overlap > 0 and 0 <= track["y_center"] - cd["y_center"] <= cd["h"] * 3:
                cd["text"] = f"{cd['text'].replace(' ', '')}-{track['text'].strip()}"
                track["included_in_clean"] = False
                track["filter_reason"] = "cd_track"
                break
    # A CD icon whose "CD1" label was not read leaves only its track number, just left of "N 番".
    for track in non_furigana_items:
        if track.get("filter_reason") or not re.fullmatch(r"\d{1,3}", track["text"].strip()):
            continue
        if any(
            other is not track and re.match(r"^\s*\d{1,2}\s*番", other["text"])
            and abs(other["y_center"] - track["y_center"]) < max(track["h"], other["h"])
            and 0 <= other["x_min"] - track["x_max"] <= other["h"] * 3
            for other in non_furigana_items
        ):
            track["included_in_clean"] = False
            track["filter_reason"] = "cd_track"
    # Lines that are only punctuation/symbol soup carry no text; bracket blanks "（　）" are kept.
    for it in non_furigana_items:
        compact = re.sub(r"\s", "", it["text"])
        if it.get("filter_reason") or len(compact) < 2 or re.fullmatch(r"[()（）［］\[\]「」_＿ー—-]+", compact):
            continue
        meaningful = len(re.findall(r"[0-9０-９A-Za-z぀-ヿ一-鿿①-⑳]", compact))
        if meaningful / len(compact) < 0.4:
            it["included_in_clean"] = False
            it["filter_reason"] = "symbols"
    non_furigana_items = [it for it in non_furigana_items if it.get("filter_reason") not in ("cd_track", "symbols")]

    # 4. Page type and coarse region blocks
    page_type = _classify_page(non_furigana_items)

    # 5. Multi-Column (XY-Cut) Detection
    # Determine if text is formatted as 2 columns
    columns_detected = 1
    sorted_items = []
    median_h = float(np.median([it["h"] for it in non_furigana_items])) if non_furigana_items else 15.0
    block_groups: List[Tuple[str, List[Dict[str, Any]]]] = []

    row_layout = page_type != "prose" and bool(non_furigana_items)
    if row_layout:
        # Exercise, answer-table, TOC and card pages: read visual rows left to right, so an
        # answer number stays on the same line as its question and choices stay together.
        gutter = _find_gutter(non_furigana_items, page_width, median_h) if enable_two_column else None
        if gutter is None:
            sections = [("row", non_furigana_items)]
        else:
            gutter_x, narrow = gutter
            columns_detected = 2
            sections = [
                ("row", [it for it in non_furigana_items if it["x_min"] < gutter_x]),
                ("sidebar" if narrow else "row", [it for it in non_furigana_items if it["x_min"] >= gutter_x]),
            ]
        for kind, group in sections:
            for row in _question_rows(group, median_h):
                if len(row) == 1 and re.fullmatch(r"\d{1,3}", row[0]["text"].strip()) and row[0]["h"] < median_h * 0.85:
                    row[0]["included_in_clean"] = False
                    row[0]["filter_reason"] = "isolated_number"  # CD track / icon digits
                    continue
                if page_type == "questions":
                    _mark_choice_number_mismatch(row)
                block_groups.append((kind, row))
        sorted_items = [item for _, row in block_groups for item in row]
    elif enable_two_column and len(non_furigana_items) >= 4:
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
        clean_line_sources = []
        for item in sorted_items:
            if clean_line_sources:
                clean_line_sources.append([])
            clean_line_sources.append([item["source_index"]])
        return {
            "clean_text": clean_text,
            "clean_line_sources": clean_line_sources,
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

    if row_layout:
        out_lines: List[str] = []
        clean_line_sources = []
        main_rows = [row for kind, row in block_groups if kind == "row"]
        side_rows = [row for kind, row in block_groups if kind == "sidebar"]
        right_edge = max((it["x_max"] for row in main_rows for it in row), default=0)
        previous_row = None
        for row in main_rows:
            text = " ".join(it["text"].strip() for it in row)
            if previous_row is not None and _continues_wrapped_line(previous_row, row, out_lines[-1], text, right_edge, median_h):
                out_lines[-1] += text
                clean_line_sources[-1].extend(it["source_index"] for it in row)
            else:
                out_lines.append(text)
                clean_line_sources.append([it["source_index"] for it in row])
            previous_row = row
        notes = _sidebar_notes(side_rows, median_h)
        if notes and out_lines:
            out_lines.append("")
            clean_line_sources.append([])
        for text, sources in notes:
            out_lines.append(f"> {text}")
            clean_line_sources.append(sources)
        clean_text = "\n".join(out_lines)
    elif page_type == "prose":
        clean_parts = [_format_prose_block(group, median_h, page_width) for _, group in block_groups]
        clean_text = "\n\n".join(part for part, _ in clean_parts if part)
        clean_line_sources = []
        for part, sources in clean_parts:
            if not part:
                continue
            if clean_line_sources:
                clean_line_sources.append([])
            clean_line_sources.extend(sources)
    else:
        clean_text = "\n\n".join("\n".join(line["text"] for line in block["lines"]) for block in blocks)
        clean_line_sources = []
        for _, group in block_groups:
            if clean_line_sources:
                clean_line_sources.append([])
            clean_line_sources.extend([[item["source_index"]] for item in group])

    return {
        "clean_text": clean_text,
        "clean_line_sources": clean_line_sources,
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
