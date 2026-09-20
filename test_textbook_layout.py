"""
Unit Tests for textbook_layout.py.
Verifies:
1. CLAHE Image Enhancement
2. Homoglyph & Sticky Boundary Corrector
3. Two-Column Sorting (XY-Cut)
4. Sentence Reflow Engine & Heading Recognition
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from textbook_layout import (
    enhance_image_for_ocr,
    correct_homoglyphs_and_boundaries,
    process_page_layout
)

def test_enhance_image_for_ocr():
    dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    enhanced = enhance_image_for_ocr(dummy_img)
    assert enhanced.shape == dummy_img.shape
    assert enhanced.dtype == np.uint8
    print("  [PASS] CLAHE Enhancement")


def test_correct_homoglyphs_and_boundaries():
    # Long vowel correction
    raw1 = "コ一ヒ一を飲む。テ一マを決める。"
    corr1, n1 = correct_homoglyphs_and_boundaries(raw1)
    assert "コーヒー" in corr1
    assert "テーマ" in corr1
    assert n1 >= 3

    # Katakana Ka & Ro & Ni
    raw2 = "ア力ウントを作成し、ヨ一口ッパに旅する。ス二ーカーを買う。"
    corr2, n2 = correct_homoglyphs_and_boundaries(raw2)
    assert "アカウント" in corr2
    assert "ヨーロッパ" in corr2
    assert "スニーカー" in corr2
    assert n2 >= 3

    # Sticky number
    raw3 = "14次の文の（ ）に入れなさい。"
    corr3, n3 = correct_homoglyphs_and_boundaries(raw3)
    assert corr3.startswith("14 次の文")
    assert n3 >= 1

    # Tilde
    raw4 = "~わけではない"
    corr4, n4 = correct_homoglyphs_and_boundaries(raw4)
    assert corr4.startswith("〜わけではない")

    print(f"  [PASS] Homoglyphs & Boundaries (Total corrections: {n1 + n2 + n3 + n4})")


def test_process_page_layout_reflow_and_two_column():
    # Simulate a two-column page with broken lines
    # Left column: x in [50, 250], Right column: x in [350, 550], Page width = 600
    txts = [
        "第1課 文法まとめ",               # Spanning header
        "日本語の勉強を始めてから、",     # Left col line 1 (no terminator)
        "半年が経ちました。",             # Left col line 2 (terminator)
        "最初は難しかったですが、",       # Right col line 1 (no terminator)
        "今は楽しく話せます。"            # Right col line 2 (terminator)
    ]
    boxes = [
        [[50, 20], [550, 20], [550, 40], [50, 40]],     # Header (w=500 > 600*0.65)
        [[60, 60], [240, 60], [240, 80], [60, 80]],     # Left 1 (y=60)
        [[60, 90], [240, 90], [240, 110], [60, 110]],   # Left 2 (y=90)
        [[360, 60], [540, 60], [540, 80], [360, 80]],   # Right 1 (y=60)
        [[360, 90], [540, 90], [540, 110], [360, 110]], # Right 2 (y=90)
    ]

    res = process_page_layout(
        txts=txts,
        boxes=boxes,
        page_width=600.0,
        page_height=800.0,
        enable_reflow=True,
        enable_two_column=True,
        enable_furigana_filter=True
    )

    clean_md = res["clean_text"]
    assert res["columns_detected"] == 2
    assert "### 第1課 文法まとめ" in clean_md
    # Lines in left col should be reflowed into one continuous sentence
    assert "日本語の勉強を始めてから、半年が経ちました。" in clean_md
    # Lines in right col should be reflowed into one continuous sentence
    assert "最初は難しかったですが、今は楽しく話せます。" in clean_md
    # Verify left col precedes right col in the text!
    left_pos = clean_md.find("半年が経ちました。")
    right_pos = clean_md.find("最初は難しかったですが")
    assert left_pos < right_pos, "Left column must appear before right column!"

    print("  [PASS] Two-Column Sorting & Sentence Reflow")


def _box(x, y, w=220, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_prose_reflow_uses_indent_and_gap_instead_of_closing_quote_alone():
    txts = [
        "本書は多くの「中、高級」",
        "の学習者にも役立ちます。",
        "次の段落は字下げして始まります。",
    ]
    boxes = [_box(60, 100, 420), _box(40, 126, 430), _box(70, 175, 400)]
    result = process_page_layout(txts, boxes, 600, 800, scores=[.99, .98, .97])
    assert result["page_type"] == "prose"
    assert "「中、高級」の学習者" in result["clean_text"]
    assert "役立ちます。\n\n次の段落" in result["clean_text"]
    assert result["blocks"][0]["lines"][0]["confidence"] == .99
    assert result["blocks"][0]["lines"][0]["bbox"] == [60, 100, 480, 120]


def test_toc_is_classified_and_rows_are_not_concatenated():
    txts = [
        "第一篇 助詞與指示", "01 單元 助詞I", "01. 〜が", "p.12", "02. 〜を", "p.15",
        "02 單元 助詞II", "06. 〜から", "p.28", "07. 〜まで", "p.30",
    ]
    boxes = [
        _box(50, 40, 500, 28), _box(50, 100, 220, 28), _box(60, 150, 150), _box(225, 150, 45),
        _box(60, 180, 150), _box(225, 180, 45), _box(330, 100, 220, 28),
        _box(340, 150, 150), _box(505, 150, 45), _box(340, 180, 150), _box(505, 180, 45),
    ]
    result = process_page_layout(txts, boxes, 600, 800, scores=[.98] * len(txts))
    assert result["page_type"] == "toc"
    assert "01. 〜が\np.12" in result["clean_text"]
    assert "p.1202." not in result["clean_text"]
    assert result["columns_detected"] == 2
    assert len(result["blocks"]) >= 3
    assert result["needs_review"] is True
    assert "complex_layout" in result["review_reasons"]


def test_question_page_and_suspicious_text_are_marked_for_review():
    txts = [
        "隨堂測驗", "次の文の（ ）に入れるのに最もよいものを選びなさい。",
        "1 似て", "2 目立って", "3 役立って", "4 合って", "媽明明明明明②",
    ]
    boxes = [_box(50, 50, 300, 30)] + [_box(60, 100 + n * 30, 430) for n in range(6)]
    result = process_page_layout(txts, boxes, 600, 800, scores=[.99, .97, .95, .94, .93, .92, .88])
    assert result["page_type"] == "questions"
    assert result["needs_review"] is True
    assert "repeated_character" in result["review_reasons"]
    suspicious = [line for block in result["blocks"] for line in block["lines"] if line["needs_review"]]
    assert suspicious[-1]["text"] == "媽明明明明明②"


def test_vocabulary_card_page_is_distinct_from_prose():
    txts = ["動詞", "022-01", "合う", "適合", "022-02", "諦める", "放棄", "動Ⅱ他4"]
    boxes = [_box(40, 40, 100, 30)] + [_box(60, 100 + n * 35, 180) for n in range(7)]
    result = process_page_layout(txts, boxes, 600, 800, scores=[.98] * len(txts))
    assert result["page_type"] == "vocabulary_cards"
    assert "complex_layout" in result["review_reasons"]


def test_structured_source_lines_preserve_filtered_text_geometry_and_spaces():
    txts = ["  頁首裝飾  ", "本文。", "かな", "123"]
    boxes = [_box(30, 5, 120, 10), _box(50, 200, 300, 24), _box(70, 230, 40, 8), _box(270, 780, 30, 15)]
    result = process_page_layout(txts, boxes, 600, 800, scores=[.8, .99, .9, .95])
    assert len(result["source_lines"]) == 4
    assert result["source_lines"][0]["raw_text"] == "  頁首裝飾  "
    assert result["source_lines"][0]["included_in_clean"] is False
    assert result["source_lines"][0]["filter_reason"] == "top_margin"
    assert result["source_lines"][1]["bbox"] == [50, 200, 350, 224]
    assert result["source_lines"][2]["filter_reason"] == "furigana"
    assert result["source_lines"][3]["filter_reason"] == "footer_page_number"
    assert [line["text"] for block in result["blocks"] for line in block["lines"]] == ["本文。"]


def test_mid_page_spanning_heading_stays_between_column_bands():
    txts = ["左上。", "右上。", "中段標題", "左下。", "右下。"]
    boxes = [
        _box(50, 100, 180), _box(350, 100, 180), _box(40, 300, 520, 30),
        _box(50, 360, 180), _box(350, 360, 180),
    ]
    result = process_page_layout(txts, boxes, 600, 800, scores=[.99] * 5)
    clean = result["clean_text"]
    assert clean.index("左上") < clean.index("中段標題") < clean.index("左下")


def test_partial_missing_geometry_is_not_reported_as_top_margin():
    result = process_page_layout(
        ["本文。", "座標なし"], [_box(50, 200, 300, 24)], 600, 800, scores=[.99, .8]
    )
    missing = result["source_lines"][1]
    assert missing["bbox"] is None
    assert missing["included_in_clean"] is False
    assert missing["filter_reason"] == "missing_geometry"
    assert missing["review_reasons"] == ["missing_geometry"]
    assert "座標なし" not in result["clean_text"]


def test_all_per_line_geometry_missing_still_preserves_source_lines():
    result = process_page_layout(["a", "b"], [None, []], 600, 800, scores=[.8, .7])
    assert [line["raw_text"] for line in result["source_lines"]] == ["a", "b"]
    assert all(line["bbox"] is None for line in result["source_lines"])
    assert result["needs_review"] is True
    assert result["review_reasons"] == ["missing_geometry"]


if __name__ == "__main__":
    test_enhance_image_for_ocr()
    test_correct_homoglyphs_and_boundaries()
    test_process_page_layout_reflow_and_two_column()
    print("\n[ALL TEXTBOOK LAYOUT UNIT TESTS PASSED!]")
