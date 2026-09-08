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


if __name__ == "__main__":
    test_enhance_image_for_ocr()
    test_correct_homoglyphs_and_boundaries()
    test_process_page_layout_reflow_and_two_column()
    print("\n[ALL TEXTBOOK LAYOUT UNIT TESTS PASSED!]")
