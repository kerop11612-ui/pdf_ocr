"""
Unit and Integration Tests for ocr_quality.py
Verifies:
1. evaluate_page_quality accuracy and thresholding
2. extract_questions_and_answers pairing and deficiency detection
3. generate_ocr_review_report structure and image links
4. save_page_preview_image PyMuPDF rendering
"""

import os
import json
import pytest
from pathlib import Path
import fitz

from ocr_quality import (
    evaluate_page_quality,
    extract_questions_and_answers,
    generate_ocr_review_report,
    save_page_preview_image
)

def test_evaluate_page_quality_high():
    text_lines = [
        "第3課 ～ようになる",
        "意味：以前はできなかったことができる状態に変化することを表す。",
        "接続：動詞可能形 ＋ ようになる",
        "例文：日本に来てから、刺身が食べられるようになりました。"
    ]
    res = evaluate_page_quality(page_no=1, text_lines=text_lines, avg_ocr_conf=0.96)
    assert res["quality"] == "high"
    assert res["score"] >= 80
    assert res["needs_review"] is False
    assert res["metrics"]["jp_ratio"] > 0.6


def test_evaluate_page_quality_low_with_anomalies():
    text_lines = [
        "14次の文の（ ）に入れるのに最もよいものを、1・2・3・4から一つ選びなさい。",
        "□ はどこですか？",
        "あ",
        "異常な文字 >2>#"
    ]
    res = evaluate_page_quality(page_no=83, text_lines=text_lines, avg_ocr_conf=0.65)
    assert res["needs_review"] is True
    assert res["score"] < 75
    # Reasons should mention sticky question numbers, garbled chars, isolated kana
    reasons_str = " ".join(res["reasons"])
    assert "黏連" in reasons_str
    assert "亂碼" in reasons_str or "替換" in reasons_str
    assert "假名單獨成行" in reasons_str


def test_extract_and_pair_qa():
    pages_data = [
        {
            "page_no": 66,
            "lines": [
                "問1 来週の会議の資料は、明日までに（ ）おいてください。",
                "① 作って ② 書いて ③ 調べて ④ 読んで",
                "問2 このペンは（ ）やすいです。",
                "① 書き ② 読み"  # Missing 3 & 4
            ]
        },
        {
            "page_no": 67,
            "lines": [
                "【解答と解説】",
                "問1: 1",
                "1番の「作っておいて」が自然な文脈です。"
            ]
        }
    ]

    qa_results = extract_questions_and_answers(pages_data)
    assert len(qa_results) == 2

    # Q1: fully paired
    q1 = next(q for q in qa_results if q["id"] == 1)
    assert q1["answer"] == "1"
    assert 66 in q1["source_pages"]
    assert 67 in q1["source_pages"]
    assert len(q1["choices"]) == 4
    assert q1["needs_review"] is False

    # Q2: missing choices (< 4) and missing answer in answer pool
    q2 = next(q for q in qa_results if q["id"] == 2)
    assert q2["answer"] is None
    assert q2["needs_review"] is True
    assert len(q2["choices"]) == 2
    assert any("選項不足" in r for r in q2["reasons"])
    assert any("未配對到" in r for r in q2["reasons"])


def test_generate_ocr_review_report_and_image(tmp_path):
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text(fitz.Point(50, 50), "Test Page Content")
    
    img_dir = tmp_path / "pages"
    saved_img = save_page_preview_image(p, page_no=83, output_dir=img_dir)
    assert saved_img.exists()
    assert saved_img.name == "page_083.png"
    doc.close()

    page_qualities = [
        {
            "page": 83,
            "quality": "low",
            "score": 58,
            "needs_review": True,
            "reasons": ["題號與句子黏連", "偵測到亂碼"],
            "metrics": {"jp_ratio": 0.55}
        }
    ]
    qa_records = [
        {
            "id": 83,
            "source_pages": [83],
            "choices": {"1": "A", "2": "B"},
            "answer": None,
            "reasons": ["選項不足 4 個", "未配對到標準答案"],
            "needs_review": True
        }
    ]

    report = generate_ocr_review_report(
        doc_id="test_book",
        page_qualities=page_qualities,
        qa_records=qa_records,
        image_rel_dir="../pages"
    )

    assert "# OCR 待確認與品質稽核報告: test_book" in report
    assert "page_083.png" in report
    assert "P.83" in report
    assert "選項不足" in report
    print("\n[ALL UNIT TESTS PASSED]")


if __name__ == "__main__":
    test_evaluate_page_quality_high()
    test_evaluate_page_quality_low_with_anomalies()
    test_extract_and_pair_qa()
    test_generate_ocr_review_report_and_image(Path("test_fixtures"))
    print("Test executed directly successfully.")
