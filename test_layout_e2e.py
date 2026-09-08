"""
End-to-End Test for Layout Preservation and Accuracy Enhancement.
Verifies:
1. CLAHE preprocessing executed without errors.
2. Homoglyphs & sticky numbers corrected in clean Markdown.
3. Raw OCR text preserved in clean text output.
4. Layout metrics JSON generated.
"""

import os
import json
import shutil
from pathlib import Path
import fitz

from directml_pipeline import process_pdf_gpu_adaptive

def create_sample_textbook_pdf(pdf_path: Path):
    doc = fitz.open()
    font_path = r"C:\Windows\Fonts\msgothic.ttc"
    has_font = Path(font_path).exists()

    # Page 1: Grammar header, Sticky Number, and Broken Lines
    p1 = doc.new_page(width=595, height=842)
    if has_font:
        p1.insert_font(fontname="msgothic", fontfile=font_path)
        fn = "msgothic"
    else:
        fn = "helv"

    p1.insert_text(fitz.Point(50, 60), "第3課 〜ようになる (JLPT N3)", fontsize=18, fontname=fn)
    p1.insert_text(fitz.Point(50, 100), "意味：以前はできなかったことができる状態に", fontsize=12, fontname=fn)
    p1.insert_text(fitz.Point(50, 120), "変化することを表す。", fontsize=12, fontname=fn)
    p1.insert_text(fitz.Point(50, 160), "14次の文を読んでコ一ヒ一を飲む。", fontsize=12, fontname=fn)

    # Page 2: Two-column layout
    p2 = doc.new_page(width=595, height=842)
    if has_font:
        p2.insert_font(fontname="msgothic", fontfile=font_path)
    # Header across top
    p2.insert_text(fitz.Point(50, 50), "【語彙と文法の比較対照】", fontsize=16, fontname=fn)
    # Left column (x=60)
    p2.insert_text(fitz.Point(60, 100), "左欄項目A：日本語の文法基礎。", fontsize=11, fontname=fn)
    p2.insert_text(fitz.Point(60, 130), "左欄項目B：動詞の活用変化。", fontsize=11, fontname=fn)
    # Right column (x=330)
    p2.insert_text(fitz.Point(330, 100), "右欄項目A：日常会話の表現。", fontsize=11, fontname=fn)
    p2.insert_text(fitz.Point(330, 130), "右欄項目B：敬語と丁寧語の違い。", fontsize=11, fontname=fn)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(pdf_path))
    doc.close()

def main():
    test_pdf = Path("test_fixtures/textbook_sample.pdf")
    test_dist = Path("test_layout_dist")
    if test_dist.exists():
        shutil.rmtree(test_dist, ignore_errors=True)

    create_sample_textbook_pdf(test_pdf)
    print(f"[1/3] 測試日文教材產生完成: {test_pdf}")

    res = process_pdf_gpu_adaptive(
        input_pdf=test_pdf,
        output_dir=test_dist,
        base_dpi=200,
        lang="japan",
        enable_clahe=True,
        enable_reflow=True,
        enable_two_column=True,
        enable_furigana_filter=True
    )

    doc_id = test_pdf.stem
    print(f"[2/3] 處理完成，檢查產物: {doc_id}")

    # 1. Check clean Markdown output
    clean_md_path = test_dist / f"{doc_id}_clean.md"
    assert clean_md_path.exists(), f"Missing {clean_md_path}"
    with open(clean_md_path, "r", encoding="utf-8") as f:
        clean_text = f.read()
    assert "# textbook_sample · 日文教材排版還原筆記" in clean_text
    # Sentence reflow check: two broken lines merged into one
    assert "以前はできなかったことができる状態に変化することを表す。" in clean_text
    # Homoglyph and sticky number check
    assert "14 次の文" in clean_text
    assert "コーヒー" in clean_text
    print(f"  [PASS] Clean Markdown: {clean_md_path.name} (Reflow & Homoglyphs verified)")

    # 2. Check raw TXT output
    raw_txt_path = test_dist / f"{doc_id}_raw.txt"
    assert raw_txt_path.exists(), f"Missing {raw_txt_path}"
    with open(raw_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    assert "Page 1" in raw_text
    print(f"  [PASS] Raw TXT: {raw_txt_path.name}")

    # 3. Check Metrics JSON
    metrics_path = test_dist / f"{doc_id}_metrics.json"
    assert metrics_path.exists(), f"Missing {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["total_pages"] == 2
    assert metrics["total_corrections"] >= 1
    assert metrics["multi_column_pages"] == 1
    print(f"  [PASS] Metrics JSON: {metrics_path.name} (Corrections: {metrics['total_corrections']}, Multi-col pages: {metrics['multi_column_pages']})")

    print("\n[ALL LAYOUT & ACCURACY E2E TESTS PASSED!]")

if __name__ == "__main__":
    main()
