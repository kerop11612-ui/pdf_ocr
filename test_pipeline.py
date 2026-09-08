"""
End-to-End Verification Test for 100MB PDF OCR & Agent Knowledge Pipeline
"""
import sys
import json
from pathlib import Path
import fitz  # PyMuPDF

def create_sample_japanese_pdf(output_path: Path):
    """Generate a sample Japanese textbook page as test fixture."""
    doc = fitz.open()
    
    # Page 1: Grammar Header & Rules
    page1 = doc.new_page()
    page1.insert_text(
        fitz.Point(50, 60),
        "第3課 ～ようになる (JLPT N3)",
        fontsize=18,
        fontname="helv"
    )
    page1.insert_text(
        fitz.Point(50, 100),
        "意味：以前はできなかったことができる状態に変化することを表す。",
        fontsize=12,
        fontname="helv"
    )
    page1.insert_text(
        fitz.Point(50, 130),
        "接続：動詞可能形 ＋ ようになる",
        fontsize=12,
        fontname="helv"
    )
    page1.insert_text(
        fitz.Point(50, 160),
        "例文：日本に来てから、刺身が食べられるようになりました。",
        fontsize=12,
        fontname="helv"
    )

    # Page 2: Grammar Table
    page2 = doc.new_page()
    page2.insert_text(
        fitz.Point(50, 60),
        "活用変化表 (Verb Conjugation Table)",
        fontsize=16,
        fontname="helv"
    )
    page2.insert_text(
        fitz.Point(50, 100),
        "辞書形 | ます形 | 可能形",
        fontsize=12,
        fontname="helv"
    )
    page2.insert_text(
        fitz.Point(50, 125),
        "書く   | 書きます | 書ける",
        fontsize=12,
        fontname="helv"
    )
    page2.insert_text(
        fitz.Point(50, 150),
        "飲む   | 飲みます | 飲める",
        fontsize=12,
        fontname="helv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"[TEST] 產生測試樣本 PDF: {output_path} (共 2 頁)")

def verify_pipeline_outputs(output_dir: Path, doc_id: str):
    """Verify all 4 canonical pipeline outputs exist and contain valid data."""
    pdf_file = output_dir / "01_pdf" / f"{doc_id}_OCR.pdf"
    json_file = output_dir / "02_structured" / f"{doc_id}.docling.json"
    md_file = output_dir / "03_obsidian" / f"{doc_id}.md"
    jsonl_file = output_dir / "04_chunks" / f"{doc_id}_chunks.jsonl"

    print("\n[VERIFY] 檢查產物完整性...")

    assert pdf_file.exists(), f"Missing PDF output: {pdf_file}"
    assert pdf_file.stat().st_size > 0, "PDF file is empty"
    print(f"  [PASS] 01_pdf: {pdf_file.name} ({pdf_file.stat().st_size} bytes)")

    assert json_file.exists(), f"Missing JSON output: {json_file}"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  [PASS] 02_structured: {json_file.name} (Valid JSON)")

    assert md_file.exists(), f"Missing Markdown output: {md_file}"
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    assert "> [!source] PDF Page" in md_content, "Markdown missing page callouts"
    print(f"  [PASS] 03_obsidian: {md_file.name} (Contains '> [!source]' Callouts)")

    assert jsonl_file.exists(), f"Missing Chunks JSONL: {jsonl_file}"
    chunk_count = 0
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                assert "text" in record, "Chunk missing 'text'"
                assert "pages" in record, "Chunk missing 'pages'"
                chunk_count += 1
    assert chunk_count > 0, "No chunks generated in JSONL"
    print(f"  [PASS] 04_chunks: {jsonl_file.name} ({chunk_count} contextualized chunks)")

    print("\n[ALL PASS] 全部四層產物驗證成功！")

if __name__ == "__main__":
    from pipeline import process_pdf_pipeline

    test_input = Path("test_fixtures/sample_n3.pdf")
    test_dist = Path("test_dist")

    create_sample_japanese_pdf(test_input)
    process_pdf_pipeline(
        input_pdf=test_input,
        output_dir=test_dist,
        lang="jpn+eng",
        jobs=2,
        batch_size=10
    )
    verify_pipeline_outputs(test_dist, doc_id="sample_n3")
