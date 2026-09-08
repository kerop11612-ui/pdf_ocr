"""
Production-grade Pipeline for 100MB+ Scanned PDF OCR & AI Agent Knowledge Extraction
Supports: OCRmyPDF (unpaper, deskew, tessdata_best) -> Docling / PyMuPDF -> Obsidian MD + RAG Chunks JSONL
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Auto-configure Windows Environment Paths ---
def setup_environment():
    """Ensure Ghostscript, Tesseract, and local tessdata are in PATH/TESSDATA_PREFIX."""
    base_dir = Path(__file__).resolve().parent
    local_tessdata = base_dir / "tessdata"

    paths_to_add = [
        r"C:\Users\kerop\gs\bin",
        r"C:\Program Files\Tesseract-OCR",
        os.path.expandvars(r"%APPDATA%\Python\Python311\Scripts"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\Scripts"),
    ]

    current_path = os.environ.get("PATH", "")
    new_paths = [p for p in paths_to_add if os.path.exists(p) and p not in current_path]
    if new_paths:
        os.environ["PATH"] = ";".join(new_paths) + ";" + current_path

    if local_tessdata.exists():
        os.environ["TESSDATA_PREFIX"] = str(local_tessdata)

setup_environment()

import fitz  # PyMuPDF

# --- Step 1: OCR Processing with OCRmyPDF ---
def run_ocrmypdf_100mb(
    input_pdf: Path,
    output_pdf: Path,
    lang: str = "jpn+eng",
    jobs: int = 2,
    mode: str = "force",
    clean: bool = True,
    deskew: bool = True,
    rotate_pages: bool = True,
    max_image_mpixels: int = 500,
    temp_dir: Optional[Path] = None
) -> Path:
    """
    Run OCRmyPDF optimized for 100MB+ large files.
    Restricts parallel jobs to prevent OOM on unpaper/Ghostscript.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "ocrmypdf",
        "-l", lang,
        f"--mode", mode,
        "--output-type", "pdfa",
        "--optimize", "1",
        "--jobs", str(jobs),
        "--max-image-mpixels", str(max_image_mpixels),
        "--tesseract-timeout", "180",
    ]

    has_unpaper = shutil.which("unpaper") is not None
    if clean and mode == "force":
        if has_unpaper:
            cmd.append("--clean")
        else:
            print("      [提示] 系統未安裝 unpaper，自動跳過 --clean 前處理。")
    if deskew and mode == "force":
        cmd.append("--deskew")
    if rotate_pages and mode == "force":
        cmd.append("--rotate-pages")

    if temp_dir:
        temp_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--temp-dir", str(temp_dir)])

    cmd.extend([str(input_pdf), str(output_pdf)])

    print(f"\n[1/3] 執行 OCRmyPDF: {input_pdf.name} (檔案大小: {input_pdf.stat().st_size / (1024*1024):.2f} MB)")
    print(f"      命令參數: jobs={jobs}, mode={mode}, lang={lang}")
    start_t = time.time()

    proc = subprocess.run(cmd, check=True)
    elapsed = time.time() - start_t
    print(f"      OCR 完成，耗時: {elapsed:.1f} 秒，輸出: {output_pdf}")
    return output_pdf


# --- Step 2: Fallback Layout Parser (Native PyMuPDF) ---
def parse_with_pymupdf(pdf_path: Path, doc_id: str) -> Dict[str, Any]:
    """
    High-speed, memory-efficient PDF layout parser (built-in fallback).
    Guarantees zero-OOM even on 500-page 100MB documents.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"\n[2/3] PyMuPDF 結構化解析 (總頁數: {total_pages})...")

    markdown_lines = [f"---\nsource: {doc_id}\ntype: grammar_notes\ntotal_pages: {total_pages}\n---\n"]
    chunks = []
    chunk_index = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_no = page_idx + 1
        page_text = page.get_text("text").strip()
        if not page_text:
            continue

        markdown_lines.append(f"\n> [!source] PDF Page {page_no}\n\n")

        # Extract blocks for layout awareness
        blocks = page.get_text("blocks")
        # Sort blocks by vertical position, then horizontal (reading order)
        blocks.sort(key=lambda b: (round(b[1] / 15) * 15, b[0]))

        current_heading = f"Page {page_no}"
        page_paragraphs = []

        for b in blocks:
            b_text = b[4].strip()
            if not b_text:
                continue

            # Heading heuristic: short line or key markers
            is_heading = False
            if len(b_text) < 40 and any(m in b_text for m in ["課", "文法", "ポイント", "～", "#", "N1", "N2", "N3"]):
                is_heading = True
                current_heading = b_text

            if is_heading:
                markdown_lines.append(f"### {b_text}\n\n")
            else:
                markdown_lines.append(f"{b_text}\n\n")

            page_paragraphs.append(b_text)

        # Build contextualized RAG Chunk
        combined_text = "\n".join(page_paragraphs)
        if combined_text:
            chunk_record = {
                "chunk_id": f"{doc_id}_{chunk_index:04d}",
                "text": f"[{doc_id} > {current_heading}]\n{combined_text}",
                "raw_text": combined_text,
                "pages": [page_no],
                "headings": [current_heading],
            }
            chunks.append(chunk_record)
            chunk_index += 1

    doc.close()
    return {
        "markdown": "".join(markdown_lines),
        "chunks": chunks,
        "structured_data": {"doc_id": doc_id, "total_pages": total_pages, "chunks_count": len(chunks)}
    }


# --- Step 3: Docling Deep Layout Parser (with Batching for 100MB files) ---
def parse_with_docling_batched(pdf_path: Path, doc_id: str, batch_size: int = 30) -> Dict[str, Any]:
    """
    Deep Layout Parser using Docling with batched paging to prevent OOM on 100MB PDFs.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, HeadingHierarchyOptions
    from docling.chunking import HybridChunker

    import torch
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # 關鍵：沿用 OCRmyPDF 高精字層
    pipeline_options.generate_parsed_pages = True
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True, max_level=3)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(pipeline_options, "accelerator_options"):
        pipeline_options.accelerator_options.device = device_str
    print(f"      [硬體] 啟用運算裝置: {device_str.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (RTX 5060 Ti 需安裝 CUDA 版 PyTorch 開啟)"))

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    doc_fitz = fitz.open(pdf_path)
    total_pages = len(doc_fitz)
    doc_fitz.close()

    print(f"\n[2/3] Docling 深度版面解析 (總頁數: {total_pages}, 批次大小: {batch_size})...")

    # If small enough, run in one go
    if total_pages <= batch_size:
        res = converter.convert(pdf_path)
        doc = res.document
        doc_dict = doc.export_to_dict()

        # Build Markdown with provenance
        md_lines = [f"---\nsource: {doc_id}\ntype: grammar_notes\ntotal_pages: {total_pages}\n---\n"]
        current_page = None
        for item, level in doc.iterate_items():
            if item.prov and item.prov[0].page_no != current_page:
                current_page = item.prov[0].page_no
                md_lines.append(f"\n> [!source] PDF Page {current_page}\n\n")
            if hasattr(item, "text"):
                prefix = "#" * level + " " if level > 0 else ""
                md_lines.append(f"{prefix}{item.text}\n\n")

        # HybridChunker
        chunker = HybridChunker(merge_peers=True)
        chunks_data = []
        for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
            enriched = chunker.contextualize(chunk=chunk)
            pages = list({p.page_no for item in chunk.meta.doc_items for p in (item.prov or [])})
            chunks_data.append({
                "chunk_id": f"{doc_id}_{i:04d}",
                "text": enriched,
                "raw_text": chunk.text,
                "pages": sorted(pages),
                "headings": getattr(chunk.meta, "headings", []),
            })

        return {
            "markdown": "".join(md_lines),
            "chunks": chunks_data,
            "structured_data": doc_dict
        }

    # Otherwise: Batching across page ranges
    all_md_lines = [f"---\nsource: {doc_id}\ntype: grammar_notes\ntotal_pages: {total_pages}\n---\n"]
    all_chunks = []
    chunk_counter = 0

    temp_batch_dir = pdf_path.parent / "_temp_batches"
    temp_batch_dir.mkdir(exist_ok=True)

    try:
        for start_p in range(0, total_pages, batch_size):
            end_p = min(start_p + batch_size, total_pages)
            print(f"      處理批次: 頁碼 {start_p + 1} ~ {end_p} / {total_pages}...")

            # Extract slice
            batch_pdf = temp_batch_dir / f"batch_{start_p+1}_{end_p}.pdf"
            src_doc = fitz.open(pdf_path)
            batch_doc = fitz.open()
            batch_doc.insert_pdf(src_doc, from_page=start_p, to_page=end_p - 1)
            batch_doc.save(batch_pdf)
            batch_doc.close()
            src_doc.close()

            # Convert slice
            res = converter.convert(batch_pdf)
            doc = res.document

            current_page = None
            for item, level in doc.iterate_items():
                if item.prov and item.prov[0].page_no:
                    actual_page = start_p + item.prov[0].page_no
                    if actual_page != current_page:
                        current_page = actual_page
                        all_md_lines.append(f"\n> [!source] PDF Page {actual_page}\n\n")
                if hasattr(item, "text"):
                    prefix = "#" * level + " " if level > 0 else ""
                    all_md_lines.append(f"{prefix}{item.text}\n\n")

            chunker = HybridChunker(merge_peers=True)
            for chunk in chunker.chunk(dl_doc=doc):
                enriched = chunker.contextualize(chunk=chunk)
                pages = [start_p + p.page_no for item in chunk.meta.doc_items for p in (item.prov or [])]
                all_chunks.append({
                    "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                    "text": enriched,
                    "raw_text": chunk.text,
                    "pages": sorted(list(set(pages))),
                    "headings": getattr(chunk.meta, "headings", []),
                })
                chunk_counter += 1

            # Clean slice
            if batch_pdf.exists():
                batch_pdf.unlink()

    finally:
        if temp_batch_dir.exists():
            shutil.rmtree(temp_batch_dir, ignore_errors=True)

    return {
        "markdown": "".join(all_md_lines),
        "chunks": all_chunks,
        "structured_data": {"doc_id": doc_id, "total_pages": total_pages, "chunks_count": len(all_chunks)}
    }


# --- Step 4: End-to-End Orchestrator ---
def process_pdf_pipeline(
    input_pdf: Path,
    output_dir: Path,
    lang: str = "jpn+eng",
    jobs: int = 2,
    batch_size: int = 30,
    force_pymupdf: bool = False
):
    """Full execution orchestrator."""
    input_pdf = Path(input_pdf).resolve()
    output_dir = Path(output_dir).resolve()
    doc_id = input_pdf.stem

    dir_pdf = output_dir / "01_pdf"
    dir_structured = output_dir / "02_structured"
    dir_obsidian = output_dir / "03_obsidian"
    dir_chunks = output_dir / "04_chunks"

    for d in [dir_pdf, dir_structured, dir_obsidian, dir_chunks]:
        d.mkdir(parents=True, exist_ok=True)

    ocr_pdf = dir_pdf / f"{doc_id}_OCR.pdf"

    # 1. OCR (if not already completed)
    if not ocr_pdf.exists():
        run_ocrmypdf_100mb(input_pdf, ocr_pdf, lang=lang, jobs=jobs)
    else:
        print(f"\n[1/3] 發現既有 OCR 成果: {ocr_pdf.name}，跳過 OCR 步驟。")

    # 2. Check if Docling is available
    use_docling = False
    if not force_pymupdf:
        try:
            import docling
            use_docling = True
        except ImportError:
            print("      [提示] 系統尚未安裝 docling，自動切換為高效 PyMuPDF 內建解析器。")

    if use_docling:
        parsed_result = parse_with_docling_batched(ocr_pdf, doc_id=doc_id, batch_size=batch_size)
    else:
        parsed_result = parse_with_pymupdf(ocr_pdf, doc_id=doc_id)

    # 3. Export Canonical Artifacts
    print(f"\n[3/3] 寫入標準知識庫資產至: {output_dir}")

    # 02_structured
    json_path = dir_structured / f"{doc_id}.docling.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_result["structured_data"], f, ensure_ascii=False, indent=2)
    print(f"      [OK] 結構化中樞: {json_path.name}")

    # 03_obsidian
    md_path = dir_obsidian / f"{doc_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(parsed_result["markdown"])
    print(f"      [OK] Obsidian 筆記: {md_path.name}")

    # 04_chunks
    jsonl_path = dir_chunks / f"{doc_id}_chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in parsed_result["chunks"]:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"      [OK] RAG 向量區塊: {jsonl_path.name} (共 {len(parsed_result['chunks'])} 個區塊)")

    print(f"\n[SUCCESS] 全部處理完成！可在 01~04 目錄中查閱成果。")


def main():
    parser = argparse.ArgumentParser(description="100MB+ 日文 PDF 轉 AI Agent 知識庫管線")
    parser.add_argument("input_pdf", type=str, help="輸入 PDF 檔案路徑")
    parser.add_argument("-o", "--output-dir", type=str, default="dist", help="輸出目錄 (預設: ./dist)")
    parser.add_argument("-l", "--lang", type=str, default="jpn+eng", help="OCR 語言模型 (預設: jpn+eng)")
    parser.add_argument("-j", "--jobs", type=int, default=2, help="並行核心數 (預設: 2, 避免 100MB 大檔 OOM)")
    parser.add_argument("-b", "--batch-size", type=int, default=30, help="Docling 分頁批次大小 (預設: 30 頁)")
    parser.add_argument("--fast", action="store_true", help="強制使用 PyMuPDF 高速引擎 (跳過 Docling)")

    args = parser.parse_args()
    input_path = Path(args.input_pdf)
    if not input_path.exists():
        print(f"[錯誤] 找不到檔案: {input_path}")
        sys.exit(1)

    process_pdf_pipeline(
        input_pdf=input_path,
        output_dir=Path(args.output_dir),
        lang=args.lang,
        jobs=args.jobs,
        batch_size=args.batch_size,
        force_pymupdf=args.fast
    )

if __name__ == "__main__":
    main()
