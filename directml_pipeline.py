"""
DirectML & CUDA GPU Pipeline for Japanese + Traditional Chinese PDF Knowledge Extraction
Features:
- Generator-based Page-by-Page Streaming for Real-Time UI Feedback & Progress Tracking
- Multilingual Support: Japanese + Traditional Chinese (zh-TW) via PP-OCRv6 Medium (18,710 CJK Dict)
- Adaptive DPI: 250 DPI Base, 350 DPI Auto-Retry on Low Confidence (< 0.72)
- Layout-Aware Sorting and Obsidian Callout Formatting
- Hardware Acceleration: CUDA (NVIDIA) > DirectML (Windows GPU) > CPU fallback
"""

import os
import sys
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Generator, Tuple, Union
import numpy as np
import fitz  # PyMuPDF
from rapidocr import RapidOCR
from rapidocr.utils.typings import ModelType
from textbook_layout import (
    enhance_image_for_ocr,
    correct_homoglyphs_and_boundaries,
    process_page_layout
)


def _unique_provider_names(provider_sets: List[List[str]]) -> List[str]:
    """依 session 出現順序去重，供人類可讀 telemetry 使用。"""
    return list(dict.fromkeys(name for providers in provider_sets for name in providers))


def _engine_provider_sets(engine: Any) -> List[List[str]]:
    """讀取 RapidOCR 各 ONNX session 的實際 provider，絕不以建構選項推測。"""
    sets: List[List[str]] = []
    for component in (getattr(engine, "text_det", None), getattr(engine, "text_rec", None), getattr(engine, "text_cls", None)):
        session = getattr(getattr(component, "session", None), "session", None)
        if session and hasattr(session, "get_providers"):
            sets.append(list(session.get_providers()))
    return sets


# --- OCR Backend Abstraction ---
class BaseOCRBackend(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        pass


class RapidOCRCUDABackend(BaseOCRBackend):
    def __init__(self, lang: str = "japan", high_accuracy: bool = True, device_id: int = 0):
        params = {
            "EngineConfig.onnxruntime.use_cuda": True,
            "EngineConfig.onnxruntime.cuda_ep_cfg": {
                "device_id": device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
            "Rec.lang_type": lang,
            "Global.log_level": "error"
        }
        if high_accuracy:
            params["Det.model_type"] = ModelType.MEDIUM
            params["Rec.model_type"] = ModelType.MEDIUM
        self.engine = RapidOCR(params=params)
        self.provider_sets = _engine_provider_sets(self.engine)
        if not self.provider_sets or any("CUDAExecutionProvider" not in providers for providers in self.provider_sets):
            raise RuntimeError(f"CUDA session provider 未啟用：{self.provider_sets}")
        self.provider_names = _unique_provider_names(self.provider_sets)
        self.backend_actual = f"CUDA GPU (device:{device_id})"

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        res = self.engine(image)
        if not res or not res.txts:
            return {"txts": [], "scores": [], "boxes": [], "avg_score": 0.0}
        scores = res.scores if res.scores else []
        avg_score = float(np.mean(scores)) if scores else 0.0
        return {
            "txts": res.txts,
            "scores": scores,
            "boxes": res.boxes,
            "avg_score": avg_score
        }


class RapidOCRDirectMLBackend(BaseOCRBackend):
    def __init__(self, lang: str = "japan", high_accuracy: bool = True):
        params = {
            "EngineConfig.onnxruntime.use_dml": True,
            "Rec.lang_type": lang,
            "Global.log_level": "error"
        }
        if high_accuracy:
            params["Det.model_type"] = ModelType.MEDIUM
            params["Rec.model_type"] = ModelType.MEDIUM
        self.engine = RapidOCR(params=params)
        self.provider_sets = _engine_provider_sets(self.engine)
        if not self.provider_sets or any("DmlExecutionProvider" not in providers for providers in self.provider_sets):
            raise RuntimeError(f"DirectML session provider 未啟用：{self.provider_sets}")
        self.provider_names = _unique_provider_names(self.provider_sets)
        self.backend_actual = "DirectML GPU"

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        res = self.engine(image)
        if not res or not res.txts:
            return {"txts": [], "scores": [], "boxes": [], "avg_score": 0.0}
        scores = res.scores if res.scores else []
        avg_score = float(np.mean(scores)) if scores else 0.0
        return {
            "txts": res.txts,
            "scores": scores,
            "boxes": res.boxes,
            "avg_score": avg_score
        }


class RapidOCRCPUBackend(BaseOCRBackend):
    def __init__(self, lang: str = "japan"):
        self.engine = RapidOCR(params={
            "EngineConfig.onnxruntime.use_dml": False,
            "EngineConfig.onnxruntime.use_cuda": False,
            "Rec.lang_type": lang,
            "Global.log_level": "error"
        })
        self.provider_sets = _engine_provider_sets(self.engine)
        self.provider_names = _unique_provider_names(self.provider_sets)
        self.backend_actual = self.provider_names[0] if self.provider_names else "CPU"

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        res = self.engine(image)
        if not res or not res.txts:
            return {"txts": [], "scores": [], "boxes": [], "avg_score": 0.0}
        scores = res.scores if res.scores else []
        avg_score = float(np.mean(scores)) if scores else 0.0
        return {
            "txts": res.txts,
            "scores": scores,
            "boxes": res.boxes,
            "avg_score": avg_score
        }


def get_ocr_backend(backend_mode: str = "auto", lang: str = "japan", high_accuracy: bool = True) -> BaseOCRBackend:
    """依使用者模式建立後端；優先序：CUDA > DirectML > CPU。auto 模式下平滑降級。"""
    valid_modes = {"auto", "gpu", "cuda", "dml", "cpu"}
    if backend_mode not in valid_modes:
        raise ValueError(f"未知後端模式: {backend_mode}，可選: {valid_modes}")

    # 1. 明確要求 CUDA
    if backend_mode == "cuda":
        return RapidOCRCUDABackend(lang=lang, high_accuracy=high_accuracy)

    # 2. 明確要求 DirectML
    if backend_mode == "dml":
        return RapidOCRDirectMLBackend(lang=lang, high_accuracy=high_accuracy)

    # 3. GPU 或 Auto 模式：優先嘗試 CUDA，次選 DirectML，末選 CPU
    if backend_mode in {"auto", "gpu"}:
        errors: list[str] = []
        try:
            backend = RapidOCRCUDABackend(lang=lang, high_accuracy=high_accuracy)
            provider_sets = getattr(backend, "provider_sets", [])
            if not provider_sets or any("CUDAExecutionProvider" not in providers for providers in provider_sets):
                raise RuntimeError(f"CUDA session provider 未啟用：{provider_sets}")
            return backend
        except Exception as exc:
            errors.append(f"CUDA: {exc}")

        try:
            backend = RapidOCRDirectMLBackend(lang=lang, high_accuracy=high_accuracy)
            provider_sets = getattr(backend, "provider_sets", [])
            if not provider_sets or any("DmlExecutionProvider" not in providers for providers in provider_sets):
                raise RuntimeError(f"DirectML session provider 未啟用：{provider_sets}")
            return backend
        except Exception as exc:
            errors.append(f"DirectML: {exc}")

        if backend_mode == "gpu":
            raise RuntimeError(f"指定 GPU 初始化失敗：{'；'.join(errors)}")

        backend = RapidOCRCPUBackend(lang=lang)
        backend.backend_actual = f"{getattr(backend, 'backend_actual', 'CPU')}（DirectML 初始化失敗後回退：{'；'.join(errors)}）"
        return backend

    return RapidOCRCPUBackend(lang=lang)


def render_page_to_np(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))


def process_pdf_gpu_adaptive_generator(
    input_pdf: Path,
    output_dir: Path,
    base_dpi: int = 250,
    adaptive_retry_dpi: int = 350,
    confidence_thresh: float = 0.72,
    lang: str = "japan",
    start_page: int = 1,
    end_page: Optional[int] = None,
    high_accuracy: bool = True,
    enable_clahe: bool = True,
    enable_reflow: bool = True,
    enable_two_column: bool = True,
    enable_furigana_filter: bool = True,
    enable_homoglyphs: bool = True,
    backend_mode: str = "auto",
    adaptive_retry: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
    stream_full_text: bool = True,
    on_page: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Generator[Tuple[float, str, str, str, bool, Optional[Dict[str, Any]]], None, None]:
    """
    Generator that processes a PDF page-by-page focusing purely on:
    1. Layout restoration (Two-column XY-Cut, sentence reflow, furigana filtering)
    2. Recognition accuracy (CLAHE contrast boost, homoglyph correction, sticky numbers)
    Yields: (progress_ratio, status_msg, current_clean_markdown, current_raw_text, is_done, final_data)
    Set stream_full_text=False to avoid rebuilding cumulative previews. on_page receives
    each completed page once; exceptions propagate so persistence failures fail the job.
    """
    input_pdf = Path(input_pdf).resolve()
    output_dir = Path(output_dir).resolve()
    doc_id = input_pdf.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    start_t = time.perf_counter()
    backend = get_ocr_backend(backend_mode=backend_mode, lang=lang, high_accuracy=high_accuracy)
    initialization_sec = time.perf_counter() - start_t
    doc = fitz.open(input_pdf)
    total_doc_pages = len(doc)

    s_idx = max(0, start_page - 1)
    e_idx = min(total_doc_pages, end_page) if end_page else total_doc_pages
    pages_to_process = list(range(s_idx, e_idx))
    process_count = len(pages_to_process)

    clean_pages_md = []
    raw_pages_txt = []
    page_stats = []
    structured_pages = []

    total_corrections = 0
    total_furigana_filtered = 0
    multi_col_pages = 0

    try:
        for step_num, idx in enumerate(pages_to_process):
            page = doc[idx]
            page_no = idx + 1

            if cancel_check and cancel_check():
                yield (step_num / process_count if process_count else 0.0, "已取消，保留已完成頁面", "\n\n".join(clean_pages_md) if stream_full_text else "", "\n\n".join(raw_pages_txt) if stream_full_text else "", True, None)
                return

            # 1. Base DPI Render & Optional CLAHE Contrast Boost
            t0 = time.perf_counter()
            img = render_page_to_np(page, base_dpi)
            render_end = time.perf_counter()
            if enable_clahe:
                img = enhance_image_for_ocr(img)
            preprocess_end = time.perf_counter()
            ocr_res = backend.recognize(img)
            recognition_end = time.perf_counter()

            retried = False
            retry_attempted = False
            retry_sec = 0.0
            selected_dpi = base_dpi
            # 2. Adaptive DPI Retry on low confidence
            if adaptive_retry and ocr_res["avg_score"] < confidence_thresh:
                retry_attempted = True
                retry_start = time.perf_counter()
                img_retry = render_page_to_np(page, adaptive_retry_dpi)
                if enable_clahe:
                    img_retry = enhance_image_for_ocr(img_retry)
                ocr_retry = backend.recognize(img_retry)
                if ocr_retry["avg_score"] > ocr_res["avg_score"]:
                    ocr_res = ocr_retry
                    retried = True
                    selected_dpi = adaptive_retry_dpi
                retry_sec = time.perf_counter() - retry_start

            txts = ocr_res.get("txts", [])
            boxes = ocr_res.get("boxes", [])

            # 3. Layout Restoration & Accuracy Correction
            scale = selected_dpi / 72.0
            p_rect = page.rect
            pw = p_rect.width * scale
            ph = p_rect.height * scale

            layout_start = time.perf_counter()
            layout_res = process_page_layout(
                txts=txts,
                boxes=boxes,
                scores=ocr_res.get("scores", []),
                page_width=pw,
                page_height=ph,
                enable_reflow=enable_reflow,
                enable_two_column=enable_two_column,
                enable_furigana_filter=enable_furigana_filter,
                enable_homoglyphs=enable_homoglyphs
            )
            layout_end = time.perf_counter()

            total_corrections += layout_res["corrections_count"]
            total_furigana_filtered += layout_res["furigana_filtered"]
            if layout_res["columns_detected"] > 1:
                multi_col_pages += 1

            if layout_res["clean_text"]:
                clean_pages_md.append(f"## Page {page_no}\n\n{layout_res['clean_text']}\n")
            if layout_res["raw_text"]:
                raw_pages_txt.append(f"=== [Page {page_no}] ===\n{layout_res['raw_text']}\n")

            page_record = {
                "page": page_no, "clean": layout_res["clean_text"].strip(), "raw": layout_res["raw_text"].strip(),
                "page_type": layout_res["page_type"], "blocks": layout_res["blocks"],
                "source_lines": layout_res["source_lines"],
                "needs_review": layout_res["needs_review"], "review_reasons": layout_res["review_reasons"],
                "page_width": pw, "page_height": ph,
            }
            structured_pages.append(page_record)
            if on_page is not None:
                on_page(page_record)

            page_stats.append({
                "page": page_no,
                "time_sec": round(layout_end - t0, 3),
                "render_sec": round(render_end - t0, 3),
                "preprocess_sec": round(preprocess_end - render_end, 3),
                "recognition_sec": round(recognition_end - preprocess_end, 3),
                "retry_sec": round(retry_sec, 3),
                "layout_sec": round(layout_end - layout_start, 3),
                "confidence": round(ocr_res["avg_score"], 3),
                "columns": layout_res["columns_detected"],
                "corrections": layout_res["corrections_count"],
                "retried": retried,
                "retry_attempted": retry_attempted,
                "retry_selected": retried,
                "lines": len(txts),
                "page_type": layout_res["page_type"],
                "needs_review": layout_res["needs_review"],
                "review_reasons": layout_res["review_reasons"],
            })

            elapsed_so_far = time.perf_counter() - start_t
            completed_steps = step_num + 1
            avg_speed = completed_steps / elapsed_so_far if elapsed_so_far > 0 else 0.0
            remaining_pages = process_count - completed_steps
            eta_sec = remaining_pages / avg_speed if avg_speed > 0 else 0.0

            status_msg = (
                f"處理中: 第 {completed_steps}/{process_count} 頁 (P.{page_no}) · "
                f"速度: {avg_speed:.1f} 頁/秒 · 預估剩餘: {eta_sec:.1f} 秒 · "
                f"同形字/黏字校正: {total_corrections} 處"
            )
            progress_ratio = completed_steps / process_count

            cur_clean = "\n\n".join(clean_pages_md) if stream_full_text else ""
            cur_raw = "\n\n".join(raw_pages_txt) if stream_full_text else ""
            yield (progress_ratio, status_msg, cur_clean, cur_raw, False, None)

        doc.close()

        full_clean_md = f"# {doc_id} · 日文教材排版還原筆記\n\n" + "\n\n".join(clean_pages_md)
        full_raw_txt = "\n\n".join(raw_pages_txt)

        clean_md_file = output_dir / f"{doc_id}_clean.md"
        with open(clean_md_file, "w", encoding="utf-8") as f:
            f.write(full_clean_md)

        raw_txt_file = output_dir / f"{doc_id}_raw.txt"
        with open(raw_txt_file, "w", encoding="utf-8") as f:
            f.write(full_raw_txt)

        structured_file = output_dir / f"{doc_id}_pages.jsonl"
        with open(structured_file, "w", encoding="utf-8", newline="\n") as f:
            for page_record in structured_pages:
                f.write(json.dumps(page_record, ensure_ascii=False) + "\n")

        total_elapsed = time.perf_counter() - start_t
        fps = process_count / total_elapsed if total_elapsed > 0 else 0.0
        metrics_data = {
            "doc_id": doc_id,
            "total_pages": total_doc_pages,
            "processed_pages": process_count,
            "total_elapsed_sec": round(total_elapsed, 2),
            "initialization_sec": round(initialization_sec, 3),
            "fps": round(fps, 2),
            "total_corrections": total_corrections,
            "total_furigana_filtered": total_furigana_filtered,
            "multi_column_pages": multi_col_pages,
            "backend_actual": (getattr(backend, "backend_actual", None) if isinstance(getattr(backend, "backend_actual", None), str) else backend.__class__.__name__),
            "page_stats": page_stats
        }
        metrics_file = output_dir / f"{doc_id}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)

        final_dict = {
            "doc_id": doc_id,
            "total_pages": total_doc_pages,
            "processed_pages": process_count,
            "total_elapsed": total_elapsed,
            "fps": fps,
            "md_path": clean_md_file,
            "raw_path": raw_txt_file,
            "metrics_path": metrics_file,
            "structured_path": structured_file,
            "markdown_content": full_clean_md,
            "raw_content": full_raw_txt,
            "metrics_data": metrics_data,
            "backend_actual": metrics_data["backend_actual"]
        }

        final_status = (
            f"處理完成！共 {process_count} 頁 (速度: {fps:.2f} 頁/秒) · "
            f"同形字/黏字校正: {total_corrections} 處 · 雙欄偵測: {multi_col_pages} 頁"
        )
        yield (1.0, final_status, full_clean_md if stream_full_text else "", full_raw_txt if stream_full_text else "", True, final_dict)
    finally:
        if not doc.is_closed:
            doc.close()


def process_pdf_gpu_adaptive(
    input_pdf: Path,
    output_dir: Path,
    base_dpi: int = 250,
    adaptive_retry_dpi: int = 350,
    confidence_thresh: float = 0.72,
    lang: str = "japan",
    start_page: int = 1,
    end_page: Optional[int] = None,
    high_accuracy: bool = True,
    enable_clahe: bool = True,
    enable_reflow: bool = True,
    enable_two_column: bool = True,
    enable_furigana_filter: bool = True,
    enable_homoglyphs: bool = True,
    backend_mode: str = "auto",
    adaptive_retry: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Dict[str, Any]:
    """Compatibility wrapper that runs the generator to completion."""
    last_res = None
    for item in process_pdf_gpu_adaptive_generator(
        input_pdf=input_pdf,
        output_dir=output_dir,
        base_dpi=base_dpi,
        adaptive_retry_dpi=adaptive_retry_dpi,
        confidence_thresh=confidence_thresh,
        lang=lang,
        start_page=start_page,
        end_page=end_page,
        high_accuracy=high_accuracy,
        enable_clahe=enable_clahe,
        enable_reflow=enable_reflow,
        enable_two_column=enable_two_column,
        enable_furigana_filter=enable_furigana_filter,
        enable_homoglyphs=enable_homoglyphs,
        backend_mode=backend_mode,
        adaptive_retry=adaptive_retry,
        cancel_check=cancel_check
    ):
        last_res = item
    return last_res[5] if last_res else {}


def process_single_page(
    doc_or_path: Union[Path, str, fitz.Document],
    page_no: int,
    base_dpi: int = 250,
    adaptive_retry_dpi: int = 350,
    confidence_thresh: float = 0.72,
    enable_clahe: bool = True,
    enable_reflow: bool = True,
    enable_two_column: bool = True,
    enable_furigana_filter: bool = True,
    enable_homoglyphs: bool = True,
    backend_mode: str = "auto",
    adaptive_retry: bool = True,
) -> Dict[str, Any]:
    """Process a single page of a PDF and return its full layout record."""
    close_doc = False
    if isinstance(doc_or_path, (str, Path)):
        doc = fitz.open(doc_or_path)
        close_doc = True
    else:
        doc = doc_or_path

    try:
        page = doc[page_no - 1]
        backend = get_ocr_backend(backend_mode=backend_mode)
        img = render_page_to_np(page, base_dpi)
        if enable_clahe:
            img = enhance_image_for_ocr(img)
        ocr_res = backend.recognize(img)
        selected_dpi = base_dpi

        if adaptive_retry and ocr_res["avg_score"] < confidence_thresh:
            img_retry = render_page_to_np(page, adaptive_retry_dpi)
            if enable_clahe:
                img_retry = enhance_image_for_ocr(img_retry)
            ocr_retry = backend.recognize(img_retry)
            if ocr_retry["avg_score"] > ocr_res["avg_score"]:
                ocr_res = ocr_retry
                selected_dpi = adaptive_retry_dpi

        scale = selected_dpi / 72.0
        p_rect = page.rect
        pw = p_rect.width * scale
        ph = p_rect.height * scale

        layout_res = process_page_layout(
            txts=ocr_res.get("txts", []),
            boxes=ocr_res.get("boxes", []),
            scores=ocr_res.get("scores", []),
            page_width=pw,
            page_height=ph,
            enable_reflow=enable_reflow,
            enable_two_column=enable_two_column,
            enable_furigana_filter=enable_furigana_filter,
            enable_homoglyphs=enable_homoglyphs,
        )

        return {
            "page": page_no,
            "clean": layout_res["clean_text"].strip(),
            "raw": layout_res["raw_text"].strip(),
            "page_type": layout_res["page_type"],
            "blocks": layout_res["blocks"],
            "source_lines": layout_res["source_lines"],
            "needs_review": layout_res["needs_review"],
            "review_reasons": layout_res["review_reasons"],
            "page_width": pw,
            "page_height": ph,
        }
    finally:
        if close_doc:
            doc.close()
