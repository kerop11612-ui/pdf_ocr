"""
Multi-Session x Multi-DPI Benchmark on RTX 5060 Ti (16GB VRAM)
Architecture: Decoupled BaseOCRBackend with Multi-Session Pooling & Adaptive DPI
"""

import os
import sys
import time
import json
import queue
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import fitz  # PyMuPDF
from rapidocr import RapidOCR

# Fix Windows CP950 console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- 1. OCR Backend Abstraction ---
class BaseOCRBackend(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        """Recognize text in an image (RGB uint8 array)."""
        pass

class RapidOCRDirectMLBackend(BaseOCRBackend):
    def __init__(self, lang: str = "japan"):
        self.engine = RapidOCR(params={
            "EngineConfig.onnxruntime.use_dml": True,
            "Rec.lang_type": lang,
            "Global.log_level": "error"
        })

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

# --- 2. Multi-Session Worker Pool ---
class MultiSessionOCRPool:
    """Pool of independent DirectML sessions to allow concurrent GPU execution."""
    def __init__(self, num_sessions: int = 1, lang: str = "japan"):
        self.num_sessions = num_sessions
        self.pool = queue.Queue()
        for _ in range(num_sessions):
            self.pool.put(RapidOCRDirectMLBackend(lang=lang))

    def process_task(self, img: np.ndarray) -> Dict[str, Any]:
        session = self.pool.get()
        try:
            return session.recognize(img)
        finally:
            self.pool.put(session)

# --- 3. Matrix Benchmark Execution ---
def render_page(doc: fitz.Document, page_idx: int, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = doc[page_idx].get_pixmap(matrix=mat, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))

def benchmark_configuration(pdf_path: Path, dpi: int, num_sessions: int, adaptive: bool = False) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # 預先在主線程渲染影像，杜絕 PyMuPDF C++ 多線程 race condition
    t_render_start = time.time()
    rendered_images = [render_page(doc, i, dpi) for i in range(total_pages)]
    total_render_sec = time.time() - t_render_start
    doc.close()

    pool = MultiSessionOCRPool(num_sessions=num_sessions)
    t_run_start = time.time()

    def process_single_page(idx: int):
        img = rendered_images[idx]
        t1 = time.time()
        result = pool.process_task(img)
        t_ocr = time.time() - t1

        retried = False
        # Adaptive DPI Rule: if confidence < 0.72, re-render at 350 DPI and re-run
        if adaptive and result["avg_score"] < 0.72:
            retry_doc = fitz.open(pdf_path)
            img_retry = render_page(retry_doc, idx, 350)
            retry_doc.close()
            res_retry = pool.process_task(img_retry)
            if res_retry["avg_score"] > result["avg_score"]:
                result = res_retry
                retried = True

        return {
            "page_idx": idx,
            "render_ms": (total_render_sec / total_pages) * 1000,
            "ocr_ms": t_ocr * 1000,
            "confidence": result["avg_score"],
            "lines": len(result["txts"]),
            "retried": retried
        }

    page_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_sessions) as executor:
        futures = [executor.submit(process_single_page, i) for i in range(total_pages)]
        for f in concurrent.futures.as_completed(futures):
            page_results.append(f.result())

    total_time = time.time() - t_run_start

    page_results.sort(key=lambda x: x["page_idx"])
    fps = total_pages / total_time
    avg_conf = np.mean([p["confidence"] for p in page_results])
    avg_render = np.mean([p["render_ms"] for p in page_results])
    avg_ocr = np.mean([p["ocr_ms"] for p in page_results])
    retries_count = sum(1 for p in page_results if p["retried"])
    proj_300p_sec = 300 / fps

    return {
        "dpi": dpi,
        "sessions": num_sessions,
        "adaptive": adaptive,
        "total_pages": total_pages,
        "total_time_sec": round(total_time, 2),
        "fps": round(fps, 2),
        "avg_render_ms": round(avg_render, 1),
        "avg_ocr_ms": round(avg_ocr, 1),
        "avg_confidence_pct": round(avg_conf * 100, 1),
        "retried_pages": retries_count,
        "projected_300p_sec": round(proj_300p_sec, 1)
    }

def main():
    test_pdf = Path("test_fixtures/benchmark_20p.pdf")
    if not test_pdf.exists():
        print(f"[錯誤] 找不到測試檔案: {test_pdf}")
        sys.exit(1)

    print("==========================================================================")
    print("  RTX 5060 Ti DirectML 矩陣基準測試 (20 頁日文教材)")
    print("  維度: DPI (250, 300) x Sessions (1, 2, 4) + Adaptive DPI (250->350)")
    print("==========================================================================")

    matrix_configs = [
        {"dpi": 250, "sessions": 1, "adaptive": False},
        {"dpi": 250, "sessions": 2, "adaptive": False},
        {"dpi": 250, "sessions": 4, "adaptive": False},
        {"dpi": 300, "sessions": 1, "adaptive": False},
        {"dpi": 300, "sessions": 2, "adaptive": False},
        {"dpi": 300, "sessions": 4, "adaptive": False},
        {"dpi": 250, "sessions": 2, "adaptive": True},  # 自適應模式
    ]

    all_results = []
    for cfg in matrix_configs:
        label = f"DPI={cfg['dpi']} | Sessions={cfg['sessions']}" + (" (自適應重跑)" if cfg["adaptive"] else "")
        print(f"\n[*] 正在測試: {label} ...")
        res = benchmark_configuration(test_pdf, dpi=cfg["dpi"], num_sessions=cfg["sessions"], adaptive=cfg["adaptive"])
        res["label"] = label
        all_results.append(res)
        print(f"    -> 總耗時: {res['total_time_sec']}s | 吞吐量: {res['fps']} 頁/秒 | 平均信心度: {res['avg_confidence_pct']}% | 300頁預估: {res['projected_300p_sec']}s")

    # Output formatted report
    out_file = Path("benchmark_matrix_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n==========================================================================")
    print("  實測矩陣匯總表 (SLA 基準)")
    print("==========================================================================")
    print(f"{'配置標籤':<30} | {'總耗時(s)':<9} | {'頁/秒(FPS)':<10} | {'信心度(%)':<9} | {'300頁預估(s)':<12}")
    print("-" * 78)
    for r in all_results:
        print(f"{r['label']:<30} | {r['total_time_sec']:<9} | {r['fps']:<10} | {r['avg_confidence_pct']:<9} | {r['projected_300p_sec']:<12}")
    print("==========================================================================")
    print(f"指標紀錄已儲存至: {out_file.name}")

if __name__ == "__main__":
    main()
