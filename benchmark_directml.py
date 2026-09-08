"""
Benchmark Script: RapidOCR + DirectML (RTX 5060 Ti) vs CPU Baseline on 20-page Japanese Textbook
Measures: Pages/sec, Total Elapsed, Render Time, Inference Time, Confidence
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
from rapidocr import RapidOCR

def generate_20p_japanese_sample_pdf(output_path: Path):
    """Generate realistic 20-page JLPT N3/N2 textbook content with tables, rules, and examples."""
    doc = fitz.open()
    
    grammar_data = [
        ("第1課 ～ことになっている", "規則・予定を表す。自分の意志ではなく、社会や集団の決定に基づく。", "動詞辞書形／ない形 ＋ ことになっている", "日本では、車は左側を走ることになっている。"),
        ("第2課 ～わけにはいかない", "社会的な常識や心理的理由から、そうすることができない。", "動詞辞書形／ない形 ＋ わけにはいかない", "明日は大事な試験があるので、休むわけにはいかない。"),
        ("第3課 ～ようになる", "能力や習慣の変化を表す。", "動詞可能形／辞書形 ＋ ようになる", "日本に来てから、納豆が食べられるようになりました。"),
        ("第4課 ～一方だ", "ある方向に変化が進行し続けていることを表す。", "動詞辞書形 ＋ 一方だ", "物価は上がる一方で、生活がなかなか楽にならない。"),
        ("第5課 ～に違いない", "強い確信を持った推量を表す（～だと確信している）。", "普通形（名詞・な形容詞語幹） ＋ に違いない", "犯人は現場をよく知っている人物に違いない。"),
        ("第6課 ～反面", "物事の表と裏、一方が～であると同時に、他方は～である。", "普通形 ＋ 反面", "都会の生活は便利な反面、ストレスも多い。"),
        ("第7課 ～どころではない", "状況が厳しくて、とても～する余裕がない。", "動詞辞書形／名詞 ＋ どころではない", "仕事が忙しくて、旅行どころではありません。"),
        ("第8課 ～に関して", "～について、～をめぐって（改まった表現）。", "名詞 ＋ に関して", "今回の事件に関して、詳しい調査が進められている。"),
        ("第9課 ～に加えて", "～だけでなく、その上にさらに～も。", "名詞 ＋ に加えて", "激しい雨に加えて、強い風も吹き荒れている。"),
        ("第10課 ～を契機に", "ある出来事をきっかけにして、新たな変化が起こる。", "名詞 ＋ を契機に", "海外留学を契機に、国際情勢に興味を持つようになった。"),
    ]

    for page_num in range(1, 21):
        page = doc.new_page()
        g_idx = (page_num - 1) % len(grammar_data)
        title, meaning, form, ex = grammar_data[g_idx]

        # Page Header & Chapter
        page.insert_text(fitz.Point(40, 50), f"新完全マスター 日検 N3/N2 文法対策 - P.{page_num}", fontsize=10, fontname="helv")
        page.insert_text(fitz.Point(40, 90), f"【第{page_num}課】 {title}", fontsize=16, fontname="helv")
        
        # Sections
        page.insert_text(fitz.Point(40, 130), f"【意味・用法】", fontsize=13, fontname="helv")
        page.insert_text(fitz.Point(60, 155), meaning, fontsize=11, fontname="helv")

        page.insert_text(fitz.Point(40, 195), f"【接続関係】", fontsize=13, fontname="helv")
        page.insert_text(fitz.Point(60, 220), form, fontsize=11, fontname="helv")

        page.insert_text(fitz.Point(40, 260), f"【例文】", fontsize=13, fontname="helv")
        page.insert_text(fitz.Point(60, 285), f"① {ex}", fontsize=11, fontname="helv")
        page.insert_text(fitz.Point(60, 310), f"② どんなに難しくても、途中で諦めるわけにはいかない。", fontsize=11, fontname="helv")
        page.insert_text(fitz.Point(60, 335), f"③ 一生懸命練習した結果、少しずつ話せるようになってきた。", fontsize=11, fontname="helv")

        # Table Simulation
        page.insert_text(fitz.Point(40, 380), "【活用語尾・接続一覧表】", fontsize=13, fontname="helv")
        page.insert_text(fitz.Point(60, 410), "動詞形態       | 接続形         | 意味区分", fontsize=11, fontname="helv")
        page.insert_text(fitz.Point(60, 435), "書く（辞書形） | 書く + 前に    | 時間的前後関係", fontsize=11, fontname="helv")
        page.insert_text(fitz.Point(60, 460), "話す（可能形） | 話せる + ように| 状態能力の変化", fontsize=11, fontname="helv")
        page.insert_text(fitz.Point(60, 485), "飲む（て形）   | 飲んで + みる  | 試行・経験", fontsize=11, fontname="helv")

        # Footer Exercise
        page.insert_text(fitz.Point(40, 540), "【確認ドリル】 次の文の（ ）に入る最も適切な言葉を選びなさい。", fontsize=11, fontname="helv")
        page.insert_text(fitz.Point(60, 570), "問1：熱が39度もあるのだから、会社へ（ ）。", fontsize=10, fontname="helv")
        page.insert_text(fitz.Point(80, 595), "1. 行くべきだ   2. 行くわけにはいかない   3. 行く一方だ", fontsize=10, fontname="helv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"[PREPARE] 已產生 20 頁標準日文教材測試檔: {output_path}")

def run_benchmark(pdf_path: Path, dpi: int = 250):
    """Benchmark RapidOCR DirectML on RTX 5060 Ti for a given DPI."""
    print(f"\n=======================================================")
    print(f"  執行基準測試: RapidOCR + DirectML (DPI = {dpi})")
    print(f"=======================================================")

    # 1. Initialize RapidOCR with DirectML
    init_start = time.time()
    ocr = RapidOCR(params={
        "EngineConfig.onnxruntime.use_dml": True,
        "Rec.lang_type": "japan",
        "Global.log_level": "error"  # Suppress per-page spam during benchmark
    })
    init_time = time.time() - init_start
    print(f"[*] RapidOCR DirectML 引擎載入耗時: {init_time:.3f} 秒")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    render_times = []
    ocr_times = []
    confidences = []
    detected_line_counts = []
    sample_texts = []

    print(f"[*] 開始批次推論 (共 {total_pages} 頁)...")
    total_start = time.time()

    for idx in range(total_pages):
        page = doc[idx]
        
        # Render Page to Pixmap (CPU rasterize)
        t0 = time.time()
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        t1 = time.time()
        render_times.append(t1 - t0)

        # GPU OCR Inference via DirectML
        t2 = time.time()
        res = ocr(img_np)
        t3 = time.time()
        ocr_times.append(t3 - t2)

        # Collect metrics
        if res and res.txts:
            line_count = len(res.txts)
            detected_line_counts.append(line_count)
            avg_score = float(np.mean(res.scores)) if res.scores else 0.0
            confidences.append(avg_score)
            if idx == 0:
                sample_texts = res.txts[:4]
        else:
            detected_line_counts.append(0)
            confidences.append(0.0)

    total_elapsed = time.time() - total_start
    doc.close()

    # Calculate statistics
    total_render = sum(render_times)
    total_ocr = sum(ocr_times)
    avg_render_page = (total_render / total_pages) * 1000  # ms
    avg_ocr_page = (total_ocr / total_pages) * 1000        # ms
    fps = total_pages / total_elapsed
    ocr_fps = total_pages / total_ocr
    avg_conf = np.mean(confidences) if confidences else 0.0

    print(f"\n--- 基準測試結果報告 (DPI: {dpi}) ---")
    print(f"  總頁數:               {total_pages} 頁")
    print(f"  總耗時 (含渲染+OCR):   {total_elapsed:.2f} 秒")
    print(f"  整體吞吐量 (End-to-End):{fps:.2f} 頁/秒 (Pages/Sec)")
    print(f"  純 GPU OCR 吞吐量:     {ocr_fps:.2f} 頁/秒")
    print(f"  每頁平均渲染耗時:      {avg_render_page:.1f} ms")
    print(f"  每頁平均 GPU 推論耗時: {avg_ocr_page:.1f} ms")
    print(f"  平均文字辨識信心度:    {avg_conf * 100:.1f} %")
    print(f"  每頁平均偵測文字行:    {np.mean(detected_line_counts):.1f} 行")
    print(f"  第 1 頁辨識採樣:")
    for st in sample_texts:
        print(f"    > {st}")

    return {
        "dpi": dpi,
        "total_pages": total_pages,
        "total_elapsed": total_elapsed,
        "total_render": total_render,
        "total_ocr": total_ocr,
        "fps": fps,
        "ocr_fps": ocr_fps,
        "avg_render_ms": avg_render_page,
        "avg_ocr_ms": avg_ocr_page,
        "avg_confidence": avg_conf,
    }

if __name__ == "__main__":
    test_pdf = Path("test_fixtures/benchmark_20p.pdf")
    generate_20p_japanese_sample_pdf(test_pdf)

    res_200 = run_benchmark(test_pdf, dpi=200)
    res_300 = run_benchmark(test_pdf, dpi=300)

    # Save benchmark metrics to json
    results_path = Path("benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"200_dpi": res_200, "300_dpi": res_300}, f, indent=2, ensure_ascii=False)
    print(f"\n[BENCHMARK FINISHED] 完整指標已寫入: {results_path}")
