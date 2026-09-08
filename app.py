"""
AuraDoc OCR - Professional AI Document & Knowledge Extraction System
Design System: ui-ux-pro-max (Developer Tool / AI GPU Dashboard - Ultra-High Contrast Dark Mode)
Palette:
- Canvas Background: #0B0F17
- Card Panels: #111827
- Borders: #1F2937 / #374151
- Primary Text: #F8FAFC (WCAG AAA Contrast > 16:1)
- Secondary Text: #94A3B8
- Accent: #3B82F6 (Electric Blue)
- Obsidian Callout: #1E293B background, #3B82F6 border, #93C5FD text
"""

import os
import sys
import json
import time
from typing import Optional
from pathlib import Path
import gradio as gr
import fitz  # PyMuPDF
from directml_pipeline import process_pdf_gpu_adaptive_generator
from pipeline import run_ocrmypdf_100mb

CUSTOM_CSS = """
/* ==========================================================================
   ui-ux-pro-max Ultra-High Contrast Dark Theme (WCAG AAA Contrast)
   ========================================================================== */

/* 全域暗色畫布 */
html, body, .gradio-container {
    background-color: #0B0F17 !important;
    color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* 頂部標題區塊 */
.auradoc-header {
    background: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-bottom: 2px solid #2563EB !important;
    padding: 18px 24px !important;
    margin-bottom: 20px !important;
    border-radius: 8px !important;
}

.auradoc-title {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    margin: 0 0 6px 0 !important;
}

/* 狀態徽章 */
.badge-status {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 8px;
}

.badge-gpu { background-color: #1E3A8A !important; color: #93C5FD !important; border: 1px solid #3B82F6 !important; }
.badge-ready { background-color: #064E3B !important; color: #6EE7B7 !important; border: 1px solid #10B981 !important; }

/* 左右核心卡片容器 */
.card-panel {
    background: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 8px !important;
    padding: 20px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4) !important;
}

/* 標題類文字強制亮白 */
h1, h2, h3, h4, 
.card-panel h1, .card-panel h2, .card-panel h3, .card-panel h4,
.card-panel .block-label {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}

/* 表單容器 (gr.File, gr.Radio, gr.Number) 深灰底色 */
.card-panel .block,
.card-panel fieldset,
.card-panel .form,
.card-panel [data-testid="block-container"],
.card-panel .wrap {
    background-color: #1F2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
}

/* 表單內所有標籤、清單、文字全面亮白高對比 */
.card-panel label,
.card-panel span,
.card-panel p,
.card-panel div,
span[data-testid="block-info"],
.card-panel .gr-radio label,
.card-panel .gr-radio span,
.card-panel label.selected > span {
    color: #F8FAFC !important;
    font-weight: 600 !important;
}

/* 輸入框 (數字框、文字框、下拉選單) 暗色底配合清晰白色字體 */
.card-panel input,
.card-panel textarea,
.card-panel select {
    background-color: #0B0F17 !important;
    color: #F8FAFC !important;
    border: 1px solid #4B5563 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    padding: 6px 10px !important;
}

/* 檔案拖曳上傳區 (gr.File) */
.card-panel [data-testid="file-upload"],
.card-panel .file-preview {
    background-color: #0B0F17 !important;
    border: 2px dashed #4B5563 !important;
    border-radius: 6px !important;
}

.card-panel [data-testid="file-upload"] *,
.card-panel .file-preview * {
    color: #F8FAFC !important;
    font-weight: 600 !important;
}

/* 分頁切換標籤 (Tabs) */
.tabs, .tab-nav {
    border-bottom: 2px solid #1F2937 !important;
}

button.tab-nav {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

button.tab-nav.selected {
    color: #60A5FA !important;
    border-bottom: 2px solid #3B82F6 !important;
}

/* 右側面板：Obsidian Markdown 預覽區 (清晰亮白字體) */
.prose, .prose *,
.markdown-text, .markdown-text *,
[data-testid="markdown"], [data-testid="markdown"] *,
.auradoc-md-preview,
.auradoc-md-preview * {
    color: #F8FAFC !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

.auradoc-md-preview h1,
.auradoc-md-preview h2,
.auradoc-md-preview h3,
.auradoc-md-preview h4 {
    color: #60A5FA !important;
    font-weight: 700 !important;
    margin-top: 16px !important;
    margin-bottom: 8px !important;
}

/* Obsidian Callout (> [!source]) 藍色科技邊框與深色強調卡片 */
.auradoc-md-preview blockquote {
    border-left: 4px solid #3B82F6 !important;
    background-color: #1E293B !important;
    padding: 10px 16px !important;
    margin: 12px 0 !important;
    border-radius: 6px !important;
    color: #93C5FD !important;
    font-weight: 600 !important;
}

.auradoc-md-preview blockquote * {
    color: #93C5FD !important;
    font-weight: 600 !important;
}

/* 程式碼與 JSON 檢視區塊 */
pre, code, .cm-editor {
    background-color: #0B0F17 !important;
    color: #38BDF8 !important;
    border: 1px solid #1F2937 !important;
}

pre *, code *, .cm-editor * {
    color: #38BDF8 !important;
}

/* 按鈕樣式 */
.btn-primary-action {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    padding: 11px 18px !important;
    cursor: pointer !important;
    transition: background-color 150ms ease !important;
}

.btn-primary-action:hover {
    background-color: #1D4ED8 !important;
}

.btn-stop-action {
    background-color: #DC2626 !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    padding: 11px 18px !important;
    cursor: pointer !important;
}
"""

def find_cached_user_pdf() -> Optional[str]:
    temp_dir = Path(os.environ.get("TEMP", "")) / "gradio"
    if temp_dir.exists():
        pdfs = list(temp_dir.glob("**/*.pdf"))
        if pdfs:
            return str(pdfs[0])
    return None

def inspect_pdf_file(file_obj):
    if not file_obj:
        return "<div style='color: #94A3B8;'>請上傳 PDF 檔案...</div>", "未載入檔案", 1, 1
    
    file_path = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj))
    size_mb = file_path.stat().st_size / (1024 * 1024)
    
    try:
        doc = fitz.open(file_path)
        pages = len(doc)
        sample_page = doc[0] if pages > 0 else None
        has_text = len(sample_page.get_text("text").strip()) > 30 if sample_page else False
        doc.close()

        status_html = (
            f"<div style='font-size: 13px; line-height: 1.6; padding: 12px; background: #1F2937; border: 1px solid #374151; border-radius: 6px; color: #F8FAFC;'>"
            f"<div><strong style='color: #60A5FA;'>檔案名稱:</strong> {file_path.name[:35]}...</div>"
            f"<div><strong style='color: #60A5FA;'>檔案規格:</strong> {size_mb:.1f} MB | 共 <strong>{pages}</strong> 頁</div>"
            f"<div><strong style='color: #60A5FA;'>文字層狀態:</strong> {'已含數位字層 (可極速抽取)' if has_text else '純掃描圖檔 (推薦 5060 Ti GPU)'}</div>"
            f"<div><strong style='color: #60A5FA;'>5060 Ti 預估:</strong> 全書約 {max(3, int(pages / 4.0))} 秒 | 前 5 頁約 2.0 秒</div>"
            f"</div>"
        )
        return status_html, f"已載入 {file_path.name} (共 {pages} 頁)", 1, min(5, pages)
    except Exception as e:
        return f"<div style='color: #F87171;'>讀取錯誤: {str(e)}</div>", "檔案無效", 1, 1

def run_auradoc_pipeline_streaming(
    file_obj,
    track_mode,
    range_mode,
    custom_start,
    custom_end,
    adaptive_dpi,
    lang_choice,
    jobs_count,
    enable_two_col,
    enable_reflow,
    enable_homoglyphs,
    enable_clahe,
    enable_furigana,
    progress=gr.Progress(track_tqdm=True)
):
    if not file_obj:
        yield "請先上傳 PDF 檔案！", "尚未執行", "", None, None, None, ""
        return

    input_path = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj)).resolve()
    doc_id = input_path.stem
    dist_dir = Path("dist") / doc_id

    doc = fitz.open(input_path)
    total_doc_pages = len(doc)
    doc.close()

    if range_mode == "⚡ 快速試跑前 5 頁 (推薦)":
        start_p, end_p = 1, min(5, total_doc_pages)
    elif range_mode == "📖 全書處理":
        start_p, end_p = 1, total_doc_pages
    else:
        start_p = max(1, int(custom_start))
        end_p = min(total_doc_pages, int(custom_end))

    proc_count = end_p - start_p + 1
    init_msg = f"初始化 5060 Ti DirectML 引擎: 第 {start_p} ~ {end_p} 頁 (共 {proc_count} 頁)..."
    progress(0.02, desc=init_msg)
    yield init_msg, "正在初始化引擎與載入頁面...", "", None, None, None, ""

    try:
        if "Track A" in track_mode:
            is_high_acc = "高精" in lang_choice or "jpn + zh-tw" in lang_choice

            for progress_ratio, status_msg, cur_clean, cur_raw, is_done, final_data in process_pdf_gpu_adaptive_generator(
                input_pdf=input_path,
                output_dir=dist_dir,
                base_dpi=250 if adaptive_dpi else 300,
                adaptive_retry_dpi=350,
                confidence_thresh=0.72,
                lang="japan" if "jpn" in lang_choice else "en",
                start_page=start_p,
                end_page=end_p,
                high_accuracy=is_high_acc,
                enable_clahe=enable_clahe,
                enable_reflow=enable_reflow,
                enable_two_column=enable_two_col,
                enable_furigana_filter=enable_furigana,
                enable_homoglyphs=enable_homoglyphs
            ):
                progress(progress_ratio, desc=status_msg)

                if not is_done:
                    live_telemetry = (
                        f"### ⚡ 5060 Ti 實時進度 (已完成 {int(progress_ratio * 100)}%)\n"
                        f"- **目前狀態:** {status_msg}\n"
                        f"- **處理範圍:** 第 {start_p} ~ {end_p} 頁 (共 {proc_count} 頁)\n"
                        f"- **排版引擎:** 雙欄偵測 ({'開' if enable_two_col else '關'}) · 標點成段 ({'開' if enable_reflow else '關'})\n"
                        f"- **精度增強:** CLAHE 對比增強 ({'開' if enable_clahe else '關'}) · 同形字校正 ({'開' if enable_homoglyphs else '關'})\n"
                    )
                    yield (
                        status_msg,
                        cur_clean,
                        cur_raw,
                        None,
                        None,
                        None,
                        live_telemetry
                    )
                else:
                    fps = final_data["fps"]
                    m_data = final_data.get("metrics_data", {})
                    telemetry = (
                        f"### ⚡ 5060 Ti 處理完成報告\n"
                        f"- **處理頁數:** 第 {start_p} ~ {end_p} 頁 (共 {proc_count} 頁)\n"
                        f"- **總耗時:** **{final_data['total_elapsed']:.2f} 秒**\n"
                        f"- **實測吞吐量:** **{fps:.2f} 頁/秒 (Pages/Sec)**\n"
                        f"- **同形字/黏字校正:** **{m_data.get('total_corrections', 0)} 處**\n"
                        f"- **雙欄版面頁數:** **{m_data.get('multi_column_pages', 0)} 頁**\n"
                        f"- **振假名降噪:** **{m_data.get('total_furigana_filtered', 0)} 處**\n"
                        f"- **硬體裝置:** NVIDIA GeForce RTX 5060 Ti (DirectML)\n"
                    )
                    yield (
                        status_msg,
                        final_data["markdown_content"],
                        final_data["raw_content"],
                        str(final_data["md_path"]),
                        str(final_data["raw_path"]),
                        str(final_data["metrics_path"]),
                        telemetry
                    )

        else:
            # Track B: Archive OCRmyPDF (Tesseract multi-language)
            progress(0.1, desc="啟動 OCRmyPDF 永久典藏 PDF/A 封裝...")
            tess_lang = "jpn+chi_tra+eng" if ("zh-tw" in lang_choice or "chi_tra" in lang_choice) else ("jpn+eng" if "jpn" in lang_choice else "eng")
            out_pdf = dist_dir / "01_pdf" / f"{doc_id}_ARCHIVE.pdf"
            start_time = time.time()
            run_ocrmypdf_100mb(
                input_pdf=input_path,
                output_pdf=out_pdf,
                lang=tess_lang,
                jobs=int(jobs_count)
            )
            elapsed = time.time() - start_time
            progress(1.0, desc="PDF/A 典藏封裝完成！")
            yield (
                f"PDF/A 典藏建立完成！耗時 {elapsed:.1f} 秒",
                f"# PDF/A 典藏檔案產出完成\n檔案路徑: `{out_pdf}`",
                "",
                str(out_pdf),
                None,
                None,
                f"PDF/A 耗時: {elapsed:.1f} 秒 (三語: {tess_lang})"
            )
    except Exception as e:
        yield (
            f"❌ 執行中斷: {str(e)}",
            f"### 錯誤詳細資訊\n```\n{str(e)}\n```",
            "",
            None,
            None,
            None,
            f"錯誤日誌: {str(e)}"
        )

# --- Gradio UI Layout ---
cached_pdf = find_cached_user_pdf()

with gr.Blocks(title="AuraDoc OCR - AI Knowledge & Archive") as demo:
    with gr.Row(elem_classes=["auradoc-header"]):
        with gr.Column():
            gr.HTML("""
                <div class="auradoc-title">AuraDoc OCR · 智慧日文教材知識庫系統</div>
                <div style="margin-top: 8px;">
                    <span class="badge-status badge-gpu">GPU: RTX 5060 Ti (16GB)</span>
                    <span class="badge-status badge-ready">語言: 日文 + 繁體中文 (jpn + zh-tw)</span>
                    <span class="badge-status badge-ready">高精核心: PP-OCRv6 Medium (18,710字庫)</span>
                    <span class="badge-status badge-ready">Tesseract: jpn + chi_tra 就緒</span>
                </div>
            """)

    with gr.Row():
        # Left Panel
        with gr.Column(scale=1, elem_classes=["card-panel"]):
            gr.Markdown("### 1. 檔案與頁碼範圍")
            file_uploader = gr.File(label="PDF 檔案 (支援 100MB+)", file_types=[".pdf"])
            if cached_pdf:
                btn_load_cache = gr.Button(
                    f"📁 點此帶入教材: 穩紮穩打N4文法 (64MB/457頁)", 
                    variant="secondary"
                )
            file_meta_box = gr.HTML("<div style='font-size: 12px; color: #94A3B8;'>載入檔案後顯示規格...</div>")

            range_choice = gr.Radio(
                choices=["⚡ 快速試跑前 5 頁 (推薦)", "📖 全書處理", "自訂頁碼範圍"],
                value="⚡ 快速試跑前 5 頁 (推薦)",
                label="處理範圍 (大檔建議先跑前 5 頁驗證)"
            )
            with gr.Row():
                custom_s = gr.Number(value=1, label="起始頁", precision=0)
                custom_e = gr.Number(value=5, label="結束頁", precision=0)

            gr.Markdown("### 2. 處理軌道選擇")
            track_choice = gr.Radio(
                choices=[
                    "Track A: AI 知識庫極速軌道 (5060 Ti DirectML GPU)",
                    "Track B: 永久典藏雙層 PDF/A 軌道 (OCRmyPDF C++)"
                ],
                value="Track A: AI 知識庫極速軌道 (5060 Ti DirectML GPU)",
                label="處理軌道"
            )

            with gr.Accordion("進階微調", open=False):
                lang_drop = gr.Dropdown(
                    choices=[
                        "jpn + zh-tw (日繁混排教材，高精 Medium 模型推薦, 信心度 95%+)",
                        "japan (純日文模型)",
                        "chinese_cht (純繁體中文模型)",
                        "eng (純英文模型)"
                    ],
                    value="jpn + zh-tw (日繁混排教材，高精 Medium 模型推薦, 信心度 95%+)",
                    label="語言模型"
                )
                adaptive_cb = gr.Checkbox(value=True, label="自適應 DPI (低信心自動 350 DPI 重跑)")
                jobs_num = gr.Slider(1, 6, value=4, step=1, label="CPU 並行數 (僅 Track B)")

            gr.Markdown("### 3. 日文排版與精準度控制")
            cb_two_col = gr.Checkbox(value=True, label="📐 智慧雙欄偵測 (XY-Cut Column Split)")
            cb_reflow = gr.Checkbox(value=True, label="✍️ 標點自然融合成段 (Sentence Reflow)")
            cb_homoglyphs = gr.Checkbox(value=True, label="🔍 日文同形字/黏字校正 (長音ー/力/口/二/題號)")
            cb_clahe = gr.Checkbox(value=True, label="⚡ CLAHE 自適應微對比 (防濁點遺失與透字)")
            cb_furigana = gr.Checkbox(value=True, label="✂️ 振假名 (Ruby) 語意降噪")

            with gr.Row():
                btn_run = gr.Button("開始執行", variant="primary", elem_classes=["btn-primary-action"], scale=3)
                btn_cancel = gr.Button("🛑 取消", variant="stop", elem_classes=["btn-stop-action"], scale=1)
                
            status_text = gr.Textbox(label="即時進度反饋", interactive=False, lines=2)

        # Right Panel
        with gr.Column(scale=2, elem_classes=["card-panel"]):
            with gr.Tabs():
                with gr.TabItem("📄 乾淨排版筆記 (Clean Markdown)"):
                    md_view = gr.Markdown(label="排版還原預覽", elem_classes=["auradoc-md-preview"])
                with gr.TabItem("🔍 原始辨識文字 (Raw OCR Text)"):
                    raw_view = gr.Code(label="原始文字對照", language="markdown")
                with gr.TabItem("⚡ 精度與排版報告"):
                    telemetry_view = gr.Markdown("### 執行後顯示實測指標")
                with gr.TabItem("💾 下載中心"):
                    with gr.Row():
                        out_clean_md = gr.File(label="下載排版還原筆記 (.md)")
                        out_raw_txt = gr.File(label="下載原始辨識純文字 (.txt)")
                    out_metrics = gr.File(label="下載排版指標報告 (.json)")

    # Events with concurrency_limit=5 to prevent queue deadlock on refresh
    file_uploader.change(fn=inspect_pdf_file, inputs=[file_uploader], outputs=[file_meta_box, status_text, custom_s, custom_e])
    run_event = btn_run.click(
        fn=run_auradoc_pipeline_streaming,
        inputs=[
            file_uploader, track_choice, range_choice, custom_s, custom_e,
            adaptive_cb, lang_drop, jobs_num,
            cb_two_col, cb_reflow, cb_homoglyphs, cb_clahe, cb_furigana
        ],
        outputs=[status_text, md_view, raw_view, out_clean_md, out_raw_txt, out_metrics, telemetry_view],
        concurrency_limit=5
    )
    if cached_pdf:
        btn_load_cache.click(
            fn=lambda: cached_pdf,
            inputs=None,
            outputs=[file_uploader]
        ).then(
            fn=inspect_pdf_file,
            inputs=[file_uploader],
            outputs=[file_meta_box, status_text, custom_s, custom_e]
        )

    btn_cancel.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

HEAD_SCRIPTS = """
<script>
document.documentElement.classList.add('dark');
localStorage.setItem('theme', 'dark');

document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') {
        const root = document.querySelector('.gradio-container') || document.querySelector('gradio-app');
        if (!root || root.children.length === 0 || root.offsetHeight === 0) {
            window.location.reload();
        }
    }
});

setInterval(function() {
    fetch('/', { method: 'HEAD', cache: 'no-store' }).catch(function() {});
}, 15000);
</script>
"""

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=5).launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=CUSTOM_CSS,
        head=HEAD_SCRIPTS,
        inbrowser=False
    )
