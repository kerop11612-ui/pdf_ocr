"""AuraDoc 的可持久化單 worker FastAPI 本機介面。"""
from __future__ import annotations

import json
import shutil
import threading
import time
import subprocess
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
ALLOWED_DOWNLOADS = {
    "clean": "clean.md",
    "raw": "raw.txt",
    "metrics": "metrics.json",
    "structured": "structured.jsonl",
    "clean_reviewed": "clean_reviewed.md",
    "review_pending": "review_pending.md",
    "obsidian": "obsidian.md",
}
FULL_TEXT_FIELDS = {"page_results", "clean_preview", "raw_preview"}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _job_summary(job: dict[str, Any], storage_path: Path | None = None) -> dict[str, Any]:
    summary = {key: value for key, value in job.items() if key not in FULL_TEXT_FIELDS}
    # Use cached stats from job.json (updated incrementally by _update_cached_stats)
    if "needs_review_count" not in summary and storage_path:
        _update_cached_stats_from_disk(job, storage_path)
        summary["needs_review_count"] = job.get("needs_review_count", 0)
        summary["page_types_summary"] = job.get("page_types_summary", {})
    return summary


def _update_cached_stats_from_disk(job: dict[str, Any], storage_path: Path) -> None:
    """One-time migration: scan pages dir and populate cached stats in job dict."""
    pages_dir = storage_path / job["id"] / "pages"
    if not pages_dir.is_dir():
        return
    needs_review_count = 0
    page_types: dict[str, int] = {}
    for pf in pages_dir.glob("*.json"):
        try:
            p = json.loads(pf.read_text(encoding="utf-8"))
            if p.get("needs_review"):
                needs_review_count += 1
            ptype = p.get("page_type") or "prose"
            page_types[ptype] = page_types.get(ptype, 0) + 1
        except Exception:
            pass
    job["needs_review_count"] = needs_review_count
    job["page_types_summary"] = page_types


class JobService:
    def __init__(self, storage: Path, start_worker: bool):
        self.storage = storage
        self.storage.mkdir(parents=True, exist_ok=True)
        self.queue: deque[str] = deque()
        self.lock = threading.RLock()
        self.cancel_events: dict[str, threading.Event] = {}
        self.stop_event = threading.Event()
        self._recover()
        if start_worker:
            threading.Thread(target=self._worker, daemon=True, name="auradoc-ocr-worker").start()

    def _recover(self) -> None:
        for status_file in self.storage.glob("*/job.json"):
            job = json.loads(status_file.read_text(encoding="utf-8"))
            if job["status"] in {"queued", "running"}:
                job["status"] = "queued"
                job["message"] = "已於服務重啟後重新排隊"
                self._save(job)
                self.queue.append(job["id"])
            self.cancel_events[job["id"]] = threading.Event()
            if job.get("cancel_requested"):
                self.cancel_events[job["id"]].set()

    def _path(self, job_id: str) -> Path:
        if not self._valid_id(job_id):
            raise HTTPException(404, "找不到工作")
        return self.storage / job_id

    @staticmethod
    def _valid_id(job_id: str) -> bool:
        try:
            return str(uuid.UUID(job_id)) == job_id
        except ValueError:
            return False

    def _save(self, job: dict[str, Any]) -> None:
        with self.lock:
            job["updated_at"] = time.time()
            path = self._path(job["id"]) / "job.json"
            temp = path.with_suffix(".tmp")
            _write_json(temp, job)
            temp.replace(path)

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            path = self._path(job_id) / "job.json"
            if not path.is_file():
                raise HTTPException(404, "找不到工作")
            return json.loads(path.read_text(encoding="utf-8"))

    def list(self, limit: int = 5) -> list[dict[str, Any]]:
        records = [_job_summary(self.get(path.parent.name), self.storage) for path in self.storage.glob("*/job.json")]
        return sorted(records, key=lambda job: job["created_at"], reverse=True)[:max(1, min(limit, 50))]

    def _write_page_result(self, job_id: str, page: dict[str, Any]) -> None:
        page_dir = self._path(job_id) / "pages"
        page_dir.mkdir(exist_ok=True)
        path = page_dir / f"{page['page']}.json"
        temp = path.with_suffix(".tmp")
        _write_json(temp, page)
        temp.replace(path)

    def create(self, upload: UploadFile, payload: bytes, mode: str, start: int | None, end: int | None, backend: str, options: dict[str, bool]) -> dict[str, Any]:
        try:
            doc = fitz.open(stream=payload, filetype="pdf")
            pages = len(doc)
            doc.close()
        except Exception as exc:
            raise HTTPException(422, f"PDF 讀取失敗：{exc}") from exc
        if pages < 1:
            raise HTTPException(422, "PDF 沒有可處理頁面")
        if mode == "first5":
            start, end = 1, min(5, pages)
        elif mode == "all":
            start, end = 1, pages
        elif mode != "custom" or start is None or end is None or start < 1 or end < start or end > pages:
            raise HTTPException(422, f"頁碼範圍必須位於 1 至 {pages}，且起頁不可大於迄頁")
        if backend not in {"auto", "gpu", "cuda", "dml", "cpu"}:
            raise HTTPException(422, "運算裝置必須是 auto、gpu、cuda、dml 或 cpu")
        job_id = str(uuid.uuid4())
        folder = self.storage / job_id
        folder.mkdir()
        (folder / "source.pdf").write_bytes(payload)
        job = {"id": job_id, "filename": Path(upload.filename or "source.pdf").name, "status": "queued", "message": "等待單一 OCR worker", "total_pages": pages, "pages": [start, end], "completed_pages": 0, "progress": 0, "backend_requested": backend, "backend_actual": None, "options": options, "outputs": {}, "created_at": time.time()}
        self._save(job)
        with self.lock:
            self.cancel_events[job_id] = threading.Event()
            self.queue.append(job_id)
        return job

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] in {"queued", "running"}:
            job["cancel_requested"] = True
            self.cancel_events.setdefault(job_id, threading.Event()).set()
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["message"] = "已取消（尚未開始）"
            else:
                job["message"] = "取消要求已收到，將保留已完成頁面"
            self._save(job)
        return self.get(job_id)

    def update_page_clean(self, job_id: str, page_num: int, clean_text: str) -> dict[str, Any]:
        with self.lock:
            path = self._path(job_id) / "pages" / f"{page_num}.json"
            if not path.is_file():
                raise HTTPException(404, "找不到指定頁面")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["clean"] = clean_text.strip()
            data["user_edited"] = True
            data["reviewed"] = True
            data["reviewed_at"] = time.time()
            _write_json(path, data)
            self._reaggregate_outputs(job_id)
            return data

    def review_page(self, job_id: str, page_num: int, reviewed: bool = True) -> dict[str, Any]:
        with self.lock:
            path = self._path(job_id) / "pages" / f"{page_num}.json"
            if not path.is_file():
                raise HTTPException(404, "找不到指定頁面")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["reviewed"] = reviewed
            data["reviewed_at"] = time.time()
            _write_json(path, data)
            self._reaggregate_outputs(job_id)
            return data

    def rerun_page(self, job_id: str, page_num: int, options_override: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            job = self.get(job_id)
            pdf_path = self._path(job_id) / "source.pdf"
            if not pdf_path.is_file():
                raise HTTPException(404, "找不到來源 PDF")
            opts = dict(job.get("options", {}))
            if options_override:
                opts.update(options_override)
            backend_mode = opts.get("backend", job.get("backend_requested", "auto"))

        from directml_pipeline import process_single_page
        page_data = process_single_page(
            pdf_path,
            page_no=page_num,
            enable_clahe=opts.get("clahe", True),
            enable_reflow=opts.get("reflow", True),
            enable_two_column=opts.get("two_column", True),
            enable_furigana_filter=opts.get("furigana", True),
            enable_homoglyphs=opts.get("homoglyphs", True),
            backend_mode=backend_mode,
            adaptive_retry=opts.get("adaptive_retry", True),
        )

        with self.lock:
            self._write_page_result(job_id, page_data)
            self._reaggregate_outputs(job_id)
            return page_data

    def _generate_auxiliary_outputs(self, job_id: str) -> None:
        folder = self._path(job_id)
        pages_dir = folder / "pages"
        if not pages_dir.is_dir():
            return
        page_files = sorted(pages_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        clean_parts: list[str] = []
        clean_reviewed_parts: list[str] = []
        review_pending_parts: list[str] = []

        for pf in page_files:
            try:
                p = json.loads(pf.read_text(encoding="utf-8"))
                p_no = p.get("page", pf.stem)
                clean_t = (p.get("clean") or "").strip()
                raw_t = (p.get("raw") or "").strip()
                if clean_t:
                    clean_parts.append(f"## Page {p_no}\n\n{clean_t}\n")
                    if not p.get("needs_review") or p.get("reviewed") or p.get("user_edited"):
                        clean_reviewed_parts.append(f"## Page {p_no}\n\n{clean_t}\n")
                if p.get("needs_review") and not (p.get("reviewed") or p.get("user_edited")):
                    reasons = ", ".join(p.get("review_reasons", [])) or "待確認"
                    review_pending_parts.append(f"## Page {p_no} (原因: {reasons})\n\n{clean_t or raw_t}\n")
            except Exception:
                pass

        if clean_reviewed_parts:
            (folder / "clean_reviewed.md").write_text("\n\n".join(clean_reviewed_parts), encoding="utf-8")
        if review_pending_parts:
            (folder / "review_pending.md").write_text("\n\n".join(review_pending_parts), encoding="utf-8")
        
        job = self.get(job_id)
        created_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.get('created_at', time.time())))
        obsidian_frontmatter = (
            f"---\n"
            f"title: \"{job.get('filename', 'document')}\"\n"
            f"date: \"{created_str}\"\n"
            f"pages: {job.get('total_pages', len(page_files))}\n"
            f"tags:\n"
            f"  - auradoc\n"
            f"  - ocr\n"
            f"---\n\n"
        )
        (folder / "obsidian.md").write_text(obsidian_frontmatter + "\n\n".join(clean_parts), encoding="utf-8")

    def _reaggregate_outputs(self, job_id: str) -> None:
        folder = self._path(job_id)
        pages_dir = folder / "pages"
        if not pages_dir.is_dir():
            return
        page_files = sorted(pages_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        clean_parts: list[str] = []
        raw_parts: list[str] = []
        structured_lines: list[str] = []
        needs_review_count = 0
        page_types: dict[str, int] = {}

        for pf in page_files:
            try:
                p = json.loads(pf.read_text(encoding="utf-8"))
                p_no = p.get("page", pf.stem)
                clean_t = (p.get("clean") or "").strip()
                raw_t = (p.get("raw") or "").strip()
                if clean_t:
                    clean_parts.append(f"## Page {p_no}\n\n{clean_t}\n")
                if raw_t:
                    raw_parts.append(f"=== [Page {p_no}] ===\n{raw_t}\n")
                if p.get("needs_review") and not (p.get("reviewed") or p.get("user_edited")):
                    needs_review_count += 1
                ptype = p.get("page_type") or "prose"
                page_types[ptype] = page_types.get(ptype, 0) + 1
                structured_lines.append(json.dumps(p, ensure_ascii=False))
            except Exception:
                pass

        (folder / "clean.md").write_text("\n\n".join(clean_parts), encoding="utf-8")
        (folder / "raw.txt").write_text("\n\n".join(raw_parts), encoding="utf-8")
        (folder / "structured.jsonl").write_text("\n".join(structured_lines) + "\n", encoding="utf-8")
        
        job = self.get(job_id)
        job["needs_review_count"] = needs_review_count
        job["page_types_summary"] = page_types
        self._save(job)
        self._generate_auxiliary_outputs(job_id)

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                job_id = self.queue.popleft() if self.queue else None
            if not job_id:
                time.sleep(.15)
                continue
            job = self.get(job_id)
            if job["status"] == "cancelled":
                continue
            self._run(job)

    def _run(self, job: dict[str, Any]) -> None:
        from directml_pipeline import process_pdf_gpu_adaptive_generator
        job["status"], job["message"] = "running", "正在初始化 OCR"
        self._save(job)
        folder = self._path(job["id"])
        options = job["options"]
        try:
            result = None
            for progress, message, clean, raw, done, final in process_pdf_gpu_adaptive_generator(
                folder / "source.pdf", folder, start_page=job["pages"][0], end_page=job["pages"][1],
                backend_mode=job["backend_requested"], cancel_check=lambda: self.cancel_events.setdefault(job["id"], threading.Event()).is_set(),
                enable_clahe=options["clahe"], enable_reflow=options["reflow"], enable_two_column=options["two_column"],
                enable_furigana_filter=options["furigana"], enable_homoglyphs=options["homoglyphs"],
                adaptive_retry=options["adaptive_retry"], stream_full_text=False,
                on_page=lambda page: self._write_page_result(job["id"], page)):
                job = _job_summary(self.get(job["id"]))
                job.update({"completed_pages": round(progress * (job["pages"][1] - job["pages"][0] + 1)), "progress": round(progress * 100), "message": message})
                if done:
                    result = final
                self._save(job)
            job = self.get(job["id"])
            if job.get("cancel_requested"):
                job.update({"status": "cancelled", "message": "已取消；已完成內容已保留"})
            elif result:
                outputs = {
                    "clean": "clean.md", "raw": "raw.txt", "metrics": "metrics.json", "structured": "structured.jsonl",
                    "clean_reviewed": "clean_reviewed.md", "review_pending": "review_pending.md", "obsidian": "obsidian.md"
                }
                shutil.copyfile(result["md_path"], folder / outputs["clean"])
                shutil.copyfile(result["raw_path"], folder / outputs["raw"])
                shutil.copyfile(result["metrics_path"], folder / outputs["metrics"])
                shutil.copyfile(result["structured_path"], folder / outputs["structured"])
                self._generate_auxiliary_outputs(job["id"])
                job.update({"status": "completed", "progress": 100, "message": "處理完成", "outputs": outputs, "backend_actual": result.get("backend_actual"), "provider_sets": result.get("provider_sets", [])})
            self._save(job)
        except Exception as exc:
            job = self.get(job["id"])
            job.update({"status": "failed", "message": str(exc)})
            self._save(job)


def create_app(storage: Path | None = None, start_worker: bool = True) -> FastAPI:
    app = FastAPI(title="AuraDoc OCR Local")
    service = JobService(storage or ROOT / "web_jobs", start_worker)
    app.state.jobs = service
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

    @app.post("/api/jobs", status_code=201)
    async def create_job(file: UploadFile = File(...), range_mode: str = Form("first5"), start_page: int | None = Form(None), end_page: int | None = Form(None), backend: str = Form("auto"), adaptive_retry: bool = Form(True), clahe: bool = Form(True), reflow: bool = Form(True), two_column: bool = Form(True), furigana: bool = Form(True), homoglyphs: bool = Form(True)):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(422, "請上傳 PDF 檔案")
        return service.create(file, await file.read(), range_mode, start_page, end_page, backend, {"adaptive_retry": adaptive_retry, "clahe": clahe, "reflow": reflow, "two_column": two_column, "furigana": furigana, "homoglyphs": homoglyphs})

    @app.get("/api/jobs")
    def list_jobs(limit: int = 5): return service.list(limit)

    @app.get("/api/hardware")
    def hardware():
        from hardware_detect import detect_hardware_summary
        return detect_hardware_summary()

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str): return _job_summary(service.get(job_id), service.storage)

    @app.get("/api/jobs/{job_id}/pages")
    def list_job_pages(job_id: str):
        service.get(job_id)
        pages_dir = service._path(job_id) / "pages"
        if not pages_dir.is_dir():
            return []
        page_files = sorted(pages_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        summaries = []
        for pf in page_files:
            try:
                p = json.loads(pf.read_text(encoding="utf-8"))
                summaries.append({
                    "page": p.get("page", int(pf.stem) if pf.stem.isdigit() else 0),
                    "page_type": p.get("page_type", "prose"),
                    "needs_review": p.get("needs_review", False),
                    "reviewed": p.get("reviewed", False),
                    "review_reasons": p.get("review_reasons", []),
                    "user_edited": p.get("user_edited", False)
                })
            except Exception:
                pass
        return summaries

    @app.get("/api/jobs/{job_id}/pages/{page}")
    def page_result(job_id: str, page: int):
        service.get(job_id)
        path = service._path(job_id) / "pages" / f"{page}.json"
        if not path.is_file(): raise HTTPException(404, "該頁尚未完成")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.put("/api/jobs/{job_id}/pages/{page}/clean")
    async def update_page_clean(job_id: str, page: int, payload: dict[str, Any]):
        clean_text = payload.get("clean", "")
        return service.update_page_clean(job_id, page, clean_text)

    @app.post("/api/jobs/{job_id}/pages/{page}/review")
    async def toggle_page_review(job_id: str, page: int, payload: dict[str, Any] | None = None):
        reviewed = (payload or {}).get("reviewed", True)
        return service.review_page(job_id, page, reviewed)

    @app.post("/api/jobs/{job_id}/pages/{page}/rerun")
    async def rerun_page(job_id: str, page: int, payload: dict[str, Any] | None = None):
        options = payload or {}
        return service.rerun_page(job_id, page, options)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str): return _job_summary(service.cancel(job_id), service.storage)

    @app.get("/api/jobs/{job_id}/download/{kind}")
    def download(job_id: str, kind: str):
        if kind not in ALLOWED_DOWNLOADS:
            raise HTTPException(404, "找不到成果")
        path = service._path(job_id) / ALLOWED_DOWNLOADS[kind]
        if not path.is_file():
            service._reaggregate_outputs(job_id)
        if not path.is_file():
            raise HTTPException(404, "成果尚未產生")
        return FileResponse(path, filename=path.name)

    @app.get("/api/jobs/{job_id}/pages/{page}/original")
    def original_page(job_id: str, page: int):
        job = service.get(job_id)
        if page < 1 or page > job["total_pages"]:
            raise HTTPException(404, "找不到頁面")
        output = service._path(job_id) / f"preview-{page}.png"
        if not output.is_file():
            doc = fitz.open(service._path(job_id) / "source.pdf")
            pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            pix.save(output)
            doc.close()
        return FileResponse(output, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    return app


app = create_app()
