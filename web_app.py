"""AuraDoc 的可持久化單 worker FastAPI 本機介面。"""
from __future__ import annotations

import json
import shutil
import threading
import time
import subprocess
import re
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
ALLOWED_DOWNLOADS = {"clean": "clean.md", "raw": "raw.txt", "metrics": "metrics.json"}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


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
        records = [self.get(path.parent.name) for path in self.storage.glob("*/job.json")]
        return sorted(records, key=lambda job: job["created_at"], reverse=True)[:max(1, min(limit, 20))]

    def _write_page_results(self, job: dict[str, Any]) -> None:
        page_dir = self._path(job["id"]) / "pages"
        page_dir.mkdir(exist_ok=True)
        for page, content in job.get("page_results", {}).items():
            path = page_dir / f"{page}.json"
            temp = path.with_suffix(".tmp")
            _write_json(temp, {"page": int(page), "clean": content.get("clean", ""), "raw": content.get("raw", "")})
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
        if backend not in {"auto", "gpu", "cpu"}:
            raise HTTPException(422, "運算裝置必須是 auto、gpu 或 cpu")
        job_id = str(uuid.uuid4())
        folder = self.storage / job_id
        folder.mkdir()
        (folder / "source.pdf").write_bytes(payload)
        job = {"id": job_id, "filename": Path(upload.filename or "source.pdf").name, "status": "queued", "message": "等待單一 OCR worker", "total_pages": pages, "pages": [start, end], "completed_pages": 0, "progress": 0, "backend_requested": backend, "backend_actual": None, "options": options, "outputs": {}, "clean_preview": "", "raw_preview": "", "created_at": time.time()}
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

    @staticmethod
    def _page_results(clean: str, raw: str) -> dict[str, dict[str, str]]:
        pages: dict[str, dict[str, str]] = {}
        for match in re.finditer(r"(?ms)^## Page (\d+)\n\n(.*?)(?=^## Page |\Z)", clean):
            pages.setdefault(match.group(1), {})["clean"] = match.group(2).strip()
        for match in re.finditer(r"(?ms)^=== \[Page (\d+)\] ===\n(.*?)(?=^=== \[Page |\Z)", raw):
            pages.setdefault(match.group(1), {})["raw"] = match.group(2).strip()
        return pages

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
                adaptive_retry=options["adaptive_retry"]):
                job = self.get(job["id"])
                job.update({"completed_pages": round(progress * (job["pages"][1] - job["pages"][0] + 1)), "progress": round(progress * 100), "message": message, "clean_preview": clean, "raw_preview": raw, "page_results": self._page_results(clean, raw)})
                if done:
                    result = final
                self._write_page_results(job)
                self._save(job)
            job = self.get(job["id"])
            if job.get("cancel_requested"):
                job.update({"status": "cancelled", "message": "已取消；已完成內容已保留"})
            elif result:
                outputs = {"clean": "clean.md", "raw": "raw.txt", "metrics": "metrics.json"}
                shutil.copyfile(result["md_path"], folder / outputs["clean"])
                shutil.copyfile(result["raw_path"], folder / outputs["raw"])
                shutil.copyfile(result["metrics_path"], folder / outputs["metrics"])
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
        import onnxruntime
        def cim(query: str) -> list[str]:
            try:
                command = ["powershell", "-NoProfile", "-Command", f"(Get-CimInstance {query} | Select-Object -ExpandProperty Name) -join '\n'"]
                return [line for line in subprocess.check_output(command, text=True, timeout=4).splitlines() if line]
            except Exception:
                return []
        return {"cpu": cim("Win32_Processor"), "gpu": cim("Win32_VideoController"), "onnxruntime_providers": onnxruntime.get_available_providers()}

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str): return service.get(job_id)

    @app.get("/api/jobs/{job_id}/pages/{page}")
    def page_result(job_id: str, page: int):
        job = service.get(job_id)
        path = service._path(job_id) / "pages" / f"{page}.json"
        if not path.is_file(): raise HTTPException(404, "該頁尚未完成")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str): return service.cancel(job_id)

    @app.get("/api/jobs/{job_id}/download/{kind}")
    def download(job_id: str, kind: str):
        if kind not in ALLOWED_DOWNLOADS:
            raise HTTPException(404, "找不到成果")
        path = service._path(job_id) / ALLOWED_DOWNLOADS[kind]
        if not path.is_file():
            raise HTTPException(404, "成果尚未產生")
        return FileResponse(path, filename=path.name)

    @app.get("/api/jobs/{job_id}/pages/{page}/original")
    def original_page(job_id: str, page: int):
        job = service.get(job_id)
        if page < 1 or page > job["total_pages"]:
            raise HTTPException(404, "找不到頁面")
        doc = fitz.open(service._path(job_id) / "source.pdf")
        pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        output = service._path(job_id) / f"preview-{page}.png"
        pix.save(output)
        doc.close()
        return FileResponse(output, media_type="image/png")

    return app


app = create_app()


