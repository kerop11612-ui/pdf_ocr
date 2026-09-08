# FastAPI 本機 Web MVP

## 範圍
- 保留 `app.py`、既有教材與成果不動；新增 `web_app.py`、`static/`、`tests/`、`run_web.ps1`。
- 每次上傳建立 UUID 工作目錄：來源、工作狀態與產物均隔離。
- 單一背景 worker 序列化 OCR；工作狀態以 JSON 持久化，刷新後可輪詢。

## 介面/API
- `POST /api/jobs`：驗證 PDF、頁數與範圍後排入佇列。
- `GET /api/jobs/{id}`、`POST /api/jobs/{id}/cancel`：查詢與取消。
- `GET /api/jobs/{id}/download/{kind}`、`GET /api/jobs/{id}/pages/{page}/original`：安全存取成果／原 PDF 頁面。
- 前端選擇前五頁、全書、自訂；auto/GPU/CPU；進階開關收合。

## 驗證
1. 先以 FastAPI TestClient 測 PDF 驗證、範圍、UUID 隔離、取消與下載路徑限制。
2. 跑既有可靠性／版面測試。
3. 以 `test_input.pdf` 實跑 CPU；GPU 只載入並記錄實際 provider，失敗按模式回退或報錯。
