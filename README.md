# AuraDoc OCR Local Web

## 啟動

```powershell
cd C:\Users\kerop\Documents\antigravity\proud-hertz
.\run_web.ps1
```

服務只監聽 `http://127.0.0.1:8766`。既有 Gradio 入口 `app.py` 保持獨立。

## 操作

1. 上傳 PDF，選前 5 頁、全書或合法自訂頁碼。
2. 選擇 `auto`、`GPU` 或 `CPU`；`auto` 僅在 DirectML 實際 session 未啟用時回退 CPU，`GPU` 會顯示初始化錯誤。
3. 建立工作後可刷新瀏覽器，以 URL hash 或最近工作列表取回進度。
4. 使用上一頁／下一頁檢視每頁原 PDF、Clean Markdown 與 Raw OCR；完成後下載 MD、TXT、JSON。

## 依賴

Python 3.11、FastAPI、Uvicorn、python-multipart、PyMuPDF、RapidOCR、onnxruntime-directml，以及原有 requirements.txt 內 OCR 依賴。

## 限制

- 一次僅執行一個 OCR worker；其他工作會排隊。
- 工作和來源 PDF 儲存在 `web_jobs/<UUID>/`，不會覆寫舊 `dist/` 成果。
- 取消會在頁面交界停止，並保留已完成頁面的預覽；未完成工作重啟服務後會重新排隊。
- UI 的 Markdown 以純文字安全呈現，不將 OCR 內容注入 HTML。
