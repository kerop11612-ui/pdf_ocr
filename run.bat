@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ======================================================
echo   AuraDoc OCR 本機校對工作台啟動中...
echo ======================================================

if not exist ".venv\Scripts\activate.bat" (
    echo [提醒] 尚未發現虛擬環境，正在自動執行初始安裝...
    call install.bat
    if %errorlevel% neq 0 exit /b 1
)

call .venv\Scripts\activate.bat

echo [啟動] 正在於 http://127.0.0.1:8000 啟動 Web 服務...
start "" "http://127.0.0.1:8000"
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000

pause
