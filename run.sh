#!/usr/bin/env bash
set -e

echo "======================================================"
echo "  AuraDoc OCR 本機校對工作台啟動中..."
echo "======================================================"

if [ ! -f ".venv/bin/activate" ]; then
    echo "[提醒] 尚未發現虛擬環境，正在自動執行初始安裝..."
    bash install.sh
fi

source .venv/bin/activate

echo "[啟動] 正在於 http://127.0.0.1:8000 啟動 Web 服務..."
if command -v xdg-open &>/dev/null; then
    xdg-open "http://127.0.0.1:8000" &
elif command -v open &>/dev/null; then
    open "http://127.0.0.1:8000" &
fi

exec python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
