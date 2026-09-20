#!/usr/bin/env bash
set -e

echo "======================================================"
echo "  AuraDoc OCR 一鍵自動安裝腳本 (Linux / macOS)"
echo "======================================================"

if ! command -v python3 &>/dev/null; then
    echo "[錯誤] 系統未找到 python3，請先安裝 Python 3.10+。"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[進度] 建立獨立 Python 虛擬環境 (.venv)..."
    python3 -m venv .venv
fi

echo "[進度] 啟動虛擬環境並執行智慧硬體探測安裝..."
source .venv/bin/activate
python setup_env.py "$@"

echo ""
echo "======================================================"
echo "  安裝完成！執行 ./run.sh 即可啟動服務。"
echo "======================================================"
