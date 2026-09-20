@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ======================================================
echo   AuraDoc OCR 一鍵自動安裝腳本 (Windows)
echo ======================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [錯誤] 系統未安裝 Python，或未將 Python 加入 PATH 環境變數！
    echo 請先安裝 Python 3.10 或更高版本：https://www.python.org/
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [進度] 建立獨立 Python 虛擬環境 (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [錯誤] 虛擬環境建立失敗！
        pause
        exit /b 1
    )
)

echo [進度] 啟動虛擬環境並執行智慧硬體探測安裝...
call .venv\Scripts\activate.bat
python setup_env.py %*

if %errorlevel% equ 0 (
    echo.
    echo ======================================================
    echo   安裝完成！日後可直接按兩下 run.bat 啟動服務。
    echo ======================================================
) else (
    echo [錯誤] 安裝過程失敗，請參閱上方記錄。
)

pause
