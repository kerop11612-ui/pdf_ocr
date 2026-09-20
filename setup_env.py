#!/usr/bin/env python3
"""AuraDoc 智慧硬體探測與相依套件安裝引導程式。

用法:
    python setup_env.py           # 自動探測硬體並安裝最佳加速版本 (推薦)
    python setup_env.py --cuda    # 強制安裝 NVIDIA CUDA 加速版 (onnxruntime-gpu)
    python setup_env.py --dml     # 強制安裝 DirectML 加速版 (onnxruntime-directml)
    python setup_env.py --cpu     # 強制安裝純 CPU 相容版 (onnxruntime)
"""

import argparse
import platform
import shutil
import subprocess
import sys

# Configure UTF-8 console output for Windows cp950/cp936 compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def print_banner():
    print("=" * 60)
    print("  AuraDoc OCR 環境與硬體加速安裝精靈")
    print("=" * 60)


def check_python_version():
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"[錯誤] Python 版本過低 ({major}.{minor})，AuraDoc 需要 Python 3.10 或更高版本。")
        sys.exit(1)
    print(f"[檢查] 偵測到 Python 版本: {sys.version.split()[0]} [OK]")


def has_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.returncode == 0
        except Exception:
            return False
    return False


def run_pip(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"[執行] {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    print_banner()
    check_python_version()

    parser = argparse.ArgumentParser(description="AuraDoc 硬體加速安裝引導")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cuda", action="store_true", help="強制安裝 NVIDIA CUDA 版本")
    group.add_argument("--dml", action="store_true", help="強制安裝 Windows DirectML 版本")
    group.add_argument("--cpu", action="store_true", help="強制安裝純 CPU 版本")
    args = parser.parse_args()

    os_type = platform.system()
    selected_target = None

    if args.cuda:
        selected_target = "requirements-cuda.txt"
        mode_desc = "NVIDIA CUDA GPU 加速模式"
    elif args.dml:
        selected_target = "requirements-directml.txt"
        mode_desc = "Windows DirectML GPU 加速模式"
    elif args.cpu:
        selected_target = "requirements-cpu.txt"
        mode_desc = "CPU 通用相容模式"
    else:
        # 自動探測模式
        print("\n[分析] 正在探測本機硬體設備…")
        try:
            from hardware_detect import detect_hardware_summary
            hw = detect_hardware_summary()
            print(f"       作業系統: {hw['os']} ({hw['arch']})")
            print(f"       中央處理器: {', '.join(hw['cpu'])}")
            print(f"       顯示卡設備: {', '.join(hw['gpu'])}")
        except Exception as err:
            print(f"       探測失敗 ({err})，將採用預設啟發式判斷。")
            hw = {}

        if has_nvidia_gpu():
            print("\n>>> 發現 NVIDIA 獨立顯示卡與驅動！")
            print(">>> 建議目標：NVIDIA CUDA 加速版 (極速推理)")
            selected_target = "requirements-cuda.txt"
            mode_desc = "NVIDIA CUDA GPU 加速模式"
        elif os_type == "Windows":
            print("\n>>> 偵測到 Windows 系統（非 NVIDIA 或缺少 CUDA 驅動）")
            print(">>> 建議目標：DirectML 加速版（支援 AMD / Intel / NVIDIA）")
            selected_target = "requirements-directml.txt"
            mode_desc = "Windows DirectML GPU 加速模式"
        else:
            print("\n>>> 建議目標：標準 CPU 相容版")
            selected_target = "requirements-cpu.txt"
            mode_desc = "CPU 通用相容模式"

    print(f"\n[準備安裝] 目標配方: {selected_target} ({mode_desc})")
    target_path = ROOT / selected_target

    if not target_path.is_file():
        print(f"[錯誤] 找不到配方檔案: {target_path}")
        sys.exit(1)

    # 升級 pip
    run_pip(["install", "--upgrade", "pip", "setuptools", "wheel"])

    # 安裝套件
    ret = run_pip(["install", "-r", str(target_path)])
    if ret == 0:
        print("\n" + "=" * 60)
        print("  [完成] AuraDoc 安裝完成！")
        print(f"  已配置運算架構: {mode_desc}")
        print("  啟動指令: python -m uvicorn web_app:app --host 127.0.0.1 --port 8000")
        print("=" * 60)
    else:
        print(f"\n[錯誤] 安裝過程出現異常（Exit Code: {ret}）。")
        sys.exit(ret)


if __name__ == "__main__":
    main()
