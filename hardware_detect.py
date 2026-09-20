"""跨平台硬體偵測模組，支援 Windows、Linux 與 macOS。

自動識別 CPU、GPU 裝置、CUDA 支援狀態與 ONNX Runtime 可用執行提供者 (Execution Providers)。
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any


def _run_cmd(cmd: list[str], timeout: float = 3.0) -> list[str]:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def detect_cpu() -> list[str]:
    system = platform.system()
    cpus: list[str] = []

    if system == "Windows":
        lines = _run_cmd(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name) -join '\n'"])
        if lines:
            cpus.extend(lines)
    elif system == "Linux":
        lines = _run_cmd(["bash", "-c", "grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2"])
        if lines:
            cpus.extend(lines)
    elif system == "Darwin":
        lines = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if lines:
            cpus.extend(lines)

    if not cpus:
        proc = platform.processor()
        if proc:
            cpus.append(proc)
        else:
            cpus.append(platform.machine() or "Generic CPU")

    return list(dict.fromkeys(cpus))


def detect_nvidia_smi() -> list[str]:
    """若有 nvidia-smi，取得所有 NVIDIA 顯示卡型號。"""
    if not shutil.which("nvidia-smi"):
        return []
    lines = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus: list[str] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if parts:
            name = parts[0]
            mem = f" ({parts[1]} MB)" if len(parts) > 1 else ""
            gpus.append(f"{name}{mem}")
    return gpus


def detect_gpu() -> list[str]:
    system = platform.system()
    gpus: list[str] = []

    # 1. 優先透過 nvidia-smi 檢測 NVIDIA GPU
    nvidia_gpus = detect_nvidia_smi()
    if nvidia_gpus:
        gpus.extend(nvidia_gpus)

    # 2. 系統層級 GPU 檢測
    if system == "Windows":
        lines = _run_cmd(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '\n'"])
        for name in lines:
            if not any(name in existing for existing in gpus):
                gpus.append(name)
    elif system == "Linux":
        lines = _run_cmd(["bash", "-c", "lspci | grep -i -E 'vga|3d|display' | cut -d: -f3"])
        for name in lines:
            if not any(name in existing for existing in gpus):
                gpus.append(name)
    elif system == "Darwin":
        lines = _run_cmd(["system_profiler", "SPDisplaysDataType"])
        for line in lines:
            if "Chipset Model:" in line:
                model = line.split(":", 1)[1].strip()
                if model and model not in gpus:
                    gpus.append(model)

    return list(dict.fromkeys(gpus)) or ["No dedicated GPU detected"]


def get_onnxruntime_providers() -> list[str]:
    try:
        import onnxruntime
        return onnxruntime.get_available_providers()
    except Exception:
        return []


def detect_hardware_summary() -> dict[str, Any]:
    cpus = detect_cpu()
    gpus = detect_gpu()
    providers = get_onnxruntime_providers()
    system = platform.system()

    has_cuda_ep = "CUDAExecutionProvider" in providers
    has_dml_ep = "DmlExecutionProvider" in providers
    has_coreml_ep = "CoreMLExecutionProvider" in providers

    # 決定推薦後端模式
    if has_cuda_ep:
        recommended = "cuda"
    elif has_dml_ep and system == "Windows":
        recommended = "dml"
    elif has_coreml_ep and system == "Darwin":
        recommended = "coreml"
    else:
        recommended = "cpu"

    return {
        "os": system,
        "os_release": platform.release(),
        "arch": platform.machine(),
        "cpu": cpus,
        "gpu": gpus,
        "onnxruntime_providers": providers,
        "has_cuda": has_cuda_ep,
        "has_dml": has_dml_ep,
        "has_coreml": has_coreml_ep,
        "recommended_backend": recommended
    }


if __name__ == "__main__":
    import json
    print(json.dumps(detect_hardware_summary(), ensure_ascii=False, indent=2))
