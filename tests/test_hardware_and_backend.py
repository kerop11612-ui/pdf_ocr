import pytest
from hardware_detect import detect_cpu, detect_gpu, detect_hardware_summary
import directml_pipeline as pipeline


def test_hardware_detect_returns_valid_dict():
    hw = detect_hardware_summary()
    assert "os" in hw
    assert "cpu" in hw and isinstance(hw["cpu"], list)
    assert "gpu" in hw and isinstance(hw["gpu"], list)
    assert "onnxruntime_providers" in hw
    assert hw["recommended_backend"] in {"cuda", "dml", "coreml", "cpu"}


def test_get_ocr_backend_cuda_fallback_to_dml_and_cpu(monkeypatch):
    # Mock CUDA failure, DirectML failure -> falls back to CPU
    class FailingCuda:
        def __init__(self, **_): raise RuntimeError("CUDA missing cuDNN")
    class FailingDml:
        def __init__(self, **_): raise RuntimeError("Dml not supported")

    monkeypatch.setattr(pipeline, "RapidOCRCUDABackend", FailingCuda)
    monkeypatch.setattr(pipeline, "RapidOCRDirectMLBackend", FailingDml)

    # auto mode should fallback to CPU
    backend = pipeline.get_ocr_backend("auto")
    assert isinstance(backend, pipeline.RapidOCRCPUBackend)
    assert "回退" in backend.backend_actual

    # gpu mode should raise RuntimeError
    with pytest.raises(RuntimeError, match="GPU"):
        pipeline.get_ocr_backend("gpu")


def test_get_ocr_backend_cuda_selection(monkeypatch):
    class FakeCudaBackend:
        backend_actual = "CUDA GPU (device:0)"
        provider_sets = [["CUDAExecutionProvider"]]
        def recognize(self, img): return {"txts": ["OK"]}

    monkeypatch.setattr(pipeline, "RapidOCRCUDABackend", lambda **_: FakeCudaBackend())

    # When CUDA succeeds, auto mode should select CUDA!
    backend = pipeline.get_ocr_backend("auto")
    assert backend.backend_actual == "CUDA GPU (device:0)"

    # Explicit cuda request
    cuda_backend = pipeline.get_ocr_backend("cuda")
    assert cuda_backend.backend_actual == "CUDA GPU (device:0)"
