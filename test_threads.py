from rapidocr import RapidOCR
import numpy as np
import concurrent.futures

o1 = RapidOCR(params={'EngineConfig.onnxruntime.use_dml': True})
o2 = RapidOCR(params={'EngineConfig.onnxruntime.use_dml': True})
img = np.ones((500, 500, 3), dtype=np.uint8) * 255

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    f1 = ex.submit(o1, img)
    f2 = ex.submit(o2, img)
    r1, r2 = f1.result(), f2.result()
    print("Concurrent DirectML Success:", r1 is not None, r2 is not None)
