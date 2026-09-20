# AuraDoc OCR — 日文與雙語教材排版還原校對工作台

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.x%20%7C%2012.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![DirectML](https://img.shields.io/badge/DirectML-Windows%20GPU-0078D7.svg)](https://learn.microsoft.com/en-us/windows/ai/directml/dml)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

**專為複雜日文／繁中教材設計的高精度本地 OCR 與視覺化排版校對工作台**  
自動適應 NVIDIA CUDA、Windows DirectML 與多核 CPU 加速

[English Overview](#english-overview) · [特色功能](#核心特色) · [一鍵啟動](#快速啟動) · [硬體加速](#硬體支援與自動適應) · [匯出格式](#匯出成果格式)

</div>

---

## 核心特色

- ⚡ **全自動硬體加速適應**：優先呼叫 **NVIDIA CUDA**，次選 **Windows DirectML**（相容 AMD / Intel / NVIDIA），底層平滑降級至多執行緒 CPU。
- 📑 **書籍與教材版面自適應還原**：
  - **雙欄 / 多欄 XY-Cut 智慧重排**：文章段落連貫重流，不再出現跨欄錯位串字。
  - **Ruby 振假名智慧濾除**：以統計字高結合純假名特徵，分離正文與漢字注音假名。
  - **版面型態自動分類**：智慧識別文章 (`prose`)、題庫測驗 (`questions`)、單字卡 (`vocabulary_cards`)、目錄 (`toc`) 與空白裝飾頁。
  - **OCR 黏字與同形字自動校正**：修正日文句讀、括號黏連及常見漢字同形異體字。
- 🔍 **視覺化校對工作台 (Proofreading Workbench)**：
  - **45% / 55% 分割畫布**：高對比原始頁面圖與整理後文字同步對照。
  - **SVG 向量文字框即時疊加**：滑鼠懸停文字，原圖對應行座標框同步高亮顯示，反之亦然。
  - **頁面全覽視覺地圖 (Minimap)**：全書進度條塊可視化，已確認、待確認與一般頁面一目了然，點擊直達。
  - **極速單鍵校對**：`Space` 一鍵標記已確認並直達下一個待審核頁；`E` 原地編輯文字；`←` / `→` 流暢翻頁。
  - **差分比對 (Diff View)**：精準呈現原始 OCR (Raw) 與版面清洗後 (Clean) 的文字差異。
  - **單頁即時重跑**：特定頁面辨識不滿意，無須整本書重新排隊，單頁 1 秒重跑。

---

## 硬體支援與自動適應

AuraDoc 在啟動與建立工作時會自動掃描您的設備規格，並決定最佳執行後端：

| 優先級 | 後端引擎 | 適用環境 | 特點與效能 |
|:---:|:---|:---|:---|
| **1 (最快)** | **NVIDIA CUDA** | 具備 NVIDIA 顯卡 (RTX / GTX 系列) | 極速推理、FP16 支援、批次加速 |
| **2 (推薦)** | **Windows DirectML** | Windows 10/11，相容 AMD、Intel 與 NVIDIA 顯卡 | 免裝 CUDA Toolkit，開箱即用 GPU 加速 |
| **3 (保底)** | **CPU Execution Provider** | 任何 x86_64 或 ARM64 處理器 | 支援多核心並行，適用輕量或伺服器環境 |

---

## 快速啟動

### 方式 A：Windows 一鍵腳本（最簡單）

1. Clone 本專案：
   ```bash
   git clone https://github.com/your-username/auradoc-ocr.git
   cd auradoc-ocr
   ```
2. 按兩下 **`run.bat`**。
   - 腳本將自動建立虛擬環境、偵測硬體安裝對應 ONNX Runtime，並自動開啟瀏覽器前往 `http://127.0.0.1:8000/`！

### 方式 B：Linux / macOS 腳本

```bash
git clone https://github.com/your-username/auradoc-ocr.git
cd auradoc-ocr
chmod +x install.sh run.sh
./run.sh
```

### 方式 C：智慧手動安裝

```bash
# 1. 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Windows 請執行: .venv\Scripts\activate

# 2. 執行硬體探測安裝精靈 (自動偵測 CUDA / DirectML / CPU)
python setup_env.py

# 3. 啟動 Web 服務
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

### 方式 D：Docker 容器部屬 (含 NVIDIA GPU 支援)

```bash
docker build -t auradoc-ocr:latest .
docker run --gpus all -p 8000:8000 -v $(pwd)/web_jobs:/app/web_jobs auradoc-ocr:latest
```

---

## 匯出成果格式

處理完成後，可於介面一鍵下載多種專業格式：

1. **`clean.md`**：整理版 Markdown，已還原分段、標題與題庫排版。
2. **`obsidian.md`**：Obsidian 專用筆記，內建 YAML Frontmatter（包含檔名、日期、頁數、標籤）。
3. **`clean_reviewed.md`**：僅包含使用者已確認或無疑義頁面的純淨文字。
4. **`review_pending.md`**：抽離所有具低信心度或異常符號的待覆核頁面，方便集中檢查。
5. **`structured.jsonl`**：每行代表一頁，包含每行文字之座標 Bounding Box、Confidence 與分塊幾何。
6. **`raw.txt`**：原始 OCR 逐行文字，不帶任何版面重組。
7. **`metrics.json`**：全書耗時統計、頁面類型分佈、同形字修正次數報表。

---

<a name="english-overview"></a>
## English Overview

**AuraDoc OCR** is a turnkey, privacy-first local OCR and proofreading workbench specifically optimized for complex Japanese and multilingual textbook layouts.

### Key Highlights
- **Hardware-Adaptive Acceleration**: Seamless fallback chain: `NVIDIA CUDA` > `DirectML (Windows GPU)` > `Multi-threaded CPU`.
- **Advanced Textbook Layout Engine**: Automatic two-column XY-cut reflow, ruby (furigana) noise suppression, homoglyph correction, and page classification (`prose`, `questions`, `vocabulary_cards`, `toc`).
- **Proofreading Workbench**: Side-by-side original image view with synchronized interactive SVG bounding boxes, instant Diff viewer, visual Minimap, and keyboard-first workflow (`Space` to review, `E` to edit).
- **Multiple Export Profiles**: Ready-to-use Markdown, Obsidian vault notes with frontmatter, structured JSONL geometry records, and review-only filtered extracts.

---

## 測試與驗證

本專案附帶完整的單元測試與端對端回歸測試套件：

```bash
python -m pytest tests test_textbook_layout.py -v
```

---

## 授權條款

本專案採用 [Apache License 2.0](LICENSE) 授權開源。
