# Task: 重構極致日文排版還原與字元精度增強系統 (UI 與核心管線)

## 目標
徹底摒棄原圖截圖、題庫配對與 RAG 向量切塊等次要包袱，將系統 100% 聚焦於：
1. **排版還原 (Layout Preservation)**：
   - 雙欄與多欄智慧偵測切分 (XY-Cut Column Split)，根除左右穿插問題。
   - 標點符號感知自然重組成句 (Sentence Reflow Engine)，消除 OCR 破碎硬換行。
   - 振假名 (Ruby) 幾何過濾/降噪，保護主句法結構。
   - 頂底邊界保護罩 (Margin Crop)，去除頁首頁尾雜訊。
2. **精準度提升 (Accuracy Boost)**：
   - CLAHE 影像自適應對比增強，防止假名濁點 (゛/゜) 遺失與底灰透字。
   - 語境約束的同形異義字自動校正 (長音符號 ー、カ/力、ロ/口、ニ/二)。
   - 數字與日文字元黏字解耦 (如 `14次の文` -> `14 次の文`)。
3. **UI 簡約高對比重構 (ui-ux-pro-max)**：
   - 左右直觀對比：乾淨排版 Markdown 預覽 vs 原始 OCR 文本預覽。
   - 實時精準度報告：顯示同形字校準次數、黏字修復數、雙欄偵測狀態與 FPS。
   - 精簡下載：一鍵導出乾淨版 `.md`、原始純文字 `.txt` 與排版指標 `.json`。

## 目標檔案
- `textbook_layout.py` [NEW]: 排版引擎 (XY-Cut 雙欄排序、句末標點融合成段、振假名清洗) 與精度引擎 (CLAHE 前處理、日文同形字/黏字校正)。(已完成)
- `directml_pipeline.py` [MODIFY]: 整合 `textbook_layout.py`，專注產出乾淨排版文字與原始純文字。(已完成)
- `app.py` [MODIFY]: 重構 UI 面板，加入排版/精度切換開關，提供排版後與原始文字雙視圖對比。(已上線)
- `test_textbook_layout.py` [NEW]: 單元測試驗證通過。(Exit Code 0)
- `test_layout_e2e.py` [NEW]: 端到端整合測試驗證通過。(Exit Code 0)
- `task.md` [MODIFY]: 進度追蹤與成果紀錄。(已更新)

## 實測成果
- `test_textbook_layout.py`：單元測試驗證 CLAHE 影像微對比增強、同形字（長音、カ/力、ロ/口、ニ/二）9 處校準、XY-Cut 雙欄分割以及句末標點自然連貫成段，全部通過。
- `test_layout_e2e.py`：端到端整合測試驗證完成，生成乾淨排版筆記、原始對照純文字與指標 JSON。
- 服務現正運行中：`http://127.0.0.1:7860` (HTTP 200 OK)。
