# Task: AuraDoc UI/UX 專業級設計系統升級 (基於 ui-ux-pro-max)

## 目標
全面導入 ui-ux-pro-max 設計規範（Slate Dark Pro + OLED 對比度），消除 Emoji 反模式、實作原圖畫布 Pan & Zoom、Minimap 圖例強化、快速鍵速查面板與平滑微互動。

## 實作項目
- [x] Phase A: 全面消除 Emoji，導入標準 Lucide 向量 SVG 圖示
  - [x] 替換工作台按鈕（標記確認、待確認跳轉、上一頁、下一頁、編輯、重跑、下載）
  - [x] 替換狀態 Badge 與 Toast 通知圖示
- [x] Phase B: 原圖畫布平移縮放引擎 (Pan & Zoom Canvas)
  - [x] 於原圖工具列加入縮放控制器（放大、縮小、100%、適應寬度）
  - [x] 實作滑鼠滾輪縮放、按住左/中鍵拖曳平移
  - [x] 確保底圖 PNG 與頂層 SVG 向量框變換矩陣完全同步
- [x] Phase C: Minimap 圖例說明與 Rich Tooltip
  - [x] 頂部加入微型圖例標籤（已確認綠、待確認紅、普通深藍）
  - [x] 強化懸停摘要提示
- [x] Phase D: 快速鍵速查面板 (Keyboard Shortcuts Cheat Sheet Modal)
  - [x] 頂部工具列加入 `?` 按鈕，支援按下 `?` 或點擊開啟 Modal
  - [x] 結構化語意 `<kbd>` 按鍵展示
- [x] Phase E: Slate Dark Pro 設計系統與微互動樣式
  - [x] 導入 Plus Jakarta Sans 與 JetBrains Mono
  - [x] 調整色系為 OLED Slate Dark（#080D1A / #0F172A / #1E293B）
  - [x] 統一過渡時間為 180ms ease，避免任何 layout shift

## 目標檔案
- `static/index.html` [MODIFY]
- `static/app.css` [MODIFY]
- `static/app.js` [MODIFY]
- `tests/test_web_app.py` [MODIFY]
