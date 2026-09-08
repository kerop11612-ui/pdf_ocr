"""
OCR Quality Scorer, QA Pairer, and Review Report Generator for Japanese Textbooks.
Features:
1. Per-page quality scoring (0-100), Japanese/noise ratios, anomaly detection.
2. OCR review report (ocr_review_report.md) with direct page image provenance.
3. Automated Question & Answer extraction and pairing (qa_pairs.jsonl).
4. Image artifact generation for visual verification.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import fitz  # PyMuPDF

# Character class definitions
RE_JP_CHAR = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
RE_HIRAGANA = re.compile(r"[\u3040-\u309F]")
RE_KATAKANA = re.compile(r"[\u30A0-\u30FF]")
RE_KANJI = re.compile(r"[\u4E00-\u9FFF]")
RE_EN_CHAR = re.compile(r"[a-zA-Z]")
RE_DIGIT = re.compile(r"[0-9]")
RE_GARBLED = re.compile(r"[□■▲▼◇◆\uFFFD]|>{2,}|\<{2,}|>\d+>#")
RE_STICKY_Q_NUM = re.compile(r"^(\d{1,3})([\u4E00-\u9FFF\u3040-\u30FF].*)")
RE_ISOLATED_KANA = re.compile(r"^[ぁ-んァ-ン]{1,2}$")
RE_CIRCLED_DIGITS = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩]")

def evaluate_page_quality(
    page_no: int,
    text_lines: List[str],
    avg_ocr_conf: float = 0.95
) -> Dict[str, Any]:
    """
    Calculates per-page Japanese OCR quality metrics.
    Returns:
    {
        "page": page_no,
        "quality": "high" | "medium" | "low",
        "score": int (0-100),
        "needs_review": bool,
        "reasons": List[str],
        "metrics": Dict[str, Any]
    }
    """
    combined_text = "\n".join(text_lines)
    total_chars = len(combined_text.replace(" ", "").replace("\n", ""))
    
    if total_chars == 0:
        return {
            "page": page_no,
            "quality": "low",
            "score": 30,
            "needs_review": True,
            "reasons": ["全頁空白或無有效文字識別"],
            "metrics": {
                "total_chars": 0,
                "jp_ratio": 0.0,
                "garbled_count": 0,
                "sticky_num_count": 0,
                "isolated_kana_count": 0
            }
        }

    jp_chars = len(RE_JP_CHAR.findall(combined_text))
    en_chars = len(RE_EN_CHAR.findall(combined_text))
    garbled_matches = RE_GARBLED.findall(combined_text)
    
    sticky_lines = []
    isolated_kana_lines = []
    isolated_numbers = []

    for line in text_lines:
        s_line = line.strip()
        if not s_line:
            continue
        m_sticky = RE_STICKY_Q_NUM.match(s_line)
        if m_sticky:
            sticky_lines.append(s_line)
        if RE_ISOLATED_KANA.match(s_line):
            isolated_kana_lines.append(s_line)
        if re.match(r"^\d+$", s_line):
            isolated_numbers.append(s_line)

    jp_ratio = jp_chars / total_chars if total_chars > 0 else 0.0
    en_ratio = en_chars / total_chars if total_chars > 0 else 0.0

    # Scoring Deduction Logic (Base 100)
    score = 100.0
    reasons = []

    # 1. Low OCR confidence penalty (up to 25 pts)
    if avg_ocr_conf < 0.85:
        conf_penalty = (0.85 - avg_ocr_conf) * 50
        score -= conf_penalty
        reasons.append(f"OCR 辨識信心度偏低 ({avg_ocr_conf:.2f})")

    # 2. Garbled character penalty (10 pts per occurrence, up to 30)
    if garbled_matches:
        g_penalty = min(30, len(garbled_matches) * 10)
        score -= g_penalty
        reasons.append(f"偵測到亂碼/未知替換字元共 {len(garbled_matches)} 處")

    # 3. Sticky question number penalty (8 pts per occurrence, up to 24)
    if sticky_lines:
        st_penalty = min(24, len(sticky_lines) * 8)
        score -= st_penalty
        reasons.append(f"題號與句子黏連 {len(sticky_lines)} 處 (例: {sticky_lines[0][:15]})")

    # 4. Isolated kana lines (possible furigana detachment, 5 pts per occurrence, up to 15)
    if isolated_kana_lines:
        ik_penalty = min(15, len(isolated_kana_lines) * 5)
        score -= ik_penalty
        reasons.append(f"假名單獨成行 (疑似注音脫落) {len(isolated_kana_lines)} 處")

    # 5. Low Japanese content on normal pages (>100 chars, jp_ratio < 0.3)
    if total_chars > 100 and jp_ratio < 0.3:
        score -= 15
        reasons.append(f"日文字元比例偏低 ({jp_ratio * 100:.1f}%)")

    score = max(0, min(100, int(round(score))))
    
    if score >= 80:
        quality = "high"
    elif score >= 60:
        quality = "medium"
    else:
        quality = "low"

    needs_review = (score < 75) or (len(garbled_matches) > 0) or (len(sticky_lines) > 0)

    return {
        "page": page_no,
        "quality": quality,
        "score": score,
        "needs_review": needs_review,
        "reasons": reasons if reasons else ["辨識品質優良"],
        "metrics": {
            "total_chars": total_chars,
            "jp_ratio": round(jp_ratio, 3),
            "en_ratio": round(en_ratio, 3),
            "garbled_count": len(garbled_matches),
            "sticky_num_count": len(sticky_lines),
            "isolated_kana_count": len(isolated_kana_lines),
            "isolated_numbers_count": len(isolated_numbers),
            "avg_confidence": round(avg_ocr_conf, 3)
        }
    }


def save_page_preview_image(page: fitz.Page, page_no: int, output_dir: Path, dpi: int = 150) -> Path:
    """Renders and saves a page image for visual verification in review reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / f"page_{page_no:03d}.png"
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(img_path))
    return img_path


def extract_questions_and_answers(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts multiple-choice questions and pairs them with answers.
    Condition for needs_review:
    - Choices count < 4
    - Unmatched answer
    - Low pairing confidence
    """
    questions = []
    answer_pool = {}  # {q_id: {"answer": ..., "page": ...}}

    circ_map = {"①": "1", "②": "2", "③": "3", "④": "4"}

    # Step 1: Parse answers pool from all pages
    for page in pages_data:
        p_no = page["page_no"]
        lines = page.get("lines", [])
        for line in lines:
            # Pattern: 1-2, 1: 2, 【正解】1, 解答 1: 3
            ans_table_matches = re.findall(r"(?:問|Q)?(\d+)[\s:：\-]+([1-4①-④])", line)
            for q_str, a_str in ans_table_matches:
                norm_a = circ_map.get(a_str, a_str)
                answer_pool[int(q_str)] = {"answer": norm_a, "page": p_no}

            m_direct = re.search(r"(?:解答|正解|答え)[\s:：]*([1-4①-④])", line)
            if m_direct:
                answer_pool.setdefault("_last_direct", []).append((p_no, circ_map.get(m_direct.group(1), m_direct.group(1))))

    # Step 2: Parse questions across pages
    q_pattern = re.compile(r"^(?:[問Q]|Question)?\s*(\d+)(?:[\.、\s\)\:・\-]+|(?=[\u4E00-\u9FFF\u3040-\u30FF]))(.+)")
    choice_pattern = re.compile(r"(?:([①②③④])|([1-4])[\.、\)])\s*([^①②③④1-4\n]+)")

    for page in pages_data:
        p_no = page["page_no"]
        lines = page.get("lines", [])
        
        current_q = None
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check question start
            qm = q_pattern.match(line_str)
            if qm and len(line_str) > 6:
                if current_q:
                    questions.append(current_q)
                q_id = int(qm.group(1))
                q_text = qm.group(2).strip()
                current_q = {
                    "id": q_id,
                    "question": q_text,
                    "choices": {},
                    "source_pages": [p_no],
                    "confidence": "high",
                    "needs_review": False,
                    "reasons": []
                }
                continue

            # Check choices within or following question
            if current_q:
                c_matches = choice_pattern.findall(line_str)
                if c_matches:
                    for m in c_matches:
                        key = m[0] if m[0] else m[1]
                        norm_key = circ_map.get(key, key)
                        val = m[2].strip()
                        if val:
                            current_q["choices"][norm_key] = val
                else:
                    if not current_q["choices"] and len(line_str) < 100:
                        current_q["question"] += " " + line_str

        if current_q:
            questions.append(current_q)

    # Step 3: Pair with answer pool & validate
    paired_results = []
    seen_ids = set()

    for q in questions:
        q_id = q["id"]
        if q_id in seen_ids:
            continue
        seen_ids.add(q_id)

        ans_info = answer_pool.get(q_id)
        if ans_info:
            q["answer"] = ans_info["answer"]
            if ans_info["page"] not in q["source_pages"]:
                q["source_pages"].append(ans_info["page"])
            q["confidence"] = "high"
        else:
            q["answer"] = None
            q["confidence"] = "low"
            q["needs_review"] = True
            q["reasons"].append("未配對到標準答案")

        if len(q["choices"]) < 4:
            q["needs_review"] = True
            q["reasons"].append(f"選項不足 4 個 (僅辨識到 {len(q['choices'])} 個)")

        q["explanation"] = ""
        paired_results.append(q)

    return paired_results


def generate_ocr_review_report(
    doc_id: str,
    page_qualities: List[Dict[str, Any]],
    qa_records: Optional[List[Dict[str, Any]]] = None,
    image_rel_dir: str = "../pages"
) -> str:
    """
    Builds the canonical ocr_review_report.md markdown file.
    Focuses on page-by-page OCR anomalies and image provenance.
    """
    total_pages = len(page_qualities)
    low_q_pages = [p for p in page_qualities if p["quality"] == "low"]
    med_q_pages = [p for p in page_qualities if p["quality"] == "medium"]
    needs_review_pages = [p for p in page_qualities if p["needs_review"]]
    unpaired_qa = [qa for qa in qa_records if qa.get("needs_review")] if qa_records else []

    report = [
        f"# OCR 待確認與品質稽核報告: {doc_id}",
        f"- **檢測總頁數:** {total_pages} 頁",
        f"- **優良 (High):** {total_pages - len(low_q_pages) - len(med_q_pages)} 頁",
        f"- **中等 (Medium):** {len(med_q_pages)} 頁",
        f"- **偏低 (Low):** {len(low_q_pages)} 頁",
        f"- **需人工覆核頁數:** {len(needs_review_pages)} 頁",
    ]
    if qa_records:
        report.append(f"- **題目待確認數:** {len(unpaired_qa)} 題")

    report.extend([
        "",
        "---",
        "## 一、 需覆核頁面總覽表",
        "",
        "| 頁碼 | 品質評分 | 等級 | 待確認原因 | 原圖對照 |",
        "| :---: | :---: | :---: | :--- | :---: |"
    ])

    for p in page_qualities:
        if p["needs_review"]:
            p_no = p["page"]
            reasons_str = "; ".join(p["reasons"])
            img_filename = f"page_{p_no:03d}.png"
            img_link = f"[{img_filename}]({image_rel_dir}/{img_filename})"
            report.append(f"| **P.{p_no}** | {p['score']} | `{p['quality']}` | {reasons_str} | {img_link} |")

    report.extend([
        "",
        "---",
        "## 二、 逐頁詳細待確認清單與原圖定位",
        ""
    ])

    for p in page_qualities:
        if p["needs_review"]:
            p_no = p["page"]
            img_filename = f"page_{p_no:03d}.png"
            img_rel_path = f"{image_rel_dir}/{img_filename}"
            report.append(f"### 原書頁面：{p_no}")
            report.append(f"- **原圖:** `{img_filename}` ([點此檢視原圖]({img_rel_path}))")
            report.append(f"- **綜合評分:** {p['score']} 分 ({p['quality'].upper()})")
            report.append(f"- **日文比例:** {p['metrics']['jp_ratio'] * 100:.1f}%")
            report.append(f"- **待確認項目:**")
            for r in p["reasons"]:
                report.append(f"  - ⚠️ {r}")
            report.append("")

    if unpaired_qa:
        report.extend([
            "---",
            "## 三、 題目與答案配對異常清單",
            "",
            "| 題號 | 來源頁面 | 辨識選項數 | 答案狀態 | 待確認原因 |",
            "| :---: | :---: | :---: | :---: | :--- |"
        ])
        for qa in unpaired_qa:
            q_id = qa["id"]
            pages_str = ", ".join(map(str, qa.get("source_pages", [])))
            c_count = len(qa.get("choices", {}))
            ans_str = qa.get("answer") or "❌ 缺"
            r_str = "; ".join(qa.get("reasons", []))
            report.append(f"| 第 {q_id} 題 | P.{pages_str} | {c_count} | {ans_str} | {r_str} |")
        report.append("")

    report.append("---")
    report.append("> [!TIP]\n> 核心原則：原始識別結果完整留存於 `04_chunks` 與 `02_structured`，本報告僅列出需人工覆核或確認之項目，請對照 `pages/` 原圖校驗。")

    return "\n".join(report)
