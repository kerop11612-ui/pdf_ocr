// AuraDoc OCR Proofreading Workbench Controller (ui-ux-pro-max edition)
(function() {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const $$ = selector => document.querySelectorAll(selector);

  // State
  let currentJobId = null;
  let currentJob = null;
  let currentPage = 1;
  let jobPages = [];
  let currentPageData = null;
  let pollTimer = null;
  let allHistory = [];
  let isEditing = false;

  // Zoom & Pan Canvas State
  let zoomScale = 1.0;
  let panX = 0;
  let panY = 0;
  let isPanning = false;
  let panStartX = 0;
  let panStartY = 0;

  const PAGE_TYPE_LABELS = {
    prose: '文章段落',
    questions: '測驗題庫',
    vocabulary_cards: '單字卡',
    toc: '書籍目錄',
    decorative_or_blank: '空白/裝飾'
  };

  const REVIEW_REASON_LABELS = {
    low_confidence: '低信心度',
    unknown_character: '未知缺字',
    repeated_character: '重複雜訊',
    mixed_symbol: '異常混雜符號',
    complex_layout: '多欄複雜排版'
  };

  const STATUS_LABELS = {
    queued: '排隊中',
    running: '處理中',
    completed: '已完成',
    cancelled: '已取消',
    failed: '失敗'
  };

  // SVG Icons Helper
  const SVG_ICONS = {
    check: '<svg class="ui-icon" viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></svg>',
    warning: '<svg class="ui-icon" viewBox="0 0 24 24"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    danger: '<svg class="ui-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    info: '<svg class="ui-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    edit: '<svg class="ui-icon" viewBox="0 0 24 24"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>',
    close: '<svg class="ui-icon" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
  };

  // Toast Notification System (SVG Iconized)
  function showToast(message, type = 'info', duration = 2800) {
    const container = $('#toast-container') || document.body;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const iconSvg = SVG_ICONS[type] || SVG_ICONS.info;
    toast.innerHTML = `${iconSvg}<span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(12px) scale(0.96)';
      toast.style.transition = 'all 0.22s cubic-bezier(0.16, 1, 0.3, 1)';
      setTimeout(() => toast.remove(), 240);
    }, duration);
  }

  // 1. Hardware Detection
  async function loadHardware() {
    try {
      const res = await fetch('/api/hardware');
      if (!res.ok) return;
      const data = await res.json();
      const unique = arr => [...new Set(arr || [])].join('、') || '無';
      const gpuText = unique(data.gpu);
      const isCuda = (data.onnxruntime_providers || []).includes('CUDAExecutionProvider');
      const isDml = (data.onnxruntime_providers || []).includes('DmlExecutionProvider');
      const accelText = isCuda ? 'NVIDIA CUDA 支援' : (isDml ? 'DirectML 支援' : 'CPU 運算');
      const badge = $('#hardware-badge');
      if (badge) {
        badge.textContent = `硬體：${gpuText.slice(0, 24)} · ${accelText}`;
      }
      const desc = $('#hardware-desc');
      if (desc) {
        desc.textContent = `CPU：${unique(data.cpu)} ｜ GPU：${gpuText} ｜ Providers：${unique(data.onnxruntime_providers)}`;
      }
    } catch (err) {
      console.warn('Hardware detection error:', err);
    }
  }

  // 2. Preset Modes & Range Toggle
  const form = $('#job-form');
  const fileInput = $('#file');
  const fileInfo = $('#file-info');
  const rangeModeInput = $('#range_mode');
  const customRangeBox = $('#custom-range-box');

  if (fileInput) {
    fileInput.addEventListener('change', async () => {
      const f = fileInput.files && fileInput.files[0];
      if (!f) {
        fileInfo.textContent = '請選擇 PDF 檔案';
        return;
      }
      try {
        const header = new TextDecoder().decode(await f.slice(0, 5).arrayBuffer());
        if (header.startsWith('%PDF-')) {
          fileInfo.textContent = `${f.name} (${(f.size / (1024 * 1024)).toFixed(1)} MB)`;
        } else {
          fileInfo.textContent = '警告：所選檔案可能非標準 PDF';
        }
      } catch {
        fileInfo.textContent = `${f.name} (${(f.size / (1024 * 1024)).toFixed(1)} MB)`;
      }
    });
  }

  $$('input[name="preset_mode"]').forEach(radio => {
    radio.addEventListener('change', () => {
      const val = radio.value;
      if (val === 'preview') {
        rangeModeInput.value = 'first5';
        customRangeBox.hidden = true;
      } else if (val === 'textbook') {
        rangeModeInput.value = 'all';
        customRangeBox.hidden = true;
      } else if (val === 'questions') {
        rangeModeInput.value = 'all';
        customRangeBox.hidden = true;
        if (form.elements['two_column']) form.elements['two_column'].checked = true;
      } else if (val === 'custom') {
        rangeModeInput.value = 'custom';
        customRangeBox.hidden = false;
      }
    });
  });

  // 3. Form Submission
  if (form) {
    form.addEventListener('submit', async e => {
      e.preventDefault();
      const submitBtn = $('#submit-btn');
      submitBtn.disabled = true;
      const data = new FormData(form);

      ['adaptive_retry', 'clahe', 'reflow', 'two_column', 'furigana', 'homoglyphs'].forEach(k => {
        if (form.elements[k]) data.set(k, String(form.elements[k].checked));
      });

      $('#status-msg').textContent = '正在上傳 PDF 並建立工作紀錄…';
      updatePipelineStage('ready');

      try {
        const res = await fetch('/api/jobs', { method: 'POST', body: data });
        const body = await res.json();
        if (!res.ok) throw new Error(body.detail || '建立工作失敗');
        currentJobId = body.id;
        history.replaceState(null, '', `#${body.id}`);
        showJob(body);
        pollJob();
        loadHistory();
        showToast('工作已排入佇列！', 'success');
      } catch (err) {
        $('#status-msg').textContent = `建立失敗：${err.message}`;
        showToast(`建立失敗：${err.message}`, 'danger');
      } finally {
        submitBtn.disabled = false;
      }
    });
  }

  // 4. Cancel Job
  const cancelBtn = $('#cancel-btn');
  if (cancelBtn) {
    cancelBtn.addEventListener('click', async () => {
      if (!currentJobId) return;
      try {
        const res = await fetch(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
        const body = await res.json();
        showJob(body);
        showToast('已送出取消要求', 'warning');
      } catch {
        $('#status-msg').textContent = '取消要求送出失敗。';
      }
    });
  }

  // 5. Pipeline Stages Rendering
  function updatePipelineStage(status) {
    const stages = $$('#pipeline-stages .stage');
    stages.forEach(s => s.classList.remove('active', 'completed'));
    if (status === 'ready' || status === 'queued') {
      const s = $('[data-stage="ready"]');
      if (s) s.classList.add('active');
    } else if (status === 'running' || status === 'init') {
      const s0 = $('[data-stage="ready"]'); if (s0) s0.classList.add('completed');
      const s1 = $('[data-stage="init"]'); if (s1) s1.classList.add('completed');
      const s2 = $('[data-stage="ocr"]'); if (s2) s2.classList.add('active');
      const s3 = $('[data-stage="layout"]'); if (s3) s3.classList.add('active');
    } else if (status === 'completed' || status === 'done') {
      stages.forEach(s => s.classList.add('completed'));
    }
  }

  // 6. Job Summary Display
  function showJob(job) {
    currentJob = job;
    currentJobId = job.id;

    $('#job-title').textContent = job.filename || '未知文件';
    $('#job-status-badge').textContent = STATUS_LABELS[job.status] || job.status;
    $('#status-msg').textContent = job.message || '';
    $('#progress').value = job.progress || 0;
    $('#progress-percent').textContent = `${job.progress || 0}%`;

    const totalInJob = job.pages ? (job.pages[1] - job.pages[0] + 1) : (job.total_pages || 0);
    $('#metric-pages').textContent = `${job.completed_pages || 0} / ${totalInJob} 頁`;
    $('#metric-review-count').textContent = typeof job.needs_review_count === 'number' ? `${job.needs_review_count} 頁` : '-';
    
    const backendLabel = job.backend_actual ? (job.backend_actual.includes('DirectML') ? 'DirectML GPU' : (job.backend_actual.includes('CUDA') ? 'CUDA GPU' : 'CPU')) : (job.backend_requested || 'Auto');
    $('#metric-backend').textContent = backendLabel;

    cancelBtn.disabled = !['queued', 'running'].includes(job.status);
    updatePipelineStage(job.status === 'completed' ? 'done' : (job.status === 'running' ? 'running' : 'ready'));

    // Downloads
    const downloadSec = $('#download-section');
    if (downloadSec) {
      const done = job.status === 'completed';
      downloadSec.hidden = !done;
      if (done) {
        $$('#download-section [data-kind]').forEach(a => {
          a.href = `/api/jobs/${job.id}/download/${a.dataset.kind}`;
        });
      }
    }

    // Workbench visibility
    const wb = $('#workbench');
    if (wb) wb.hidden = false;

    // Refresh pages metadata and load page
    loadJobPagesList();
  }

  // 7. Pages Metadata & Workbench Loading
  async function loadJobPagesList() {
    if (!currentJobId) return;
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/pages`);
      if (res.ok) {
        jobPages = await res.json();
      }
    } catch {
      jobPages = [];
    }

    renderMinimap();

    if (!currentPage && currentJob && currentJob.pages) {
      currentPage = currentJob.pages[0];
    }
    loadCurrentPage();
  }

  async function loadCurrentPage() {
    if (!currentJobId || !currentPage) return;
    const pageNum = currentPage;

    $('#page-label').textContent = `第 ${pageNum} 頁（共 ${currentJob?.pages?.[1] || '?'} 頁）`;
    const jumpInput = $('#page-jump-input');
    if (jumpInput) jumpInput.value = pageNum;

    $('#original-img').src = `/api/jobs/${currentJobId}/pages/${pageNum}/original`;

    // Fetch single page JSON
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/pages/${pageNum}`);
      if (!res.ok) throw new Error('該頁尚未產生');
      currentPageData = await res.json();
      renderPageDetails(currentPageData);
    } catch (err) {
      currentPageData = null;
      renderEmptyPage(err.message);
    }

    updateNavButtons();
    renderMinimap();
  }

  function renderPageDetails(data) {
    // Type badge
    const badge = $('#page-type-badge');
    badge.textContent = PAGE_TYPE_LABELS[data.page_type] || data.page_type || '一般';

    // Review status button
    const btnReview = $('#btn-toggle-reviewed');
    const labelReview = $('#label-toggle-reviewed');
    if (btnReview && labelReview) {
      const isReviewed = Boolean(data.reviewed);
      btnReview.classList.toggle('is-reviewed', isReviewed);
      labelReview.textContent = isReviewed ? '已確認 (Space 取消)' : '標記已確認 (Space)';
    }

    // Review reasons
    const container = $('#review-reasons-container');
    container.innerHTML = '';
    const reasons = data.review_reasons || [];
    if (reasons.length === 0 && !data.needs_review) {
      const span = document.createElement('span');
      span.className = 'type-badge';
      span.style.color = '#34d399';
      span.innerHTML = `${SVG_ICONS.check} 品質良好`;
      container.appendChild(span);
    } else {
      reasons.forEach(r => {
        const tag = document.createElement('span');
        tag.className = 'reason-tag';
        tag.innerHTML = `${SVG_ICONS.warning} ${REVIEW_REASON_LABELS[r] || r}`;
        container.appendChild(tag);
      });
    }

    // Clean text view
    const cleanView = $('#clean-view');
    cleanView.innerHTML = '';
    const lines = (data.clean || '').split('\n');
    lines.forEach((lineText, idx) => {
      const lineSpan = document.createElement('div');
      lineSpan.className = 'text-line';
      lineSpan.dataset.lineIdx = idx;
      lineSpan.textContent = lineText || ' ';
      lineSpan.addEventListener('mouseenter', () => highlightMatchingSvgBox(idx));
      lineSpan.addEventListener('mouseleave', () => clearSvgBoxHighlight());
      cleanView.appendChild(lineSpan);
    });

    // Clean Editor
    const editor = $('#clean-editor');
    if (editor) editor.value = data.clean || '';

    // Raw text view
    const rawView = $('#raw-view');
    if (rawView) rawView.textContent = data.raw || '無原始識別文字。';

    // Diff view
    renderDiffView(data.raw || '', data.clean || '');

    // Render SVG Bounding Boxes Overlay
    renderSvgOverlay(data);
  }

  function renderEmptyPage(msg) {
    $('#page-type-badge').textContent = '處理中';
    $('#review-reasons-container').innerHTML = '';
    $('#clean-view').textContent = msg || '此頁尚在辨識中…';
    $('#raw-view').textContent = msg || '此頁尚在辨識中…';
    $('#diff-view').innerHTML = '<span class="muted">尚無比對資料</span>';
    const svg = $('#original-svg');
    if (svg) svg.innerHTML = '';
  }

  // 8. Visual Page Minimap
  function renderMinimap() {
    const container = $('#page-minimap');
    if (!container) return;
    container.innerHTML = '';

    const pages = jobPages.length ? jobPages : (currentJob?.pages ? Array.from({ length: currentJob.pages[1] - currentJob.pages[0] + 1 }, (_, i) => ({ page: currentJob.pages[0] + i })) : []);
    let reviewedCount = 0;

    pages.forEach(p => {
      const cell = document.createElement('div');
      cell.className = 'minimap-cell';
      cell.textContent = p.page;

      if (p.page === currentPage) {
        cell.classList.add('cell-current');
      }

      if (p.reviewed || (p.user_edited && !p.needs_review)) {
        cell.classList.add('cell-reviewed');
        reviewedCount++;
      } else if (p.needs_review) {
        cell.classList.add('cell-needs-review');
      } else {
        cell.classList.add('cell-normal');
      }

      const ptype = PAGE_TYPE_LABELS[p.page_type] || p.page_type || '';
      const revStatus = p.reviewed ? ' (已確認)' : (p.needs_review ? ' (待審核)' : '');
      cell.title = `第 ${p.page} 頁 · ${ptype}${revStatus}`;

      cell.onclick = () => {
        currentPage = p.page;
        loadCurrentPage();
      };
      container.appendChild(cell);
    });

    const stats = $('#minimap-stats');
    if (stats && pages.length) {
      const pct = Math.round((reviewedCount / pages.length) * 100);
      stats.textContent = `已確認 ${reviewedCount} / ${pages.length} 頁 (${pct}%)`;
    }
  }

  // 9. SVG Overlay Bounding Boxes
  function renderSvgOverlay(data) {
    const svg = $('#original-svg');
    if (!svg) return;
    svg.innerHTML = '';

    const lines = data.source_lines || [];
    if (lines.length === 0) return;

    let pw = data.page_width;
    let ph = data.page_height;
    if (!pw || !ph) {
      pw = Math.max(...lines.map(l => (l.bbox ? l.bbox[2] : 1000)), 1000);
      ph = Math.max(...lines.map(l => (l.bbox ? l.bbox[3] : 1400)), 1400);
    }
    svg.setAttribute('viewBox', `0 0 ${pw} ${ph}`);

    lines.forEach((line, idx) => {
      if (!line.bbox) return;
      const [x1, y1, x2, y2] = line.bbox;
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', x1);
      rect.setAttribute('y', y1);
      rect.setAttribute('width', Math.max(x2 - x1, 4));
      rect.setAttribute('height', Math.max(y2 - y1, 4));
      rect.dataset.lineIdx = idx;

      if (line.included_in_clean === false) {
        rect.classList.add('bbox-filtered');
      } else if (line.needs_review) {
        rect.classList.add('bbox-review');
      } else {
        rect.classList.add('bbox-clean');
      }

      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      const reasons = line.review_reasons?.map(r => REVIEW_REASON_LABELS[r] || r).join(', ');
      title.textContent = `${line.text}\n${line.filter_reason ? '過濾原因: ' + line.filter_reason : (reasons ? '待確認: ' + reasons : '信心度: ' + Math.round((line.confidence || 0) * 100) + '%')}`;
      rect.appendChild(title);

      rect.addEventListener('mouseenter', () => {
        rect.classList.add('active');
        highlightMatchingTextLine(idx);
      });
      rect.addEventListener('mouseleave', () => {
        rect.classList.remove('active');
        clearTextLineHighlight();
      });
      rect.addEventListener('click', () => {
        scrollTextLineIntoView(idx);
      });

      svg.appendChild(rect);
    });
  }

  function highlightMatchingSvgBox(idx) {
    const rect = $(`#original-svg rect[data-line-idx="${idx}"]`);
    if (rect) rect.classList.add('active');
  }

  function clearSvgBoxHighlight() {
    $$('#original-svg rect.active').forEach(r => r.classList.remove('active'));
  }

  function highlightMatchingTextLine(idx) {
    const line = $(`#clean-view .text-line[data-line-idx="${idx}"]`);
    if (line) line.classList.add('active');
  }

  function clearTextLineHighlight() {
    $$('#clean-view .text-line.active').forEach(l => l.classList.remove('active'));
  }

  function scrollTextLineIntoView(idx) {
    const line = $(`#clean-view .text-line[data-line-idx="${idx}"]`);
    if (line) {
      line.scrollIntoView({ behavior: 'smooth', block: 'center' });
      line.classList.add('active');
      setTimeout(() => line.classList.remove('active'), 1500);
    }
  }

  // Toggle SVG Overlay
  const toggleOverlay = $('#toggle-overlay');
  if (toggleOverlay) {
    toggleOverlay.addEventListener('change', () => {
      const svg = $('#original-svg');
      if (svg) svg.style.display = toggleOverlay.checked ? 'block' : 'none';
    });
  }

  // Toggle Collapse Original Image on Mobile
  const toggleCollapseBtn = $('#toggle-collapse-img');
  if (toggleCollapseBtn) {
    toggleCollapseBtn.addEventListener('click', () => {
      const vp = $('#original-viewport');
      if (vp) {
        const isHidden = vp.style.display === 'none';
        vp.style.display = isHidden ? 'block' : 'none';
        toggleCollapseBtn.textContent = isHidden ? '收合原圖' : '展開原圖';
      }
    });
  }

  // 10. Pan & Zoom Engine for Original Canvas
  const originalViewport = $('#original-viewport');
  const originalContainer = $('#original-container');
  const zoomLevelIndicator = $('#zoom-level');

  function updateCanvasTransform() {
    if (originalContainer) {
      originalContainer.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomScale})`;
    }
    if (zoomLevelIndicator) {
      zoomLevelIndicator.textContent = `${Math.round(zoomScale * 100)}%`;
    }
  }

  if (originalViewport && originalContainer) {
    // Wheel Zoom
    originalViewport.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = originalViewport.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      const newScale = Math.max(0.4, Math.min(4.0, zoomScale * factor));

      // Zoom towards cursor
      panX = mouseX - (mouseX - panX) * (newScale / zoomScale);
      panY = mouseY - (mouseY - panY) * (newScale / zoomScale);
      zoomScale = newScale;

      updateCanvasTransform();
    }, { passive: false });

    // Drag Pan
    originalViewport.addEventListener('mousedown', e => {
      // Don't drag if clicking directly on a selectable bounding box
      if (e.target.tagName === 'rect') return;
      isPanning = true;
      panStartX = e.clientX - panX;
      panStartY = e.clientY - panY;
      originalViewport.classList.add('panning');
    });

    window.addEventListener('mousemove', e => {
      if (!isPanning) return;
      panX = e.clientX - panStartX;
      panY = e.clientY - panStartY;
      updateCanvasTransform();
    });

    window.addEventListener('mouseup', () => {
      if (isPanning) {
        isPanning = false;
        originalViewport.classList.remove('panning');
      }
    });

    // Zoom Buttons
    $('#btn-zoom-in')?.addEventListener('click', () => {
      zoomScale = Math.min(4.0, zoomScale * 1.25);
      updateCanvasTransform();
    });

    $('#btn-zoom-out')?.addEventListener('click', () => {
      zoomScale = Math.max(0.4, zoomScale / 1.25);
      updateCanvasTransform();
    });

    $('#btn-zoom-reset')?.addEventListener('click', () => {
      zoomScale = 1.0;
      panX = 0;
      panY = 0;
      updateCanvasTransform();
    });

    $('#btn-zoom-fit')?.addEventListener('click', () => {
      zoomScale = 1.0;
      panX = 0;
      panY = 0;
      updateCanvasTransform();
    });
  }

  // 11. Diff View Rendering
  function renderDiffView(rawText, cleanText) {
    const diffView = $('#diff-view');
    if (!diffView) return;
    diffView.innerHTML = '';

    const rawLines = rawText.split('\n').map(s => s.trim()).filter(Boolean);
    const cleanLines = cleanText.split('\n').map(s => s.trim()).filter(Boolean);

    const maxLen = Math.max(rawLines.length, cleanLines.length);
    for (let i = 0; i < maxLen; i++) {
      const r = rawLines[i] || '';
      const c = cleanLines[i] || '';

      if (r === c) {
        const d = document.createElement('div');
        d.className = 'diff-line diff-same';
        d.textContent = `  ${c}`;
        diffView.appendChild(d);
      } else {
        if (r) {
          const dDel = document.createElement('div');
          dDel.className = 'diff-line diff-del';
          dDel.textContent = `- [Raw]   ${r}`;
          diffView.appendChild(dDel);
        }
        if (c) {
          const dAdd = document.createElement('div');
          dAdd.className = 'diff-line diff-add';
          dAdd.textContent = `+ [Clean] ${c}`;
          diffView.appendChild(dAdd);
        }
      }
    }
  }

  // 12. Tabs Switching (Clean / Raw / Diff)
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.tab;

      $$('.tab-pane').forEach(p => p.hidden = true);
      const pane = $(`#pane-${target}`);
      if (pane) pane.hidden = false;
    });
  });

  // 13. Clean Text In-Place Editing
  const btnEdit = $('#btn-edit-clean');
  const btnSave = $('#btn-save-clean');
  const cleanView = $('#clean-view');
  const cleanEditor = $('#clean-editor');
  const labelEdit = $('#label-edit-clean');

  if (btnEdit && btnSave && cleanView && cleanEditor) {
    btnEdit.addEventListener('click', () => {
      isEditing = !isEditing;
      if (isEditing) {
        cleanView.hidden = true;
        cleanEditor.hidden = false;
        if (labelEdit) labelEdit.textContent = '取消編輯';
        btnSave.hidden = false;
        cleanEditor.focus();
      } else {
        cleanView.hidden = false;
        cleanEditor.hidden = true;
        if (labelEdit) labelEdit.textContent = '編輯 (E)';
        btnSave.hidden = true;
      }
    });

    btnSave.addEventListener('click', async () => {
      if (!currentJobId || !currentPage) return;
      btnSave.disabled = true;
      try {
        const res = await fetch(`/api/jobs/${currentJobId}/pages/${currentPage}/clean`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ clean: cleanEditor.value })
        });
        if (!res.ok) throw new Error('儲存失敗');
        const updated = await res.json();
        currentPageData = updated;
        renderPageDetails(updated);
        renderMinimap();

        isEditing = false;
        cleanView.hidden = false;
        cleanEditor.hidden = true;
        if (labelEdit) labelEdit.textContent = '編輯 (E)';
        btnSave.hidden = true;
        showToast('修改已儲存並自動標記為已確認！', 'success');
      } catch (err) {
        showToast(`儲存修改失敗：${err.message}`, 'danger');
      } finally {
        btnSave.disabled = false;
      }
    });
  }

  // 14. Toggle Page Review Status
  async function toggleCurrentPageReview(autoJump = false) {
    if (!currentJobId || !currentPage || !currentPageData) return;
    const nextState = !currentPageData.reviewed;
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/pages/${currentPage}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviewed: nextState })
      });
      if (!res.ok) throw new Error('狀態更新失敗');
      const updated = await res.json();
      currentPageData.reviewed = updated.reviewed;
      const targetInList = jobPages.find(p => p.page === currentPage);
      if (targetInList) targetInList.reviewed = updated.reviewed;

      renderPageDetails(currentPageData);
      renderMinimap();

      if (nextState) {
        showToast(`第 ${currentPage} 頁已標記為確認！`, 'success');
        if (autoJump) {
          const nextReview = jobPages.find(p => p.needs_review && !p.reviewed && p.page > currentPage) ||
                             jobPages.find(p => p.needs_review && !p.reviewed);
          if (nextReview) {
            currentPage = nextReview.page;
            loadCurrentPage();
          } else {
            $('#next-page')?.click();
          }
        }
      } else {
        showToast(`第 ${currentPage} 頁已取消確認標記。`, 'info');
      }
    } catch (err) {
      showToast(`標記失敗：${err.message}`, 'danger');
    }
  }

  $('#btn-toggle-reviewed')?.addEventListener('click', () => toggleCurrentPageReview(false));

  // 15. Single Page Rerun
  const btnRerun = $('#btn-rerun-page');
  if (btnRerun) {
    btnRerun.addEventListener('click', async () => {
      if (!currentJobId || !currentPage) return;
      if (!confirm(`確定要重新執行第 ${currentPage} 頁的辨識與排版嗎？`)) return;
      btnRerun.disabled = true;
      btnRerun.textContent = '處理中…';
      try {
        const res = await fetch(`/api/jobs/${currentJobId}/pages/${currentPage}/rerun`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        if (!res.ok) throw new Error('單頁重新執行失敗');
        const updated = await res.json();
        currentPageData = updated;
        renderPageDetails(updated);
        renderMinimap();
        showToast(`第 ${currentPage} 頁已重新辨識完成！`, 'success');
      } catch (err) {
        showToast(`重跑失敗：${err.message}`, 'danger');
      } finally {
        btnRerun.disabled = false;
        btnRerun.innerHTML = `<svg class="ui-icon" viewBox="0 0 24 24"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg> 重跑本頁`;
      }
    });
  }

  // 16. Page Jump Input
  const jumpInput = $('#page-jump-input');
  if (jumpInput) {
    jumpInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        const val = parseInt(jumpInput.value, 10);
        if (val >= 1 && currentJob?.pages && val >= currentJob.pages[0] && val <= currentJob.pages[1]) {
          currentPage = val;
          loadCurrentPage();
        } else {
          showToast(`頁碼必須介於 ${currentJob?.pages?.[0] || 1} 至 ${currentJob?.pages?.[1] || '?'} 之間`, 'warning');
        }
      }
    });
  }

  // 17. Navigation & Filtering
  function getFilteredPages() {
    const filter = $('#page-filter')?.value || 'all';
    if (!jobPages.length && currentJob?.pages) {
      const arr = [];
      for (let i = currentJob.pages[0]; i <= currentJob.pages[1]; i++) arr.push({ page: i, needs_review: false });
      return arr;
    }
    if (filter === 'all') return jobPages;
    if (filter === 'review') return jobPages.filter(p => p.needs_review && !p.reviewed);
    return jobPages.filter(p => p.page_type === filter);
  }

  function updateNavButtons() {
    const list = getFilteredPages();
    const curIdx = list.findIndex(p => p.page === currentPage);
    const prevBtn = $('#prev-page');
    const nextBtn = $('#next-page');

    if (prevBtn) prevBtn.disabled = curIdx <= 0;
    if (nextBtn) nextBtn.disabled = curIdx < 0 || curIdx >= list.length - 1;
  }

  $('#prev-page')?.addEventListener('click', () => {
    const list = getFilteredPages();
    const curIdx = list.findIndex(p => p.page === currentPage);
    if (curIdx > 0) {
      currentPage = list[curIdx - 1].page;
      loadCurrentPage();
    }
  });

  $('#next-page')?.addEventListener('click', () => {
    const list = getFilteredPages();
    const curIdx = list.findIndex(p => p.page === currentPage);
    if (curIdx >= 0 && curIdx < list.length - 1) {
      currentPage = list[curIdx + 1].page;
      loadCurrentPage();
    }
  });

  // Jump to Next Review Page
  $('#jump-next-review')?.addEventListener('click', () => {
    const pendingReviewPages = jobPages.filter(p => p.needs_review && !p.reviewed);
    if (!pendingReviewPages.length) {
      showToast('目前沒有待確認的頁面！', 'success');
      return;
    }
    const next = pendingReviewPages.find(p => p.page > currentPage) || pendingReviewPages[0];
    currentPage = next.page;
    loadCurrentPage();
    showToast(`已跳轉至待確認第 ${currentPage} 頁`, 'info');
  });

  $('#page-filter')?.addEventListener('change', () => {
    const list = getFilteredPages();
    if (list.length > 0 && !list.some(p => p.page === currentPage)) {
      currentPage = list[0].page;
    }
    loadCurrentPage();
  });

  // 18. Shortcuts Modal Controls
  const shortcutsModal = $('#shortcuts-modal');
  const btnHelp = $('#btn-shortcuts-help');
  const btnCloseModal = $('#modal-close');

  function openShortcutsModal() {
    if (shortcutsModal) shortcutsModal.hidden = false;
  }
  function closeShortcutsModal() {
    if (shortcutsModal) shortcutsModal.hidden = true;
  }

  btnHelp?.addEventListener('click', openShortcutsModal);
  btnCloseModal?.addEventListener('click', closeShortcutsModal);
  shortcutsModal?.addEventListener('click', e => {
    if (e.target === shortcutsModal) closeShortcutsModal();
  });

  // 19. Keyboard Shortcuts
  window.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
      if (e.key === 'Escape') {
        e.target.blur();
      }
      return;
    }

    if (e.key === '?' || (e.shiftKey && e.key === '/')) {
      e.preventDefault();
      if (shortcutsModal) {
        shortcutsModal.hidden ? openShortcutsModal() : closeShortcutsModal();
      }
      return;
    }

    if (e.key === 'Escape') {
      if (shortcutsModal && !shortcutsModal.hidden) {
        closeShortcutsModal();
        return;
      }
    }

    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      $('#prev-page')?.click();
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      $('#next-page')?.click();
    } else if (e.key === 'j' || e.key === 'J') {
      e.preventDefault();
      $('#jump-next-review')?.click();
    } else if (e.key === 'e' || e.key === 'E') {
      e.preventDefault();
      $('#btn-edit-clean')?.click();
    } else if (e.code === 'Space') {
      e.preventDefault();
      toggleCurrentPageReview(true);
    }
  });

  // 20. Polling
  async function pollJob() {
    if (pollTimer) clearTimeout(pollTimer);
    if (!currentJobId) return;

    try {
      const res = await fetch(`/api/jobs/${currentJobId}`);
      if (!res.ok) throw new Error('讀取狀態失敗');
      const job = await res.json();
      showJob(job);
      if (['queued', 'running'].includes(job.status)) {
        pollTimer = setTimeout(pollJob, 1000);
      } else {
        loadHistory();
      }
    } catch {
      pollTimer = setTimeout(pollJob, 2000);
    }
  }

  // 21. Recent Jobs List
  async function loadHistory() {
    try {
      const res = await fetch('/api/jobs?limit=50');
      if (!res.ok) return;
      allHistory = await res.json();
      renderHistoryList(allHistory);
    } catch (err) {
      console.warn('Load history error:', err);
    }
  }

  function renderHistoryList(jobs) {
    const container = $('#history-list');
    if (!container) return;
    container.innerHTML = '';

    const searchTerm = ($('#history-search')?.value || '').toLowerCase().trim();
    const filtered = jobs.filter(j => (j.filename || '').toLowerCase().includes(searchTerm));

    if (!filtered.length) {
      container.innerHTML = '<p class="muted">無相符的工作紀錄。</p>';
      return;
    }

    filtered.forEach(j => {
      const item = document.createElement('div');
      item.className = 'history-item';

      const left = document.createElement('div');
      left.className = 'history-left';

      const title = document.createElement('h4');
      title.textContent = j.filename;
      left.appendChild(title);

      const meta = document.createElement('div');
      meta.className = 'history-meta';

      const statusPill = document.createElement('span');
      statusPill.className = `status-pill status-${j.status}`;
      statusPill.textContent = STATUS_LABELS[j.status] || j.status;
      meta.appendChild(statusPill);

      const pagesSpan = document.createElement('span');
      pagesSpan.textContent = `頁數: ${j.pages ? j.pages[0] + '–' + j.pages[1] : (j.total_pages || '?')} 頁`;
      meta.appendChild(pagesSpan);

      const backendSpan = document.createElement('span');
      backendSpan.textContent = j.backend_actual ? (j.backend_actual.includes('DirectML') ? 'DirectML GPU' : (j.backend_actual.includes('CUDA') ? 'CUDA GPU' : 'CPU')) : (j.backend_requested || 'Auto');
      meta.appendChild(backendSpan);

      if (typeof j.needs_review_count === 'number' && j.needs_review_count > 0) {
        const revSpan = document.createElement('span');
        revSpan.className = 'badge-review';
        revSpan.textContent = `待確認 ${j.needs_review_count} 頁`;
        meta.appendChild(revSpan);
      }

      const timeSpan = document.createElement('span');
      timeSpan.textContent = new Date((j.updated_at || j.created_at) * 1000).toLocaleString('zh-TW', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      meta.appendChild(timeSpan);

      left.appendChild(meta);
      item.appendChild(left);

      // Actions
      const actions = document.createElement('div');
      actions.className = 'history-actions';

      const openBtn = document.createElement('button');
      openBtn.className = 'secondary-btn';
      openBtn.textContent = '開啟工作台';
      openBtn.onclick = () => {
        currentJobId = j.id;
        currentPage = j.pages ? j.pages[0] : 1;
        history.replaceState(null, '', `#${j.id}`);
        pollJob();
        $('#workbench')?.scrollIntoView({ behavior: 'smooth' });
      };
      actions.appendChild(openBtn);

      const rerunBtn = document.createElement('button');
      rerunBtn.className = 'text-btn';
      rerunBtn.textContent = '沿用參數重跑';
      rerunBtn.onclick = () => {
        if (j.backend_requested) {
          const radio = $(`input[name="backend"][value="${j.backend_requested}"]`);
          if (radio) radio.checked = true;
        }
        if (j.options) {
          Object.keys(j.options).forEach(k => {
            if (form.elements[k]) form.elements[k].checked = Boolean(j.options[k]);
          });
        }
        form?.scrollIntoView({ behavior: 'smooth' });
      };
      actions.appendChild(rerunBtn);

      item.appendChild(actions);
      container.appendChild(item);
    });
  }

  $('#history-search')?.addEventListener('input', () => {
    renderHistoryList(allHistory);
  });

  // Initialization
  loadHardware();
  loadHistory();
  if (location.hash) {
    currentJobId = location.hash.slice(1);
    pollJob();
  }
})();
