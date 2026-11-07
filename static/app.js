// AuraDoc 校對工作台
(function () {
  'use strict';

  const $ = s => document.querySelector(s);
  const $$ = s => document.querySelectorAll(s);

  // ---------- State ----------
  let currentJobId = null;
  let currentJob = null;
  let currentPage = null;
  let loadedPage = null;          // page number whose data is rendered
  let jobPages = [];
  let currentPageData = null;
  let pollTimer = null;
  let allHistory = [];
  let isEditing = false;
  let rerunArmed = null;          // timeout id while the rerun button awaits a second click
  let allDoneAnnounced = false;
  let reviewInFlight = false;     // blocks double Space presses from sending two requests
  let saveInFlight = false;
  let lastScrolledPage = null;    // minimap only auto-scrolls when the current page changes
  let historyTimer = null;

  // Turn a failed response into a readable message (FastAPI 422 details are arrays,
  // proxies may return HTML).
  async function errorMessage(res, fallback) {
    try {
      const body = await res.json();
      const d = body?.detail;
      if (typeof d === 'string') return d;
      if (Array.isArray(d)) return d.map(x => x?.msg || String(x)).join('；');
    } catch { /* not JSON */ }
    return `${fallback}（HTTP ${res.status}）`;
  }

  let zoomScale = 1, panX = 0, panY = 0;
  let isPanning = false, panStartX = 0, panStartY = 0;

  const PAGE_TYPE_LABELS = {
    prose: '文章', questions: '題目', vocabulary_cards: '單字卡',
    toc: '目錄', decorative_or_blank: '空白／裝飾'
  };
  const REVIEW_REASON_LABELS = {
    low_confidence: '信心度低', unknown_character: '有缺字', repeated_character: '重複字元',
    mixed_symbol: '符號混雜', choice_number_mismatch: '選項編號可能誤讀',
    complex_layout: '版面複雜', missing_geometry: '缺少位置資訊'
  };
  const STATUS_LABELS = {
    queued: '排隊中', running: '辨識中', completed: '已完成', cancelled: '已停止', failed: '失敗'
  };

  // ---------- Toast ----------
  function showToast(message, type = 'info', action = null, duration = 3200) {
    const container = $('#toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', type === 'danger' ? 'alert' : 'status');
    const text = document.createElement('span');
    text.textContent = message;
    toast.appendChild(text);
    let timer;
    const dismiss = () => {
      clearTimeout(timer);
      toast.classList.add('leaving');
      setTimeout(() => toast.remove(), 180);
    };
    if (action) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'toast-action';
      btn.textContent = action.label;
      btn.addEventListener('click', () => { dismiss(); action.onClick(); });
      toast.appendChild(btn);
      duration = Math.max(duration, 5000);
    }
    container.appendChild(toast);
    timer = setTimeout(dismiss, duration);
  }

  const backendLabel = j => j.backend_actual
    ? (j.backend_actual.includes('DirectML') ? 'DirectML' : (j.backend_actual.includes('CUDA') ? 'CUDA' : 'CPU'))
    : ({ auto: '自動', gpu: 'GPU', cuda: 'CUDA', dml: 'DirectML', cpu: 'CPU' }[j.backend_requested] || '自動');

  // ---------- Hardware ----------
  async function loadHardware() {
    try {
      const res = await fetch('/api/hardware');
      if (!res.ok) return;
      const data = await res.json();
      const providers = data.onnxruntime_providers || [];
      const gpu = [...new Set(data.gpu || [])];
      const accel = providers.includes('CUDAExecutionProvider') ? 'CUDA'
        : providers.includes('DmlExecutionProvider') ? 'DirectML' : 'CPU';
      const gpuName = (gpu[0] || '').replace(/^NVIDIA\s+/, '').replace(/^GeForce\s+/, '');
      $('#hardware-badge').textContent = gpuName ? `${gpuName} · ${accel}` : accel;
      $('#hardware-desc').textContent =
        `CPU：${[...new Set(data.cpu || [])].join('、') || '—'}　GPU：${gpu.join('、') || '—'}　Providers：${providers.join('、') || '—'}`;
    } catch { /* hardware info is optional */ }
  }

  // ---------- Routing ----------
  function route() {
    const id = location.hash.slice(1);
    const inWorkbench = Boolean(id);
    $('#view-home').hidden = inWorkbench;
    $('#view-workbench').hidden = !inWorkbench;
    $('#btn-home').hidden = !inWorkbench;
    if (inWorkbench) {
      if (id !== currentJobId) {
        currentJobId = id;
        currentJob = null;
        currentPage = null;
        loadedPage = null;
        currentPageData = null;
        jobPages = [];
        allDoneAnnounced = false;
        lastScrolledPage = null;
        $('#page-filter').value = 'all';
        resetZoom();
      }
      clearTimeout(historyTimer);
      pollJob();
    } else {
      clearTimeout(pollTimer);
      currentJobId = null;
      currentJob = null;
      document.title = 'AuraDoc 校對工作台';
      loadHistory();
    }
  }

  function goHome() {
    if (!confirmDiscardEdits()) return;
    exitEditMode();
    location.hash = '';
  }

  window.addEventListener('hashchange', () => {
    if (isDirty() && !confirm('有尚未儲存的修改，確定要離開嗎？')) {
      history.replaceState(null, '', `#${currentJobId}`);
      return;
    }
    exitEditMode();
    route();
  });
  $('#btn-home').addEventListener('click', goHome);
  $('#brand-link').addEventListener('click', e => { e.preventDefault(); goHome(); });
  window.addEventListener('beforeunload', e => {
    if (isDirty()) { e.preventDefault(); e.returnValue = ''; }
  });

  // ---------- New job form ----------
  const form = $('#job-form');
  const fileInput = $('#file');
  const fileInfo = $('#file-info');
  const dropZone = $('#drop-zone');
  const submitBtn = $('#submit-btn');

  fileInput.addEventListener('change', () => {
    const f = fileInput.files && fileInput.files[0];
    submitBtn.disabled = !f;
    dropZone.classList.toggle('has-file', Boolean(f));
    fileInfo.textContent = f ? `${f.name}（${(f.size / 1048576).toFixed(1)} MB）` : '選擇或拖入 PDF';
  });

  let dragDepth = 0;
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  dropZone.addEventListener('dragenter', e => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault(); dragDepth += 1; dropZone.classList.add('is-dragging');
  });
  dropZone.addEventListener('dragover', e => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault(); e.dataTransfer.dropEffect = 'copy';
  });
  dropZone.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dropZone.classList.remove('is-dragging');
  });
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dragDepth = 0; dropZone.classList.remove('is-dragging');
    const files = [...(e.dataTransfer?.files || [])];
    if (files.length !== 1 || !files[0].name.toLowerCase().endsWith('.pdf')) {
      showToast('請一次拖入一個 PDF 檔案', 'warning');
      return;
    }
    const t = new DataTransfer();
    t.items.add(files[0]);
    fileInput.files = t.files;
    fileInput.dispatchEvent(new Event('change'));
  });
  ['dragover', 'drop'].forEach(type => window.addEventListener(type, e => {
    if (e.dataTransfer?.types.includes('Files')) e.preventDefault();
  }));

  $$('input[name="range_mode"]').forEach(r => r.addEventListener('change', () => {
    $('#custom-range-box').hidden = form.elements.range_mode.value !== 'custom';
  }));

  form.addEventListener('submit', async e => {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = '上傳中…';
    const data = new FormData(form);
    ['adaptive_retry', 'clahe', 'reflow', 'two_column', 'furigana', 'homoglyphs'].forEach(k => {
      data.set(k, String(form.elements[k].checked));
    });
    try {
      const res = await fetch('/api/jobs', { method: 'POST', body: data });
      if (!res.ok) throw new Error(await errorMessage(res, '建立工作失敗'));
      const body = await res.json();
      form.reset();
      fileInput.dispatchEvent(new Event('change'));
      $('#custom-range-box').hidden = true;
      location.hash = body.id;
    } catch (err) {
      showToast(`無法建立工作：${err.message}`, 'danger');
      submitBtn.disabled = !fileInput.files?.length;
    } finally {
      submitBtn.textContent = '開始辨識';
    }
  });

  // ---------- Job header & polling ----------
  $('#cancel-btn').addEventListener('click', async () => {
    if (!currentJobId) return;
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
      if (!res.ok) throw new Error(await errorMessage(res, '停止要求送出失敗'));
      renderJobHeader(await res.json());
      showToast('已要求停止，已完成的頁面會保留', 'info');
    } catch (err) {
      showToast(err.message || '停止要求送出失敗', 'danger');
    }
  });

  function renderJobHeader(job) {
    currentJob = job;
    const active = ['queued', 'running'].includes(job.status);
    $('#job-title').textContent = job.filename || '未命名文件';
    document.title = `${job.filename || '校對'} · AuraDoc`;
    const pill = $('#job-status-badge');
    pill.textContent = STATUS_LABELS[job.status] || job.status;
    pill.className = `status-pill status-${job.status}`;
    $('#status-msg').textContent = active || job.status === 'failed' ? (job.message || '') : `${backendLabel(job)} · ${job.pages[0]}–${job.pages[1]} 頁`;
    $('#progress-wrap').hidden = !active;
    $('#progress').value = job.progress || 0;
    $('#progress-percent').textContent = `${job.progress || 0}%`;
    $('#cancel-btn').hidden = !active;
    const done = job.status === 'completed' || job.status === 'cancelled';
    $('#download-section').hidden = !done;
    $$('#download-section [data-kind]').forEach(a => {
      a.href = `/api/jobs/${job.id}/download/${a.dataset.kind}`;
    });
    $('#page-total').textContent = `/ ${job.pages[1]}`;
    $('#page-jump-input').min = job.pages[0];
    $('#page-jump-input').max = job.pages[1];
  }

  async function pollJob() {
    clearTimeout(pollTimer);
    const id = currentJobId;
    if (!id) return;
    try {
      const res = await fetch(`/api/jobs/${id}`);
      if (res.status === 404) {
        showToast('找不到這個工作', 'danger');
        location.hash = '';
        return;
      }
      if (!res.ok) throw new Error();
      const job = await res.json();
      if (id !== currentJobId) return;
      renderJobHeader(job);
      await refreshPagesList();
      if (currentPage === null) currentPage = job.pages[0];
      if (loadedPage !== currentPage || !currentPageData) await loadCurrentPage();
      if (['queued', 'running'].includes(job.status)) pollTimer = setTimeout(pollJob, 1200);
    } catch {
      pollTimer = setTimeout(pollJob, 2500);
    }
  }

  async function refreshPagesList() {
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/pages`);
      if (res.ok) jobPages = await res.json();
    } catch { /* keep previous list */ }
    renderMinimap();
    updateNavButtons();
  }

  // ---------- Page loading ----------
  function goToPage(page) {
    if (page === currentPage && loadedPage === page) return;
    if (!confirmDiscardEdits()) return;
    exitEditMode();
    currentPage = page;
    loadCurrentPage();
  }

  async function loadCurrentPage() {
    if (!currentJobId || currentPage === null) return;
    const pageNum = currentPage;
    const jump = $('#page-jump-input');
    // Polling reloads a still-processing page every tick: don't wipe what the user
    // is typing in the jump box, and don't disarm a rerun they're confirming.
    if (loadedPage !== pageNum || document.activeElement !== jump) jump.value = pageNum;
    if (loadedPage !== pageNum) disarmRerun();
    const img = $('#original-img');
    const src = `/api/jobs/${currentJobId}/pages/${pageNum}/original`;
    if (!img.src.endsWith(src)) img.src = src;
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/pages/${pageNum}`);
      if (!res.ok) throw new Error('這一頁還在辨識中…');
      const data = await res.json();
      if (pageNum !== currentPage) return;
      currentPageData = data;
      loadedPage = pageNum;
      renderPageDetails(data);
    } catch (err) {
      if (pageNum !== currentPage) return;
      currentPageData = null;
      loadedPage = null;
      renderEmptyPage(err.message);
    }
    renderMinimap();
    updateNavButtons();
  }

  function renderPageDetails(data) {
    $('#page-type-badge').textContent = PAGE_TYPE_LABELS[data.page_type] || data.page_type || '—';
    updateReviewButton();

    const reasons = $('#review-reasons-container');
    reasons.textContent = '';
    const list = data.review_reasons || [];
    if (!list.length && !data.needs_review) {
      const ok = document.createElement('span');
      ok.className = 'reason ok';
      ok.textContent = '辨識品質良好';
      reasons.appendChild(ok);
    } else {
      list.forEach(r => {
        const tag = document.createElement('span');
        tag.className = 'reason';
        tag.textContent = REVIEW_REASON_LABELS[r] || r;
        reasons.appendChild(tag);
      });
    }

    const cleanView = $('#clean-view');
    cleanView.textContent = '';
    const text = data.clean || '';
    if (!text.trim()) {
      cleanView.innerHTML = '<p class="empty">這一頁沒有可整理的文字。</p>';
    } else {
      text.split('\n').forEach((lineText, idx) => {
        const line = document.createElement('div');
        line.className = 'text-line';
        line.dataset.lineIdx = idx;
        line.textContent = lineText || ' ';
        const sources = data.clean_line_sources?.[idx] || [];
        line.addEventListener('mouseenter', () => highlightBoxes(sources));
        line.addEventListener('mouseleave', clearBoxHighlight);
        cleanView.appendChild(line);
      });
    }
    $('#clean-editor').value = text;
    $('#raw-view').textContent = data.raw || '沒有原始辨識文字。';
    renderDiffView(data.raw || '', text);
    renderSvgOverlay(data);
  }

  function renderEmptyPage(msg) {
    $('#page-type-badge').textContent = '—';
    $('#review-reasons-container').textContent = '';
    updateReviewButton();
    $('#clean-view').innerHTML = '';
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = msg;
    $('#clean-view').appendChild(p);
    $('#raw-view').textContent = msg;
    $('#diff-view').textContent = '';
    $('#original-svg').innerHTML = '';
  }

  function updateReviewButton() {
    const btn = $('#btn-toggle-reviewed');
    const reviewed = Boolean(currentPageData?.reviewed);
    btn.disabled = !currentPageData;
    btn.classList.toggle('is-reviewed', reviewed);
    $('#label-toggle-reviewed').textContent = reviewed ? '取消確認' : '確認，下一頁';
    btn.title = reviewed ? '取消本頁的確認（Space）' : '確認本頁並前往下一頁（Space）';
  }

  // ---------- Minimap ----------
  // Every page of the requested range: processed ones from the server, the rest as
  // "waiting" placeholders (previously unprocessed pages vanished once one finished).
  function pagesInRange() {
    if (!currentJob) return jobPages;
    const [a, b] = currentJob.pages;
    const done = new Map(jobPages.map(p => [p.page, p]));
    return Array.from({ length: b - a + 1 }, (_, i) => done.get(a + i) || { page: a + i, pending: true });
  }

  const isPending = p => p.needs_review && !p.reviewed && !p.user_edited;

  function renderMinimap() {
    const container = $('#page-minimap');
    const pages = pagesInRange();
    const frag = document.createDocumentFragment();
    let reviewed = 0;
    let pending = 0;
    pages.forEach(p => {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = 'cell';
      cell.textContent = p.page;
      let state = '一般';
      if (p.pending) { cell.classList.add('cell-waiting'); state = '尚未辨識'; }
      else if (p.reviewed) { cell.classList.add('cell-reviewed'); state = '已確認'; reviewed++; }
      else if (p.needs_review) { cell.classList.add('cell-needs-review'); state = '待確認'; pending++; }
      if (p.page === currentPage) { cell.classList.add('cell-current'); cell.setAttribute('aria-current', 'page'); }
      const type = PAGE_TYPE_LABELS[p.page_type] || '';
      cell.setAttribute('aria-label', `第 ${p.page} 頁，${state}${type ? '，' + type : ''}`);
      cell.title = `第 ${p.page} 頁 · ${state}${type ? ' · ' + type : ''}`;
      cell.addEventListener('click', () => goToPage(p.page));
      frag.appendChild(cell);
    });
    container.replaceChildren(frag);
    $('#minimap-stats').textContent = jobPages.length
      ? (pending ? `待確認 ${pending} 頁 · 已確認 ${reviewed} 頁` : `沒有待確認頁 · 已確認 ${reviewed} 頁`)
      : '';
    // Scroll only the strip, and only when the page changed: scrollIntoView on every
    // 1.2 s poll yanked the minimap (and sometimes the window) back while browsing.
    const cur = container.querySelector('.cell-current');
    if (cur && lastScrolledPage !== currentPage) {
      lastScrolledPage = currentPage;
      const c = container.getBoundingClientRect();
      const r = cur.getBoundingClientRect();
      if (r.left < c.left || r.right > c.right) container.scrollLeft += r.left - c.left - (c.width - r.width) / 2;
      if (r.top < c.top || r.bottom > c.bottom) container.scrollTop += r.top - c.top - (c.height - r.height) / 2;
    }
  }

  // ---------- Navigation ----------
  function filteredPages() {
    const filter = $('#page-filter').value;
    const pages = pagesInRange();
    if (filter === 'all') return pages;
    if (filter === 'review') return pages.filter(isPending);
    return pages.filter(p => p.page_type === filter);
  }

  function neighbour(step) {
    const list = filteredPages();
    if (!list.length) return null;
    const idx = list.findIndex(p => p.page === currentPage);
    if (idx === -1) {
      // current page is outside the filter: pick the nearest one in the requested direction
      const cand = step > 0 ? list.find(p => p.page > currentPage) : [...list].reverse().find(p => p.page < currentPage);
      return cand ? cand.page : null;
    }
    const next = list[idx + step];
    return next ? next.page : null;
  }

  function updateNavButtons() {
    $('#prev-page').disabled = neighbour(-1) === null;
    $('#next-page').disabled = neighbour(1) === null;
  }

  function nextPendingPage(after) {
    const pending = jobPages.filter(isPending);
    return (pending.find(p => p.page > after) || pending[0])?.page ?? null;
  }

  $('#prev-page').addEventListener('click', () => { const p = neighbour(-1); if (p !== null) goToPage(p); });
  $('#next-page').addEventListener('click', () => { const p = neighbour(1); if (p !== null) goToPage(p); });

  $('#jump-next-review').addEventListener('click', () => {
    const p = nextPendingPage(currentPage);
    if (p === null) { showToast('沒有待確認的頁面了', 'success'); return; }
    goToPage(p);
  });

  $('#page-jump-input').addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    const val = parseInt(e.target.value, 10);
    const [a, b] = currentJob?.pages || [1, 1];
    if (val >= a && val <= b) { goToPage(val); e.target.blur(); }
    else showToast(`請輸入 ${a} 到 ${b} 之間的頁碼`, 'warning');
  });

  $('#page-filter').addEventListener('change', () => {
    const list = filteredPages();
    updateNavButtons();
    if (list.length && !list.some(p => p.page === currentPage)) goToPage(list[0].page);
    else if (!list.length) showToast('沒有符合條件的頁面', 'info');
  });

  // ---------- Review ----------
  async function setReviewed(page, reviewed) {
    const res = await fetch(`/api/jobs/${currentJobId}/pages/${page}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewed })
    });
    if (!res.ok) throw new Error(await errorMessage(res, '狀態更新失敗'));
    const updated = await res.json();
    const entry = jobPages.find(p => p.page === page);
    if (entry) { entry.reviewed = updated.reviewed; entry.user_edited = Boolean(updated.user_edited); }
    if (currentPageData && page === currentPage) currentPageData.reviewed = updated.reviewed;
    return updated;
  }

  async function toggleReview() {
    if (!currentPageData || isEditing || reviewInFlight) return;
    const page = currentPage;
    reviewInFlight = true;
    try {
      if (currentPageData.reviewed) {
        await setReviewed(page, false);
        updateReviewButton();
        renderMinimap();
        showToast(`第 ${page} 頁已取消確認`, 'info');
        return;
      }
      await setReviewed(page, true);
      updateReviewButton();
      renderMinimap();
      const undo = { label: '復原', onClick: async () => {
        try {
          await setReviewed(page, false);
          goToPage(page);
          updateReviewButton();
          renderMinimap();
        } catch (err) { showToast(`復原失敗：${err.message}`, 'danger'); }
      } };
      const target = nextPendingPage(page);
      if (target !== null) {
        showToast(`第 ${page} 頁已確認`, 'success', undo);
        goToPage(target);
      } else {
        if (!allDoneAnnounced) {
          allDoneAnnounced = true;
          showToast('全部頁面已確認', 'success', undo);
        } else {
          showToast(`第 ${page} 頁已確認`, 'success', undo);
        }
        const next = neighbour(1);
        if (next !== null) goToPage(next);
      }
    } catch (err) {
      showToast(`無法更新：${err.message}`, 'danger');
    } finally {
      reviewInFlight = false;
    }
  }

  $('#btn-toggle-reviewed').addEventListener('click', toggleReview);

  // ---------- Editing ----------
  const cleanView = $('#clean-view');
  const editor = $('#clean-editor');
  const btnEdit = $('#btn-edit-clean');
  const btnSave = $('#btn-save-clean');
  const btnCancelEdit = $('#btn-cancel-edit');

  const isDirty = () => isEditing && editor.value !== (currentPageData?.clean || '');
  function confirmDiscardEdits() {
    return !isDirty() || confirm('有尚未儲存的修改，確定要放棄嗎？');
  }

  function showTab(name) {
    $$('.tab-btn').forEach(b => {
      const on = b.dataset.tab === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', String(on));
    });
    $$('.tab-pane').forEach(p => { p.hidden = p.id !== `pane-${name}`; });
  }
  $$('.tab-btn').forEach(b => b.addEventListener('click', () => {
    if (isEditing && b.dataset.tab !== 'clean') return;
    showTab(b.dataset.tab);
  }));

  function enterEditMode() {
    if (!currentPageData || isEditing) return;
    showTab('clean');
    isEditing = true;
    editor.value = currentPageData.clean || '';
    cleanView.hidden = true;
    editor.hidden = false;
    btnEdit.hidden = true;
    btnSave.hidden = false;
    btnCancelEdit.hidden = false;
    $('#view-workbench').classList.add('is-editing');
    editor.focus();
  }

  function exitEditMode() {
    if (!isEditing) return;
    isEditing = false;
    cleanView.hidden = false;
    editor.hidden = true;
    btnEdit.hidden = false;
    btnSave.hidden = true;
    btnCancelEdit.hidden = true;
    $('#view-workbench').classList.remove('is-editing');
  }

  async function saveEdit() {
    if (!isEditing || saveInFlight) return;   // Ctrl+S bypasses the disabled button
    saveInFlight = true;
    btnSave.disabled = true;
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/pages/${currentPage}/clean`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clean: editor.value })
      });
      if (!res.ok) throw new Error(await errorMessage(res, '儲存失敗'));
      currentPageData = await res.json();
      exitEditMode();
      renderPageDetails(currentPageData);
      await refreshPagesList();
      showToast('已儲存，本頁同時標記為已確認', 'success');
    } catch (err) {
      showToast(`儲存失敗：${err.message}`, 'danger');
    } finally {
      saveInFlight = false;
      btnSave.disabled = false;
    }
  }

  btnEdit.addEventListener('click', enterEditMode);
  btnSave.addEventListener('click', saveEdit);
  btnCancelEdit.addEventListener('click', () => { if (confirmDiscardEdits()) exitEditMode(); });

  // ---------- Rerun (two-step) ----------
  const btnRerun = $('#btn-rerun-page');
  function disarmRerun() {
    clearTimeout(rerunArmed);
    rerunArmed = null;
    btnRerun.classList.remove('armed');
    $('#label-rerun').textContent = '重新辨識';
  }
  btnRerun.addEventListener('click', async () => {
    if (!currentJobId || currentPage === null) return;
    if (!rerunArmed) {
      btnRerun.classList.add('armed');
      $('#label-rerun').textContent = currentPageData?.user_edited ? '會覆蓋你的修改，再按一次' : '再按一次確認';
      rerunArmed = setTimeout(disarmRerun, 4000);
      return;
    }
    disarmRerun();
    if (!confirmDiscardEdits()) return;
    exitEditMode();
    const page = currentPage;
    btnRerun.disabled = true;
    $('#label-rerun').textContent = '辨識中…';
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/pages/${page}/rerun`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
      });
      if (!res.ok) throw new Error(await errorMessage(res, '重新辨識失敗'));
      const updated = await res.json();
      if (page === currentPage) { currentPageData = updated; renderPageDetails(updated); }
      await refreshPagesList();
      showToast(`第 ${page} 頁已重新辨識`, 'success');
    } catch (err) {
      showToast(err.message, 'danger');
    } finally {
      btnRerun.disabled = false;
      $('#label-rerun').textContent = '重新辨識';
    }
  });

  // ---------- Overlay & sync highlight ----------
  $('#toggle-overlay').addEventListener('change', e => {
    $('#original-svg').style.display = e.target.checked ? '' : 'none';
  });

  function renderSvgOverlay(data) {
    const svg = $('#original-svg');
    svg.innerHTML = '';
    const lines = data.source_lines || [];
    if (!lines.length) return;
    let pw = data.page_width, ph = data.page_height;
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
      rect.classList.add(line.included_in_clean === false ? 'bbox-filtered' : (line.needs_review ? 'bbox-review' : 'bbox-clean'));
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      const reasons = (line.review_reasons || []).map(r => REVIEW_REASON_LABELS[r] || r).join('、');
      title.textContent = `${line.text}\n${line.filter_reason ? '已略過：' + line.filter_reason
        : reasons ? '需留意：' + reasons : '信心度 ' + Math.round((line.confidence || 0) * 100) + '%'}`;
      rect.appendChild(title);
      rect.addEventListener('mouseenter', () => { rect.classList.add('active'); highlightLines(idx, data.clean_line_sources); });
      rect.addEventListener('mouseleave', () => { rect.classList.remove('active'); clearLineHighlight(); });
      rect.addEventListener('click', () => scrollLineIntoView(idx, data.clean_line_sources));
      svg.appendChild(rect);
    });
  }

  function highlightBoxes(indices) {
    indices.forEach(i => $(`#original-svg rect[data-line-idx="${i}"]`)?.classList.add('active'));
  }
  function clearBoxHighlight() {
    $$('#original-svg rect.active').forEach(r => r.classList.remove('active'));
  }
  function linesForSource(sourceIdx, mapping) {
    if (!Array.isArray(mapping)) return [];
    return mapping.flatMap((sources, i) => (sources?.includes(sourceIdx) ? [i] : []));
  }
  function highlightLines(sourceIdx, mapping) {
    linesForSource(sourceIdx, mapping).forEach(i => $(`#clean-view .text-line[data-line-idx="${i}"]`)?.classList.add('active'));
  }
  function clearLineHighlight() {
    $$('#clean-view .text-line.active').forEach(l => l.classList.remove('active'));
  }
  function scrollLineIntoView(sourceIdx, mapping) {
    const i = linesForSource(sourceIdx, mapping)[0];
    if (i === undefined) return;
    showTab('clean');
    const line = $(`#clean-view .text-line[data-line-idx="${i}"]`);
    if (!line) return;
    line.scrollIntoView({ behavior: 'smooth', block: 'center' });
    line.classList.add('active');
    setTimeout(() => line.classList.remove('active'), 1500);
  }

  // ---------- Zoom & pan ----------
  const viewport = $('#original-viewport');
  const canvas = $('#original-container');
  function applyTransform() {
    canvas.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomScale})`;
    $('#zoom-level').textContent = `${Math.round(zoomScale * 100)}%`;
  }
  function resetZoom() { zoomScale = 1; panX = 0; panY = 0; applyTransform(); }
  function zoomAt(factor, cx, cy) {
    const next = Math.max(0.5, Math.min(4, zoomScale * factor));
    panX = cx - (cx - panX) * (next / zoomScale);
    panY = cy - (cy - panY) * (next / zoomScale);
    zoomScale = next;
    applyTransform();
  }
  viewport.addEventListener('wheel', e => {
    e.preventDefault();
    const r = viewport.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  viewport.addEventListener('mousedown', e => {
    if (e.button !== 0 || e.target.tagName === 'rect') return;
    isPanning = true;
    panStartX = e.clientX - panX;
    panStartY = e.clientY - panY;
    viewport.classList.add('panning');
  });
  window.addEventListener('mousemove', e => {
    if (!isPanning) return;
    panX = e.clientX - panStartX;
    panY = e.clientY - panStartY;
    applyTransform();
  });
  window.addEventListener('mouseup', () => { isPanning = false; viewport.classList.remove('panning'); });
  viewport.addEventListener('dblclick', resetZoom);
  $('#zoom-level').addEventListener('click', resetZoom);
  $('#btn-zoom-in').addEventListener('click', () => zoomAt(1.25, viewport.clientWidth / 2, viewport.clientHeight / 2));
  $('#btn-zoom-out').addEventListener('click', () => zoomAt(0.8, viewport.clientWidth / 2, viewport.clientHeight / 2));

  // ---------- Diff ----------
  function renderDiffView(rawText, cleanText) {
    const view = $('#diff-view');
    view.textContent = '';
    const raw = rawText.split('\n').map(s => s.trim()).filter(Boolean);
    const clean = cleanText.split('\n').map(s => s.trim()).filter(Boolean);
    const add = (cls, text) => {
      const d = document.createElement('div');
      d.className = `diff-line ${cls}`;
      d.textContent = text;
      view.appendChild(d);
    };
    for (let i = 0; i < Math.max(raw.length, clean.length); i++) {
      const r = raw[i] || '', c = clean[i] || '';
      if (r === c) add('diff-same', c);
      else {
        if (r) add('diff-del', r);
        if (c) add('diff-add', c);
      }
    }
    if (!view.childElementCount) add('diff-same', '沒有差異');
  }

  // ---------- Shortcuts modal ----------
  const modal = $('#shortcuts-modal');
  let modalReturnFocus = null;
  function openModal() { modalReturnFocus = document.activeElement; modal.hidden = false; $('#modal-close').focus(); }
  function closeModal() { modal.hidden = true; modalReturnFocus?.focus?.(); }
  $('#btn-shortcuts-help').addEventListener('click', openModal);
  $('#modal-close').addEventListener('click', closeModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });

  // ---------- Keyboard ----------
  window.addEventListener('keydown', e => {
    if ($('#view-workbench').hidden) return;

    if (isEditing) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); saveEdit(); }
      else if (e.key === 'Escape') { e.preventDefault(); if (confirmDiscardEdits()) exitEditMode(); }
      return;
    }
    if (!modal.hidden) {
      if (e.key === 'Escape' || e.key === '?') { e.preventDefault(); closeModal(); }
      return;
    }
    if (e.ctrlKey || e.altKey || e.metaKey) return;
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
      if (e.key === 'Escape') e.target.blur();
      return;
    }

    switch (e.key) {
      case ' ':
        // Space always means "confirm" here, even when a button has focus.
        e.preventDefault();
        if (!e.repeat) toggleReview();
        break;
      case 'ArrowLeft': e.preventDefault(); $('#prev-page').click(); break;
      case 'ArrowRight': e.preventDefault(); $('#next-page').click(); break;
      case 'j': case 'J': e.preventDefault(); $('#jump-next-review').click(); break;
      case 'e': case 'E': e.preventDefault(); enterEditMode(); break;
      case '?': e.preventDefault(); openModal(); break;
      default: break;
    }
  });
  // Stop Space from also activating a focused button on keyup.
  window.addEventListener('keyup', e => {
    if (e.key === ' ' && !$('#view-workbench').hidden && e.target.tagName === 'BUTTON') e.preventDefault();
  });

  // ---------- History ----------
  async function loadHistory() {
    clearTimeout(historyTimer);
    try {
      const res = await fetch('/api/jobs?limit=50');
      if (!res.ok) return;
      allHistory = await res.json();
      renderHistoryList();
    } catch { /* ignore */ }
    // Keep running jobs' status/progress fresh on the home view.
    if (!currentJobId && allHistory.some(j => ['queued', 'running'].includes(j.status))) {
      historyTimer = setTimeout(loadHistory, 3000);
    }
  }

  function renderHistoryList() {
    const container = $('#history-list');
    const term = $('#history-search').value.toLowerCase().trim();
    const jobs = allHistory.filter(j => (j.filename || '').toLowerCase().includes(term));
    container.textContent = '';
    if (!jobs.length) {
      const p = document.createElement('p');
      p.className = 'empty';
      p.textContent = allHistory.length ? '沒有符合的工作。' : '還沒有工作，從左邊上傳第一個 PDF 吧。';
      container.appendChild(p);
      return;
    }
    jobs.forEach(j => {
      const item = document.createElement('div');
      item.className = 'history-item';

      const open = document.createElement('a');
      open.className = 'history-open';
      open.href = `#${j.id}`;
      const name = document.createElement('strong');
      name.textContent = j.filename;
      const meta = document.createElement('span');
      meta.className = 'history-meta';
      const pill = document.createElement('span');
      pill.className = `status-pill status-${j.status}`;
      pill.textContent = STATUS_LABELS[j.status] || j.status;
      meta.appendChild(pill);
      const parts = [
        j.pages ? `${j.pages[0]}–${j.pages[1]} 頁` : '',
        backendLabel(j),
        new Date((j.updated_at || j.created_at) * 1000).toLocaleString('zh-TW', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })
      ].filter(Boolean);
      const info = document.createElement('span');
      info.textContent = parts.join(' · ');
      meta.appendChild(info);
      if (j.needs_review_count > 0) {
        const rev = document.createElement('span');
        rev.className = 'pending-count';
        rev.textContent = `待確認 ${j.needs_review_count}`;
        meta.appendChild(rev);
      }
      open.append(name, meta);
      item.appendChild(open);

      const reuse = document.createElement('button');
      reuse.type = 'button';
      reuse.className = 'link-btn';
      reuse.textContent = '沿用設定';
      reuse.title = '把這個工作的設定套用到新增辨識表單';
      reuse.addEventListener('click', () => {
        const backend = form.querySelector(`input[name="backend"][value="${j.backend_requested}"]`);
        if (backend) backend.checked = true;
        Object.entries(j.options || {}).forEach(([k, v]) => { if (form.elements[k]) form.elements[k].checked = Boolean(v); });
        form.querySelector('details.advanced').open = true;
        dropZone.focus();
        showToast('已套用設定，選擇 PDF 後即可開始', 'info');
      });
      item.appendChild(reuse);
      container.appendChild(item);
    });
  }
  $('#history-search').addEventListener('input', renderHistoryList);

  // ---------- Init ----------
  loadHardware();
  route();
})();
