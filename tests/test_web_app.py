import io
import json
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient


def make_pdf() -> bytes:
    document = fitz.open()
    for n in range(3):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {n + 1}")
    data = document.tobytes()
    document.close()
    return data


def test_upload_creates_uuid_isolated_queued_job(tmp_path):
    from web_app import create_app
    app = create_app(tmp_path, start_worker=False)
    client = TestClient(app)
    response = client.post('/api/jobs', files={'file': ('book.pdf', make_pdf(), 'application/pdf')}, data={'range_mode': 'first5', 'backend': 'cpu'})
    assert response.status_code == 201
    job = response.json()
    assert job['status'] == 'queued'
    assert job['total_pages'] == 3
    assert job['pages'] == [1, 3]
    assert len(job['id']) == 36
    assert (tmp_path / job['id'] / 'source.pdf').is_file()
    assert client.get(f"/api/jobs/{job['id']}").json()['filename'] == 'book.pdf'


def test_rejects_out_of_range_custom_pages(tmp_path):
    from web_app import create_app
    client = TestClient(create_app(tmp_path, start_worker=False))
    response = client.post('/api/jobs', files={'file': ('book.pdf', make_pdf(), 'application/pdf')}, data={'range_mode': 'custom', 'start_page': '3', 'end_page': '4'})
    assert response.status_code == 422
    assert '頁碼範圍' in response.json()['detail']


def test_cancel_preserves_partial_state_and_download_is_scoped(tmp_path):
    from web_app import create_app
    app = create_app(tmp_path, start_worker=False)
    client = TestClient(app)
    job = client.post('/api/jobs', files={'file': ('book.pdf', make_pdf(), 'application/pdf')}).json()
    response = client.post(f"/api/jobs/{job['id']}/cancel")
    assert response.status_code == 200
    assert response.json()['status'] == 'cancelled'
    assert client.get(f"/api/jobs/{job['id']}/download/../source.pdf").status_code == 404

def test_history_lists_newest_jobs_and_page_result_is_scoped(tmp_path):
    from web_app import create_app
    app = create_app(tmp_path, start_worker=False)
    client = TestClient(app)
    first = client.post('/api/jobs', files={'file': ('one.pdf', make_pdf(), 'application/pdf')}).json()
    second = client.post('/api/jobs', files={'file': ('two.pdf', make_pdf(), 'application/pdf')}).json()
    history = client.get('/api/jobs').json()
    assert [item['id'] for item in history[:2]] == [second['id'], first['id']]
    assert client.get(f"/api/jobs/{first['id']}/pages/1").status_code == 404


def test_cancel_uses_durable_signal_without_losing_status(tmp_path):
    from web_app import create_app
    app = create_app(tmp_path, start_worker=False)
    client = TestClient(app)
    job = client.post('/api/jobs', files={'file': ('book.pdf', make_pdf(), 'application/pdf')}).json()
    cancelled = client.post(f"/api/jobs/{job['id']}/cancel").json()
    assert cancelled['cancel_requested'] is True
    assert app.state.jobs.cancel_events[job['id']].is_set()
    assert client.get(f"/api/jobs/{job['id']}").json()['status'] == 'cancelled'


def test_hardware_endpoint_returns_observed_provider_list(tmp_path):
    from web_app import create_app
    client = TestClient(create_app(tmp_path, start_worker=False))
    data = client.get('/api/hardware').json()
    assert isinstance(data['onnxruntime_providers'], list)

def test_auto_falls_back_and_gpu_errors_when_session_lacks_directml(monkeypatch):
    import directml_pipeline as pipeline
    class FakeGpu:
        backend_actual = 'CPUExecutionProvider'
        provider_sets = [['CPUExecutionProvider']]
    monkeypatch.setattr(pipeline, 'RapidOCRDirectMLBackend', lambda **_: FakeGpu())
    class FakeCpu: backend_actual = 'CPU'
    cpu = FakeCpu()
    monkeypatch.setattr(pipeline, 'RapidOCRCPUBackend', lambda **_: cpu)
    assert pipeline.get_ocr_backend('auto') is cpu
    with pytest.raises(RuntimeError, match='指定 GPU'):
        pipeline.get_ocr_backend('gpu')

def test_running_cancel_keeps_completed_page_preview(tmp_path, monkeypatch):
    import threading, time
    import directml_pipeline
    from web_app import create_app
    yielded = threading.Event()
    def stream(*args, **kwargs):
        kwargs['on_page']({'page': 1, 'clean': '乾淨文字', 'raw': '原始文字'})
        yield .5, '處理中', '## Page 1\n\n乾淨文字\n', '=== [Page 1] ===\n原始文字\n', False, None
        yielded.set()
        while not kwargs['cancel_check'](): time.sleep(.01)
        yield .5, '已取消', '## Page 1\n\n乾淨文字\n', '=== [Page 1] ===\n原始文字\n', True, None
    monkeypatch.setattr(directml_pipeline, 'process_pdf_gpu_adaptive_generator', stream)
    app = create_app(tmp_path, start_worker=True); client = TestClient(app)
    job = client.post('/api/jobs', files={'file': ('book.pdf', make_pdf(), 'application/pdf')}).json()
    assert yielded.wait(2)
    client.post(f"/api/jobs/{job['id']}/cancel")
    for _ in range(30):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state['status'] == 'cancelled': break
        time.sleep(.03)
    assert state['status'] == 'cancelled'
    assert client.get(f"/api/jobs/{job['id']}/pages/1").json()['clean'] == '乾淨文字'

def test_completed_new_job_writes_each_page_json(tmp_path, monkeypatch):
    import threading, time
    import directml_pipeline
    from web_app import create_app
    def stream(source, output, **kwargs):
        output = Path(output)
        kwargs['on_page']({'page': 1, 'clean': '乾淨文字', 'raw': '原始文字'})
        yield 1, '完成', '## Page 1\n\n乾淨文字\n', '=== [Page 1] ===\n原始文字\n', False, None
        md, raw, metrics, structured = output/'native.md', output/'native.txt', output/'native.json', output/'native.jsonl'
        md.write_text('完整 clean', encoding='utf-8'); raw.write_text('完整 raw', encoding='utf-8'); metrics.write_text('{}', encoding='utf-8')
        structured.write_text('{"page":1}\n', encoding='utf-8')
        yield 1, '完成', '## Page 1\n\n乾淨文字\n', '=== [Page 1] ===\n原始文字\n', True, {'md_path':md,'raw_path':raw,'metrics_path':metrics,'structured_path':structured,'backend_actual':'CPU'}
    monkeypatch.setattr(directml_pipeline, 'process_pdf_gpu_adaptive_generator', stream)
    app=create_app(tmp_path,start_worker=True); client=TestClient(app)
    job=client.post('/api/jobs',files={'file':('book.pdf',make_pdf(),'application/pdf')},data={'range_mode':'custom','start_page':'1','end_page':'1'}).json()
    for _ in range(30):
        state=client.get(f"/api/jobs/{job['id']}").json()
        if state['status']=='completed': break
        time.sleep(.03)
    page_file=tmp_path/job['id']/'pages'/'1.json'
    assert page_file.is_file()
    assert json.loads(page_file.read_text(encoding='utf-8'))['clean']=='乾淨文字'
    assert client.get(f"/api/jobs/{job['id']}/download/structured").text.replace('\r\n', '\n') == '{"page":1}\n'


def test_history_defaults_to_five_newest_records(tmp_path):
    from web_app import create_app
    client=TestClient(create_app(tmp_path,start_worker=False))
    for n in range(6): client.post('/api/jobs',files={'file':(f'{n}.pdf',make_pdf(),'application/pdf')})
    assert len(client.get('/api/jobs').json())==5



def test_provider_name_summary_deduplicates_session_providers():
    from directml_pipeline import _unique_provider_names
    assert _unique_provider_names([['CPUExecutionProvider'], ['CPUExecutionProvider']]) == ['CPUExecutionProvider']
    assert _unique_provider_names([['DmlExecutionProvider','CPUExecutionProvider'], ['DmlExecutionProvider','CPUExecutionProvider']]) == ['DmlExecutionProvider','CPUExecutionProvider']


def test_proofreading_pages_and_clean_update(tmp_path):
    from web_app import create_app
    app = create_app(tmp_path, start_worker=False)
    client = TestClient(app)
    job = client.post('/api/jobs', files={'file': ('book.pdf', make_pdf(), 'application/pdf')}).json()
    job_id = job['id']

    # Mock page 1 and page 2 results
    pages_dir = tmp_path / job_id / 'pages'
    pages_dir.mkdir(parents=True, exist_ok=True)
    p1 = {'page': 1, 'clean': '原始第1頁', 'raw': '原始 raw 1', 'page_type': 'prose', 'needs_review': False, 'review_reasons': []}
    p2 = {'page': 2, 'clean': '原始第2頁', 'raw': '原始 raw 2', 'page_type': 'questions', 'needs_review': True, 'review_reasons': ['low_confidence']}
    (pages_dir / '1.json').write_text(json.dumps(p1, ensure_ascii=False), encoding='utf-8')
    (pages_dir / '2.json').write_text(json.dumps(p2, ensure_ascii=False), encoding='utf-8')

    # 1. Test job summary includes review stats
    job_info = client.get(f"/api/jobs/{job_id}").json()
    assert job_info['needs_review_count'] == 1
    assert job_info['page_types_summary']['questions'] == 1
    assert job_info['page_types_summary']['prose'] == 1

    # 2. Test list job pages summary
    pages_list = client.get(f"/api/jobs/{job_id}/pages").json()
    assert len(pages_list) == 2
    assert pages_list[0]['page'] == 1
    assert pages_list[1]['needs_review'] is True
    assert pages_list[1]['page_type'] == 'questions'

    # 3. Test update page clean
    edit_res = client.put(f"/api/jobs/{job_id}/pages/2/clean", json={'clean': '使用者修訂後的第2頁'})
    assert edit_res.status_code == 200
    assert edit_res.json()['clean'] == '使用者修訂後的第2頁'
    assert edit_res.json()['user_edited'] is True

    # 4. Verify clean.md and downloads re-aggregated
    clean_md = client.get(f"/api/jobs/{job_id}/download/clean").text
    assert '使用者修訂後的第2頁' in clean_md
    assert '原始第1頁' in clean_md

    # 5. Verify auxiliary downloads (obsidian, clean_reviewed)
    obsidian_md = client.get(f"/api/jobs/{job_id}/download/obsidian").text
    assert 'tags:' in obsidian_md
    assert 'auradoc' in obsidian_md


def test_single_page_rerun_endpoint(tmp_path, monkeypatch):
    import directml_pipeline
    from web_app import create_app
    def fake_process_single_page(*args, **kwargs):
        return {
            'page': kwargs.get('page_no', 1),
            'clean': '重新執行的單頁 Clean',
            'raw': '重新執行的單頁 Raw',
            'page_type': 'prose',
            'blocks': [],
            'source_lines': [],
            'needs_review': False,
            'review_reasons': [],
            'page_width': 1000,
            'page_height': 1400
        }
    monkeypatch.setattr(directml_pipeline, 'process_single_page', fake_process_single_page)

    app = create_app(tmp_path, start_worker=False)
    client = TestClient(app)
    job = client.post('/api/jobs', files={'file': ('book.pdf', make_pdf(), 'application/pdf')}).json()
    job_id = job['id']

    rerun_res = client.post(f"/api/jobs/{job_id}/pages/1/rerun", json={'clahe': True})
    assert rerun_res.status_code == 200
    data = rerun_res.json()
    assert data['clean'] == '重新執行的單頁 Clean'
    assert data['page'] == 1

    # Check page file persisted
    p1_file = tmp_path / job_id / 'pages' / '1.json'
    assert p1_file.is_file()
    assert json.loads(p1_file.read_text(encoding='utf-8'))['clean'] == '重新執行的單頁 Clean'


def test_toggle_page_review_and_preview_cache(tmp_path):
    from web_app import create_app
    app = create_app(tmp_path, start_worker=False)
    client = TestClient(app)
    job = client.post('/api/jobs', files={'file': ('book.pdf', make_pdf(), 'application/pdf')}).json()
    job_id = job['id']

    # Mock page 1
    pages_dir = tmp_path / job_id / 'pages'
    pages_dir.mkdir(parents=True, exist_ok=True)
    p1 = {'page': 1, 'clean': '文字', 'raw': '文字', 'page_type': 'prose', 'needs_review': True, 'review_reasons': ['low_confidence']}
    (pages_dir / '1.json').write_text(json.dumps(p1, ensure_ascii=False), encoding='utf-8')

    # 1. Toggle review
    res = client.post(f"/api/jobs/{job_id}/pages/1/review", json={'reviewed': True})
    assert res.status_code == 200
    assert res.json()['reviewed'] is True

    # Check pages summary has reviewed: True
    pages_list = client.get(f"/api/jobs/{job_id}/pages").json()
    assert pages_list[0]['reviewed'] is True

    # 2. Test original page preview generation & caching
    prev_res1 = client.get(f"/api/jobs/{job_id}/pages/1/original")
    assert prev_res1.status_code == 200
    assert 'max-age=86400' in prev_res1.headers.get('Cache-Control', '')
    preview_file = tmp_path / job_id / 'preview-1.png'
    assert preview_file.is_file()

    # Second fetch should reuse preview file
    prev_res2 = client.get(f"/api/jobs/{job_id}/pages/1/original")
    assert prev_res2.status_code == 200



