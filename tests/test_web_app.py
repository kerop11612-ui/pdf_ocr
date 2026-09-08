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
        yield 1, '完成', '## Page 1\n\n乾淨文字\n', '=== [Page 1] ===\n原始文字\n', False, None
        md, raw, metrics = output/'native.md', output/'native.txt', output/'native.json'
        md.write_text('完整 clean', encoding='utf-8'); raw.write_text('完整 raw', encoding='utf-8'); metrics.write_text('{}', encoding='utf-8')
        yield 1, '完成', '## Page 1\n\n乾淨文字\n', '=== [Page 1] ===\n原始文字\n', True, {'md_path':md,'raw_path':raw,'metrics_path':metrics,'backend_actual':'CPU'}
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


def test_history_defaults_to_five_newest_records(tmp_path):
    from web_app import create_app
    client=TestClient(create_app(tmp_path,start_worker=False))
    for n in range(6): client.post('/api/jobs',files={'file':(f'{n}.pdf',make_pdf(),'application/pdf')})
    assert len(client.get('/api/jobs').json())==5



def test_provider_name_summary_deduplicates_session_providers():
    from directml_pipeline import _unique_provider_names
    assert _unique_provider_names([['CPUExecutionProvider'], ['CPUExecutionProvider']]) == ['CPUExecutionProvider']
    assert _unique_provider_names([['DmlExecutionProvider','CPUExecutionProvider'], ['DmlExecutionProvider','CPUExecutionProvider']]) == ['DmlExecutionProvider','CPUExecutionProvider']
