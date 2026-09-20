"""Real pipeline/storage flow with only OCR and rendering replaced by fixtures."""
import io
import json

import fitz
import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

import directml_pipeline as pipeline


def make_source(path, pages):
    with fitz.open() as doc:
        for _ in range(pages):
            doc.new_page(width=600, height=800)
        doc.save(path)


@pytest.fixture
def fixed_ocr(monkeypatch):
    class Backend:
        backend_actual = 'CPU'
        def recognize(self, image):
            return dict(
                txts=['本文。'],
                boxes=np.asarray([[[50, 300], [200, 300], [200, 320], [50, 320]]], dtype=np.float32),
                scores=np.asarray([.95], dtype=np.float32), avg_score=.95,
            )
    monkeypatch.setattr(pipeline, 'get_ocr_backend', lambda **kwargs: Backend())
    monkeypatch.setattr(pipeline, 'render_page_to_np', lambda *args: None)


@pytest.mark.parametrize('pages', [5, 50, 200])
def test_web_writes_each_page_once_and_keeps_status_small(tmp_path, monkeypatch, fixed_ocr, pages):
    import web_app
    source = tmp_path / 'book.pdf'
    make_source(source, pages)
    app = web_app.create_app(tmp_path / 'jobs', start_worker=False)
    service = app.state.jobs
    job = service.create(UploadFile(filename='book.pdf', file=io.BytesIO()), source.read_bytes(), 'all', None, None, 'cpu',
                         dict(clahe=False, reflow=True, two_column=True, furigana=True, homoglyphs=True, adaptive_retry=False))
    writes = []
    original_write = web_app._write_json
    def track_write(path, value):
        if path.parent.name == 'pages':
            writes.append(path.name)
        original_write(path, value)
    monkeypatch.setattr(web_app, '_write_json', track_write)
    service._run(job)
    client = TestClient(app)
    response = client.get(f"/api/jobs/{job['id']}")
    assert response.json()['status'] == 'completed'
    assert len(writes) == pages
    assert len(set(writes)) == pages
    assert len(response.content) < 2000
    stored = json.loads((service.storage / job['id'] / 'job.json').read_text(encoding='utf-8'))
    for heavy_key in ('page_results', 'clean_preview', 'raw_preview'):
        assert heavy_key not in stored
        assert heavy_key not in response.json()
    last_page = client.get(f"/api/jobs/{job['id']}/pages/{pages}").json()
    assert last_page['raw'] == '本文。'
    assert last_page['page_type'] == 'prose'
    assert last_page['blocks'][0]['lines'][0]['bbox'] == [50.0, 300.0, 200.0, 320.0]
    expected = '\n\n'.join(f'=== [Page {n}] ===\n本文。\n' for n in range(1, pages + 1))
    assert client.get(f"/api/jobs/{job['id']}/download/raw").text.replace('\r\n', '\n') == expected
    assert len(client.get('/api/jobs').content) < 2000


def test_legacy_status_response_excludes_full_text_without_rewriting_files(tmp_path):
    from web_app import create_app
    app = create_app(tmp_path, start_worker=False)
    job_id = '11111111-1111-4111-8111-111111111111'
    folder = tmp_path / job_id
    folder.mkdir()
    payload = json.dumps(dict(id=job_id, created_at=1, status='completed', clean_preview='x' * 10000,
                              raw_preview='x' * 10000, page_results={'1': {'raw': 'x' * 10000}}))
    (folder / 'job.json').write_text(payload, encoding='utf-8')
    client = TestClient(app)
    assert len(client.get(f'/api/jobs/{job_id}').content) < 500
    assert len(client.get('/api/jobs').content) < 500
    assert (folder / 'job.json').read_text(encoding='utf-8') == payload


def test_incremental_events_match_legacy_downloads(tmp_path, fixed_ocr):
    source = tmp_path / 'book.pdf'
    make_source(source, 3)
    legacy = list(pipeline.process_pdf_gpu_adaptive_generator(source, tmp_path / 'legacy', enable_clahe=False))
    events = []
    incremental = list(pipeline.process_pdf_gpu_adaptive_generator(
        source, tmp_path / 'incremental', enable_clahe=False, stream_full_text=False, on_page=events.append))
    assert [page['page'] for page in events] == [1, 2, 3]
    assert all(page['raw'] == '本文。' for page in events)
    assert all(page['page_type'] == 'prose' for page in events)
    assert all(page['blocks'] for page in events)
    assert all(page['source_lines'] for page in events)
    assert all(page['needs_review'] is False for page in events)
    assert all(item[2:4] == ('', '') for item in incremental)
    for key in ('md_path', 'raw_path'):
        assert incremental[-1][5][key].read_bytes() == legacy[-1][5][key].read_bytes()
    structured = incremental[-1][5]['structured_path']
    records = [json.loads(line) for line in structured.read_text(encoding='utf-8').splitlines()]
    assert [record['page'] for record in records] == [1, 2, 3]
    assert records[0]['blocks'][0]['lines'][0]['confidence'] == pytest.approx(.95)
    assert '## Page 3' in legacy[-1][2]


@pytest.mark.parametrize('stop_mode', ['cancel', 'write_error'])
def test_incremental_job_keeps_saved_pages_when_stopped(tmp_path, monkeypatch, fixed_ocr, stop_mode):
    import web_app
    source = tmp_path / 'book.pdf'
    make_source(source, 3)
    app = web_app.create_app(tmp_path / 'jobs', start_worker=False)
    service = app.state.jobs
    job = service.create(UploadFile(filename='book.pdf', file=io.BytesIO()), source.read_bytes(), 'all', None, None, 'cpu',
                         dict(clahe=False, reflow=True, two_column=True, furigana=True, homoglyphs=True, adaptive_retry=False))
    original_write = web_app._write_json
    def controlled_write(path, value):
        if path.parent.name == 'pages' and path.stem == '2' and stop_mode == 'write_error':
            raise OSError('disk full fixture')
        original_write(path, value)
        if path.parent.name == 'pages' and stop_mode == 'cancel':
            service.cancel(job['id'])
    monkeypatch.setattr(web_app, '_write_json', controlled_write)
    service._run(job)
    client = TestClient(app)
    state = client.get(f"/api/jobs/{job['id']}").json()
    assert state['status'] == ('cancelled' if stop_mode == 'cancel' else 'failed')
    assert state['completed_pages'] == 1
    assert client.get(f"/api/jobs/{job['id']}/pages/1").json()['raw'] == '本文。'
    assert client.get(f"/api/jobs/{job['id']}/pages/2").status_code == 404
    assert not list((service.storage / job['id'] / 'pages').glob('*.tmp'))


@pytest.mark.parametrize('retry_score', [None, .1, .9])
def test_timing_includes_retry_layout_and_initialization(tmp_path, monkeypatch, retry_score):
    source = tmp_path / 'book.pdf'
    make_source(source, 1)
    clock = [0.0]
    def spend(seconds, value=None):
        clock[0] += seconds
        return value
    scores = iter([.95] if retry_score is None else [.5, retry_score])
    class Backend:
        backend_actual = 'CPU'
        def recognize(self, image):
            return spend(7, dict(txts=['本文。'], boxes=[], scores=[], avg_score=next(scores)))
    monkeypatch.setattr(pipeline.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(pipeline, 'get_ocr_backend', lambda **kwargs: spend(5, Backend()))
    monkeypatch.setattr(pipeline, 'render_page_to_np', lambda *args: spend(2))
    monkeypatch.setattr(pipeline, 'enhance_image_for_ocr', lambda image: spend(3))
    monkeypatch.setattr(pipeline, 'process_page_layout', lambda **kwargs: spend(11, dict(
        clean_text='本文。', raw_text='本文。', corrections_count=0, furigana_filtered=0, columns_detected=1,
        page_type='prose', blocks=[], source_lines=[], needs_review=False, review_reasons=[])))
    result = pipeline.process_pdf_gpu_adaptive(source, tmp_path / 'out')
    metrics = result['metrics_data']
    page = metrics['page_stats'][0]
    assert page['time_sec'] == (23 if retry_score is None else 35)
    assert page['render_sec'] == 2
    assert page['preprocess_sec'] == 3
    assert page['recognition_sec'] == 7
    assert page['layout_sec'] == 11
    assert page['retry_sec'] == (0 if retry_score is None else 12)
    assert page['retry_attempted'] is (retry_score is not None)
    assert page['retry_selected'] is (retry_score == .9)
    assert page['retried'] is (retry_score == .9)  # Preserve the legacy field's meaning.
    assert metrics['initialization_sec'] == 5
    assert metrics['total_elapsed_sec'] == (28 if retry_score is None else 40)
