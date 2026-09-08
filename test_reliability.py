import unittest
from textbook_layout import correct_homoglyphs_and_boundaries, process_page_layout

class ReliabilityTests(unittest.TestCase):
    def test_correct_words_unchanged(self):
        for word in ['入力フォーム', '人口ピラミッド', 'シリーズ一覧']:
            with self.subTest(word=word):
                self.assertEqual(correct_homoglyphs_and_boundaries(word), (word, 0))

    def test_raw_preserves_missing_geometry_and_whitespace(self):
        texts = [' 頁首 ', '', '本文。']
        result = process_page_layout(texts, [], 600, 800)
        self.assertEqual(result['raw_text'], '\n'.join(texts))

    def test_correction_switch(self):
        result = process_page_layout(['コ一ヒ一'], [[[50,100],[200,100],[200,120],[50,120]]], 600,800, enable_homoglyphs=False)
        self.assertEqual(result['clean_text'], 'コ一ヒ一')
        self.assertEqual(result['corrections_count'], 0)

    def test_raw_precedes_filtering(self):
        texts = ['頁首', '本文。']
        boxes = [[[50,y],[200,y],[200,y+20],[50,y+20]] for y in [5,100]]
        result = process_page_layout(texts, boxes,600,800)
        self.assertEqual(result['raw_text'], '\n'.join(texts))


class PipelineReliabilityTests(unittest.TestCase):
    def test_retry_coordinates_and_switch(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import fitz
        import directml_pipeline as pipeline
        for retry_score in [0.9, 0.1]:
            with self.subTest(retry_score=retry_score), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'input.pdf'
                with fitz.open() as doc:
                    doc.new_page(width=600, height=800)
                    doc.save(source)
                def result(score):
                    return dict(txts=['コ一ヒ一'], boxes=[[[50,300],[200,300],[200,320],[50,320]]], scores=[score], avg_score=score)
                with patch.object(pipeline, 'get_ocr_backend') as backend, patch.object(pipeline, 'render_page_to_np'), patch.object(pipeline, 'process_page_layout', wraps=process_page_layout) as layout:
                    backend.return_value.recognize.side_effect = [result(0.5), result(retry_score)]
                    output = pipeline.process_pdf_gpu_adaptive(source, Path(tmp)/'out', enable_clahe=False, enable_homoglyphs=False)
                    expected_dpi = 350 if retry_score > 0.5 else 250
                    self.assertAlmostEqual(layout.call_args.kwargs['page_width'], 600 * expected_dpi / 72)
                    self.assertEqual(output['processed_pages'], 1)
                    self.assertIn('コ一ヒ一', output['markdown_content'])
                    self.assertEqual(output['metrics_data']['total_corrections'], 0)

if __name__ == '__main__':
    unittest.main()
