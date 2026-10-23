"""Regression checks using the shipped checkpoint and Streamlit's test runner."""
import sys
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class IntakeTests(unittest.TestCase):
    def test_intake_does_not_import_inference(self):
        # Run this class first, in a fresh process.
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertNotIn('inference', sys.modules)
        app.text_input(key='input_pname').set_value('Test Patient').run()
        self.assertFalse(app.exception)
        self.assertNotIn('inference', sys.modules)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from inference import load_medfusionnet_model, predict_with_multiscale_gradcam
        cls.predict = staticmethod(predict_with_multiscale_gradcam)
        cls.model, cls.metadata, cls.device = load_medfusionnet_model(
            str(ROOT / 'checkpoints/medfusionnet_best.pt'))
        cls.image = Image.new('RGB', (384, 256), (180, 125, 110))

    def run_prediction(self, heatmap=False, threshold=0):
        return self.predict(self.model, self.image, self.metadata['image_size'],
                            self.device, include_heatmap=heatmap,
                            confidence_threshold=threshold)

    def test_heatmap_preserves_predictions_and_hooks(self):
        plain = self.run_prediction()
        seen = []
        handle = self.model.convnext_features[5].register_forward_hook(
            lambda *args: seen.append(True))
        try:
            explained = self.run_prediction(True)
            self.assertIn(handle.id, self.model.convnext_features[5]._forward_hooks)
        finally:
            handle.remove()
        np.testing.assert_allclose(plain[0], explained[0], atol=1e-6)
        self.assertIsNone(plain[1])
        self.assertTrue(np.isfinite(explained[1]).all())
        self.assertEqual(explained[1].shape, (self.metadata['image_size'],)*2)
        self.assertFalse(self.model.convnext_features[5]._forward_hooks)
        self.assertIsNone(self.run_prediction(True, threshold=1.1)[1])

    def test_concurrent_calls(self):
        images = [self.image, Image.new('RGB', (256, 384), (80, 50, 30))]
        def analyze(image):
            return self.predict(self.model, image, self.metadata['image_size'],
                                self.device, include_heatmap=True, confidence_threshold=0)
        expected = [analyze(image) for image in images]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(analyze, images))
        for actual, reference in zip(results, expected):
            for index in (0, 1, 2):
                np.testing.assert_allclose(actual[index], reference[index], atol=1e-6)
        self.assertFalse(self.model.convnext_features[7]._forward_hooks)

    def test_failed_heatmap_removes_its_hooks(self):
        original = self.model.forward
        calls = 0
        def fail_second_pass(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('simulated heatmap failure')
            return original(*args, **kwargs)
        with patch.object(self.model, 'forward', side_effect=fail_second_pass):
            with self.assertRaisesRegex(RuntimeError, 'simulated heatmap failure'):
                self.run_prediction(True)
        self.assertFalse(self.model.convnext_features[5]._forward_hooks)
        self.assertFalse(self.model.convnext_features[7]._forward_hooks)
        self.assertTrue(np.isfinite(self.run_prediction()[0]).all())

    def test_results_and_on_demand_reports(self):
        probs, _, weights, idx = self.run_prediction()
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        app.session_state.current_page = 'results'
        app.session_state.analysis_results = dict(
            active_image=self.image, image_source_label='test.jpg',
            patient_id='TEST', patient_name='Test Patient', patient_age='40',
            anatomic_site='Back / Torso', probabilities=probs, cam=None,
            fusion_weights=weights, predicted_index=idx,
            predicted_code=self.metadata['classes'][idx], confidence=float(probs[idx]),
            meta={'name':'Test class', 'category':'Test', 'is_cancer':False},
            is_confident=True, is_cancer=False, classes=self.metadata['classes'],
            image_size=self.metadata['image_size'])
        app.run()
        self.assertFalse(app.exception)
        self.assertNotIn('_cached_pdf', app.session_state)
        app.button(key='prepare_reports').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state['_cached_pdf'].startswith(b'%PDF'))
        old_pdf = app.session_state['_cached_pdf']
        app.text_area(key='results_clinical_notes').set_value('Updated notes').run()
        self.assertFalse(app.exception)
        self.assertEqual(old_pdf, app.session_state['_cached_pdf'])
        self.assertEqual(len(app.get('download_button')), 0)
        app.button(key='prepare_reports').click().run(timeout=30)
        self.assertIn('Updated notes', app.session_state['_cached_txt'])
        app.session_state.analysis_results['cam'] = np.ones((self.metadata['image_size'],)*2)
        app.run()
        self.assertFalse(app.exception)
        app.selectbox(key='results_cmap_palette').set_value('jet (Classic)').run()
        app.button(key='prepare_reports').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state['_cached_pdf'].startswith(b'%PDF'))
        app.button(key='btn_nav_back_top').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.current_page, 'intake')


if __name__ == '__main__':
    unittest.main()
