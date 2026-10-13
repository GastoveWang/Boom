# 檔案功能：驗證新影片推論的 30 FPS、歷史窗口、輸出影片、事件與檔案保護。
# 所屬分工：tests：使用暫存合成影片與微型權重，不修改使用者資料。
# 輸入／輸出：不同 FPS 影片及本機 checkpoint → unittest 結果。
# 執行方式：python benchmark/ASTT/tests/test_video_inference.py。
# 設定與說明：測試推論流程；不代表正式模型準確率。

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from benchmark.ASTT.src.model import VisionTransformer
from benchmark.ASTT.src.annotations import Source, sample_indices
from benchmark.ASTT.src.video_inference import infer_video, pixel_box, video_frames
from benchmark.ASTT.src.ui.inference_view import VIEW_WIDTH, VIEW_HEIGHT, render_results
from argparse import Namespace


class VideoInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory(prefix='astt_video_test_')
        self.root = Path(self.temporary.name)
        self.config = dict(image_size=32, patch_size=16, emb_dim=32, mlp_dim=64,
                           num_heads=4, spatial_layers=1, temporal_layers=1, num_frames=4, dropout_rate=0.)
        self.checkpoint = self.root / 'best.pth'
        torch.save(dict(model=VisionTransformer(**self.config).state_dict(), model_config=self.config,
                        registration=False), self.checkpoint)

    def tearDown(self):
        self.temporary.cleanup()

    def video(self, fps=15., count=12):
        path = self.root / f'source_{fps}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(count):
            writer.write(np.full((48, 64, 3), index * 10, np.uint8))
        writer.release()
        return path

    def options(self, video, **extra):
        options = dict(video=video, checkpoint=self.checkpoint, output_dir=self.root / 'results',
                       experiment_name='video_run', device='cpu', batch_size=3, num_workers=0,
                       registration=False, threshold=0., min_area=1, morph_kernel=1,
                       min_event_frames=2, event_iou=.3, event_max_gap=2, display=False,
                       save_heatmaps=True, max_frames=None)
        return Namespace(**{**options, **extra})

    def test_resampling_matches_annotation_for_low_and_high_fps(self):
        for fps in (15., 30., 60.):
            source = Source.inspect(self.video(fps))
            cap = cv2.VideoCapture(source.path)
            raw = []
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                raw.append(image)
            cap.release()
            converted = list(video_frames(source, source.frame_count))
            self.assertEqual(len(converted), source.frame_count)
            for (index, source_index, image), expected in zip(converted, sample_indices(source)):
                self.assertEqual(source_index, expected)
                np.testing.assert_array_equal(image, raw[expected])

    def test_cli_writes_complete_movie_scores_and_original_pixel_events(self):
        video = self.video()
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / 'benchmark/ASTT/inference.py'),
                                 '--video', str(video), '--checkpoint', str(self.checkpoint),
                                 '--output-dir', str(self.root / 'results'), '--experiment-name', 'video_run',
                                 '--device', 'cpu', '--batch-size', '3', '--threshold', '0', '--min-area', '1',
                                 '--morph-kernel', '1', '--min-event-frames', '2', '--save-heatmaps'],
                                check=True, capture_output=True)
        output = self.root / 'results' / 'astt' / 'video_run'
        cap = cv2.VideoCapture(str(output / 'annotated.mp4'))
        try:
            self.assertEqual(round(cap.get(cv2.CAP_PROP_FPS)), 30)
            count = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                self.assertEqual(image.shape[:2], (VIEW_HEIGHT, VIEW_WIDTH))
                count += 1
            self.assertEqual(count, 24)
        finally:
            cap.release()
        scores = np.load(output / 'scores.npy')
        self.assertEqual(scores.shape, (24,))
        self.assertTrue(np.isnan(scores[:4]).all())
        self.assertTrue(np.isfinite(scores[4:]).all())
        with (output / 'frame_scores.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        np.testing.assert_array_equal([int(row['source_frame_index']) for row in rows], np.arange(24) // 2)
        np.testing.assert_allclose([float(row['time_seconds']) for row in rows], np.arange(24) / 30)
        with (output / 'events.csv').open(newline='', encoding='utf-8') as stream:
            events = list(csv.DictReader(stream))
        self.assertTrue(events)
        self.assertEqual((int(events[0]['x2']), int(events[0]['y2'])), (64, 48))
        self.assertEqual(events[0]['status'], 'candidate')
        metrics = json.loads((output / 'metrics.json').read_text())
        self.assertIsNone(metrics['auc'])
        self.assertEqual(metrics['labelled_frames'], 0)
        self.assertEqual(metrics['evaluated_frames'], 20)
        self.assertTrue(metrics['processed_full_video'])
        self.assertEqual(metrics['bbox_coordinate_space'], 'source_video_pixels')
        self.assertEqual((metrics['source_width'], metrics['source_height']), (64, 48))
        self.assertEqual((metrics['output_width'], metrics['output_height']), (VIEW_WIDTH, VIEW_HEIGHT))
        self.assertEqual(len(list((output / 'heatmaps').glob('*.npy'))), 20)
        self.assertEqual(hashlib.sha256(video.read_bytes()).hexdigest(), digest)
        self.assertIn('[Video]', result.stdout.decode())
        with self.assertRaises(FileExistsError):
            infer_video(self.options(video))

    def test_history_excludes_future_and_bounded_run(self):
        video = self.video(30.)
        source = Source.inspect(video)
        raw = [image for _, _, image in video_frames(source, source.frame_count)]
        inputs = []
        original_forward = VisionTransformer.forward
        def recording_forward(model, value):
            inputs.extend(value.cpu().clone())
            return original_forward(model, value)
        with patch.object(VisionTransformer, 'forward', recording_forward):
            metrics = infer_video(self.options(video, max_frames=9))
        self.assertEqual(metrics['processed_frames'], 9)
        self.assertFalse(metrics['processed_full_video'])
        self.assertEqual(len(inputs), 5)
        for target_index, value in enumerate(inputs, 4):
            expected = np.stack([cv2.resize(raw[i], (32, 32)) for i in range(target_index - 4, target_index)])
            tensor = torch.from_numpy(expected.transpose(0, 3, 1, 2).copy()).float() / 127.5 - 1
            torch.testing.assert_close(value, tensor)

    def test_bad_options_and_registration_do_not_create_outputs(self):
        video = self.video()
        for options in [dict(max_frames=4), dict(max_frames=0), dict(registration=True),
                        dict(heatmap_max=0), dict(heatmap_max=float('nan'))]:
            with self.assertRaises(ValueError):
                infer_video(self.options(video, **options))
        self.assertFalse((self.root / 'results').exists())
        self.assertEqual(pixel_box((2, 3, 16, 20), 640, 480, 32), (40, 45, 320, 300))

    def test_preview_stop_keeps_only_written_frames(self):
        video = self.video(30.)
        with patch('cv2.imshow'), patch('cv2.waitKey', side_effect=[-1, -1, -1, -1, ord('q')]), patch('cv2.destroyWindow'):
            metrics = infer_video(self.options(video, display=True))
        output = self.root / 'results' / 'astt' / 'video_run'
        self.assertEqual(metrics['processed_frames'], 5)
        self.assertEqual(metrics['evaluated_frames'], 1)
        self.assertTrue(metrics['stopped_early'])
        self.assertFalse(metrics['processed_full_video'])
        self.assertEqual(np.load(output / 'scores.npy').shape, (5,))
        self.assertEqual(json.loads((output / 'progress.json').read_text())['stage'], 'Stopped')
        cap = cv2.VideoCapture(str(output / 'annotated.mp4'))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
        cap.release()

    def test_result_view_contains_original_heatmap_overlay_and_past_scores(self):
        image = np.full((480, 640, 3), 100, np.uint8)
        error = np.zeros((32, 32), np.float32)
        error[:, 16:] = .25
        scores = np.array([np.nan, np.nan, np.nan, np.nan, .01, .08, .02, 9.], np.float32)
        original_scores = scores.copy()
        displayed = [(1, (320, 50, 600, 400), True)]
        rendered = render_results(image, error, displayed, scores, 6, 30, 'candidate', .1, .25)
        self.assertEqual(rendered.shape, (VIEW_HEIGHT, VIEW_WIDTH, 3))
        # Same scene point: untouched original, colored heatmap, and a blended overlay.
        np.testing.assert_array_equal(rendered[300, 400], [100, 100, 100])
        self.assertFalse(np.array_equal(rendered[300, 1040], rendered[300, 400]))
        self.assertFalse(np.array_equal(rendered[300, 1680], rendered[300, 400]))
        self.assertFalse(np.array_equal(rendered[300, 1680], rendered[300, 1040]))
        np.testing.assert_array_equal(scores, original_scores)
        scores[7] = .001  # A future score must not change the displayed history.
        future_changed = render_results(image, error, displayed, scores, 6, 30, 'candidate', .1, .25)
        np.testing.assert_array_equal(rendered, future_changed)
        scores[5] = .04
        past_changed = render_results(image, error, displayed, scores, 6, 30, 'candidate', .1, .25)
        self.assertFalse(np.array_equal(rendered[638:821], past_changed[638:821]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
