# 檔案功能：驗證人工異常區段、抽幀標籤對齊、來源分組、安全匯出與 GUI 操作。
# 所屬分工：tests：資料準備工具的可執行 unittest。
# 輸入／輸出：系統暫存影片／圖片 → 測試結果；不使用或修改正式資料。
# 執行方式：python benchmark/ASTT/tests/test_annotations.py。
# 設定與說明：GUI 測試只建立隱藏 Tk 視窗；不開啟使用者的影片或應用程式。

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from benchmark.ASTT.src.runtime.annotations import (
    TARGET_FPS, ExportCancelled, Source, export_dataset, load_project, normalize_intervals,
    plan_split, sample_indices, save_project, timeline_count,
)
from benchmark.ASTT.src.runtime.dataset import VideoDataset


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='astt_annotations_test_')
        self.root = Path(self.temporary.name)
        self.sources = []
        for name in ('normal', 'change_a', 'change_b'):
            path = self.root / f'{name}.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20., (32, 32))
            self.assertTrue(writer.isOpened())
            for i in range(40):
                writer.write(np.full((32, 32, 3), i * 5, dtype=np.uint8))
            writer.release()
            source = Source.inspect(path)
            source.reviewed = True
            if name != 'normal':
                source.intervals = [(12, 19)]
            self.sources.append(source)

    def tearDown(self):
        self.temporary.cleanup()

    def test_intervals_and_unmarked_normal(self):
        source = self.sources[1]
        source.intervals = [(7, 9), (4, 7), (12, 12)]
        self.assertEqual(normalize_intervals(source.intervals, 40), [(4, 9), (12, 12)])
        labels = source.labels()
        expected = np.zeros(60, np.uint8)
        expected[4:10] = 1
        expected[12] = 1
        np.testing.assert_array_equal(labels, expected)
        with self.assertRaises(ValueError):
            normalize_intervals([(9, 7)], 40)
        with self.assertRaises(ValueError):
            normalize_intervals([(0, 40)], 40)

    def test_30fps_timeline_keeps_marked_single_frames_and_endpoints(self):
        source = self.sources[1]
        self.assertEqual(source.source_frame_count, 40)
        self.assertEqual(source.frame_count, 60)
        source.intervals = [(0, 0), (1, 1), (13, 13), (59, 59)]
        indices = sample_indices(source)
        np.testing.assert_array_equal(indices, np.floor(np.arange(60) * 20 / 30).astype(int))
        labels = source.labels()
        np.testing.assert_array_equal(np.flatnonzero(labels), [0, 1, 13, 59])

    def test_other_video_rates_and_exported_images(self):
        for fps, count in [(15., 30), (30., 60), (60., 120), (29.97, 60)]:
            with self.subTest(fps=fps):
                path = self.root / f'rate_{fps}.mp4'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (32, 32))
                self.assertTrue(writer.isOpened())
                for i in range(count):
                    writer.write(np.full((32, 32, 3), i * 2, np.uint8))
                writer.release()
                source = Source.inspect(path)
                source.reviewed = True
                source.intervals = [(10, 19)]
                expected_count = timeline_count(count, source.fps)
                self.assertEqual(source.frame_count, expected_count)
                expected = np.minimum(np.floor(np.arange(expected_count) * source.fps / 30 + 1e-9).astype(int), count - 1)
                np.testing.assert_array_equal(sample_indices(source), expected)
                report = export_dataset([self.sources[0], source], self.root / f'export_{fps}')
                folder = Path(report['output']) / 'astt' / 'val' / 'frames' / source.name
                cap = cv2.VideoCapture(str(path))
                raw = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    raw.append(frame)
                cap.release()
                for k, index in enumerate(expected):
                    ok, encoded = cv2.imencode('.jpg', raw[index])
                    self.assertTrue(ok)
                    self.assertEqual((folder / f'frame_{k:08d}.jpg').read_bytes(), encoded.tobytes())

    def test_grouping_review_and_reproducibility(self):
        same_origin = Source.inspect(self.sources[0].path)
        same_origin.path = str(self.root / 'same_origin.mp4')
        Path(same_origin.path).write_bytes(Path(self.sources[0].path).read_bytes())
        stat = Path(same_origin.path).stat()
        same_origin.size_bytes, same_origin.mtime_ns = stat.st_size, stat.st_mtime_ns
        same_origin.name = 'another_clip'
        same_origin.reviewed = True
        sources = [*self.sources, same_origin]
        plan = plan_split(sources)
        self.assertEqual(plan['assignments'][same_origin.id], plan['assignments'][self.sources[0].id])
        self.assertEqual(plan['assignments'], plan_split(sources)['assignments'])
        self.assertEqual(plan['assignments'][self.sources[0].id], 'train')
        self.assertEqual(plan['selection'], 'auc')
        self.sources[0].reviewed = False
        with self.assertRaisesRegex(ValueError, '完成此來源標記'):
            plan_split(self.sources)
        with self.assertRaises(ValueError):
            plan_split(self.sources[:1])

    def test_export_loads_in_astt_and_preserves_sources(self):
        original = {s.path: hashlib.sha256(Path(s.path).read_bytes()).hexdigest() for s in self.sources}
        report = export_dataset(self.sources, self.root / 'datasets')
        dataset = Path(report['output'])
        self.assertEqual(dataset.name, 'artillery-shell_v1')
        self.assertTrue((dataset / 'annotations.json').is_file())
        self.assertTrue((dataset / 'split_report.json').is_file())
        self.assertEqual(set(report['summary']), {'train', 'val'})
        self.assertFalse((dataset / 'astt' / 'test').exists())
        self.assertIn('--split val', (dataset / 'README.md').read_text(encoding='utf-8'))
        self.assertEqual(len(load_project(dataset / 'annotations.json')), 3)
        for source in self.sources:
            split = report['assignments'][source.id]
            split_root = dataset / 'astt' / split
            root = split_root if split == 'train' else split_root / 'frames'
            labels = np.load(split_root / 'test_frame_masks' / f'{source.name}.npy')
            expected = source.labels()
            np.testing.assert_array_equal(labels, expected)
            self.assertEqual(len(labels), source.frame_count)
            with (root / source.name / 'frames.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            np.testing.assert_array_equal([int(row['source_frame_index']) for row in rows], sample_indices(source))
            np.testing.assert_allclose([float(row['time_seconds']) for row in rows], np.arange(source.frame_count) / TARGET_FPS)
            np.testing.assert_allclose([float(row['source_time_seconds']) for row in rows], sample_indices(source) / source.fps)
            np.testing.assert_array_equal([int(row['label']) for row in rows], labels)
            ds = VideoDataset(root, 32, 4, split_root / 'test_frame_masks', normal_only=split == 'train')
            self.assertGreater(len(ds), 0)
            if split == 'train':
                for video, start in ds.samples:
                    self.assertFalse(ds.labels[video][start:start + 5].any())
            self.assertEqual(len(list((root / source.name).glob('*.jpg'))), len(labels))
        for path, digest in original.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        second = export_dataset(self.sources, self.root / 'datasets')
        self.assertNotEqual(report['output'], second['output'])
        self.assertTrue((dataset / 'README.md').is_file())
        self.assertEqual(report['frame_policy'], 'fixed_30fps')
        self.assertEqual(report['output_fps'], 30.)

    def test_training_excludes_abnormal_input_and_target(self):
        # All source groups have anomalies; train still receives only all-normal windows.
        self.sources[0].intervals = [(12, 19)]
        report = export_dataset(self.sources, self.root / 'mixed')
        root = Path(report['output']) / 'astt' / 'train'
        ds = VideoDataset(root, 32, 4, root / 'test_frame_masks', normal_only=True)
        self.assertEqual(len(ds), report['summary']['train']['normal_windows'])
        self.assertGreater(len(ds), 0)
        for video, start in ds.samples:
            self.assertFalse(ds.labels[video][start:start + 5].any())

    def test_existing_frames_preserve_gaps_and_normal_labels(self):
        folder = self.root / 'frames'
        folder.mkdir()
        for number in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]:
            cv2.imwrite(str(folder / f'frame_{number:06d}.png'), np.zeros((32, 32, 3), np.uint8))
        source = Source.inspect(folder)
        source.reviewed = True
        self.sources[0] = source
        report = export_dataset(self.sources, self.root / 'from_frames')
        root = Path(report['output']) / 'astt' / 'train'
        ds = VideoDataset(root, 32, 4, root / 'test_frame_masks', normal_only=True)
        self.assertEqual(sum(video == source.name for video, start in ds.samples), 3)
        self.assertTrue((root / source.name / 'frame_000007.png').is_file())
        self.assertIsNone(source.fps)
        np.testing.assert_array_equal(np.load(root / 'test_frame_masks' / 'frames.npy'), np.zeros(11, np.uint8))

    def test_holdouts_use_scorable_sources_and_reject_name_collisions(self):
        folder = self.root / 'gaps'
        folder.mkdir()
        for number in [1, 3, 5, 7, 9, 11]:
            cv2.imwrite(str(folder / f'frame_{number:06d}.jpg'), np.zeros((32, 32, 3), np.uint8))
        gaps = Source.inspect(folder)
        gaps.reviewed = True
        plan = plan_split([gaps, *self.sources])
        self.assertEqual(plan['assignments'][gaps.id], 'train')
        self.assertGreater(plan['summary']['val']['scored_normal'], 0)
        self.sources[1].name = self.sources[0].name.upper()
        with self.assertRaisesRegex(ValueError, '來源名稱'):
            plan_split(self.sources)

    def test_two_groups_and_seven_three_ratio(self):
        two = plan_split(self.sources[:2])
        self.assertEqual(two['group_counts'], {'train': 1, 'val': 1})
        self.assertEqual(two['ratios'], [7., 3.])
        for i in range(7):
            path = self.root / f'extra_{i}.mp4'
            path.write_bytes(Path(self.sources[0].path).read_bytes())
            source = Source.inspect(path)
            source.reviewed = True
            self.sources.append(source)
        plan = plan_split(self.sources)
        self.assertEqual(plan['group_counts'], {'train': 7, 'val': 3})
        self.assertEqual(plan['assignments'], plan_split(list(reversed(self.sources)))['assignments'])
        for ratios in [(7, 3, 0), (7, 0), (7,)]:
            with self.assertRaises(ValueError):
                plan_split(self.sources, ratios=ratios)

    def test_exported_train_val_only_runs_training_and_val_inference(self):
        report = export_dataset(self.sources[:2], self.root / 'ready')
        data = report['output']
        base = PROJECT_ROOT / 'benchmark' / 'ASTT'
        common = ['--data-dir', data, '--output-dir', str(self.root / 'results'),
                  '--device', 'cpu', '--batch-size', '8']
        tiny = ['--epochs', '1', '--image-size', '32', '--patch-size', '16', '--emb-dim', '32',
                '--mlp-dim', '64', '--num-heads', '4', '--spatial-layers', '1', '--temporal-layers', '1']
        subprocess.run([sys.executable, str(base / 'train.py'), *common, *tiny,
                        '--selection', report['selection'], '--experiment-name', 'prepared_train'],
                       check=True, capture_output=True)
        checkpoint = self.root / 'results' / 'astt' / 'prepared_train' / 'checkpoints' / 'best.pth'
        self.assertTrue(checkpoint.is_file())
        subprocess.run([sys.executable, str(base / 'test.py'), *common, '--split', 'val',
                        '--checkpoint', str(checkpoint), '--experiment-name', 'prepared_val'],
                       check=True, capture_output=True)
        metrics = json.loads((self.root / 'results' / 'astt' / 'prepared_val' / 'metrics.json').read_text())
        self.assertGreater(metrics['labelled_frames'], 0)
        self.assertIsNotNone(metrics['auc'])

    def test_cancel_and_bad_image_leave_no_partial_dataset(self):
        parent = self.root / 'cancel'
        parent.mkdir()
        existing = parent / 'artillery-shell_v1'
        existing.mkdir()
        (existing / 'keep.txt').write_text('keep', encoding='utf-8')
        event = threading.Event()
        event.set()
        with self.assertRaises(ExportCancelled):
            export_dataset(self.sources, parent, cancel=event)
        self.assertEqual(sorted(p.name for p in parent.iterdir()), ['artillery-shell_v1'])
        self.assertEqual((existing / 'keep.txt').read_text(), 'keep')
        folder = self.root / 'broken_frames'
        folder.mkdir()
        for i in range(6):
            cv2.imwrite(str(folder / f'frame_{i:06d}.jpg'), np.zeros((32, 32, 3), np.uint8))
        bad = Source.inspect(folder)
        bad.reviewed = True
        (folder / 'frame_000003.jpg').write_bytes(b'not an image')
        with self.assertRaises(ValueError):
            export_dataset([bad, *self.sources[1:]], parent)
        self.assertEqual(sorted(p.name for p in parent.iterdir()), ['artillery-shell_v1'])

    def test_preflight_cli_keeps_model_settings_and_balances_val(self):
        import torch
        report = export_dataset(self.sources[:2], self.root / 'preflight_data')
        base = PROJECT_ROOT / 'benchmark' / 'ASTT'
        command = [sys.executable, str(base / 'train.py'), '--data-dir', report['output'],
                   '--output-dir', str(self.root / 'results'), '--experiment-name', 'short_check',
                   '--device', 'cpu', '--batch-size', '4', '--epochs', '1', '--selection', 'auc',
                   '--image-size', '32', '--patch-size', '16', '--emb-dim', '32', '--mlp-dim', '64',
                   '--num-heads', '4', '--spatial-layers', '1', '--temporal-layers', '1', '--preflight']
        result = subprocess.run(command, check=True, capture_output=True)
        output = self.root / 'results' / 'astt' / 'short_check'
        state = torch.load(output / 'checkpoints' / 'best.pth', map_location='cpu', weights_only=True)
        self.assertTrue(state['preflight'])
        self.assertEqual(state['model_config']['emb_dim'], 32)
        self.assertEqual(state['steps_per_epoch'], 4)
        metrics = json.loads((output / 'metrics.json').read_text())
        self.assertEqual(metrics['evaluated_frames'], 8)
        self.assertIsNotNone(metrics['auc'])
        self.assertTrue(metrics['preflight'])
        self.assertEqual(json.loads((output / 'progress.json').read_text())['stage'], 'Completed')
        self.assertIn('[Train epoch 1/1] 1/4', result.stdout.decode())
        self.assertIn('[Val epoch 1/1] 2/2', result.stdout.decode())
        command[command.index('--epochs') + 1] = '30'
        invalid = subprocess.run(command, capture_output=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn('Preflight requires --epochs 1', invalid.stderr.decode())

    def test_project_and_cli_preview(self):
        project = self.root / 'annotations.json'
        save_project(project, self.sources)
        restored = load_project(project)
        self.assertEqual(restored[1].intervals, [(12, 19)])
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / 'benchmark/ASTT/tools/prepare_dataset.py'),
                                 '--project', str(project), '--preview'], check=True, capture_output=True)
        report = json.loads(result.stdout.decode('utf-8'))
        self.assertEqual(sum(report['group_counts'].values()), 3)
        data = json.loads(project.read_text(encoding='utf-8'))
        data['unmarked_policy'] = 'unknown'
        project.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_project(project)

    def test_old_project_migration_and_30fps_schema(self):
        project = self.root / 'old.json'
        save_project(project, self.sources)
        data = json.loads(project.read_text(encoding='utf-8'))
        self.assertEqual(data['version'], 2)
        self.assertEqual(data['annotation_fps'], 30.)
        data['annotation_fps'] = 20.
        project.write_text(json.dumps(data), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '30 FPS'):
            load_project(project)
        data['version'] = 1
        data.pop('annotation_fps')
        data.pop('interval_domain')
        for item in data['sources']:
            item['frame_count'] = item.pop('source_frame_count')
        project.write_text(json.dumps(data), encoding='utf-8')
        original = project.read_bytes()
        restored = load_project(project)
        self.assertEqual(project.read_bytes(), original)
        self.assertFalse(any(s.reviewed for s in restored))
        self.assertEqual(restored[1].source_frame_count, 40)
        self.assertEqual(restored[1].frame_count, 60)
        mapping = sample_indices(restored[1])
        np.testing.assert_array_equal(restored[1].labels(), ((mapping >= 12) & (mapping <= 19)).astype(np.uint8))
        new = self.root / 'new.json'
        save_project(new, restored)
        np.testing.assert_array_equal(load_project(new)[1].labels(), restored[1].labels())

    def test_gui_marks_ranges_autosaves_and_edits(self):
        import tkinter as tk
        from benchmark.ASTT.src.ui.annotation_app import AnnotationApp
        root = tk.Tk()
        root.withdraw()
        app = AnnotationApp(root)
        app.project_path = self.root / 'gui_annotations.json'
        try:
            self.assertEqual(app.ratios.get(), '7:3')
            self.assertEqual(app.dataset_name.get(), 'artillery-shell')
            self.assertEqual(app.options()['ratios'], (7., 3.))
            self.assertNotIn('fps', app.options())
            app.add_paths([s.path for s in self.sources])
            root.update()
            self.assertIsNotNone(app.current)
            app.seek(3)
            self.assertEqual(app.capture_next, 3)  # Timeline index 3 is original index 2 at 20 FPS.
            self.assertIn('00:00.10', app.position_text(3))
            self.assertEqual(app.play_delay(), 33)
            app.jump(1)
            self.assertEqual(app.frame_index, 33)
            app.seek(4)
            app.mark_start()
            app.seek(7)
            app.mark_end()
            app.finish_source()
            self.assertEqual(app.current.intervals, [(4, 7)])
            self.assertTrue(app.current.reviewed)
            self.assertEqual(load_project(app.project_path)[0].intervals, [(4, 7)])
            app.interval_list.selection_set('0')
            app.delete_interval()
            self.assertFalse(app.current.reviewed)
            self.assertEqual(app.current.intervals, [])
            app.toggle_play()
            app.slider_seek('12')
            self.assertFalse(app.playing)
            self.assertEqual(app.frame_index, 12)
            self.assertTrue(app.canvas.find_all())
            app.seek(0)
            app.playing = True
            # Even a large elapsed clock value cannot make playback jump over source frames.
            with patch('benchmark.ASTT.src.ui.annotation_app.time.perf_counter', return_value=999999):
                app.tick()
            self.assertEqual(app.frame_index, 1)
            app.pause()
        finally:
            app.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
