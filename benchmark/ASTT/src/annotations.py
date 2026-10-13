# 檔案功能：人工區段標記、30 FPS 時間軸、來源分組與安全版本匯出。
"""Source-grouped dataset preparation; unmarked frames are explicitly normal."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
import math
import random
import re
import shutil
import threading
import uuid

import cv2
import numpy as np

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
SPLITS = ('train', 'val')
TARGET_FPS = 30.


def timeline_count(count, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('來源影片 FPS 無效。')
    return int(math.ceil(count * TARGET_FPS / fps - 1e-9))


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', Path(path).name)]


def safe_name(name, allow_spaces=False):
    pattern = r'[^\w. -]+' if allow_spaces else r'[^\w.-]+'
    name = re.sub(pattern, '_', str(name)).strip(' ._')
    if not name:
        raise ValueError('請輸入有效的資料集／來源名稱。')
    if name.casefold() == 'test_frame_masks' or name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        name = 'source_' + name
    return name


def normalize_intervals(intervals, frame_count):
    normalized = []
    for pair in intervals:
        if len(pair) != 2 or any(type(v) is not int for v in pair):
            raise ValueError('區段必須是整數影格起點、終點。')
        a, b = pair
        if not 0 <= a <= b < frame_count:
            raise ValueError(f'區段 {a + 1}–{b + 1} 超出來源影格範圍。')
        normalized.append((a, b))
    merged = []
    for a, b in sorted(normalized):
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


@dataclass
class Source:
    id: str
    path: str
    name: str
    kind: str
    frame_count: int
    fps: float | None
    origin: str
    intervals: list = field(default_factory=list)
    reviewed: bool = False
    frame_files: list = field(default_factory=list)
    size_bytes: int = 0
    mtime_ns: int = 0
    source_frame_count: int = 0

    @classmethod
    def inspect(cls, path):
        path = Path(path).resolve()
        if path.is_dir():
            frames = sorted((p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
            if len(frames) < 5:
                raise ValueError(f'{path.name}：至少需要五張影格。請選擇直接包含圖片的資料夾。')
            return cls(uuid.uuid4().hex, str(path), safe_name(path.name), 'frames', len(frames), None, path.name,
                       frame_files=frames, source_frame_count=len(frames))
        if not path.is_file():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            ok, _ = cap.read()
            if not cap.isOpened() or not ok or count < 5 or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f'{path.name}：無法讀取影片或影格率／長度無效。')
        finally:
            cap.release()
        stat = path.stat()
        converted_count = timeline_count(count, fps)
        if converted_count < 5:
            raise ValueError(f'{path.name}：轉為 30 FPS 後至少需要五幀。')
        return cls(uuid.uuid4().hex, str(path), safe_name(path.stem), 'video', converted_count, fps, path.stem,
                   size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, source_frame_count=count)

    def validate(self, check_disk=False):
        if self.kind not in {'video', 'frames'} or type(self.frame_count) is not int or self.frame_count < 5:
            raise ValueError('標記專案的來源類型／影格數無效。')
        if not re.fullmatch(r'[a-f0-9]{32}', self.id) or safe_name(self.name) != self.name or not self.origin.strip():
            raise ValueError('標記專案的來源 ID／名稱／來源群組無效。')
        if type(self.reviewed) is not bool:
            raise ValueError('完成標記狀態無效。')
        self.intervals = normalize_intervals(self.intervals, self.frame_count)
        path = Path(self.path)
        if self.kind == 'video':
            if self.fps is None or not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError('影片 FPS 無效。')
            if type(self.source_frame_count) is not int or self.source_frame_count < 1 or self.frame_count != timeline_count(self.source_frame_count, self.fps):
                raise ValueError('來源影格數與 30 FPS 時間軸不一致。')
            if check_disk:
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (self.size_bytes, self.mtime_ns):
                    raise ValueError(f'{self.name}：來源影片已修改，請重新加入並標記。')
        else:
            if len(self.frame_files) != self.frame_count or len(set(self.frame_files)) != self.frame_count:
                raise ValueError('影格檔名列表與影格數不一致。')
            if any(Path(n).name != n or '\\' in n or Path(n).suffix.lower() not in IMAGE_SUFFIXES for n in self.frame_files):
                raise ValueError('影格列表必須是圖片檔名，不能包含其他路徑。')
            if check_disk:
                actual = sorted((p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
                if actual != self.frame_files:
                    raise ValueError(f'{self.name}：影格列表已改變，請重新加入並標記。')

    def labels(self, indices=None):
        self.validate()
        indices = np.arange(self.frame_count) if indices is None else np.asarray(indices, dtype=np.int64)
        values = np.zeros(len(indices), dtype=np.uint8)
        for a, b in self.intervals:
            values[(indices >= a) & (indices <= b)] = 1
        return values


def save_project(path, sources):
    path = Path(path).resolve()
    for source in sources:
        source.validate()
    data = {'version': 2, 'unmarked_policy': 'normal', 'interval_end': 'inclusive', 'annotation_fps': TARGET_FPS,
            'interval_domain': '30fps_timeline',
            'sources': [asdict(s) for s in sources]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_project(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    version = data.get('version')
    if version not in (1, 2) or data.get('unmarked_policy') != 'normal' or data.get('interval_end') != 'inclusive':
        raise ValueError('請選擇本工具保存的標記專案；不直接讀取 LabelMe JSON。')
    if version == 2 and (data.get('annotation_fps') != TARGET_FPS or data.get('interval_domain') != '30fps_timeline'):
        raise ValueError('此工具的標記時間軸必須是 30 FPS。')
    sources = [Source(**item) for item in data['sources']]
    ids, paths, names = set(), set(), set()
    for source in sources:
        if version == 1:
            original_count = source.frame_count
            old_intervals = normalize_intervals(source.intervals, original_count)
            source.source_frame_count = original_count
            source.frame_count = timeline_count(original_count, source.fps or TARGET_FPS)
            indices = sample_indices(source)
            mask = np.zeros(len(indices), dtype=np.uint8)
            for a, b in old_intervals:
                mask[(indices >= a) & (indices <= b)] = 1
            transitions = np.diff(np.r_[0, mask, 0].astype(np.int8))
            source.intervals = list(zip(np.flatnonzero(transitions == 1).tolist(),
                                        (np.flatnonzero(transitions == -1) - 1).tolist()))
            source.reviewed = False
        source.validate(check_disk=True)
        canonical = str(Path(source.path).resolve()).casefold()
        if source.id in ids or canonical in paths or source.name.casefold() in names:
            raise ValueError('標記專案有重複的來源 ID、路徑或匯出名稱。')
        ids.add(source.id)
        paths.add(canonical)
        names.add(source.name.casefold())
    return sources


def sample_indices(source):
    """One source mapping per 30 FPS timeline frame; hold the latest available source frame."""
    indices = np.arange(source.frame_count, dtype=np.int64)
    if source.kind == 'frames':
        return indices
    return np.minimum(np.floor(indices * source.fps / TARGET_FPS + 1e-9).astype(np.int64),
                      source.source_frame_count - 1)


def window_targets(source, indices):
    valid = np.ones(max(0, len(indices) - 4), dtype=bool)
    if source.kind == 'frames':
        numbers = [re.search(r'(\d+)$', Path(n).stem) for n in source.frame_files]
        numbers = [int(m.group(1)) if m else None for m in numbers]
        selected = [numbers[i] for i in indices]
        for i in range(len(valid)):
            window = selected[i:i + 5]
            if all(n is not None for n in window):
                valid[i] = all(b == a + 1 for a, b in zip(window, window[1:]))
    return np.flatnonzero(valid) + 4


def source_stats(source):
    indices = sample_indices(source)
    labels = source.labels()
    targets = window_targets(source, indices)
    normal = sum(not labels[t - 4:t + 1].any() for t in targets)
    return {'frames': len(indices), 'abnormal_frames': int(labels.sum()), 'normal_windows': int(normal),
            'scored_normal': int((labels[targets] == 0).sum()), 'scored_abnormal': int((labels[targets] == 1).sum())}


def plan_split(sources, ratios=(7., 3.), seed=2026):
    if len(ratios) != 2 or any(not math.isfinite(v) or v <= 0 for v in ratios):
        raise ValueError('請提供 train／val 兩個大於零的比例，例如 7、3。')
    if len({s.name.casefold() for s in sources}) != len(sources) or len({s.id for s in sources}) != len(sources):
        raise ValueError('來源名稱與 ID 不能重複。')
    if len({str(Path(s.path).resolve()).casefold() for s in sources}) != len(sources):
        raise ValueError('不能重複加入同一個來源。')
    groups, stats = {}, {}
    for source in sources:
        source.validate(check_disk=True)
        if not source.reviewed:
            raise ValueError(f'{source.name}：請先按「完成此來源標記」。未標記區段會視為正常。')
        stats[source.id] = source_stats(source)
        groups.setdefault(source.origin.strip().casefold(), []).append(source)
    if len(groups) < 2:
        raise ValueError('至少需要兩個獨立來源群組，才能分成 train／val。同一原影片的片段請用相同來源群組。')
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    group_stats = {k: {field: sum(stats[s.id][field] for s in groups[k]) for field in stats[groups[k][0].id]} for k in keys}
    eligible = [k for k in keys if group_stats[k]['normal_windows']]
    if not eligible:
        raise ValueError('沒有可訓練的連續五幀正常窗口。請加入正常影片或修正異常區段。')
    # Reserve a normal-capable train group; prefer fully normal sources so anomalies remain in held-out splits.
    train_first = min(eligible, key=lambda k: (group_stats[k]['abnormal_frames'] > 0,
                                               group_stats[k]['abnormal_frames'] / group_stats[k]['frames']))
    allocation = {train_first: 'train'}
    available = [k for k in keys if k != train_first]
    for split in ('val',):
        preferred = sorted((k for k in available if group_stats[k]['scored_normal'] or group_stats[k]['scored_abnormal']),
                           key=lambda k: (not (group_stats[k]['scored_normal'] and group_stats[k]['scored_abnormal']),
                                                     not group_stats[k]['scored_abnormal']))
        if not preferred:
            raise ValueError(f'沒有可分配到 {split} 的連續五幀窗口，請加入更多獨立連續來源。')
        chosen = preferred[0]
        allocation[chosen] = split
        available.remove(chosen)
    desired = np.asarray(ratios, dtype=float) / sum(ratios) * len(groups)
    counts = {split: 1 for split in SPLITS}
    for key in available:
        split = max(SPLITS, key=lambda s: desired[SPLITS.index(s)] - counts[s])
        allocation[key] = split
        counts[split] += 1
    assignments = {s.id: allocation[s.origin.strip().casefold()] for s in sources}
    summary = {split: {field: sum(stats[s.id][field] for s in sources if assignments[s.id] == split)
                       for field in ('frames', 'abnormal_frames', 'normal_windows', 'scored_normal', 'scored_abnormal')} for split in SPLITS}
    warnings = []
    for split in ('val',):
        if not summary[split]['scored_normal'] or not summary[split]['scored_abnormal']:
            warnings.append(f'{split} 可評分影格未同時包含正常／異常，AUROC 會是 null。')
        if not summary[split]['scored_normal'] and not summary[split]['scored_abnormal']:
            raise ValueError(f'{split} 沒有連續五幀窗口，請檢查來源缺幀。')
    selection = 'auc' if summary['val']['scored_normal'] and summary['val']['scored_abnormal'] else 'mse'
    if selection == 'mse' and summary['val']['scored_abnormal']:
        warnings.append('val 只有異常可評分影格，MSE 選模也會包含異常；建議加入有正常畫面的驗證來源。')
    return {'assignments': assignments, 'source_stats': stats, 'summary': summary, 'group_counts': counts,
            'ratios': list(ratios), 'ratio_unit': 'source_groups', 'seed': seed, 'frame_policy': 'fixed_30fps', 'output_fps': TARGET_FPS,
            'selection': selection, 'warnings': warnings}


class ExportCancelled(Exception):
    pass


def export_dataset(sources, output_parent, name='artillery-shell', ratios=(7., 3.), seed=2026, progress=None, cancel=None):
    plan = plan_split(sources, ratios, seed)
    parent = Path(output_parent).resolve()
    for source in sources:
        source_path = Path(source.path).resolve()
        if source.kind == 'frames' and (parent == source_path or parent.is_relative_to(source_path)):
            raise ValueError('輸出位置不能放在來源影格資料夾內。')
    name = safe_name(name, allow_spaces=True)
    parent.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        output = parent / f'{name}_v{version}'
        try:
            output.mkdir(exist_ok=False)
            break
        except FileExistsError:
            version += 1
    # Only this exclusively created directory is ever removed on failure/cancellation.
    assert output.resolve().parent == parent and output.name == f'{name}_v{version}'
    cancel = cancel or threading.Event()
    try:
        for split in SPLITS:
            root = output / 'astt' / split
            (root if split == 'train' else root / 'frames').mkdir(parents=True)
            (root / 'test_frame_masks').mkdir()
        save_project(output / 'annotations.json', sources)
        for number, source in enumerate(sources):
            if cancel.is_set():
                raise ExportCancelled('已取消匯出。')
            split = plan['assignments'][source.id]
            root = output / 'astt' / split
            folder = (root if split == 'train' else root / 'frames') / source.name
            folder.mkdir()
            indices = sample_indices(source)
            labels = source.labels()
            cap = cv2.VideoCapture(source.path) if source.kind == 'video' else None
            try:
                with (folder / 'frames.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['filename', 'source_frame_index', 'time_seconds', 'label', 'source_time_seconds'])
                    written = 0
                    decoded_index = -1
                    for i in indices.tolist():
                        if cancel.is_set():
                            raise ExportCancelled('已取消匯出。')
                        if cap is not None:
                            while decoded_index < i:
                                if cancel.is_set():
                                    raise ExportCancelled('已取消匯出。')
                                ok, image = cap.read()
                                decoded_index += 1
                                if not ok:
                                    raise ValueError(f'{source.name}：原始第 {decoded_index + 1} 幀解碼失敗，未產生不完整資料集。')
                        if source.kind == 'frames':
                            filename = source.frame_files[i]
                            image_path = Path(source.path) / filename
                            if cv2.imread(str(image_path)) is None:
                                raise ValueError(f'無法讀取 {image_path}')
                            shutil.copy2(image_path, folder / filename)
                        else:
                            filename = f'frame_{written:08d}.jpg'
                            if not cv2.imwrite(str(folder / filename), image):
                                raise IOError(f'無法寫入 {filename}')
                        writer.writerow([filename, i, written / TARGET_FPS, int(labels[written]), i / source.fps if source.fps else ''])
                        written += 1
                        if progress and (written % 100 == 0 or written == len(indices)):
                            progress(f'{number + 1}/{len(sources)} {source.name} → {split}：{written}/{len(indices)} 幀')
                if written != len(labels):
                    raise ValueError('輸出影格與標籤長度不一致。')
                np.save(root / 'test_frame_masks' / f'{source.name}.npy', labels, allow_pickle=False)
            finally:
                if cap is not None:
                    cap.release()
        if cancel.is_set():
            raise ExportCancelled('已取消匯出。')
        report = {**plan, 'output': str(output), 'unmarked_policy': 'normal',
                  'sources': [{'id': s.id, 'path': s.path, 'name': s.name, 'origin': s.origin, 'source_fps': s.fps,
                               'split': plan['assignments'][s.id]} for s in sources]}
        (output / 'split_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        experiment = f'astt_{safe_name(output.name)}'
        results_root = Path(__file__).resolve().parents[1] / 'output'
        (output / 'README.md').write_text(
            '<!-- 檔案功能：本次人工標記資料集的使用方式。 -->\n\n'
            '# ASTT 標記資料集\n\n依來源群組分 train／val，預設 7:3；未標記正常 0，標記異常 1。\n'
            '標記與輸出固定 30 FPS；高幀率取樣、低幀率重複前一原始影格補齊，標籤對應同一時間軸。\n'
            '圖片資料夾視為已是 30 FPS 序列；原始影片不會被修改。\n'
            'train 保存完整影片與標籤，ASTT loader 會排除 input＋target 五幀中含異常的窗口。\n'
            '所有片段依來源群組分配；比例按群組數近似，不是隨機切單張圖片。\n\n'
            f'從 Boom 根目錄執行：\n\n```powershell\n'
            f'python benchmark/ASTT/train.py --data-dir "{output}" --output-dir "{results_root}" --selection {plan["selection"]} --experiment-name "{experiment}" --device cuda --batch-size 8 --epochs 30\n'
            f'python benchmark/ASTT/test.py --split val --data-dir "{output}" --output-dir "{results_root}" --checkpoint "{results_root / "astt" / experiment / "checkpoints" / "best.pth"}" --experiment-name "{experiment}_val" --device cuda --batch-size 8\n```\n'
            'val 用於選模與查看效果，本版本沒有獨立 test。\n'
            '如已執行過同名實驗，請更換 experiment-name，並對應修改測試的 checkpoint 路徑。\n', encoding='utf-8')
        return report
    except BaseException:
        # Do not touch pre-existing datasets or source files.
        assert output.resolve().parent == parent and not output.is_symlink()
        shutil.rmtree(output)
        raise
