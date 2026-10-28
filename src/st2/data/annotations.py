# 檔案功能：ST2 手動逐幀區域標註、來源分組及版本化資料匯出。
"""Per-frame ROI labels with fixed-position seeding and versioned export."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import re
import shutil
import threading
import uuid

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / 'data/st2'
OUTPUT_ROOT = PROJECT_ROOT / 'output/st2'
ANNOTATIONS_ROOT = OUTPUT_ROOT / 'annotations'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', Path(name).name)]


def safe_name(name):
    value = re.sub(r'[^\w.-]+', '_', str(name)).strip(' ._')
    if not value:
        raise ValueError('請輸入有效的資料集名稱。')
    if value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        value = 'dataset_' + value
    return value


def _fingerprint(path, files):
    paths = [path / name for name in files] if files else [path]
    digest = hashlib.sha256()
    for item in paths:
        stat = item.stat()
        digest.update(f'{item.name}\t{stat.st_size}\t{stat.st_mtime_ns}\n'.encode('utf-8'))
    return digest.hexdigest()


def merge_intervals(intervals, count):
    merged = []
    for pair in sorted(intervals):
        if len(pair) != 2 or any(type(v) is not int for v in pair) or not 0 <= pair[0] <= pair[1] < count:
            raise ValueError('區段必須是範圍內的整數起點、終點，且起點不可晚於終點。')
        start, end = pair
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


@dataclass
class SmokeSource:
    id: str
    path: str
    name: str
    origin: str
    kind: str
    frame_count: int
    fps: float
    fingerprint: str
    frame_files: list = field(default_factory=list)
    intervals: list = field(default_factory=list)
    ignored: list = field(default_factory=list)
    boxes: dict = field(default_factory=dict)

    @classmethod
    def inspect(cls, path, images_fps=30.):
        path = Path(path).resolve()
        if path.is_dir():
            files = sorted((p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
            if not files or not math.isfinite(images_fps) or images_fps <= 0:
                raise ValueError('請選直接包含連續圖片的資料夾，並指定有效的圖片序列 FPS。')
            if cv2.imread(str(path / files[0])) is None:
                raise ValueError('第一張圖片無法讀取。')
            return cls(uuid.uuid4().hex, str(path), path.name, path.name, 'frames', len(files),
                       float(images_fps), _fingerprint(path, files), frame_files=files)
        if not path.is_file():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        try:
            count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
            ok, _ = cap.read()
            if not ok or count < 1 or not math.isfinite(fps) or fps <= 0:
                raise ValueError('影片無法讀取或缺少可用的幀數／FPS。')
        finally:
            cap.release()
        return cls(uuid.uuid4().hex, str(path), path.stem, path.stem, 'video', count, fps, _fingerprint(path, []))

    def validate(self, check_disk=False):
        if self.kind not in {'video', 'frames'} or type(self.frame_count) is not int or self.frame_count < 1:
            raise ValueError('來源類型／幀數無效。')
        if not re.fullmatch(r'[a-f0-9]{32}', self.id) or not self.name.strip() or not self.origin.strip():
            raise ValueError('來源 ID、名稱或群組無效。')
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('來源 FPS 無效。')
        self.intervals = merge_intervals(self.intervals, self.frame_count)
        self.ignored = merge_intervals(self.ignored, self.frame_count)
        if self.kind == 'frames':
            if len(self.frame_files) != self.frame_count or len(set(self.frame_files)) != self.frame_count:
                raise ValueError('圖片列表與幀數不一致。')
            if any(Path(n).name != n or '\\' in n or Path(n).suffix.lower() not in IMAGE_SUFFIXES for n in self.frame_files):
                raise ValueError('圖片列表包含無效檔名。')
        elif self.frame_files:
            raise ValueError('影片來源不應包含圖片檔名列表。')
        for key, boxes in self.boxes.items():
            if not str(key).isdigit() or str(int(key)) != str(key) or not 0 <= int(key) < self.frame_count:
                raise ValueError('框的影格索引無效。')
            for box in boxes:
                if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in box):
                    raise ValueError('BBox 必須包含四個有限座標。')
                if not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
                    raise ValueError('BBox 為相對原圖的 xyxy，須有正面積且在 [0,1] 內。')
        if check_disk:
            path = Path(self.path)
            if self.kind == 'frames':
                files = sorted((p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
                if files != self.frame_files:
                    raise ValueError(f'{self.name} 的圖片列表已改變。')
            if _fingerprint(path, self.frame_files) != self.fingerprint:
                raise ValueError(f'{self.name} 的原始來源已修改，請重新加入。')

    def labels(self):
        values = np.zeros(self.frame_count, dtype=np.int8)
        for start, end in self.intervals:
            values[start:end + 1] = 1
        for start, end in self.ignored:
            values[start:end + 1] = -1
        return values

    def missing_boxes(self):
        return [int(i) for i in np.flatnonzero(self.labels() == 1) if not self.boxes.get(str(i))]

    def validate_annotations(self):
        self.validate(check_disk=True)
        labels = self.labels()
        if any(boxes and labels[int(key)] == 0 for key, boxes in self.boxes.items()):
            raise ValueError('有框的影格不在有物體區段內，請先修正區段。')
        missing = self.missing_boxes()
        if missing:
            raise ValueError(f'{self.name}：有物體區段還有 {len(missing)} 幀未框選，第一個是第 {missing[0] + 1} 幀。')

    def set_boxes(self, index, boxes):
        if not 0 <= index < self.frame_count:
            raise ValueError('影格索引超出範圍。')
        if self.labels()[index] == -1:
            raise ValueError('此幀在略過區段內，請先移除略過區段。')
        previous = self.boxes.get(str(index))
        self.boxes[str(index)] = [list(box) for box in boxes]
        try:
            self.validate()
        except Exception:
            if previous is None:
                self.boxes.pop(str(index), None)
            else:
                self.boxes[str(index)] = previous
            raise
        if boxes and self.labels()[index] == 0:
            self.intervals = merge_intervals([*self.intervals, [index, index]], self.frame_count)

    def seed_boxes(self, template_index, start, end):
        """Copy the current template to empty frames; preserve existing work."""
        merge_intervals([[start, end]], self.frame_count)
        self.validate()
        if type(template_index) is not int or not 0 <= template_index < self.frame_count:
            raise ValueError('模板影格索引無效。')
        template = self.boxes.get(str(template_index))
        if not template or self.labels()[template_index] != 1:
            raise ValueError('請先在目前影格畫好 BBox，再套用到區段。')
        self.intervals = merge_intervals([*self.intervals, [start, end]], self.frame_count)
        labels = self.labels()
        counts = {'filled': 0, 'kept': 0, 'ignored': 0}
        for index in range(start, end + 1):
            if labels[index] == -1:
                counts['ignored'] += 1
            elif self.boxes.get(str(index)):
                counts['kept'] += 1
            else:
                self.boxes[str(index)] = [box[:] for box in template]
                counts['filled'] += 1
        return counts


@dataclass
class SmokeProject:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sources: list = field(default_factory=list)

    def validate(self, check_disk=False):
        if not re.fullmatch(r'[a-f0-9]{32}', self.id):
            raise ValueError('標註專案 ID 無效。')
        seen = set()
        for source in self.sources:
            source.validate(check_disk)
            key = str(Path(source.path).resolve()).casefold()
            if source.id in seen or key in seen:
                raise ValueError('來源 ID 或路徑重複。')
            seen.update((source.id, key))


def save_project(path, project):
    path = Path(path).resolve()
    project.validate()
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8-sig'))
        if existing.get('schema') not in {'boom.astt_bbox.v1', 'boom.astt_bbox.v2', 'boom.astt_bbox.v3', 'st2.annotations.v1'} or existing.get('id') != project.id:
            raise FileExistsError('此檔案屬於其他專案，請使用新的保存名稱。')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    data = {'schema': 'st2.annotations.v1', 'bbox_mode': 'per_frame',
            'frame_policy': 'source_native', 'unmarked_policy': 'normal_outside_intervals', **asdict(project)}
    try:
        temporary.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_project(path):
    data = json.loads(Path(path).resolve().read_text(encoding='utf-8-sig'))
    accepted = {('st2.annotations.v1', 'per_frame', 'normal_outside_intervals'),
                ('boom.astt_bbox.v1', 'manual_per_frame', 'normal_after_source_review'),
                ('boom.astt_bbox.v2', 'per_frame_with_review', 'normal_after_source_review'),
                ('boom.astt_bbox.v3', 'per_frame', 'normal_outside_intervals')}
    if ((data.get('schema'), data.get('bbox_mode'), data.get('unmarked_policy')) not in accepted
            or data.get('frame_policy') != 'source_native'):
        raise ValueError('請選擇 Boom ST2 標註專案，不支援 ASTT 區段標籤或其他 JSON。')
    sources = []
    for row in data['sources']:
        row = dict(row)
        if data['schema'] not in ('boom.astt_bbox.v3', 'st2.annotations.v1'):
            row.pop('reviewed', None)
            row.pop('pending_frames', None)
        sources.append(SmokeSource(**row))
    project = SmokeProject(data['id'], sources)
    project.validate(check_disk=True)
    return project


def sample_targets(source, history_frames):
    numbers = []
    if source.kind == 'frames':
        matches = [re.search(r'(\d+)$', Path(name).stem) for name in source.frame_files]
        if all(matches):
            numbers = [int(match.group(1)) for match in matches]
    labels = source.labels()
    targets = []
    for index in range(history_frames - 1, source.frame_count):
        first = index - history_frames + 1
        if (labels[first:index + 1] == -1).any():
            continue
        if numbers and any(numbers[i + 1] != numbers[i] + 1 for i in range(first, index)):
            continue
        targets.append(index)
    return targets


def plan_split(project, ratios=(7., 3.), seed=2026, history_frames=4):
    if type(history_frames) is not int or history_frames < 1:
        raise ValueError('歷史窗口幀數必須是正整數。')
    if len(ratios) != 2 or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError('請提供正數的 train／val 比例，例如 7:3。')
    project.validate(check_disk=True)
    groups, stats = {}, {}
    for source in project.sources:
        source.validate_annotations()
        labels = source.labels()
        targets = sample_targets(source, history_frames)
        stats[source.id] = {'frames': source.frame_count, 'roi_frames': int((labels == 1).sum()),
                            'ignored_frames': int((labels == -1).sum()), 'samples': len(targets),
                            'positive_samples': int(sum(labels[i] == 1 for i in targets)),
                            'max_boxes': max((len(v) for v in source.boxes.values()), default=0)}
        groups.setdefault(source.origin.strip().casefold(), []).append(source)
    if len(groups) < 2:
        raise ValueError('至少需要兩個獨立來源群組；同一影片的片段請使用相同群組。')
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    val_count = max(1, min(len(keys) - 1, round(len(keys) * ratios[1] / sum(ratios))))
    val, train = keys[:val_count], keys[val_count:]
    def positive(key):
        return any(stats[s.id]['positive_samples'] for s in groups[key])
    if not any(positive(k) for k in train):
        candidate = next((k for k in val if positive(k)), None)
        if candidate is None:
            raise ValueError('沒有可訓練的有框窗口；請確認區段、逐幀 BBox 與歷史幀數。')
        swapped = train[0]
        train[0], val[val.index(candidate)] = candidate, swapped
    if not any(positive(k) for k in val) and sum(positive(k) for k in train) > 1:
        candidate = next(k for k in train if positive(k))
        swapped = val[0]
        val[0], train[train.index(candidate)] = candidate, swapped
    assignments = {s.id: ('train' if key in train else 'val') for key, members in groups.items() for s in members}
    summary = {split: {field: sum(stats[s.id][field] for s in project.sources if assignments[s.id] == split)
                       for field in ('frames', 'roi_frames', 'ignored_frames', 'samples', 'positive_samples')}
               for split in ('train', 'val')}
    if not summary['val']['samples']:
        raise ValueError('val 沒有可用的連續窗口，請調整來源／歷史幀數或略過區段。')
    warnings = []
    if not summary['val']['positive_samples']:
        warnings.append('val 沒有有框窗口，無法評估目標區域的召回或 AP。')
    return {'assignments': assignments, 'source_stats': stats, 'summary': summary,
            'group_counts': {'train': len(train), 'val': len(val)}, 'ratios': list(ratios), 'seed': seed,
            'history_frames': history_frames, 'frame_policy': 'source_native', 'warnings': warnings}


class ExportCancelled(Exception):
    pass


def export_dataset(project, output_parent=DATASET_ROOT, name='st2', ratios=(7., 3.), seed=2026,
                   history_frames=4, progress=None, cancel=None):
    report = plan_split(project, ratios, seed, history_frames)
    parent = Path(output_parent).resolve()
    for source in project.sources:
        source_path = Path(source.path).resolve()
        if source.kind == 'frames' and (parent == source_path or parent.is_relative_to(source_path)):
            raise ValueError('輸出位置不能在來源圖片資料夾內。')
    parent.mkdir(parents=True, exist_ok=True)
    name = safe_name(name)
    version = 1
    while True:
        output = parent / f'{name}_v{version}'
        try:
            output.mkdir(exist_ok=False)
            break
        except FileExistsError:
            version += 1
    assert output.resolve().parent == parent and not output.is_symlink()
    cancel = cancel or threading.Event()
    def cancelled():
        if cancel.is_set():
            raise ExportCancelled('已取消匯出。')
    try:
        save_project(output / 'project.json', project)
        for split in ('train', 'val'):
            (output / split).mkdir()
        for number, source in enumerate(project.sources):
            cancelled()
            split = report['assignments'][source.id]
            folder = output / split / f'{safe_name(source.name)}_{source.id[:8]}'
            images = folder / 'images'
            images.mkdir(parents=True)
            labels = source.labels()
            targets = set(sample_targets(source, history_frames))
            cap = cv2.VideoCapture(source.path) if source.kind == 'video' else None
            filenames = []
            try:
                with (folder / 'frames.jsonl').open('x', encoding='utf-8', newline='\n') as annotations, \
                     (folder / 'samples.jsonl').open('x', encoding='utf-8', newline='\n') as samples, \
                     (folder / 'frames.csv').open('x', encoding='utf-8', newline='') as manifest:
                    writer = csv.writer(manifest)
                    writer.writerow(['filename', 'source_frame_index', 'source_filename', 'time_seconds', 'label'])
                    for index in range(source.frame_count):
                        cancelled()
                        if cap is not None:
                            ok, image = cap.read()
                            suffix, source_filename = '.jpg', ''
                        else:
                            source_filename = source.frame_files[index]
                            image = cv2.imread(str(Path(source.path) / source_filename))
                            ok, suffix = image is not None, Path(source_filename).suffix.lower()
                        if not ok:
                            raise ValueError(f'{source.name}：第 {index + 1} 幀無法讀取，取消不完整版本。')
                        filename = f'images/frame_{index:08d}{suffix}'
                        if cap is not None:
                            if not cv2.imwrite(str(folder / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                                raise OSError('無法寫入匯出圖片。')
                        else:
                            shutil.copy2(Path(source.path) / source_filename, folder / filename)
                        height, width = image.shape[:2]
                        boxes = source.boxes.get(str(index), []) if labels[index] == 1 else []
                        state = 'roi' if labels[index] == 1 else 'normal' if labels[index] == 0 else 'ignore'
                        row = {'frame_index': index, 'time_seconds': index / source.fps, 'image': filename,
                               'width': width, 'height': height, 'state': state, 'roi_xyxy': boxes}
                        annotations.write(json.dumps(row, ensure_ascii=False) + '\n')
                        writer.writerow([filename, index, source_filename, index / source.fps, int(labels[index])])
                        filenames.append(filename)
                        if index in targets:
                            normalized = [[(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1] for x1,y1,x2,y2 in boxes]
                            samples.write(json.dumps({'target_frame': index,
                                'input_images': filenames[index-history_frames+1:index+1],
                                'roi_xyxy': boxes, 'target': {'labels': [0] * len(boxes), 'boxes': normalized}}) + '\n')
                        if progress and (index % 100 == 0 or index + 1 == source.frame_count):
                            progress(f'{number + 1}/{len(project.sources)} {source.name} → {split}：{index + 1}/{source.frame_count} 幀')
                np.save(folder / 'labels.npy', labels, allow_pickle=False)
                source.validate(check_disk=True)
                (folder / 'source.json').write_bytes(json.dumps({'id': source.id, 'source': source.path,
                    'origin': source.origin, 'fps': source.fps, 'kind': source.kind}, ensure_ascii=False, indent=2).encode('utf-8'))
            finally:
                if cap is not None:
                    cap.release()
        cancelled()
        report['output'] = str(output)
        (output / 'split_report.json').write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8'))
        (output / 'README.md').write_bytes(('''# Boom ST2 資料集

時間區段內的框由使用者逐幀標註，或同位置套用後微調；所有區段內的框均可作監督，不設已檢查狀態，略過的框不作監督。原始來源未修改，影片保持原始固定幀率。
labels.npy：1=關注區域、0=區段外的無目標影格、-1=略過。略過影格不進入訓練窗口。
frames.jsonl 的 roi_xyxy 是相對原圖 [0,1] 座標；samples.jsonl 的 target.boxes 是 cxcywh。
框代表使用者關注區域，不代表物體類別；target.labels 固定 0，供單一區域的框監督使用。
ST2 v3 trainer 不使用框資訊；只依整幀 normal 狀態選擇正常訓練窗口。框保留供未來研究。
所有來源依群組分配 train／val，沒有獨立 test。本版本完整保留正常歷史 context。
ST2 訓練入口：python src/st2 train --data <本資料集路徑>。
詳見 doc/ST2.md。不要把此格式當成 ASTT 原版資料。
''').encode('utf-8'))
        return report
    except BaseException:
        assert output.resolve().parent == parent and not output.is_symlink()
        shutil.rmtree(output)
        raise
