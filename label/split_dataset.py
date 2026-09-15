"""Split paired YOLO detection data 70:30, balancing every class."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from uuid import uuid4

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import yaml
import re
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

DATA_ROOT = Path(__file__).resolve().parents[1] / "data/yolo_datasets"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def dataset_roots(source):
    """Accept standard YOLO folders or images/annotations together at the root."""
    if (source / "images").is_dir():
        return source / "images", source / "labels"
    if source.is_dir() and any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in source.iterdir()):
        return source, source
    raise ValueError(f"No images found: expected {source / 'images'} or image files directly in {source}")


def read_dataset(source, min_samples=2, classes_file=None):
    classes_file = Path(classes_file) if classes_file is not None else source / "classes.txt"
    if not classes_file.is_file():
        raise FileNotFoundError(f"Missing classes.txt: {classes_file}; one class name per line, starting with class 0")
    class_names = [line.strip() for line in classes_file.read_text(encoding="utf-8-sig").splitlines()]
    if not class_names or any(not name for name in class_names):
        raise ValueError(f"classes.txt must contain one non-empty class name per line: {classes_file}")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"Duplicate class names in: {classes_file}")
    names = dict(enumerate(class_names))
    images_root, labels_root = dataset_roots(source)
    flat = images_root == source
    samples, missing, used_labels = [], [], set()
    identities, hashes = set(), {}
    duplicate_groups = defaultdict(list)
    for image in sorted(images_root.glob("*") if flat else images_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image.relative_to(images_root)
        label = (labels_root / relative).with_suffix(".txt")
        if not label.is_file():
            missing.append(relative.as_posix())
        identity = relative.with_suffix("").as_posix().casefold()
        if identity in identities:
            raise ValueError(f"Ambiguous image/label stem: {relative}")
        identities.add(identity)
        counts = Counter()
        annotation = label.read_text(encoding="utf-8-sig") if label.is_file() else ""
        for number, line in enumerate(annotation.splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Expected YOLO detection 'class x y w h': {label}:{number}")
            category = int(parts[0])
            box = list(map(float, parts[1:]))
            if category not in names or not all(math.isfinite(v) and 0 <= v <= 1 for v in box) or min(box[2:]) <= 0:
                raise ValueError(f"Invalid class or normalized box: {label}:{number}")
            counts[category] += 1
        # Keep duplicate content together rather than leaking across splits.
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in hashes and hashes[digest] != annotation.strip():
            raise ValueError(f"Identical image has conflicting annotations: {image}")
        hashes[digest] = annotation.strip()
        duplicate_groups[digest].append(len(samples))
        samples.append((relative, counts))
        used_labels.add(label.resolve())
    label_files = labels_root.glob("*.txt") if flat else labels_root.rglob("*.txt")
    metadata = {"classes.txt", "train.txt", "val.txt", "valid.txt", "test.txt"} if flat else set()
    orphan = sorted(str(p.relative_to(labels_root)) for p in label_files
                    if p.name.lower() not in metadata and p.resolve() not in used_labels)
    if orphan:
        raise ValueError(f"Labels without matching images: {orphan[:10]}")
    if len(samples) < min_samples:
        raise ValueError(f"Need at least {min_samples} images in dataset: {source}")
    return names, samples, missing, list(duplicate_groups.values())


def stratify(samples, names, seed, duplicate_groups=None):
    """Keep whole images together; enforce floor/ceil 70% for every metric.

    Group identical class-count vectors to keep the integer problem small.
    Both class-containing image counts and box counts are constrained. If
    co-occurrence prevents the constraints, fail instead of silently relaxing.
    """
    groups = defaultdict(list)
    units = duplicate_groups if duplicate_groups is not None else [[i] for i in range(len(samples))]
    for indices in units:
        counts = samples[indices[0]][1]
        groups[(len(indices), *(counts.get(c, 0) for c in names))].append(indices)
    keys = sorted(groups)
    capacities = np.array([len(groups[k]) for k in keys])
    matrix = [[k[0] for k in keys]]
    metrics = ["images"]
    for c in names:
        matrix.extend([[k[0]*int(k[c+1] > 0) for k in keys], [k[0]*k[c+1] for k in keys]])
        metrics.extend([f"class_{c}_images", f"class_{c}_boxes"])
    matrix.append([k[0]*int(not any(k[1:])) for k in keys])
    metrics.append("background_images")
    a = np.asarray(matrix, dtype=float)
    totals = a @ capacities
    targets = totals*.7
    low, high = np.floor(targets+1e-9), np.ceil(targets-1e-9)
    low[0] = high[0] = (7*len(samples)+5)//10
    # Minimize absolute rounding error, with rare-class metrics equally weighted.
    m, n = a.shape
    zeros = np.zeros((m, m))
    constraint = LinearConstraint(
        np.vstack([np.hstack([a, zeros]), np.hstack([a, -np.eye(m)]), np.hstack([-a, -np.eye(m)])]),
        np.concatenate([low, np.full(2*m, -np.inf)]),
        np.concatenate([high, targets, -targets]),
    )
    result = milp(
        c=np.concatenate([np.zeros(n), 1/np.maximum(totals, 1)]),
        integrality=np.concatenate([np.ones(n), np.zeros(m)]),
        bounds=Bounds(np.zeros(n+m), np.concatenate([capacities, np.full(m, np.inf)])),
        constraints=constraint, options={"time_limit": 60.0},
    )
    if not result.success:
        raise ValueError("Cannot satisfy 70:30 rounding for total and every class (images and boxes). "
                         "Multi-label co-occurrence may conflict, or solver timed out: " + result.message)
    allocation = np.rint(result.x[:n]).astype(int)
    actual = a @ allocation
    if not (np.all(actual >= low) and np.all(actual <= high)):
        raise ValueError("Solver allocation failed ratio validation")
    rng = random.Random(seed)
    train = set()
    for key, count in zip(keys, allocation):
        indices = groups[key].copy()
        rng.shuffle(indices)
        for unit in indices[:count]:
            train.update(unit)
    stats = {name: {"total": int(total), "train": int(value), "val": int(total-value),
                    "train_fraction": float(value/total) if total else None,
                    "exact_7_3": int(total)*7 == int(value)*10}
             for name, total, value in zip(metrics, totals, actual)}
    return train, stats


def version_paths(parent, name):
    """Choose a new version from this dataset family's existing directories."""
    if not name or name in {".", ".."} or any(c in name for c in '<>:"/\\|?*') or name.endswith((".", " ")):
        raise ValueError("請輸入有效的資料集名稱，例如 dataset 或 smoke。")
    parent = Path(parent).resolve()
    if parent.exists() and not parent.is_dir():
        raise ValueError("輸出位置必須是資料夾。")
    pattern = re.compile(re.escape(name) + r"_v(\d+)$", re.IGNORECASE)
    versions = [(int(match[1]), p) for p in parent.iterdir()
                if (match := pattern.fullmatch(p.name))] if parent.exists() else []
    latest = max(versions, key=lambda item: item[0]) if versions else None
    version = latest[0] + 1 if latest else 1
    return parent / f"{name}_v{version}", version, latest[1] if latest else None


def build_dataset(sources, classes_file, output_parent, name="dataset", seed=42,
                  dry_run=False, log=print):
    """Merge the latest version and newly selected sources into the next version."""
    sources = [Path(p).resolve() for p in sources]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("請加入至少一個來源，且不要重複選取同一資料夾。")
    if any(a in b.parents for a in sources for b in sources if a != b):
        raise ValueError("來源資料夾不能互相包含，請選擇各自的資料集根目錄。")
    if not classes_file or not Path(classes_file).is_file():
        raise ValueError("請選擇有效的 classes.txt。")
    parent = Path(output_parent).resolve()
    output, version, previous = version_paths(parent, name)
    if previous:
        if not previous.is_dir():
            raise ValueError(f"上一版路徑不是資料夾，請檢查：{previous}")
        if previous in sources:
            raise ValueError("上一版會自動加入，來源清單請只加入新增資料。")
        sources.insert(0, previous)
        log(f"自動合併上一版：{previous}")
        if any(a in b.parents for a in sources for b in sources if a != b):
            raise ValueError("新增來源不能位於上一版資料集內。")
    else:
        log("尚無舊版本，將建立第一版。")
    if any(parent == s or s in parent.parents for s in sources):
        raise ValueError("輸出位置不能位於來源資料夾內。")

    samples, origins, missing = [], [], []
    hashes, groups = {}, defaultdict(list)
    source_counts = []
    for source in sources:
        log(f"讀取來源：{source}")
        names, entries, absent, _ = read_dataset(source, min_samples=1, classes_file=classes_file)
        if source == previous:
            if (source / "classes.txt").is_file():
                previous_names, _, _, _ = read_dataset(source, min_samples=1)
                if previous_names != names:
                    raise ValueError("選取的 classes.txt 與上一版類別名稱或順序不同，無法直接合併。")
            else:
                log("上一版沒有 classes.txt，使用本次選取的類別檔解讀標註。")
        images_root, labels_root = dataset_roots(source)
        source_counts.append({"path": str(source), "images": len(entries),
                              "role": "previous" if source == previous else "new"})
        log(f"  {'上一版' if source == previous else '新增來源'}：{len(entries)} 張")
        missing.extend(str(images_root / p) for p in absent)
        for relative, counts in entries:
            image = images_root / relative
            label = (labels_root / relative).with_suffix(".txt")
            annotation = label.read_text(encoding="utf-8-sig").strip() if label.is_file() else ""
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != annotation:
                raise ValueError(f"相同圖片的標註不一致：{image}")
            hashes[digest] = annotation
            groups[digest].append(len(samples))
            # Global numbering avoids collisions without per-source subfolders.
            destination = Path(f"image_{len(samples) + 1:08d}{image.suffix.lower()}")
            samples.append((destination, counts))
            origins.append((image, annotation))
    if len(samples) < 2:
        raise ValueError("合併後至少需要 2 張圖片。")
    log(f"共 {len(sources)} 個來源、{len(samples)} 張圖片，計算 7:3 分配中…")
    previous_count = sum(s["images"] for s in source_counts if s["role"] == "previous")
    log(f"上一版 {previous_count} 張｜新增 {len(samples) - previous_count} 張")
    train, stats = stratify(samples, names, seed, list(groups.values()))
    if any(output == s or output in s.parents for s in sources):
        raise ValueError("輸出資料夾不能包含來源。")
    report = {"output": str(output), "version": version, "dataset_name": name,
              "previous_version": str(previous) if previous else None,
              "previous_images": previous_count, "new_images": len(samples) - previous_count,
              "sources": source_counts, "seed": seed, "ratio": "7:3", "names": names,
              "metrics": stats, "filled_empty_labels": missing,
              "duplicate_groups": [[samples[i][0].name for i in group] for group in groups.values() if len(group) > 1],
              "assignments": [{"source_image": str(image), "image": samples[i][0].name,
                               "split": "train" if i in train else "val"}
                              for i, (image, _) in enumerate(origins)]}
    log(f"訓練：{len(train)} 張｜驗證：{len(samples) - len(train)} 張")
    for category, class_name in names.items():
        metric = stats[f"class_{category}_images"]
        log(f"{category} {class_name}：共 {metric['total']} 張，train {metric['train']} / val {metric['val']}")
    log(f"缺少標註、將補空白標註的圖片：{len(missing)} 張")
    log(f"{'預計輸出' if dry_run else '輸出'}：{output}")
    if dry_run:
        return report

    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{output.name}.staging_{uuid4().hex}"
    stage.mkdir(exist_ok=False)
    try:
        for kind in ("images", "labels"):
            for split in ("train", "val"):
                (stage / kind / split).mkdir(parents=True)
        for i, (image, annotation) in enumerate(origins):
            split = "train" if i in train else "val"
            destination = samples[i][0]
            shutil.copy2(image, stage / "images" / split / destination)
            (stage / "labels" / split / destination.with_suffix(".txt")).write_text(annotation + ("\n" if annotation else ""), encoding="utf-8")
            if (i + 1) % 100 == 0:
                log(f"已保存 {i + 1} / {len(samples)} 張")
        (stage / "classes.txt").write_text("\n".join(names.values()) + "\n", encoding="utf-8")
        config = {"path": output.as_posix(), "train": "images/train", "val": "images/val", "names": names}
        (stage / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (stage / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _, verified, absent, _ = read_dataset(stage)
        if len(verified) != len(samples) or absent:
            raise ValueError("輸出配對驗證失敗。")
        # Only promote a completed dataset; existing versions are never replaced.
        if output.exists():
            raise FileExistsError(f"版本資料夾已被建立，請重新執行：{output}")
        if stage.resolve().parent != parent or output.resolve().parent != parent:
            raise ValueError("暫存與版本資料夾必須位於指定輸出位置。")
        stage.rename(output)
    except Exception as exc:
        raise RuntimeError(f"建立失敗，來源保持不變；暫存結果保留於 {stage}\n{exc}") from exc
    log("完成：data.yaml、classes.txt 與 split_report.json 已產生。")
    return report



class SplitDatasetUI:
    def __init__(self, root):
        self.root = root
        self.busy = False
        self.events = queue.Queue()
        self.controls = []
        root.title("Boom｜資料集合併與切分")
        root.geometry("900x790")
        root.minsize(850, 730)
        root.protocol("WM_DELETE_WINDOW", self.close)
        panel = ttk.Frame(root, padding=20)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="資料集合併與切分", font=("Microsoft JhengHei UI", 19, "bold")).pack(anchor="w")
        ttk.Label(panel, text="加入資料集 → 選擇類別與輸出位置 → 預檢或切分。train 70% / val 30%。").pack(anchor="w", pady=(6, 16))

        ttk.Label(panel, text="1. 本次新增資料（可加入多個資料夾，上一版會自動合併）").pack(anchor="w")
        self.sources = tk.Listbox(panel, height=6, selectmode="extended", exportselection=False)
        self.sources.pack(fill="x", pady=6)
        self.controls.append(self.sources)
        buttons = ttk.Frame(panel)
        buttons.pack(fill="x")
        self.button(buttons, "加入資料夾", self.add_source).pack(side="left")
        self.button(buttons, "移除選取", self.remove_sources).pack(side="left", padx=8)
        ttk.Label(panel, text="支援 images/labels 分開，或圖片與標註在同一層。").pack(anchor="w", pady=(6, 14))

        shared = DATA_ROOT / "classes.txt"
        self.classes = tk.StringVar(value=str(shared) if shared.is_file() else "")
        self.path_row(panel, "2. 共用 classes.txt", self.classes, self.choose_classes)
        ttk.Label(panel, text="必選：每行一個類別，從 class 0 開始；所有來源使用這份類別表。").pack(anchor="w", pady=(0, 12))

        self.output_parent = tk.StringVar(value=str(DATA_ROOT))
        self.path_row(panel, "3. 輸出位置", self.output_parent, self.choose_output)
        naming = ttk.Frame(panel)
        naming.pack(fill="x", pady=(0, 8))
        ttk.Label(naming, text="資料集名稱").pack(side="left")
        self.output_name = tk.StringVar(value="dataset")
        entry = ttk.Entry(naming, textvariable=self.output_name, width=30)
        entry.pack(side="left", padx=8)
        self.controls.append(entry)
        ttk.Label(naming, text="隨機種子").pack(side="left", padx=(12, 0))
        self.seed = tk.StringVar(value="42")
        entry = ttk.Entry(naming, textvariable=self.seed, width=9)
        entry.pack(side="left", padx=8)
        self.controls.append(entry)
        ttk.Label(panel, text="合併同名稱的最新版與新增資料，輸出 dataset_v1、dataset_v2…，保留舊版本。").pack(anchor="w")

        self.summary = tk.StringVar(value="預檢後顯示圖片總數、train / val 張數與版本資料夾。")
        ttk.Label(panel, textvariable=self.summary, wraplength=830).pack(anchor="w", pady=(8, 0))
        actions = ttk.Frame(panel)
        actions.pack(fill="x", pady=14)
        self.button(actions, "預檢（不寫入）", lambda: self.start(True)).pack(side="left")
        self.button(actions, "開始切分", lambda: self.start(False)).pack(side="left", padx=8)
        self.status = tk.StringVar(value="等待選擇資料集")
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 8))
        self.log = ScrolledText(panel, height=12, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True)
        root.after(100, self.poll)

    def button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.controls.append(button)
        return button

    def path_row(self, parent, title, variable, command):
        ttk.Label(parent, text=title).pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=6)
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side="left", fill="x", expand=True)
        self.controls.append(entry)
        self.button(row, "瀏覽…", command).pack(side="left", padx=(8, 0))

    def add_source(self):
        selected = filedialog.askdirectory(parent=self.root, title="選擇資料集（可再次加入其他資料夾）", initialdir=str(DATA_ROOT))
        if selected:
            path = str(Path(selected).resolve())
            if path not in self.sources.get(0, "end"):
                self.sources.insert("end", path)

    def remove_sources(self):
        for index in reversed(self.sources.curselection()):
            self.sources.delete(index)

    def choose_classes(self):
        selected = filedialog.askopenfilename(parent=self.root, title="選擇類別檔", initialdir=str(DATA_ROOT), filetypes=[("文字檔", "*.txt")])
        if selected:
            self.classes.set(selected)

    def choose_output(self):
        selected = filedialog.askdirectory(parent=self.root, title="選擇輸出資料夾的上層位置", initialdir=self.output_parent.get())
        if selected:
            self.output_parent.set(selected)

    def build_options(self, dry_run):
        sources = self.sources.get(0, "end")
        if not sources:
            raise ValueError("請至少加入一個新增資料夾。")
        classes = self.classes.get().strip()
        if not classes or not Path(classes).is_file():
            raise ValueError("請選擇 classes.txt。")
        if not self.output_parent.get().strip():
            raise ValueError("請選擇輸出位置。")
        try:
            seed = int(self.seed.get())
        except ValueError:
            raise ValueError("隨機種子必須是整數。") from None
        name = self.output_name.get().strip()
        version_paths(self.output_parent.get(), name)
        return dict(sources=sources, classes_file=classes,
                    output_parent=self.output_parent.get(), name=name, seed=seed,
                    dry_run=dry_run)

    def start(self, dry_run):
        if self.busy:
            return
        try:
            options = self.build_options(dry_run)
        except ValueError as exc:
            messagebox.showerror("請檢查設定", str(exc), parent=self.root)
            return
        self.busy = True
        for widget in self.controls:
            widget.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.status.set("預檢中…" if dry_run else "切分中…")
        self.progress.start()
        threading.Thread(target=self.worker, args=(options,), daemon=True).start()

    def worker(self, options):
        try:
            report = build_dataset(**options, log=lambda line: self.events.put(("log", line + "\n")))
            self.events.put(("report", report))
            code = 0
        except Exception as exc:
            self.events.put(("log", f"執行失敗：{exc}\n"))
            code = 1
        self.events.put(("done", (code, options["dry_run"])))

    def poll(self):
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind == "log":
                self.log.configure(state="normal")
                self.log.insert("end", value)
                self.log.see("end")
                self.log.configure(state="disabled")
            elif kind == "report":
                metrics = value["metrics"]["images"]
                self.summary.set(f"上一版 {value['previous_images']} 張＋新增 {value['new_images']} 張＝{metrics['total']} 張｜train {metrics['train']} / val {metrics['val']}\n{value['output']}")
            else:
                code, dry_run = value
                self.busy = False
                self.progress.stop()
                for widget in self.controls:
                    widget.configure(state="normal")
                if code == 0 and not dry_run:
                    self.sources.delete(0, "end")
                self.status.set(("預檢通過，尚未寫入" if dry_run else "切分完成") if code == 0 else "執行失敗，請查看下方紀錄")
        self.root.after(100, self.poll)

    def close(self):
        if self.busy:
            messagebox.showinfo("處理中", "請等待處理完成後再關閉視窗。", parent=self.root)
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    SplitDatasetUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
