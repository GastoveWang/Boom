"""Split paired YOLO detection data 70:30, balancing every class."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from datetime import datetime
from uuid import uuid4

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
import yaml

DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "data/yolo_datasets/detect_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_dataset(source):
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8-sig"))
    names = config["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs in data.yaml must be contiguous from zero")
    names = dict(sorted(names.items()))
    images_root, labels_root = source / "images", source / "labels"
    samples, missing, used_labels = [], [], set()
    identities, hashes = set(), {}
    duplicate_groups = defaultdict(list)
    for image in sorted(images_root.rglob("*")):
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
    orphan = sorted(str(p.relative_to(labels_root)) for p in labels_root.rglob("*.txt") if p.resolve() not in used_labels)
    if orphan:
        raise ValueError(f"Labels without matching images: {orphan[:10]}")
    if len(samples) < 2:
        raise ValueError("Need at least two paired images/labels")
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


def run(source, output, seed=42, dry_run=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be separate from the source dataset")
    if output.exists() and not dry_run:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    names, samples, missing, duplicate_groups = read_dataset(source)
    train, stats = stratify(samples, names, seed, duplicate_groups)
    # Remove old split prefixes so repeated runs do not nest train/train/...
    destinations = []
    seen = set()
    for relative, _ in samples:
        parts = list(relative.parts)
        while len(parts) > 1 and parts[0] in {"train", "val", "valid", "test"}:
            parts.pop(0)
        relative_out = Path(*parts)
        key = relative_out.with_suffix("").as_posix().casefold()
        if key in seen:
            raise ValueError(f"Conflicting filenames after removing split prefixes: {relative_out}")
        seen.add(key)
        destinations.append(relative_out)
    report = {"source": str(source), "output": str(output), "seed": seed,
              "ratio": "7:3", "names": names, "metrics": stats,
              "filled_empty_labels": missing,
              "missing_label_policy": "background: create empty label in output only",
              "duplicate_groups": [[samples[i][0].as_posix() for i in group] for group in duplicate_groups if len(group)>1],
              "rounding": "nearest total; floor/ceil 70% for each class's image and box counts",
              "assignments": [{"image": p.as_posix(), "split": "train" if i in train else "val"}
                              for i, (p, _) in enumerate(samples)]}
    print(f"Images: {len(samples)}; train: {len(train)}; val: {len(samples)-len(train)}; empty labels to create: {len(missing)}")
    print(f"Background: {stats['background_images']}")
    for c, name in names.items():
        print(f"{c} {name}: images {stats[f'class_{c}_images']}; boxes {stats[f'class_{c}_boxes']}")
    if dry_run:
        return report
    output.mkdir(parents=True, exist_ok=False)
    for i, (relative, _) in enumerate(samples):
        split = "train" if i in train else "val"
        for kind, rel in (("images", relative), ("labels", relative.with_suffix(".txt"))):
            relative_out = destinations[i] if kind == "images" else destinations[i].with_suffix(".txt")
            dest = output / kind / split / relative_out
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = source / kind / rel
            if kind == "labels" and not src.is_file():
                dest.write_text("", encoding="utf-8")
            else:
                shutil.copy2(src, dest)
    config = {"path": output.as_posix(), "train": "images/train", "val": "images/val", "names": names}
    (output / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output / 'data.yaml'}")
    return report


def run_in_place(source, seed=42, dry_run=False):
    """Build in a sibling directory, then swap with rollback and retain backup."""
    source = Path(source).resolve()
    if not (source / "data.yaml").is_file() or not (source / "images").is_dir():
        raise ValueError(f"Not a dataset directory: {source}")
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    stage = source.with_name(source.name + ".staging_" + suffix)
    backup = source.with_name(source.name + ".backup_" + suffix)
    # Verify absolute rename targets stay next to the explicitly selected dataset.
    if any(p.resolve().parent != source.parent or p.exists() for p in (stage, backup)):
        raise ValueError("Unsafe or occupied staging/backup path")
    report = run(source, stage, seed, dry_run)
    report.update(output=str(source), backup=str(backup), mode="in_place")
    if dry_run:
        print(f"Would replace {source} after backing up to {backup}")
        return report
    # Preserve auxiliary user files; split-specific lists are replaced by YAML paths.
    replaced = {"images", "labels", "data.yaml", "train.txt", "val.txt", "test.txt", "split_report.json"}
    for item in source.iterdir():
        if item.name in replaced:
            continue
        if item.is_dir():
            shutil.copytree(item, stage/item.name)
        else:
            shutil.copy2(item, stage/item.name)
    _, staged_samples, missing, _ = read_dataset(stage)
    if len(staged_samples) != report["metrics"]["images"]["total"] or missing:
        raise ValueError(f"Staged dataset validation failed; source unchanged. Inspect {stage}")
    config = yaml.safe_load((stage/"data.yaml").read_text(encoding="utf-8"))
    config["path"] = source.as_posix()
    (stage/"data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (stage/"split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    source.rename(backup)
    try:
        stage.rename(source)
    except OSError:
        backup.rename(source)
        raise
    print(f"Updated: {source}\nBackup: {backup}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, help="Dataset folder; updated in place by default")
    parser.add_argument("--source", type=Path, help="Alternative to positional folder")
    parser.add_argument("--output", type=Path, help="Optional separate output; disables in-place update")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate/count without writing files")
    args = parser.parse_args()
    if args.folder and args.source:
        parser.error("Use either folder or --source, not both")
    source = args.folder or args.source
    if source is None:
        parser.error("Provide the dataset folder to update")
    if args.output and args.output.resolve() != source.resolve():
        run(source, args.output, args.seed, args.dry_run)
    else:
        run_in_place(source, args.seed, args.dry_run)


if __name__ == "__main__":
    main()
