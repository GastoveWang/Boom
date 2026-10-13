"""Persist feature correspondences and report their geometric quality."""
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .geometry import transform


def match_directory(output, stage):
    if output is None:
        return None
    Path(output).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=stage + "_", dir=output))


def render_matches(image, reference, a, b, inliers):
    """Grayscale comparison; green inliers, red outliers/unverified points."""
    height = min(720, image.shape[0], reference.shape[0])
    def gray_bgr(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        width = max(1, round(gray.shape[1] * height / gray.shape[0]))
        scaled = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        scale = np.array([width / gray.shape[1], height / gray.shape[0]])
        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR), scale

    (left, left_scale), (right, right_scale) = gray_bgr(image), gray_bgr(reference)
    canvas = np.concatenate((left, right), axis=1)
    for index, (p, q) in enumerate(zip(a, b)):
        if not np.isfinite([*p, *q]).all():
            continue
        p = (p + .5) * left_scale - .5
        q = (q + .5) * right_scale - .5
        start = tuple(np.round(p).astype(int))
        end = (int(round(q[0])) + left.shape[1], int(round(q[1])))
        color = (0, 255, 0) if inliers[index] else (0, 0, 255)
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
    return canvas


def geometric_quality(a, b, h, selected, shape):
    count = int(selected.sum())
    error, coverage = None, 0.0
    if h is not None and count:
        residual = transform(a[selected], h) - b[selected]
        value = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        error = value if np.isfinite(value) else None
        if count >= 3:
            coverage = float(cv2.contourArea(cv2.convexHull(a[selected].astype(np.float32)))
                             / (shape[0] * shape[1]))
    return {"matches": len(a), "inliers": count,
            "inlier_ratio": count / len(a) if len(a) else 0.0,
            "reprojection_error_px": error, "coverage": coverage}


def write_match(directory, stage, image, reference, a, b, selected, quality,
                *, name="matches", **metadata):
    data = {"stage": stage, **metadata, **quality,
            "line_colors": {"green": "RANSAC inlier", "red": "RANSAC outlier or unverified match"}}
    if directory is not None:
        path = Path(directory) / (name + ".jpg")
        if not cv2.imwrite(str(path), render_matches(image, reference, a, b, selected)):
            raise OSError(f"Cannot write matching result: {path}")
        data["image"] = str(path)
        (Path(directory) / (name + ".json")).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    details = []
    for key in ("frame", "time_sec", "matcher_backend", "matches", "inliers", "inlier_ratio",
                "reprojection_error_px", "coverage", "confidence", "mean_match_confidence",
                "elapsed_ms", "accepted", "reason", "center_gps", "prior_distance_m"):
        if key in data:
            value = data[key]
            display = f"{value:.4f}" if isinstance(value, float) else str(value)
            details.append(f"{key}={display}")
    print(f"[MATCH] {stage}: " + ", ".join(details), flush=True)
    if metadata.get("filepath"):
        print(f"[MATCH] Map: {metadata['filepath']}", flush=True)
    if data.get("image"):
        print(f"[MATCH] Image: {data['image']}", flush=True)
    return data
