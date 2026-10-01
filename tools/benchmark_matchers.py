# 檔案功能：同影像對比較 EDM 與 SuperPoint/LightGlue 的匹配表現。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""Compare Boom's SuperPoint+LightGlue and EDM on the same image pair.

Run from the repository root after installing Boom and the optional runtimes.
This reports matching and geometric-consistency metrics, not ground-truth pose
accuracy. Both backends receive the same 640x480 grayscale inputs.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from boom.config.defaults import MapMatchingConfig
from boom.localization.feature_matching import create_feature_backend


def run(backend, first, second, warmup, repeats, ransac_threshold):
    def synchronize():
        torch = getattr(backend, "torch", None)
        if torch is not None and backend.device == "cuda":
            torch.cuda.synchronize()

    def match_once():
        a = backend.extract(first)
        b = backend.extract(second)
        return backend.match(a, b)

    for _ in range(warmup):
        match_once()
    synchronize()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        points0, points1, scores = match_once()
        synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    count = len(points0)
    inliers = np.zeros(count, bool)
    error = None
    if count >= 4:
        homography, selection = cv2.findHomography(points0, points1, cv2.RANSAC,
                                                  ransac_threshold)
        if homography is not None and selection is not None:
            inliers = selection.ravel().astype(bool)
            if inliers.any():
                projected = cv2.perspectiveTransform(
                    points0[inliers].reshape(-1, 1, 2), homography).reshape(-1, 2)
                error = float(np.sqrt(np.mean(np.sum(
                    (projected - points1[inliers])**2, axis=1))))
    return {"matches": count, "inliers": int(inliers.sum()),
            "inlier_ratio": float(inliers.mean()) if count else 0.0,
            "inlier_reprojection_rmse_px": error,
            "median_match_confidence": float(np.median(scores)) if scores is not None and len(scores) else None,
            "median_ms": float(np.median(durations)), "mean_ms": float(np.mean(durations))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image0", type=Path)
    parser.add_argument("image1", type=Path)
    parser.add_argument("--edm-model-path", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ransac-threshold-px", type=float, default=3.0)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.ransac_threshold_px <= 0:
        parser.error("warmup must be nonnegative; repeats and RANSAC threshold must be positive")
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
              for path in (args.image0, args.image1)]
    if any(image is None for image in images):
        parser.error("both image paths must point to readable images")
    first, second = (cv2.resize(image, (640, 480)) for image in images)
    config = MapMatchingConfig(device=args.device, edm_model_path=args.edm_model_path)
    results = {}
    for name in ("superpoint_lightglue", "edm"):
        backend = create_feature_backend(replace(config, matcher_backend=name))
        results[name] = run(backend, first, second, args.warmup, args.repeats,
                            args.ransac_threshold_px)
    print(json.dumps({"input_size": [640, 480], "device": args.device,
                      "warmup": args.warmup, "repeats": args.repeats,
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
