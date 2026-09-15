"""Offline media discovery and input resolution."""
from __future__ import annotations

from artillery.common.defaults import PROJECT_ROOT
from artillery.common.defaults import VIDEO_PATH
from pathlib import Path
from typing import Optional
import cv2
from artillery.common.defaults import IMAGE_PATH
from artillery.common.events import GeoReference
from .reference import read_geo_reference_from_image


def resolve_video_path(image_path: Path) -> Path:
    if VIDEO_PATH:
        return Path(VIDEO_PATH)

    image_stem = image_path.stem
    if not image_stem.endswith("_P"):
        raise ValueError(
            f"Image filename must end with '_P': {image_path.name}")

    video_stem = f"{image_stem[:-2]}_V"
    for suffix in (".MP4", ".mp4", ".MOV", ".mov", ".AVI", ".avi"):
        candidate = image_path.with_name(f"{video_stem}{suffix}")
        if candidate.exists():
            return candidate

    fuzzy_candidates = sorted(image_path.parent.glob(f"{video_stem}.*"))
    if fuzzy_candidates:
        return fuzzy_candidates[0]

    raise FileNotFoundError(
        f"Could not find matching video for {image_path.name}. Expected something like {video_stem}.MP4"
    )


def _default_video_path() -> Optional[Path]:
    data_dir = PROJECT_ROOT / "data"
    for pattern in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"):
        matches = sorted(data_dir.rglob(pattern)) if data_dir.exists() else []
        if matches:
            return matches[0]
    return None


def resolve_inputs(args):
    image_path = args.reference_image or (Path(IMAGE_PATH) if IMAGE_PATH else None)
    if args.map_dir and (image_path is None or args.map_mpp <= 0):
        raise ValueError("Map localization requires --reference-image and positive --map-mpp; use --no-map to disable localization.")
    if image_path is not None and not image_path.exists():
        raise FileNotFoundError(f"Reference image does not exist: {image_path}")
    if image_path is not None and not args.map_dir:
        geo = read_geo_reference_from_image(image_path)
    else:
        geo = GeoReference()
        if not args.map_dir:
            print("[WARN] No reference image supplied; using default geo reference.")

    video_path = args.video or (Path(VIDEO_PATH) if VIDEO_PATH else None)
    if video_path is None and image_path is not None:
        video_path = resolve_video_path(image_path)
    if video_path is None:
        video_path = _default_video_path()
    if video_path is None:
        raise FileNotFoundError("No input video found. Use --video or place a video under data/.")
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    return image_path, video_path, geo


class VideoSource:
    """Capture lifecycle, including cleanup when downstream startup fails."""

    def __init__(self, path, start_frame=0):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Failed to open video: {path}")
        if start_frame > 0:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self):
        return self.capture.read()

    def close(self):
        self.capture.release()
