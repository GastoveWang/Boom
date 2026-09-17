"""
==============================================================================
Boom 離線執行期 - 輸入媒體解析與視訊擷取生命週期 (inputs.py)
==============================================================================

【檔案定位】
本檔案隸屬於 `artillery.offline.src.runtime` 執行期模組，負責在管線啟動前
解析無人機輸入影片與參考照片，並提供安全的 OpenCV 視訊擷取物件包裝。

【核心功能】
1. 視訊與照片智慧匹配：
   - 支援無人機命名慣例（如 DJI_001_V.MP4 與 DJI_001_P.JPG）自動關聯。
   - 若未指定 `--reference-image`，自動在影片同目錄下尋找成對照片。
   - 若無照片且未明確要求地圖，自動降級為純偵測模式（--no-map），避免拋出異常。
2. 視訊擷取資源管理 (`VideoSource`)：
   - 管理 OpenCV `cv2.VideoCapture` 的開啟、跳轉起始影格、影格率偵測與安全關閉釋放。

【核心類別與函式清單】
- `resolve_inputs(args)`: 解析命令列參數中的輸入媒體，回傳 `(image_path, video_path, geo)`。
- `resolve_video_path(image_path)`: 依據照片檔名推導並尋找對應的影片檔。
- `resolve_paired_image(video_path)`: 依據影片檔名自動尋找同目錄對應的參考照片檔。
- `_default_video_path()`: 若未指定影片，自動在 `data/` 下搜尋第一個可用的影片。
- `VideoSource`: 視訊擷取來源包裝類別，支援 context cleanup 與影格跳轉。

【相依模組】
- 上游：被 `artillery.offline.src.runtime.runner` 於執行起點調用。
- 下游：依賴 `artillery.common.defaults`、`artillery.common.events.GeoReference`、`reference.py`。
==============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import cv2

from boom.config.defaults import IMAGE_PATH, PROJECT_ROOT, VIDEO_PATH
from boom.core.events import GeoReference
from .reference import read_geo_reference_from_image


def resolve_video_path(image_path: Path) -> Path:
    """依據照片檔名（例如 DJI_001_P.JPG）自動推斷尋找同目錄下的對應影片（DJI_001_V.MP4）。"""
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


def resolve_paired_image(video_path: Path) -> Optional[Path]:
    """依據影片檔名（例如 DJI_001_V.MP4）嘗試尋找同目錄下成對的 DJI 照片（DJI_001_P.JPG）。"""
    video_stem = video_path.stem
    if video_stem.endswith("_V"):
        image_stem = f"{video_stem[:-2]}_P"
        for suffix in (".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"):
            candidate = video_path.with_name(f"{image_stem}{suffix}")
            if candidate.exists():
                return candidate
    return None


def _default_video_path() -> Optional[Path]:
    """在 data/ 目錄下自動搜尋第一個存在的影片檔案。"""
    data_dir = PROJECT_ROOT / "data"
    for pattern in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"):
        matches = sorted(data_dir.rglob(pattern)) if data_dir.exists() else []
        if matches:
            return matches[0]
    return None


def resolve_inputs(args):
    """解析輸入參數中的影片與參考照片，提供智慧成對與安全回退。"""
    video_path = args.video or (Path(VIDEO_PATH) if VIDEO_PATH else None)
    image_path = args.reference_image or (Path(IMAGE_PATH) if IMAGE_PATH else None)

    # 若未指定影片但有指定照片，推導影片
    if video_path is None and image_path is not None:
        video_path = resolve_video_path(image_path)
    # 若仍無影片，搜尋預設路徑
    if video_path is None:
        video_path = _default_video_path()
    if video_path is None:
        raise FileNotFoundError("No input video found. Use --video or place a video under data/.")
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    # 若未指定照片，嘗試自動配對成對照片
    if image_path is None:
        auto_img = resolve_paired_image(video_path)
        if auto_img is not None:
            image_path = auto_img
            print(f"[INFO] 自動配對同目錄無人機參考照片: {image_path.name}")

    # 若啟用了地圖模式但找不到參考照片，自動降級為純偵測無地圖模式 (--no-map)，避免終止程式
    if args.map_dir and image_path is None:
        print("[INFO] 未指定或找不到參考照片；自動切換至純偵測無地圖模式 (--no-map)。")
        args.map_dir = None

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

    return image_path, video_path, geo


class VideoSource:
    """OpenCV 視訊擷取生命週期包裝，確保在錯誤或提早結束時正確關閉串流。"""

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
