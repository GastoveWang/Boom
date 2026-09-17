"""
==============================================================================
Boom 離線執行期 - 輸出目錄解析與座標日誌寫入 (output.py)
==============================================================================

【檔案定位】
本檔案負責離線處理結果的持久化儲存，包含建立非覆蓋式編號輸出目錄、
初始化 OpenCV 視訊編碼器、以及輸出包含經緯度與像素位置的事件座標文字報表。

【核心功能】
1. 非覆蓋目錄命名 (`resolve_output_paths`)：
   - 遵循 `<video_stem>_<method>_<nn>` 命名格式（例如 `DJI_001_V_pidnet_01`），
     自動遞增序號，絕對不覆蓋使用者既有實驗與推論結果。
2. 視訊寫入器包裝 (`VideoOutput`)：
   - 延遲初始化 `cv2.VideoWriter`（以首格畫面動態獲取寬高），支援 mp4v 編碼。
3. 座標日誌輸出 (`write_coordinate_log`)：
   - 格式化輸出各事件的起煙影格、時間戳記、確認延遲、像素座標、WGS84 經緯度、
     TWD97 座標與定位品質狀態。

【核心類別與函式清單】
- `resolve_output_paths(...)`: 自動生成專屬輸出子目錄與目標路徑。
- `VideoOutput`: 視訊編碼輸出器，支援 context 安全關閉。
- `write_coordinate_log(...)`: 輸出事件座標文字檔。
==============================================================================
"""
from __future__ import annotations
import cv2

from boom.config.defaults import OUTPUT_ROOT
from boom.core.events import LoggedEvent
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
import math


def resolve_output_paths(
    video_stem: str,
    detector_name: str,
    output_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Return paths in a method-labelled folder without overwriting prior runs."""
    root = OUTPUT_ROOT if output_root is None else output_root
    method_names = {
        "pidnet": "pidnet",
        "motion": "optical_flow",
        "fusion": "optical_pidnet",
    }
    method_name = method_names.get(detector_name, detector_name)
    folder_stem = f"{video_stem}_{method_name}"
    output_dir = root / folder_stem
    sequence = 2
    while output_dir.exists():
        output_dir = root / f"{folder_stem}_{sequence:02d}"
        sequence += 1
    output_dir.mkdir(parents=True, exist_ok=False)

    return (
        output_dir / f"output_{video_stem}.mp4",
        output_dir / f"{video_stem}_impact_coordinates.txt",
    )


def write_coordinate_log(log_path: Path, events: List[LoggedEvent]) -> Path:
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            if not math.isfinite(event.lon) or not math.isfinite(event.lat):
                fh.write(f"ID {event.event_id}: t={event.timestamp_sec:.2f}s, "
                         f"confidence={event.confidence:.3f}, coordinate=unavailable, "
                         f"impact_px={event.impact_point}\n")
                continue
            fh.write(
                f"ID {event.event_id}: "
                f"t={event.timestamp_sec:.2f}s, "
                f"confidence={event.confidence:.3f}, "
                f"coordinate_status={event.coordinate_status}, "
                f"confirmation_delay={event.confirmation_delay_sec:.3f}s, "
                f"impact_px=({event.impact_point[0]:.1f}, {event.impact_point[1]:.1f}), "
                f"TWD97(E={event.easting:.2f}, N={event.northing:.2f}), "
                f"WGS84(lat={event.lat:.6f}, lon={event.lon:.6f})\n"
            )
    return log_path


class VideoOutput:
    """Encode output after the composed frame dimensions become known."""

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.writer = None

    def write(self, frame):
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot write output video: {self.path}")
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
