"""
==============================================================================
Boom 離線執行期 - 終端進度報告與結果統計 (progress.py)
==============================================================================

【檔案定位】
本檔案負責離線視訊處理過程中的命令列進度呈現與最終辨識統計摘要報告。

【核心功能】
1. 實時進度回報 (`report_progress`)：
   - 定期印出處理影格數、百分比、目前 FPS 與剩餘預估時間 (ETA)。
2. 結算統計報告 (`report_result`)：
   - 彙總處理總影格、花費時間、平均處理 FPS、偵測到的煙霧事件數量清單、
     輸出影片路徑與座標文字檔儲存位置。
==============================================================================
"""
from __future__ import annotations





def _format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


import time


def report_progress(processed, frame_idx, target_frames, fps, started_at):
    elapsed = max(time.monotonic() - started_at, 1e-6)
    processing_fps = processed / elapsed
    progress = processed / target_frames if target_frames else 0.0
    remaining = (
        (target_frames - processed) / processing_fps
        if target_frames and processing_fps > 0 else 0.0
    )
    print(
        f"[PROGRESS] {processed}/{target_frames or '?'} frames "
        f"({progress * 100:6.2f}%) "
        f"| video={_format_duration(frame_idx / fps)} "
        f"| speed={processing_fps:.2f} fps "
        f"| elapsed={_format_duration(elapsed)} "
        f"| ETA={_format_duration(remaining)}",
        flush=True,
    )


def report_result(processed, history, detector, image_path, video_path, geo, localizer, output_video_path, log_path, detector_name):
    print(f"[INFO] Processed frames: {processed}")
    print(f"[INFO] Confirmed events: {len(history.all_events)}")
    guard_summary = detector.camera_guard_summary()
    if guard_summary:
        summary_text = ", ".join(
            f"{reason}={count}" for reason, count in sorted(guard_summary.items()))
        print(f"[INFO] Camera guard skipped frames: {summary_text}")
    print(f"[INFO] Image metadata source: {image_path or 'configured defaults'}")
    if localizer:
        print(f"[INFO] GPS photo anchor: lon={localizer.photo.lon:.7f}, lat={localizer.photo.lat:.7f}")
        print(f"[INFO] Localization status: {localizer.status}; ground-plane estimates")
    else:
        print(
            f"[INFO] Geo reference: lon={geo.lon:.7f}, lat={geo.lat:.7f}, "
            f"alt={geo.alt:.3f}, yaw={geo.yaw:.3f}, pitch={geo.pitch:.3f}, roll={geo.roll:.3f}"
        )
    print(f"[INFO] Input video: {video_path}")
    print(f"[INFO] Detector: {detector_name}")
    print(f"[INFO] Output video: {output_video_path}")
    print(f"[INFO] Coordinate log: {log_path}")
