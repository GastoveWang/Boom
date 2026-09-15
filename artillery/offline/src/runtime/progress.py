"""Offline video progress formatting."""
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
