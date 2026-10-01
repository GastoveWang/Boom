# 檔案功能：離線影格處理、偵測、定位與輸出主迴圈。
# 執行設定：doc/RUN_SETTINGS.md；非入口模組由對應 runner 傳入設定。
"""
==============================================================================
Boom 離線執行期 - 主管線排程與生命週期調度 (runner.py)
==============================================================================

【檔案定位】
本檔案為 Boom 離線處理的核心引擎。協調無人機輸入讀取、相機姿態追蹤與地圖配準、
演算法推論、事件歷史生命週期管理、畫面合成以及結果錄製寫入。

【核心工作流程】
1. 資源初始化：解析視訊來源、建立偵測器與輸出目錄、初始化 MapLocalizer。
2. 逐格循序處理迴圈：
   - 來源擷取 (`source.read()`)
   - 當格姿態與視覺里程計計算 (`localizer.process()`)
   - 煙霧演算法推論 (`detector.process_frame()`)
   - 事件確認取出與地面座標綁定 (`localize_event()`)
   - 事件逾期清理 (`history.expire()`)
   - 多面板畫面合成 (`compose_frame()`)
   - 編碼寫入與視窗互動 (`output.write()`, `cv2.imshow()`)
3. 輸出結算：寫入地面座標 log、列印最終辨識統計報表。

【相依模組】
- 上游：被 `pipelines/main.py` 與 `run_offline.py` 調用。
- 下游：調用 `inputs`, `events`, `output`, `progress`, `localization.MapLocalizer`, `ui.composition`。
==============================================================================
"""
from contextlib import ExitStack
import time

import cv2

from boom.detection import create_detector
from .async_output import AsyncVideoOutput
from .live_preview import run_with_preview
from .events import EventHistory, localize_event
from .inputs import VideoSource, resolve_inputs
from .output import VideoOutput, resolve_output_paths, write_coordinate_log
from .progress import _format_duration, report_progress, report_result
from boom.ui.composition import compose_frame


def run(args):
    return run_with_preview(lambda preview: _run(args, preview),
                            enabled=args.display,
                            max_width=getattr(args, "display_width", 1280))


def _run(args, preview):
    image_path, video_path, geo = resolve_inputs(args)
    with ExitStack() as resources:
        source = VideoSource(video_path, args.start_frame)
        resources.callback(source.close)
        detector = create_detector(args, source.fps)
        output_video_path, coordinate_log_path = resolve_output_paths(video_path.stem, args.detector)
        output = VideoOutput(output_video_path, detector.fps)
        if getattr(args, "writer_queue", 2) > 0:
            output = AsyncVideoOutput(output, getattr(args, "writer_queue", 2))
        resources.callback(output.close)

        localizer = None
        if args.map_dir:
            from boom.localization import MapLocalizer
            from boom.pipelines.cli import map_matching_config
            localizer = MapLocalizer(args.map_dir, image_path, output_video_path.parent,
                                     args.drone_icon, args.map_mpp, args.ground_height, args.allow_gps_seed,
                                     matching_config=map_matching_config(args))
            resources.callback(localizer.close)
        history = EventHistory(detector.fps, args.panel_hold_sec, args.box_hold_sec)
        target_frames = max(source.frame_count-args.start_frame, 0) if source.frame_count else 0
        if args.max_frames > 0:
            target_frames = min(target_frames, args.max_frames) if target_frames else args.max_frames
        frame_idx, processed = args.start_frame, 0
        started_at = time.monotonic()
        print("[INFO] Display: " + ("enabled" if args.display else "disabled (headless)"))
        if target_frames:
            print(f"[INFO] Workload: {target_frames} frames ({_format_duration(target_frames/detector.fps)} of video)")

        while not preview.stop.is_set():
            for flags in preview.take_scroll():
                if localizer is not None:
                    factor = 0.85 if flags > 0 else 1.18
                    localizer.map_span_m = min(6000.0, max(150.0, localizer.map_span_m * factor))
            ok, frame = source.read()
            if not ok:
                break
            if localizer:
                localizer.process(frame, frame_idx, detector.fps)
            visualization = detector.process_frame(frame, frame_idx)
            for event in detector.consume_pending_confirmations():
                history.append(localize_event(event, frame.shape, visualization.shape, detector.fps, geo, localizer))
            history.expire(frame_idx)
            combined = compose_frame(visualization, history, video_path.name, localizer)
            output.write(combined)
            if localizer and processed == 0:
                cv2.imwrite(str(output_video_path.parent / "ui_preview.jpg"), combined)
            preview.publish(combined)

            frame_idx += 1
            processed += 1
            if processed % max(args.report_every, 1) == 0 or (target_frames and processed >= target_frames):
                report_progress(processed, frame_idx, target_frames, detector.fps, started_at)
            if args.max_frames > 0 and processed >= args.max_frames:
                break

    elapsed = max(time.monotonic() - started_at, 1e-6)
    print(f"[INFO] Processing including output drain: {processed / elapsed:.2f} fps ({elapsed:.2f}s)")
    log_path = write_coordinate_log(coordinate_log_path, history.all_events)
    report_result(processed, history, detector, image_path, video_path, geo,
                  localizer, output_video_path, log_path, args.detector)
