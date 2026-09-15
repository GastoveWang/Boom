"""Offline orchestration: source -> localization -> detector -> events -> UI/output."""
from contextlib import ExitStack
import time

import cv2

from artillery.common.detector_factory import create_detector
from .events import EventHistory, localize_event
from .inputs import VideoSource, resolve_inputs
from .output import VideoOutput, resolve_output_paths, write_coordinate_log
from .progress import _format_duration, report_progress, report_result
from ..ui.composition import compose_frame


def run(args):
    image_path, video_path, geo = resolve_inputs(args)
    with ExitStack() as resources:
        source = VideoSource(video_path, args.start_frame)
        resources.callback(source.close)
        detector = create_detector(args, source.fps)
        output_video_path, coordinate_log_path = resolve_output_paths(video_path.stem, args.detector)
        output = VideoOutput(output_video_path, detector.fps)
        resources.callback(output.close)

        localizer = None
        if args.map_dir:
            from artillery.offline.src.localization import MapLocalizer
            localizer = MapLocalizer(args.map_dir, image_path, output_video_path.parent,
                                     args.drone_icon, args.map_mpp, args.ground_height, args.allow_gps_seed)
            resources.callback(localizer.close)
        window_name = "Boom - Wilderness smoke detection"
        if args.display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            def _on_mouse(event, x, y, flags, param):
                if localizer is None:
                    return
                if event == cv2.EVENT_MOUSEWHEEL:
                    if flags > 0:
                        localizer.map_span_m = max(150.0, localizer.map_span_m * 0.85)
                    else:
                        localizer.map_span_m = min(6000.0, localizer.map_span_m * 1.18)

            cv2.setMouseCallback(window_name, _on_mouse)
            resources.callback(cv2.destroyAllWindows)

        history = EventHistory(detector.fps, args.panel_hold_sec, args.box_hold_sec)
        target_frames = max(source.frame_count-args.start_frame, 0) if source.frame_count else 0
        if args.max_frames > 0:
            target_frames = min(target_frames, args.max_frames) if target_frames else args.max_frames
        frame_idx, processed = args.start_frame, 0
        started_at = time.monotonic()
        print("[INFO] Display: " + ("enabled" if args.display else "disabled (headless)"))
        if target_frames:
            print(f"[INFO] Workload: {target_frames} frames ({_format_duration(target_frames/detector.fps)} of video)")

        while True:
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
            if args.display:
                cv2.imshow(window_name, combined)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            frame_idx += 1
            processed += 1
            if processed % max(args.report_every, 1) == 0 or (target_frames and processed >= target_frames):
                report_progress(processed, frame_idx, target_frames, detector.fps, started_at)
            if args.max_frames > 0 and processed >= args.max_frames:
                break

    log_path = write_coordinate_log(coordinate_log_path, history.all_events)
    report_result(processed, history, detector, image_path, video_path, geo,
                  localizer, output_video_path, log_path, args.detector)
