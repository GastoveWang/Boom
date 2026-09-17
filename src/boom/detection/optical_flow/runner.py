from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
from boom.detection.optical_flow.types import (
    DetectorConfig, VideoRunSummary, DetectionArtifact
)
from boom.detection.optical_flow.detector import InstantSmokeDustDetector

def run_video(
    video_path: Path,
    config: DetectorConfig,
    start_frame: int = 0,
    max_frames: int = -1,
    display_enabled: bool = False,
    output_video_path: Optional[Path] = None,
    detection_root: Optional[Path] = None,
    report_every: int = 300,
) -> Tuple[VideoRunSummary, List[DetectionArtifact]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = InstantSmokeDustDetector(config, fps)

    writer = None
    frame_idx = start_frame
    processed = 0
    confirmed_count = 0
    detections: List[DetectionArtifact] = []

    if output_video_path is not None:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            vis = detector.process_frame(frame, frame_idx)

            if writer is None and output_video_path is not None:
                height, width = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_video_path), fourcc, detector.fps, (width, height))

            if writer is not None:
                writer.write(vis)

            for event in detector.consume_pending_confirmations():
                confirmed_count += 1
                if detection_root is not None:
                    detections.append(_save_detection_snapshot(
                        video_path, detection_root, vis, event))

            if display_enabled:
                show_frame = vis
                if config.display_scale != 1.0:
                    show_frame = cv2.resize(
                        vis,
                        None,
                        fx=config.display_scale,
                        fy=config.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow("Instant Smoke / Dust POC", show_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1
            processed += 1
            if report_every > 0 and processed % report_every == 0:
                print(
                    f"[INFO] {video_path.name}: processed {processed} frames")
            if max_frames > 0 and processed >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display_enabled:
            cv2.destroyAllWindows()

    summary = VideoRunSummary(
        video_name=video_path.name,
        processed_frames=processed,
        confirmed_events=confirmed_count,
        snapshots_saved=len(detections) * 2,
        output_video_path=output_video_path,
    )
    return summary, detections


def run_batch(
    input_dir: Path,
    output_root: Path,
    config: DetectorConfig,
    video_glob: str,
    start_frame: int = 0,
    max_frames: int = -1,
    report_every: int = 300,
) -> Tuple[List[VideoRunSummary], List[DetectionArtifact]]:
    videos = sorted(input_dir.glob(video_glob))
    if not videos:
        raise FileNotFoundError(
            f"No videos found in {input_dir} matching {video_glob}")

    annotated_dir, detection_dir, logs_dir = _ensure_batch_subdirs(output_root)
    all_summaries: List[VideoRunSummary] = []
    all_detections: List[DetectionArtifact] = []

    print(f"[INFO] Batch input dir: {input_dir}")
    print(f"[INFO] Batch output dir: {output_root}")
    print(f"[INFO] Found {len(videos)} videos")

    for idx, video_path in enumerate(videos, start=1):
        print(f"[INFO] [{idx}/{len(videos)}] Processing {video_path.name}")
        output_video_path = annotated_dir / f"{video_path.stem}__annotated.mp4"
        summary, detections = run_video(
            video_path=video_path,
            config=config,
            start_frame=start_frame,
            max_frames=max_frames,
            display_enabled=False,
            output_video_path=output_video_path,
            detection_root=detection_dir,
            report_every=report_every,
        )
        all_summaries.append(summary)
        all_detections.extend(detections)
        print(
            f"[INFO] Finished {video_path.name}: "
            f"frames={summary.processed_frames}, events={summary.confirmed_events}, "
            f"snapshots={summary.snapshots_saved}"
        )

    _write_batch_csvs(logs_dir, all_summaries, all_detections)
    return all_summaries, all_detections


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="POC detector for instantaneous dust/smoke anomalies under drone ego-motion."
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Input video path. Defaults to first *.mp4 in CWD.")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="Process all videos in this directory.")
    parser.add_argument("--video-glob", type=str,
                        default="*.mp4", help="Glob used in batch mode.")
    parser.add_argument("--output", type=str, default=None,
                        help="Single-video annotated output path.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Batch output root or single-video artifact root.")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start processing from a specific frame.")
    parser.add_argument("--max-frames", type=int, default=-1,
                        help="Process only the first N frames after start-frame.")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable cv2.imshow window.")
    parser.add_argument("--report-every", type=int, default=300,
                        help="Print progress every N processed frames.")

    parser.add_argument("--downscale", type=float, default=0.5,
                        help="Spatial downsampling factor.")
    parser.add_argument("--stride", type=int, default=5,
                        help="Frame gap N used for T vs T-N differencing.")
    parser.add_argument("--diff-threshold", type=int, default=40,
                        help="Threshold for absolute difference image.")
    parser.add_argument("--min-blob-area", type=float,
                        default=220.0, help="Minimum candidate contour area.")
    parser.add_argument("--min-growth-ratio", type=float,
                        default=2.2, help="Short-term area explosion threshold.")
    parser.add_argument("--stop-growth-ratio", type=float, default=1.08,
                        help="Silence event when growth falls below this.")
    parser.add_argument("--max-age-sec", type=float, default=3.0,
                        help="Maximum active lifetime per confirmed event.")
    parser.add_argument("--track-match-distance", type=float,
                        default=120.0, help="Track association radius in pixels.")
    parser.add_argument("--max-missed-frames", type=int, default=6,
                        help="How many missing frames an event can survive.")
    parser.add_argument(
        "--min-dimension-growth-ratio",
        type=float,
        default=1.08,
        help="Minimum width/height growth used to distinguish expansion from translation.",
    )
    parser.add_argument(
        "--max-center-shift-ratio",
        type=float,
        default=0.65,
        help="Reject translation-dominant motion when centroid shifts too much relative to blob size.",
    )
    parser.add_argument(
        "--split-blob-area",
        type=float,
        default=2500.0,
        help="Try watershed splitting when a motion blob becomes larger than this area.",
    )
    parser.add_argument("--disable-flow-check", action="store_true",
                        help="Disable radial optical-flow validation.")
    parser.add_argument("--hide-debug-tiles", action="store_true",
                        help="Hide absdiff / mask thumbnails.")
    parser.add_argument("--show-candidates", action="store_true",
                        help="Draw orange candidate boxes for debugging.")
    return parser


def config_from_args(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig(
        downscale=args.downscale,
        stride_frames=max(args.stride, 1),
        diff_threshold=args.diff_threshold,
        min_blob_area=args.min_blob_area,
        min_growth_ratio=args.min_growth_ratio,
        stop_growth_ratio=args.stop_growth_ratio,
        max_event_age_sec=args.max_age_sec,
        track_match_distance=args.track_match_distance,
        max_missed_frames=args.max_missed_frames,
        min_dimension_growth_ratio=args.min_dimension_growth_ratio,
        max_center_shift_ratio=args.max_center_shift_ratio,
        split_blob_area=args.split_blob_area,
        enable_radial_flow_check=not args.disable_flow_check,
        draw_debug_tiles=not args.hide_debug_tiles,
        show_candidate_boxes=args.show_candidates,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = config_from_args(args)

    if args.batch_dir:
        input_dir = Path(args.batch_dir)
        if not input_dir.exists():
            raise FileNotFoundError(
                f"Batch input directory does not exist: {input_dir}")
        output_root = Path(
            args.output_dir) if args.output_dir else _default_batch_output_dir(input_dir)
        summaries, detections = run_batch(
            input_dir=input_dir,
            output_root=output_root,
            config=config,
            video_glob=args.video_glob,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            report_every=args.report_every,
        )
        print(
            f"[INFO] Batch complete: videos={len(summaries)}, "
            f"confirmed_events={sum(s.confirmed_events for s in summaries)}, "
            f"saved_images={sum(s.snapshots_saved for s in summaries)}"
        )
        print(f"[INFO] Output root: {output_root}")
        print(
            f"[INFO] Detection log: {output_root / 'logs' / 'detections.csv'}")
        return

    video_path = Path(args.video) if args.video else _default_video_from_cwd()
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(
            "No input video found. Use --video or place an .mp4 in the current directory.")

    output_video_path = Path(args.output) if args.output else None
    detection_root = None
    if args.output_dir:
        output_root = Path(args.output_dir)
        annotated_dir, detection_dir, _ = _ensure_batch_subdirs(output_root)
        detection_root = detection_dir
        if output_video_path is None:
            output_video_path = annotated_dir / \
                f"{video_path.stem}__annotated.mp4"

    summary, detections = run_video(
        video_path=video_path,
        config=config,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        display_enabled=not args.no_display,
        output_video_path=output_video_path,
        detection_root=detection_root,
        report_every=args.report_every,
    )
    print(
        f"[INFO] Finished {summary.video_name}: "
        f"frames={summary.processed_frames}, events={summary.confirmed_events}, "
        f"snapshots={summary.snapshots_saved}"
    )
    if detections:
        print(f"[INFO] Detection frames saved under: {detection_root}")


if __name__ == "__main__":
    main()



