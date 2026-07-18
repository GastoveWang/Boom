from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2


@dataclass
class DetectionRow:
    video_name: str
    event_id: int
    frame_idx: int
    timestamp_sec: float
    area: float
    growth_ratio: float
    radial_ratio: Optional[float]
    full_frame_path: Path
    crop_path: Path


@dataclass
class DetectionCluster:
    video_name: str
    detections: List[DetectionRow] = field(default_factory=list)
    fps: float = 30.0

    @property
    def start_frame(self) -> int:
        return min(row.frame_idx for row in self.detections)

    @property
    def end_frame(self) -> int:
        return max(row.frame_idx for row in self.detections)

    @property
    def start_sec(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_sec(self) -> float:
        return self.end_frame / self.fps

    @property
    def representative(self) -> DetectionRow:
        return max(self.detections, key=lambda row: (row.growth_ratio, row.area))


def read_detections(csv_path: Path) -> List[DetectionRow]:
    rows: List[DetectionRow] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            radial = raw["radial_ratio"].strip()
            rows.append(
                DetectionRow(
                    video_name=raw["video_name"],
                    event_id=int(raw["event_id"]),
                    frame_idx=int(raw["frame_idx"]),
                    timestamp_sec=float(raw["timestamp_sec"]),
                    area=float(raw["area"]),
                    growth_ratio=float(raw["growth_ratio"]),
                    radial_ratio=float(radial) if radial else None,
                    full_frame_path=Path(raw["full_frame_path"]),
                    crop_path=Path(raw["crop_path"]),
                )
            )
    return rows


def cluster_detections(
    rows: List[DetectionRow],
    fps_by_video: Dict[str, float],
    merge_gap_sec: float,
) -> List[DetectionCluster]:
    grouped: Dict[str, List[DetectionRow]] = {}
    for row in rows:
        grouped.setdefault(row.video_name, []).append(row)

    clusters: List[DetectionCluster] = []
    for video_name, items in grouped.items():
        fps = fps_by_video[video_name]
        merge_gap_frames = max(int(round(merge_gap_sec * fps)), 1)
        items.sort(key=lambda row: row.frame_idx)
        current: Optional[DetectionCluster] = None
        for row in items:
            if current is None:
                current = DetectionCluster(video_name=video_name, fps=fps, detections=[row])
                continue
            if row.frame_idx <= current.end_frame + merge_gap_frames:
                current.detections.append(row)
            else:
                clusters.append(current)
                current = DetectionCluster(video_name=video_name, fps=fps, detections=[row])
        if current is not None:
            clusters.append(current)
    return clusters


def extract_cluster_clip(
    video_path: Path,
    cluster: DetectionCluster,
    output_path: Path,
    pre_roll_sec: float,
    post_roll_sec: float,
    downscale: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or cluster.fps
    start_frame = max(int(cluster.start_frame - round(pre_roll_sec * fps)), 0)
    end_frame = min(
        int(cluster.end_frame + round(post_roll_sec * fps)),
        max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) - 1, 0),
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame_idx = start_frame
        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if downscale != 1.0:
                frame = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)

            stamp = f"{video_path.name}  f={frame_idx}  t={frame_idx / fps:.2f}s"
            cv2.putText(frame, stamp, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"cluster={cluster.start_sec:.2f}-{cluster.end_sec:.2f}s hits={len(cluster.detections)}",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def collect_fps(videos_root: Path, video_names: List[str]) -> Dict[str, float]:
    fps_map: Dict[str, float] = {}
    for video_name in sorted(set(video_names)):
        video_path = videos_root / video_name
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps_map[video_name] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    return fps_map


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Group raw detections into short review clips.")
    parser.add_argument("--detections-csv", required=True, type=str, help="Path to detections.csv")
    parser.add_argument("--videos-root", type=str, default=".", help="Directory containing source videos.")
    parser.add_argument(
        "--review-videos-root",
        type=str,
        default=None,
        help="Optional directory containing annotated videos to cut review clips from.",
    )
    parser.add_argument("--output-dir", required=True, type=str, help="Where grouped review clips will be written.")
    parser.add_argument("--merge-gap-sec", type=float, default=0.5, help="Merge detections that are this close in time.")
    parser.add_argument("--pre-roll-sec", type=float, default=1.0, help="Seconds before cluster start to include.")
    parser.add_argument("--post-roll-sec", type=float, default=1.0, help="Seconds after cluster end to include.")
    parser.add_argument("--downscale", type=float, default=1.0, help="Downscale review clips for quick viewing.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    detections_csv = Path(args.detections_csv)
    videos_root = Path(args.videos_root)
    review_videos_root = Path(args.review_videos_root) if args.review_videos_root else None
    output_dir = Path(args.output_dir)

    rows = read_detections(detections_csv)
    fps_map = collect_fps(videos_root, [row.video_name for row in rows])
    clusters = cluster_detections(rows, fps_map, args.merge_gap_sec)

    clips_dir = output_dir / "clips"
    refs_dir = output_dir / "representatives"
    clips_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    feedback_rows = []
    per_video_index: Dict[str, int] = {}

    for cluster in clusters:
        per_video_index[cluster.video_name] = per_video_index.get(cluster.video_name, 0) + 1
        cluster_idx = per_video_index[cluster.video_name]
        stem = f"{Path(cluster.video_name).stem}__cluster{cluster_idx:03d}__t{cluster.start_sec:08.2f}-{cluster.end_sec:08.2f}"

        clip_path = clips_dir / f"{stem}.mp4"
        review_video_path = videos_root / cluster.video_name
        if review_videos_root is not None:
            annotated_candidate = review_videos_root / f"{Path(cluster.video_name).stem}__annotated.mp4"
            if annotated_candidate.exists():
                review_video_path = annotated_candidate
        extract_cluster_clip(
            video_path=review_video_path,
            cluster=cluster,
            output_path=clip_path,
            pre_roll_sec=args.pre_roll_sec,
            post_roll_sec=args.post_roll_sec,
            downscale=args.downscale,
        )

        rep = cluster.representative
        full_target = refs_dir / f"{stem}__full.jpg"
        crop_target = refs_dir / f"{stem}__crop.jpg"
        if rep.full_frame_path.exists():
            shutil.copy2(rep.full_frame_path, full_target)
        if rep.crop_path.exists():
            shutil.copy2(rep.crop_path, crop_target)

        summary_rows.append(
            {
                "video_name": cluster.video_name,
                "cluster_id": cluster_idx,
                "start_frame": cluster.start_frame,
                "end_frame": cluster.end_frame,
                "start_sec": f"{cluster.start_sec:.3f}",
                "end_sec": f"{cluster.end_sec:.3f}",
                "detections_in_cluster": len(cluster.detections),
                "representative_event_id": rep.event_id,
                "representative_growth_ratio": f"{rep.growth_ratio:.3f}",
                "representative_area": f"{rep.area:.3f}",
                "review_source_video": str(review_video_path),
                "clip_path": str(clip_path),
                "representative_crop": str(crop_target) if crop_target.exists() else "",
                "representative_full": str(full_target) if full_target.exists() else "",
            }
        )
        feedback_rows.append(
            {
                "video_name": cluster.video_name,
                "cluster_id": cluster_idx,
                "start_sec": f"{cluster.start_sec:.3f}",
                "end_sec": f"{cluster.end_sec:.3f}",
                "clip_path": str(clip_path),
                "representative_crop": str(crop_target) if crop_target.exists() else "",
                "review_label": "",
                "review_notes": "",
            }
        )
        print(
            f"[INFO] {cluster.video_name} cluster {cluster_idx}: "
            f"{cluster.start_sec:.2f}-{cluster.end_sec:.2f}s "
            f"hits={len(cluster.detections)}"
        )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    with (output_dir / "detected_review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(feedback_rows[0].keys()) if feedback_rows else [])
        if feedback_rows:
            writer.writeheader()
            writer.writerows(feedback_rows)
    with (output_dir / "missed_events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "timestamp_sec",
                "frame_idx",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_name": "",
                "timestamp_sec": "",
                "frame_idx": "",
                "bbox_x": "",
                "bbox_y": "",
                "bbox_w": "",
                "bbox_h": "",
                "notes": "fill only for real explosions the detector completely missed",
            }
        )
    print(f"[INFO] Wrote {len(summary_rows)} review clips to {output_dir}")


if __name__ == "__main__":
    main()
