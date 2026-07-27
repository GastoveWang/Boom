"""MOD-IR based smoke detection for UAV videos.

Paper-inspired stages:
ORB + Hamming KNN -> RANSAC homography -> registered frame difference ->
Kapur maximum-entropy threshold -> Quick-shift region refinement.

The smoke-specific appearance and temporal filters are an application extension;
the source paper detects generic moving objects rather than classifying smoke.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from geo_ui import (
    LocatedEvent,
    build_impact_panel,
    locate_event,
    resolve_geo_reference,
)

try:
    from skimage.segmentation import quickshift
except ImportError:  # The OpenCV fallback keeps the detector runnable.
    quickshift = None


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass
class Config:
    process_scale: float = 0.30
    segment_scale: float = 0.28
    max_features: int = 2500
    ratio_test: float = 0.75
    min_matches: int = 16
    min_inliers: int = 10
    ransac_threshold: float = 4.0
    min_threshold: int = 12
    max_threshold: int = 90
    min_area: float = 90.0
    max_area_ratio: float = 0.25
    segment_foreground_ratio: float = 0.12
    max_saturation: float = 155.0
    min_brightness: float = 30.0
    max_edge_density: float = 0.24
    persistence_frames: int = 3
    hold_frames: int = 20
    match_distance: float = 90.0
    show_debug_overlay: bool = False


@dataclass
class Candidate:
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: float
    mean_saturation: float
    edge_density: float


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    first_frame: int
    last_frame: int
    hits: int = 1
    confirmed: bool = False
    emitted: bool = False


def kapur_threshold(gray: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> int:
    """Return the threshold maximizing background + foreground entropy."""
    pixels = gray[valid_mask > 0] if valid_mask is not None else gray.ravel()
    if pixels.size == 0:
        return 0
    hist = np.bincount(pixels, minlength=256).astype(np.float64)
    probability = hist / max(hist.sum(), 1.0)
    cumulative = np.cumsum(probability)
    p_log_p = np.zeros_like(probability)
    positive = probability > 0
    p_log_p[positive] = probability[positive] * np.log(probability[positive])
    cumulative_p_log_p = np.cumsum(p_log_p)
    total_p_log_p = cumulative_p_log_p[-1]

    scores = np.full(256, -np.inf, dtype=np.float64)
    for threshold in range(1, 255):
        p_background = cumulative[threshold]
        p_foreground = 1.0 - p_background
        if p_background <= 1e-12 or p_foreground <= 1e-12:
            continue
        h_background = np.log(p_background) - cumulative_p_log_p[threshold] / p_background
        h_foreground = (
            np.log(p_foreground)
            - (total_p_log_p - cumulative_p_log_p[threshold]) / p_foreground
        )
        scores[threshold] = h_background + h_foreground
    return int(np.argmax(scores))


class ModIRSmokeDetector:
    def __init__(self, config: Config, fps: float) -> None:
        self.config = config
        self.fps = fps if fps > 0 else 30.0
        self.orb = cv2.ORB_create(
            nfeatures=config.max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=12,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.previous_gray: Optional[np.ndarray] = None
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1
        self.last_threshold = 0
        self.last_matches = 0
        self.last_inliers = 0
        self.segmentation_method = "quick-shift" if quickshift is not None else "mean-shift fallback"

    def process(self, frame: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray, list[Track]]:
        small = cv2.resize(
            frame, None, fx=self.config.process_scale, fy=self.config.process_scale,
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = np.zeros_like(gray)
        newly_confirmed: list[Track] = []

        if self.previous_gray is not None:
            aligned, valid = self._register_previous(self.previous_gray, gray)
            if aligned is not None and valid is not None:
                difference = cv2.absdiff(gray, aligned)
                difference[valid == 0] = 0
                threshold = kapur_threshold(difference, valid)
                threshold = int(np.clip(
                    threshold, self.config.min_threshold, self.config.max_threshold
                ))
                self.last_threshold = threshold
                mask = self._foreground_mask(difference, valid, threshold)
                mask = self._refine_regions(small, mask)
                candidates = self._smoke_candidates(small, mask)
                newly_confirmed = self._update_tracks(candidates, frame_idx)
            else:
                self._age_tracks(frame_idx)

        self.previous_gray = gray
        annotated = self._draw(small, mask, frame_idx)
        return annotated, mask, newly_confirmed

    def _register_previous(
        self, previous: np.ndarray, current: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        kp_prev, desc_prev = self.orb.detectAndCompute(previous, None)
        kp_curr, desc_curr = self.orb.detectAndCompute(current, None)
        if desc_prev is None or desc_curr is None or len(desc_prev) < 2 or len(desc_curr) < 2:
            return None, None
        pairs = self.matcher.knnMatch(desc_prev, desc_curr, k=2)
        good = [
            pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < self.config.ratio_test * pair[1].distance
        ]
        self.last_matches = len(good)
        if len(good) < self.config.min_matches:
            return None, None

        src = np.float32([kp_prev[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_curr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(
            src, dst, cv2.RANSAC, self.config.ransac_threshold
        )
        self.last_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if homography is None or self.last_inliers < self.config.min_inliers:
            return None, None

        height, width = current.shape
        aligned = cv2.warpPerspective(previous, homography, (width, height))
        source_valid = np.full_like(previous, 255)
        valid = cv2.warpPerspective(source_valid, homography, (width, height))
        valid = cv2.erode(valid, np.ones((7, 7), np.uint8), iterations=1)
        return aligned, valid

    @staticmethod
    def _foreground_mask(difference: np.ndarray, valid: np.ndarray, threshold: int) -> np.ndarray:
        _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, valid)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        return mask

    def _refine_regions(self, frame: np.ndarray, foreground: np.ndarray) -> np.ndarray:
        scale = self.config.segment_scale
        seg_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        seg_mask = cv2.resize(
            foreground, (seg_frame.shape[1], seg_frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        if quickshift is not None:
            rgb = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
            labels = quickshift(rgb, ratio=0.65, kernel_size=3, max_dist=7, convert2lab=True)
        else:
            filtered = cv2.pyrMeanShiftFiltering(seg_frame, sp=8, sr=18, maxLevel=1)
            quantized = (filtered // 24).astype(np.int32)
            labels = quantized[:, :, 0] * 121 + quantized[:, :, 1] * 11 + quantized[:, :, 2]

        refined = np.zeros_like(seg_mask)
        for label in np.unique(labels):
            region = labels == label
            if region.sum() < 8:
                continue
            ratio = float(np.count_nonzero(seg_mask[region])) / float(region.sum())
            if ratio >= self.config.segment_foreground_ratio:
                refined[region] = 255
        refined = cv2.resize(
            refined, (foreground.shape[1], foreground.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return cv2.bitwise_or(foreground, refined)

    def _smoke_candidates(self, frame: np.ndarray, mask: np.ndarray) -> list[Candidate]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(frame.shape[0] * frame.shape[1])
        candidates: list[Candidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_area or area > frame_area * self.config.max_area_ratio:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            roi_mask = np.zeros((height, width), np.uint8)
            shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(roi_mask, [shifted], -1, 255, -1)
            sat = float(cv2.mean(hsv[y:y + height, x:x + width, 1], mask=roi_mask)[0])
            val = float(cv2.mean(hsv[y:y + height, x:x + width, 2], mask=roi_mask)[0])
            edge_count = np.count_nonzero(cv2.bitwise_and(edges[y:y + height, x:x + width], roi_mask))
            edge_density = float(edge_count) / max(float(np.count_nonzero(roi_mask)), 1.0)
            if sat > self.config.max_saturation or val < self.config.min_brightness:
                continue
            if edge_density > self.config.max_edge_density:
                continue
            moments = cv2.moments(contour)
            if moments["m00"]:
                centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            else:
                centroid = (x + width / 2.0, y + height / 2.0)
            candidates.append(Candidate((x, y, width, height), centroid, area, sat, edge_density))
        return candidates

    def _update_tracks(self, candidates: list[Candidate], frame_idx: int) -> list[Track]:
        unmatched = set(range(len(candidates)))
        updated: set[int] = set()
        for track_id, track in list(self.tracks.items()):
            best_idx = None
            best_distance = float("inf")
            for idx in unmatched:
                candidate = candidates[idx]
                distance = float(np.hypot(
                    track.centroid[0] - candidate.centroid[0],
                    track.centroid[1] - candidate.centroid[1],
                ))
                if distance < self.config.match_distance and distance < best_distance:
                    best_idx, best_distance = idx, distance
            if best_idx is None:
                continue
            candidate = candidates[best_idx]
            unmatched.remove(best_idx)
            track.bbox = candidate.bbox
            track.centroid = candidate.centroid
            track.last_frame = frame_idx
            track.hits += 1
            track.confirmed = track.hits >= self.config.persistence_frames
            updated.add(track_id)

        for idx in unmatched:
            candidate = candidates[idx]
            track = Track(
                self.next_track_id, candidate.bbox, candidate.centroid, frame_idx, frame_idx
            )
            self.tracks[track.track_id] = track
            updated.add(track.track_id)
            self.next_track_id += 1

        self._age_tracks(frame_idx)
        newly_confirmed = []
        for track_id in updated:
            track = self.tracks.get(track_id)
            if track and track.confirmed and not track.emitted:
                track.emitted = True
                newly_confirmed.append(track)
        return newly_confirmed

    def _age_tracks(self, frame_idx: int) -> None:
        self.tracks = {
            track_id: track for track_id, track in self.tracks.items()
            if frame_idx - track.last_frame <= self.config.hold_frames
        }

    def _draw(self, frame: np.ndarray, mask: np.ndarray, frame_idx: int) -> np.ndarray:
        canvas = frame.copy()
        overlay = np.zeros_like(canvas)
        overlay[:, :, 2] = mask
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.24, 0.0)
        for track in self.tracks.values():
            if not track.confirmed:
                continue
            x, y, width, height = track.bbox
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(
                canvas, f"SMOKE {track.track_id}", (x, max(y - 8, 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA,
            )
        if self.config.show_debug_overlay:
            status = (
                f"frame={frame_idx} th={self.last_threshold} matches={self.last_matches} "
                f"inliers={self.last_inliers} segments={self.segmentation_method}"
            )
            cv2.putText(canvas, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return canvas


def discover_videos(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def run_video(
    video_path: Path,
    output_root: Path,
    config: Config,
    max_frames: int,
    display: bool,
    geo,
    event_display_sec: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    detector = ModIRSmokeDetector(config, fps)

    output_dir = output_root / video_path.stem
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "annotated.mp4"
    csv_path = output_dir / "detections.csv"
    writer = None
    rows: list[dict[str, object]] = []
    recent_events: list[LocatedEvent] = []
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, _, events = detector.process(frame, processed)
            for event in events:
                located = locate_event(
                    event.track_id,
                    processed,
                    fps,
                    event.bbox,
                    config.process_scale,
                    source_width,
                    source_height,
                    geo,
                )
                recent_events.append(located)

            keep_frames = max(int(round(event_display_sec * fps)), 1)
            recent_events = [
                event for event in recent_events
                if processed - event.frame_idx <= keep_frames
            ]
            panel = build_impact_panel(
                annotated.shape[0], recent_events, video_path.name, geo is not None
            )
            annotated = np.hstack((annotated, panel))
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
            writer.write(annotated)

            for event in events:
                located = next(item for item in recent_events if item.event_id == event.track_id)
                snapshot = snapshots_dir / f"smoke_{event.track_id:04d}_f{processed:06d}.jpg"
                cv2.imwrite(str(snapshot), annotated)
                x, y, width, height = event.bbox
                rows.append({
                    "event_id": event.track_id,
                    "frame_idx": processed,
                    "timestamp_sec": f"{processed / fps:.3f}",
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": width,
                    "bbox_h": height,
                    "twd97_e": "" if located.easting is None else f"{located.easting:.2f}",
                    "twd97_n": "" if located.northing is None else f"{located.northing:.2f}",
                    "longitude": "" if located.lon is None else f"{located.lon:.7f}",
                    "latitude": "" if located.lat is None else f"{located.lat:.7f}",
                    "snapshot": str(snapshot),
                })

            if display:
                cv2.imshow("MOD-IR Smoke Detection", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            processed += 1
            if processed % 30 == 0:
                print(f"[INFO] {video_path.name}: {processed}/{frame_count or '?'} frames")
            if max_frames > 0 and processed >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    fieldnames = [
        "event_id", "frame_idx", "timestamp_sec", "bbox_x", "bbox_y",
        "bbox_w", "bbox_h", "twd97_e", "twd97_n", "longitude", "latitude",
        "snapshot",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    print(f"[DONE] {video_path.name}: frames={processed}, smoke_events={len(rows)}")
    print(f"[DONE] Output: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MOD-IR based UAV smoke detector")
    parser.add_argument("--video", type=Path, default=None, help="指定單一影片；預設處理 data/ 內全部影片")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/mod_ir_smoke"))
    parser.add_argument("--max-frames", type=int, default=-1, help="測試時限制處理幀數")
    parser.add_argument("--display", action="store_true", help="顯示 OpenCV 即時視窗")
    parser.add_argument("--process-scale", type=float, default=0.30)
    parser.add_argument("--min-area", type=float, default=90.0)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--reference-image", type=Path, default=None,
                        help="含 DJI GPS/姿態 EXIF-XMP 的定位參考照片")
    parser.add_argument("--drone-lon", type=float, default=None)
    parser.add_argument("--drone-lat", type=float, default=None)
    parser.add_argument("--drone-alt", type=float, default=None)
    parser.add_argument("--drone-yaw", type=float, default=None)
    parser.add_argument("--drone-pitch", type=float, default=None)
    parser.add_argument("--drone-roll", type=float, default=None)
    parser.add_argument("--event-display-sec", type=float, default=10.0)
    parser.add_argument("--debug-overlay", action="store_true",
                        help="在影片左上角顯示偵測除錯資訊")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Config(
        process_scale=args.process_scale,
        min_area=args.min_area,
        persistence_frames=max(args.persistence, 1),
        show_debug_overlay=args.debug_overlay,
    )
    geo = resolve_geo_reference(
        args.reference_image,
        args.drone_lon,
        args.drone_lat,
        args.drone_alt,
        args.drone_yaw,
        args.drone_pitch,
        args.drone_roll,
    )
    videos = [args.video] if args.video else discover_videos(args.data_dir)
    missing = [path for path in videos if not path.exists()]
    if missing:
        raise FileNotFoundError(f"找不到影片：{missing[0]}")
    if not videos:
        raise FileNotFoundError(f"{args.data_dir} 中找不到支援的影片")
    print(f"[INFO] Region refinement: {'Quick-shift' if quickshift is not None else 'OpenCV mean-shift fallback'}")
    for video_path in videos:
        run_video(
            video_path, args.output_dir, config, args.max_frames, args.display,
            geo, args.event_display_sec,
        )


if __name__ == "__main__":
    main()
