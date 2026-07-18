from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


Color = Tuple[int, int, int]


@dataclass
class LabelEntry:
    label_id: int
    kind: str
    center_frame: int
    start_frame: int
    end_frame: int
    timestamp_sec: float
    bbox: Optional[Tuple[int, int, int, int]]


class FrameReader:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        self.last_frame_idx = -1
        self.last_frame = None

    def read(self, frame_idx: int):
        frame_idx = max(frame_idx, 0)
        if self.last_frame_idx >= 0 and frame_idx == self.last_frame_idx and self.last_frame is not None:
            return self.last_frame.copy()
        if self.last_frame_idx >= 0 and frame_idx == self.last_frame_idx + 1:
            ok, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
        if not ok:
            raise EOFError(frame_idx)
        self.last_frame_idx = frame_idx
        self.last_frame = frame
        return frame.copy()

    def release(self) -> None:
        self.cap.release()


class ExplosionLabeler:
    WINDOW = "Explosion Labeler"

    def __init__(
        self,
        video_path: Path,
        output_path: Path,
        window_sec: float,
        max_display_width: int,
        max_display_height: int,
    ) -> None:
        self.video_path = video_path
        self.output_path = output_path
        self.reader = FrameReader(video_path)

        self.fps = self.reader.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.reader.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.src_width = int(self.reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.src_height = int(self.reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        scale_w = max_display_width / max(self.src_width, 1)
        scale_h = max_display_height / max(self.src_height, 1)
        self.display_scale = min(1.0, scale_w, scale_h)
        self.tolerance_frames = max(2, int(round(window_sec * self.fps)))

        self.frame_idx = 0
        self.playing = False
        self.drag_start = None
        self.pending_bbox: Optional[Tuple[int, int, int, int]] = None
        self.labels: List[LabelEntry] = []
        self.next_label_id = 1

        self._load_existing()

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)

    def _load_existing(self) -> None:
        if not self.output_path.exists():
            return
        data = json.loads(self.output_path.read_text(encoding="utf-8"))
        for raw in data.get("labels", []):
            bbox = tuple(raw["bbox"]) if raw.get("bbox") else None
            self.labels.append(
                LabelEntry(
                    label_id=int(raw["label_id"]),
                    kind=str(raw["kind"]),
                    center_frame=int(raw["center_frame"]),
                    start_frame=int(raw["start_frame"]),
                    end_frame=int(raw["end_frame"]),
                    timestamp_sec=float(raw["timestamp_sec"]),
                    bbox=bbox,
                )
            )
        self.next_label_id = 1 + max((label.label_id for label in self.labels), default=0)

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_path": str(self.video_path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "frame_size": [self.src_width, self.src_height],
            "tolerance_frames": self.tolerance_frames,
            "labels": [
                {
                    "label_id": label.label_id,
                    "kind": label.kind,
                    "center_frame": label.center_frame,
                    "start_frame": label.start_frame,
                    "end_frame": label.end_frame,
                    "timestamp_sec": label.timestamp_sec,
                    "bbox": list(label.bbox) if label.bbox else None,
                }
                for label in self.labels
            ],
        }
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run(self) -> None:
        try:
            while True:
                frame = self.reader.read(self.frame_idx)
                canvas = self._render(frame)
                cv2.imshow(self.WINDOW, canvas)
                # Keep the UI responsive even while paused so mouse-based box
                # drawing works without requiring an extra key press.
                key = cv2.waitKey(20) & 0xFF
                if self._handle_key(key):
                    break
                if self.playing:
                    self.frame_idx = min(self.frame_idx + 1, max(self.frame_count - 1, 0))
        finally:
            self.save()
            self.reader.release()
            cv2.destroyAllWindows()

    def _render(self, frame):
        display = frame
        if self.display_scale != 1.0:
            display = cv2.resize(
                frame,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_AREA,
            )

        for label in self.labels:
            active = label.start_frame <= self.frame_idx <= label.end_frame
            color = (0, 255, 0) if label.kind == "explosion" else (0, 165, 255)
            thickness = 2 if active else 1
            if label.bbox:
                x, y, w, h = self._scale_bbox(label.bbox)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            if active:
                cv2.putText(
                    display,
                    f"{label.kind}:{label.label_id}",
                    (12, 110 + 24 * (label.label_id % 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if self.pending_bbox:
            x, y, w, h = self._scale_bbox(self.pending_bbox)
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 255, 0), 2)

        info = [
            f"{self.video_path.name}",
            f"frame={self.frame_idx}/{max(self.frame_count - 1, 0)}",
            f"time={self.frame_idx / self.fps:.3f}s  tol=+/-{self.tolerance_frames}f",
            f"labels={len(self.labels)}  mode={'play' if self.playing else 'pause'}",
            "keys: space play, a/d +-1, j/l +-10, u/o +-60, c explosion, f nuisance, x clear box, r remove, s save, q quit",
        ]
        y = 28
        for line in info:
            cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28
        return display

    def _handle_key(self, key: int) -> bool:
        if key == 255:
            return False
        if key in (27, ord("q")):
            return True
        if key == ord(" "):
            self.playing = not self.playing
        elif key == ord("a"):
            self.playing = False
            self.frame_idx = max(self.frame_idx - 1, 0)
        elif key == ord("d"):
            self.playing = False
            self.frame_idx = min(self.frame_idx + 1, self.frame_count - 1)
        elif key == ord("j"):
            self.playing = False
            self.frame_idx = max(self.frame_idx - 10, 0)
        elif key == ord("l"):
            self.playing = False
            self.frame_idx = min(self.frame_idx + 10, self.frame_count - 1)
        elif key == ord("u"):
            self.playing = False
            self.frame_idx = max(self.frame_idx - 60, 0)
        elif key == ord("o"):
            self.playing = False
            self.frame_idx = min(self.frame_idx + 60, self.frame_count - 1)
        elif key == ord("-"):
            self.tolerance_frames = max(self.tolerance_frames - 1, 0)
        elif key in (ord("="), ord("+")):
            self.tolerance_frames += 1
        elif key == ord("x"):
            self.pending_bbox = None
        elif key == ord("s"):
            self.save()
            print(f"[INFO] Saved labels to {self.output_path}")
        elif key == ord("c"):
            self._commit_label("explosion")
        elif key == ord("f"):
            self._commit_label("nuisance")
        elif key == ord("r"):
            self._remove_nearest_label()
        elif key == ord("n"):
            self._jump_to_neighbor_label(next_label=True)
        elif key == ord("p"):
            self._jump_to_neighbor_label(next_label=False)
        return False

    def _commit_label(self, kind: str) -> None:
        start_frame = max(self.frame_idx - self.tolerance_frames, 0)
        end_frame = min(self.frame_idx + self.tolerance_frames, max(self.frame_count - 1, 0))
        label = LabelEntry(
            label_id=self.next_label_id,
            kind=kind,
            center_frame=self.frame_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            timestamp_sec=self.frame_idx / self.fps,
            bbox=self.pending_bbox,
        )
        self.labels.append(label)
        self.labels.sort(key=lambda item: (item.center_frame, item.label_id))
        self.next_label_id += 1
        self.save()
        print(
            f"[INFO] Added {kind} label {label.label_id} at frame {label.center_frame} "
            f"({label.timestamp_sec:.3f}s), bbox={label.bbox}"
        )

    def _remove_nearest_label(self) -> None:
        if not self.labels:
            return
        nearest = min(self.labels, key=lambda item: abs(item.center_frame - self.frame_idx))
        if abs(nearest.center_frame - self.frame_idx) > max(self.tolerance_frames, 5):
            return
        self.labels = [label for label in self.labels if label.label_id != nearest.label_id]
        self.save()
        print(f"[INFO] Removed label {nearest.label_id}")

    def _jump_to_neighbor_label(self, next_label: bool) -> None:
        if not self.labels:
            return
        ordered = sorted(self.labels, key=lambda item: item.center_frame)
        if next_label:
            for label in ordered:
                if label.center_frame > self.frame_idx:
                    self.frame_idx = label.center_frame
                    return
            self.frame_idx = ordered[-1].center_frame
        else:
            for label in reversed(ordered):
                if label.center_frame < self.frame_idx:
                    self.frame_idx = label.center_frame
                    return
            self.frame_idx = ordered[0].center_frame

    def _scale_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox
        return (
            int(round(x * self.display_scale)),
            int(round(y * self.display_scale)),
            int(round(w * self.display_scale)),
            int(round(h * self.display_scale)),
        )

    def _unscale_point(self, x: int, y: int) -> Tuple[int, int]:
        if self.display_scale == 0:
            return x, y
        return int(round(x / self.display_scale)), int(round(y / self.display_scale))

    def _on_mouse(self, event, x, y, flags, param) -> None:
        sx, sy = self._unscale_point(x, y)
        sx = max(0, min(sx, max(self.src_width - 1, 0)))
        sy = max(0, min(sy, max(self.src_height - 1, 0)))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (sx, sy)
            self.pending_bbox = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.pending_bbox = _normalize_bbox(self.drag_start, (sx, sy))
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            bbox = _normalize_bbox(self.drag_start, (sx, sy))
            self.drag_start = None
            if bbox[2] < 4 or bbox[3] < 4:
                self.pending_bbox = None
            else:
                self.pending_bbox = bbox


def _normalize_bbox(p0: Tuple[int, int], p1: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x0, y0 = p0
    x1, y1 = p1
    x_min = min(x0, x1)
    y_min = min(y0, y1)
    x_max = max(x0, x1)
    y_max = max(y0, y1)
    return x_min, y_min, x_max - x_min, y_max - y_min


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual labeler for explosion timestamps and rough ROIs.")
    parser.add_argument("--video", required=True, type=str, help="Input video path.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
    parser.add_argument(
        "--window-sec",
        type=float,
        default=0.10,
        help="Initial tolerance window around the current frame. The UI saves +/- this many seconds as valid positives.",
    )
    parser.add_argument("--max-display-width", type=int, default=1600, help="Maximum display width.")
    parser.add_argument("--max-display-height", type=int, default=900, help="Maximum display height.")
    return parser


def default_output_path(video_path: Path) -> Path:
    return Path("labels") / f"{video_path.stem}.labels.json"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = Path(args.output) if args.output else default_output_path(video_path)
    labeler = ExplosionLabeler(
        video_path=video_path,
        output_path=output_path,
        window_sec=args.window_sec,
        max_display_width=args.max_display_width,
        max_display_height=args.max_display_height,
    )
    labeler.run()


if __name__ == "__main__":
    main()
