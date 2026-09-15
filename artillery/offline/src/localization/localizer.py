"""Localization session facade, event projection and JSONL output."""
from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image

from . import registration, visual_odometry
from .geo_map import GeoMap
from .geometry import estimate_pose, plane_homography, transform
from .map_panel import render_panel
from .photo import PhotoPose, read_bgr, resize_work


class MapLocalizer:
    """Runner facade holding session state and coordinating localization modules."""

    def __init__(self, root, reference, output, icon, mpp=0.75, ground_height=None, allow_gps_seed=False):
        self.output = output
        self.photo = PhotoPose.read(reference, ground_height)
        self.map = GeoMap(root, self.photo, output, mpp)
        self.reference, _ = resize_work(read_bgr(reference))
        self.ref_gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        self.ref_k = self.photo.camera_matrix(self.reference.shape[1], self.reference.shape[0])
        self.rotation = self.photo.rotation()
        self.center = np.array([0., 0., self.photo.height])
        self.ref_h = plane_homography(self.ref_k, self.rotation, self.center)
        self.sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher()
        self.status = "INITIALIZING"
        self.quality = {}
        self.history = {}
        self.trajectory = []
        self.events = []
        self.prev_gray = None
        self.frame_size = None
        self.points = None
        self.world_points = None
        self.last_relocalization = -1000
        self.last_map_check = -1000
        self.last_map_fix = None
        self.allow_gps_seed = allow_gps_seed
        self.video_focal_scale = None
        self.map_span_m = 1100.0
        self.icon = None
        if icon is not None and hasattr(icon, "exists") and icon.exists():
            with Image.open(icon) as im:
                self.icon = np.array(im.convert("RGBA"))
        self.log = (output / "localization.jsonl").open("w", encoding="utf-8")
        self.register_reference()

    def record(self, data):
        self.log.write(json.dumps(data, ensure_ascii=False, allow_nan=False)+"\n")
        self.log.flush()

    def matches(self, a, b, mask_a=None, mask_b=None):
        return registration.matches(self, a, b, mask_a, mask_b)

    def register_reference(self):
        return registration.register_reference(self)

    def reference_link(self, gray, k):
        return registration.reference_link(self, gray, k)

    def map_correction(self, gray, k):
        return registration.map_correction(self, gray, k)

    def fit_pose(self, world, pixels, k, max_shift, previous=None):
        result = estimate_pose(world, pixels, k, max_shift,
                               self.center if previous is None else previous, self.photo.height)
        if result is None:
            return False
        self.rotation, self.center, self.quality = result
        return True

    def replenish(self, gray, k):
        return visual_odometry.replenish(self, gray, k)

    def process(self, frame, frame_idx, fps):
        return visual_odometry.process(self, frame, frame_idx, fps)

    def panel(self, height, width):
        return render_panel(self, height, width)

    def event_position(self, event_id, frame_idx, pixel):
        sample = self.history.get(frame_idx)
        if sample is None:
            self.record({"stage": "event", "id": event_id, "frame": frame_idx, "status": "unavailable"})
            return None
        if float(sample[0][2] @ np.array([pixel[0], pixel[1], 1.])) <= 0:
            self.record({"stage": "event", "id": event_id, "frame": frame_idx, "status": "ray_above_ground_horizon"})
            return None
        world = transform([pixel], sample[0])[0]
        if not np.all(np.isfinite(world)) or not self.map.contains(world) or np.linalg.norm(world-sample[1][:2]) > 1800:
            self.record({"stage": "event", "id": event_id, "frame": frame_idx, "status": "outside_map_or_ground_range"})
            return None
        self.events.append((event_id, world))
        lon, lat = self.map.to_gps(world)
        self.record({"stage": "event", "id": event_id, "frame": frame_idx,
                     "status": "ground_plane_estimate", "gps": [lon, lat],
                     "map_alignment": "visual" if self.reference_valid else "gps_metadata_only"})
        return lon, lat

    def close(self):
        self.log.close()
