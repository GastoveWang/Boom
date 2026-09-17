"""
==============================================================================
Boom 航拍定位 - 地圖定位協調外觀與 Session 管理 (localizer.py)
==============================================================================

【檔案定位】
本檔案為 `artillery.offline.src.localization` 模組的核心外觀（Facade）。
對外提供給執行期 `runner.py` 呼叫的統一協調類別 `MapLocalizer`，封裝所有底層
圖資讀取、特徵匹配、視覺里程計與地面座標投影細節。

【核心功能】
1. Session 生命週期管理：
   - 載入參考照片之 GPS/姿態資訊，初始化 GeoTIFF 航拍圖資索引。
   - 透過 `NearbyMapLocalizer` 執行照片與航拍地圖局部匹配；若視覺比對失敗且啟用
     `--allow-gps-seed`，則以照片 GPS 作為暫估起始點。
2. 逐格姿態追蹤與視覺里程計 (`process`)：
   - 每格計算特徵追蹤，推算無人機軌跡與相機姿態矩陣。
3. 煙霧事件地面座標解算 (`event_position`)：
   - 將起煙影格的煙霧源像素反投射至 GeoTIFF 地平表面，輸出 WGS84 經緯度與 TWD97 座標。
4. 地圖視覺化面板輸出 (`render_panel`)：
   - 輸出包含無人機當前位置、航跡歷史、視野錐體與煙霧事件標記的地圖預覽。
==============================================================================
"""
from __future__ import annotations

import json
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from . import registration, visual_odometry
from .geo_map import GeoMap
from .geometry import camera_heading, estimate_pose, plane_homography, transform
from .map_panel import render_panel
from .photo import PhotoPose, read_bgr, resize_work
from .nearby import NearbyMapLocalizer
from boom.interfaces.localizer import BaseLocalizer


class MapLocalizer(BaseLocalizer):
    """Runner facade holding session state and coordinating localization modules."""

    def __init__(self, root, reference, output, icon, mpp=0.75, ground_height=None,
                 allow_gps_seed=False, matching_config=None):
        self.output = output
        self.photo = PhotoPose.read(reference, ground_height)
        self.nearby = NearbyMapLocalizer(root, matching_config)
        self.map = GeoMap(root, self.photo, output, mpp, nearby=self.nearby)
        self.nearby.submap = self.map.region_selection["selected_submap"]
        self.reference, _ = resize_work(read_bgr(reference))
        self.ref_gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        self.ref_k = self.photo.camera_matrix(self.reference.shape[1], self.reference.shape[0])
        self.rotation = self.photo.rotation()
        self.center = np.array([0., 0., self.photo.height])
        self.ref_h = plane_homography(self.ref_k, self.rotation, self.center)
        self.sift = self.nearby.sift
        self.matcher = self.nearby.matcher
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

    @property
    def heading_deg(self):
        """Validated aircraft heading under the camera/body alignment assumption."""
        if not self.reference_valid or self.status not in ("REFERENCE_MATCHED", "VO_TRACKING"):
            return None
        return camera_heading(self.rotation)

    def matches(self, a, b, mask_a=None, mask_b=None):
        return registration.matches(self, a, b, mask_a, mask_b)

    def register_reference(self):
        result = self.localize(self.reference, self.photo.lon, self.photo.lat)
        accepted = False
        if result.success:
            gps = result.roi.pixel_to_geo(result.map_points[result.inliers])
            world = np.array([self.map.to_en(lon, lat) for lon, lat in gps])
            accepted = self.fit_pose(world, result.image_points[result.inliers], self.ref_k,
                                     max_shift=self.nearby.config.search_radius_m)
            if accepted:
                self.ref_h = plane_homography(self.ref_k, self.rotation, self.center)
        self.record({"stage": "nearby_map_match", "status": result.reason,
                     "candidates": result.candidates, "pose_accepted": accepted})
        if not accepted:
            # Preserve the established attitude-based oblique-photo rectification.
            # Match a nearby crop while keeping the complete display map loaded.
            return registration.register_reference(self)
        self.reference_valid = accepted
        self.status = "REFERENCE_MATCHED" if accepted else "REFERENCE_MATCH_FAILED"
        self.record({"stage": "photo_initialization", "status": self.status,
                     "map_match": result.reason, "candidates": result.candidates,
                     "quality": self.quality, "heading_deg": self.heading_deg,
                     "heading_source": "visual_pose_camera_aligned_with_body",
                     "submap": self.nearby.submap})
        self.ref_center, self.ref_rotation = self.center.copy(), self.rotation.copy()
        print(f"[LOCALIZATION] {self.status}: map_match={result.reason}", flush=True)

    def localize(self, frame, longitude, latitude, *, search_radius_m=None):
        """Reusable frame/prior API; does not overwrite the VO pose or history."""
        return self.nearby.localize(frame, longitude, latitude,
                                   search_radius_m=search_radius_m, debug_dir=self.output)

    @staticmethod
    def pixel_to_geo(result, u, v):
        return result.pixel_to_geo(u, v)

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

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> dict:
        return self.process(frame_bgr, frame_idx, fps=30.0)

    def estimate_ground_coordinates(self, pixel_xy: Tuple[float, float], frame_idx: int) -> Optional[dict]:
        res = self.event_position(event_id=0, frame_idx=frame_idx, pixel=pixel_xy)
        if res is None:
            return None
        lon, lat = res
        return {"wgs84": (lon, lat)}

    def render_map_panel(self, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        h, w = target_size if target_size else (500, 500)
        return self.panel(h, w)

    def close(self):
        self.log.close()
