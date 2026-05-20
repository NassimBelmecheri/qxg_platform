from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from qxg_platform.domain import TrackedObject
from qxg_platform.yolo_models import ensure_yolo_model

LOGGER = logging.getLogger(__name__)


class DetectionTracker:
    def __init__(self, detection_config: dict[str, Any], reasoning_mode: str = "3d"):
        self.config = detection_config
        self.reasoning_mode = reasoning_mode
        self.objects: dict[int, TrackedObject] = {}
        self.frame_count = 0
        self.enabled = bool(detection_config.get("enabled", True))
        self.model = None
        self.classes: dict[int, str] = {}
        self._last_empty_log_frame = 0
        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install qxg-platform[ml] to enable YOLO detection") from exc

        weights = ensure_yolo_model(self.config)
        LOGGER.info("Loading YOLO model weights=%s", weights)
        self.model = YOLO(str(weights))
        self.classes = self.model.names
        LOGGER.info("YOLO model loaded with %s classes", len(self.classes))

    def process_frame(
        self, frame: np.ndarray, world_info: Any | None = None
    ) -> list[TrackedObject]:
        if not self.enabled:
            return []
        if self.model is None:
            raise RuntimeError("Detection model is not initialized")

        self.frame_count += 1
        tracker_path = self.config.get("tracker_config")
        results = self.model.track(
            frame,
            conf=float(self.config.get("confidence_threshold", 0.3)),
            imgsz=int(self.config.get("image_size", 1024)),
            verbose=False,
            persist=bool(self.config.get("persist_tracks", True)),
            tracker=str(tracker_path) if tracker_path else None,
        )

        for obj in self.objects.values():
            obj.is_active = False

        active: list[TrackedObject] = []
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            self._log_empty_result("no boxes returned")
            return active

        allowed_classes = set(self.config.get("classes", []))
        box_data = boxes.cpu().numpy()
        raw_count = len(box_data)
        kept_count = 0
        seen_tracking_ids: set[int] = set()
        kept_detections: list[tuple[str, np.ndarray]] = []
        duplicate_iou_threshold = float(self.config.get("duplicate_iou_threshold", 0.85))
        for index in range(raw_count):
            tracking_id = self._tracking_id_for_box(box_data, index)
            if tracking_id in seen_tracking_ids:
                LOGGER.debug(
                    "Skipping duplicate detection with repeated tracking_id=%s at frame=%s",
                    tracking_id,
                    self.frame_count,
                )
                continue
            category = self.classes[int(box_data.cls[index])]
            if allowed_classes and category not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box_data.xyxy[index])
            bbox = np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)
            if is_duplicate_detection(
                category, bbox, kept_detections, duplicate_iou_threshold
            ):
                LOGGER.debug(
                    "Skipping duplicate %s detection at frame=%s bbox=%s",
                    category,
                    self.frame_count,
                    bbox.tolist(),
                )
                continue
            kept_count += 1
            seen_tracking_ids.add(tracking_id)
            kept_detections.append((category, bbox))
            obj = self.objects.get(tracking_id)
            if obj is None:
                obj = TrackedObject(
                    tracking_id,
                    category,
                    bbox,
                    float(box_data.conf[index]),
                    self.frame_count,
                )
                self.objects[tracking_id] = obj

            obj.is_active = True
            obj.category = category
            obj.bbox = bbox
            obj.confidence = float(box_data.conf[index])
            obj.last_seen_frame = self.frame_count

            if self.reasoning_mode == "3d" and world_info is not None:
                self._update_3d_state(obj, world_info)
            obj.update_history(time.time())
            active.append(obj)

        if kept_count == 0:
            self._log_empty_result(
                f"{raw_count} raw detections, 0 kept after class filter {sorted(allowed_classes)}"
            )
        elif self.frame_count % 30 == 0:
            LOGGER.info(
                "Detection frame=%s raw=%s kept=%s",
                self.frame_count,
                raw_count,
                kept_count,
            )

        for object_id in [
            object_id
            for object_id, obj in self.objects.items()
            if not obj.is_active and self.frame_count - obj.last_seen_frame > 50
        ]:
            del self.objects[object_id]

        return active

    def _tracking_id_for_box(self, box_data: Any, index: int) -> int:
        if getattr(box_data, "id", None) is not None and box_data.id[index] is not None:
            return int(box_data.id[index])
        # Some Ultralytics tracker/model combinations return boxes before IDs are assigned.
        # Use stable per-frame slots so detections are still visible and graphable.
        return int(index + 1)

    def _log_empty_result(self, reason: str) -> None:
        if self.frame_count - self._last_empty_log_frame >= 30:
            LOGGER.warning("No active detections at frame=%s: %s", self.frame_count, reason)
            self._last_empty_log_frame = self.frame_count

    def _update_3d_state(self, obj: TrackedObject, world_info: Any) -> None:
        intrinsics = None
        if hasattr(world_info, "profile"):
            try:
                intrinsics = world_info.profile.as_video_stream_profile().get_intrinsics()
            except AttributeError:
                intrinsics = None
        elif hasattr(world_info, "get_intrinsics"):
            intrinsics = world_info.get_intrinsics()
        if intrinsics is None:
            return

        depth = robust_depth(world_info, obj.bbox)
        if depth <= 0.1:
            return

        x, y, w, h = obj.bbox
        center_x = x + w / 2
        center_y = y + h / 2
        cam_x = (center_x - intrinsics.ppx) * depth / intrinsics.fx
        cam_y = (center_y - intrinsics.ppy) * depth / intrinsics.fy
        p_cam_center = np.array([cam_x, cam_y, depth], dtype=float)
        obj.set_world_coord(p_cam_center)
        obj.real_width = float(w / intrinsics.fx * depth)
        obj.real_height = float(h / intrinsics.fy * depth)
        obj.real_depth = obj.real_width * 0.8
        obj.set_attribute("p_cam_center", p_cam_center)
        obj.set_attribute(
            "camera_intrinsics",
            {
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "ppx": float(intrinsics.ppx),
                "ppy": float(intrinsics.ppy),
            },
        )
        obj.bev_center = np.array([cam_x, depth], dtype=float)
        obj.bev_bbox = np.array(
            [cam_x - obj.real_width / 2, depth, cam_x + obj.real_width / 2, depth + obj.real_depth],
            dtype=float,
        )


def is_duplicate_detection(
    category: str,
    bbox: np.ndarray,
    kept_detections: list[tuple[str, np.ndarray]],
    iou_threshold: float,
) -> bool:
    if iou_threshold <= 0:
        return False
    return any(
        category == kept_category and bbox_iou(bbox, kept_bbox) >= iou_threshold
        for kept_category, kept_bbox in kept_detections
    )


def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_x1, left_y1, left_w, left_h = map(float, left)
    right_x1, right_y1, right_w, right_h = map(float, right)
    left_x2 = left_x1 + max(0.0, left_w)
    left_y2 = left_y1 + max(0.0, left_h)
    right_x2 = right_x1 + max(0.0, right_w)
    right_y2 = right_y1 + max(0.0, right_h)

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    left_area = max(0.0, left_w) * max(0.0, left_h)
    right_area = max(0.0, right_w) * max(0.0, right_h)
    union = left_area + right_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def robust_depth(world_info: Any, bbox: np.ndarray) -> float:
    depth_frame = (
        world_info.get_depth_frame()
        if hasattr(world_info, "get_depth_frame")
        else world_info
    )
    if not hasattr(depth_frame, "get_data"):
        return 0.0
    depth_map = np.asarray(depth_frame.get_data())
    if depth_map.ndim != 2:
        return 0.0
    x, y, w, h = map(int, bbox)
    x1, x2 = max(0, x + w // 4), min(depth_map.shape[1], x + 3 * w // 4)
    y1, y2 = max(0, y + h // 4), min(depth_map.shape[0], y + 3 * h // 4)
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    valid = roi[roi > 0]
    if valid.size == 0:
        return 0.0
    value = float(np.median(valid))
    return value / 1000.0 if depth_map.dtype == np.uint16 else value
