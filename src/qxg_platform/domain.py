from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrackedObject:
    tracking_id: int
    category: str
    bbox: np.ndarray
    confidence: float = 1.0
    frame_id: int = 0
    history_size: int = 100
    world_coord: np.ndarray | None = None
    bev_bbox: np.ndarray | None = None
    bev_center: np.ndarray | None = None
    real_width: float | None = None
    real_depth: float | None = None
    real_height: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def __post_init__(self) -> None:
        self.last_seen_frame = self.frame_id
        self.position_history: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.timestamp_history: deque[float] = deque(maxlen=self.history_size)

    def set_world_coord(self, coord: np.ndarray) -> None:
        self.world_coord = np.asarray(coord, dtype=float)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def get_attribute(self, key: str) -> Any:
        return self.attributes.get(key)

    def update_history(self, timestamp: float) -> None:
        if self.world_coord is not None:
            position = np.asarray(self.world_coord, dtype=float).reshape(-1)[:3]
        else:
            x, y, w, h = self.bbox
            position = np.array([x + w / 2, y + h / 2, 0.0], dtype=float)
        self.position_history.append(position)
        self.timestamp_history.append(float(timestamp))


def create_camera_object(config: dict) -> TrackedObject | None:
    camera_config = config.get("camera", {})
    if not camera_config.get("include_camera_object", True):
        return None
    width = float(camera_config.get("width_m", 0.5))
    depth = float(camera_config.get("depth_m", 0.5))
    camera = TrackedObject(
        tracking_id=int(camera_config.get("camera_id", 0)),
        category="camera",
        bbox=np.array([0, 0, 1, 1], dtype=float),
        confidence=1.0,
        frame_id=0,
    )
    camera.set_world_coord(np.array([0.0, 0.0, 0.0], dtype=float))
    camera.bev_center = np.array([0.0, 0.0], dtype=float)
    camera.bev_bbox = np.array([-width / 2, 0.0, width / 2, depth], dtype=float)
    return camera
