from __future__ import annotations

import math
from collections import Counter

import cv2
import numpy as np

from qxg_platform.domain import TrackedObject


class Visualizer:
    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.window_name = str(config.get("window_name", "QXG Platform"))
        self.width = int(config.get("dashboard_width", 1440))
        self.height = int(config.get("dashboard_height", 900))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            "bg": (24, 26, 30),
            "panel": (35, 38, 45),
            "muted": (150, 156, 166),
            "text": (232, 235, 240),
            "accent": (0, 215, 255),
            "object": (255, 190, 60),
            "camera": (90, 110, 255),
            "edge": (130, 136, 148),
            "danger": (80, 90, 255),
        }
        if self.enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)

    def display(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        relevant_objects: list[TrackedObject],
        relations: dict[tuple[int, int], dict],
    ) -> int:
        if not self.enabled:
            return -1

        dashboard = np.full((self.height, self.width, 3), self.colors["bg"], dtype=np.uint8)
        left_w = int(self.width * 0.58)
        top_h = int(self.height * 0.64)
        right_w = self.width - left_w
        bottom_h = self.height - top_h

        camera_panel = dashboard[0:top_h, 0:left_w]
        bev_panel = dashboard[top_h : self.height, 0 : left_w // 2]
        graph_panel = dashboard[top_h : self.height, left_w // 2 : left_w]
        side_panel = dashboard[0 : self.height, left_w : self.width]

        relevant_ids = {obj.tracking_id for obj in relevant_objects}
        self._draw_camera(camera_panel, frame, objects, relevant_ids)
        self._draw_bev(bev_panel, objects, relevant_ids)
        self._draw_graph(graph_panel, objects, relations, relevant_ids)
        self._draw_side_panel(side_panel, objects, relevant_objects, relations)
        self._draw_panel_borders(dashboard, left_w, top_h, right_w, bottom_h)

        cv2.imshow(self.window_name, dashboard)
        return cv2.waitKey(1) & 0xFF

    def _draw_camera(
        self,
        panel: np.ndarray,
        frame: np.ndarray,
        objects: list[TrackedObject],
        relevant_ids: set[int],
    ) -> None:
        self._fill_panel(panel, "Camera")
        view, offset_x, offset_y, scale = self._letterbox_info(
            frame, panel.shape[1], panel.shape[0] - 42
        )
        y_offset = 42
        panel[y_offset : y_offset + view.shape[0], 0 : view.shape[1]] = view
        for obj in objects:
            if obj.category == "camera":
                continue
            x, y, w, h = [int(v) for v in obj.bbox]
            color = (
                self.colors["accent"]
                if obj.tracking_id in relevant_ids
                else self.colors["object"]
            )
            p1 = (int(offset_x + x * scale), int(y_offset + offset_y + y * scale))
            p2 = (
                int(offset_x + (x + w) * scale),
                int(y_offset + offset_y + (y + h) * scale),
            )
            if not self._draw_3d_box(panel, obj, y_offset, offset_x, offset_y, scale, color):
                cv2.rectangle(panel, p1, p2, color, 2)
            self._label(
                panel,
                f"{obj.category} {obj.tracking_id}",
                p1[0],
                max(62, p1[1] - 8),
                color,
            )

    def _draw_bev(
        self,
        panel: np.ndarray,
        objects: list[TrackedObject],
        relevant_ids: set[int],
    ) -> None:
        self._fill_panel(panel, "Bird's-Eye View")
        h, w = panel.shape[:2]
        origin = np.array([w // 2, h - 36])
        scale = min(w / 16.0, h / 12.0)
        cv2.line(panel, tuple(origin), (origin[0], 52), self.colors["muted"], 1)
        cv2.line(panel, (20, origin[1]), (w - 20, origin[1]), self.colors["muted"], 1)
        cv2.circle(panel, tuple(origin), 8, self.colors["camera"], -1)
        self._label(panel, "ego", origin[0] + 10, origin[1] - 8, self.colors["text"], 0.45)

        for obj in objects:
            if obj.bev_center is None and obj.bev_bbox is None:
                continue
            color = (
                self.colors["accent"]
                if obj.tracking_id in relevant_ids
                else self.colors["object"]
            )
            if obj.bev_bbox is not None:
                x1, z1, x2, z2 = obj.bev_bbox
                p1 = self._bev_to_pixel(origin, scale, x1, z1)
                p2 = self._bev_to_pixel(origin, scale, x2, z2)
                cv2.rectangle(
                    panel,
                    (min(p1[0], p2[0]), min(p1[1], p2[1])),
                    (max(p1[0], p2[0]), max(p1[1], p2[1])),
                    color,
                    2,
                )
                self._label(panel, str(obj.tracking_id), p1[0], p1[1] - 6, color, 0.45)

    def _draw_graph(
        self,
        panel: np.ndarray,
        objects: list[TrackedObject],
        relations: dict[tuple[int, int], dict],
        relevant_ids: set[int],
    ) -> None:
        self._fill_panel(panel, "Qualitative Graph")
        ids = sorted({obj.tracking_id for obj in objects})
        if not ids:
            self._label(panel, "No active graph yet", 20, 72, self.colors["muted"])
            return
        positions = self._node_positions(ids, panel.shape[1], panel.shape[0])
        for (left, right), relation in relations.items():
            if left in positions and right in positions:
                color = (
                    self.colors["accent"]
                    if left in relevant_ids or right in relevant_ids
                    else self.colors["edge"]
                )
                cv2.line(panel, positions[left], positions[right], color, 2)
                midpoint = (
                    (positions[left][0] + positions[right][0]) // 2,
                    (positions[left][1] + positions[right][1]) // 2,
                )
                if "RA" in relation:
                    self._label(
                        panel,
                        f"RA {relation['RA']}",
                        midpoint[0] - 34,
                        midpoint[1] - 6,
                        color,
                        0.4,
                    )
        category_by_id = {obj.tracking_id: obj.category for obj in objects}
        for node_id, point in positions.items():
            is_relevant = node_id in relevant_ids
            is_camera = category_by_id.get(node_id) == "camera"
            color_key = "camera" if is_camera else "accent" if is_relevant else "object"
            color = self.colors[color_key]
            cv2.circle(panel, point, 17, color, -1)
            cv2.circle(panel, point, 17, self.colors["text"], 1)
            text = str(node_id)
            cv2.putText(panel, text, (point[0] - 8, point[1] + 6), self.font, 0.5, (20, 20, 20), 1)

    def _draw_side_panel(
        self,
        panel: np.ndarray,
        objects: list[TrackedObject],
        relevant_objects: list[TrackedObject],
        relations: dict[tuple[int, int], dict],
    ) -> None:
        self._fill_panel(panel, "Runtime")
        y = 70
        metrics = [
            ("objects", len(objects)),
            ("relevant", len(relevant_objects)),
            ("relations", len(relations)),
        ]
        for label, value in metrics:
            self._metric(panel, label, str(value), 24, y)
            y += 58

        y += 10
        self._label(panel, "Classes", 24, y, self.colors["text"], 0.65)
        y += 28
        for category, count in Counter(obj.category for obj in objects).most_common(8):
            self._label(panel, f"{category:<14} {count}", 28, y, self.colors["muted"], 0.52)
            y += 24

        y += 18
        self._label(panel, "Latest Relations", 24, y, self.colors["text"], 0.65)
        y += 28
        if not relations:
            self._label(panel, "Waiting for two tracked objects", 28, y, self.colors["muted"], 0.5)
            return
        for (left, right), relation in list(relations.items())[-9:]:
            ra = relation.get("RA", "-")
            distance = relation.get("distance", "-")
            qtc_x = relation.get("QTC_x", "-")
            text = f"{left}-{right}: RA={ra}, D={distance}, Q={qtc_x}"
            self._label(panel, text[:46], 28, y, self.colors["muted"], 0.48)
            y += 23

    def _fill_panel(self, panel: np.ndarray, title: str) -> None:
        panel[:] = self.colors["panel"]
        cv2.putText(panel, title, (18, 28), self.font, 0.72, self.colors["text"], 2)

    def _draw_panel_borders(
        self, dashboard: np.ndarray, left_w: int, top_h: int, right_w: int, bottom_h: int
    ) -> None:
        del right_w, bottom_h
        cv2.line(dashboard, (left_w, 0), (left_w, self.height), self.colors["bg"], 3)
        cv2.line(dashboard, (0, top_h), (left_w, top_h), self.colors["bg"], 3)
        cv2.line(dashboard, (left_w // 2, top_h), (left_w // 2, self.height), self.colors["bg"], 3)

    def _metric(self, panel: np.ndarray, label: str, value: str, x: int, y: int) -> None:
        cv2.rectangle(panel, (x, y - 30), (panel.shape[1] - 24, y + 12), (45, 49, 58), -1)
        self._label(panel, label.upper(), x + 12, y - 6, self.colors["muted"], 0.45)
        self._label(panel, value, panel.shape[1] - 90, y - 2, self.colors["accent"], 0.72)

    def _label(
        self,
        panel: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        scale: float = 0.55,
    ) -> None:
        cv2.putText(panel, text, (int(x), int(y)), self.font, scale, color, 1, cv2.LINE_AA)

    def _letterbox(self, frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        canvas, _, _, _ = self._letterbox_info(frame, target_w, target_h)
        return canvas

    def _letterbox_info(
        self, frame: np.ndarray, target_w: int, target_h: int
    ) -> tuple[np.ndarray, int, int, float]:
        frame_h, frame_w = frame.shape[:2]
        scale = min(target_w / frame_w, target_h / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), self.colors["bg"], dtype=np.uint8)
        y = (target_h - new_h) // 2
        x = (target_w - new_w) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas, x, y, scale

    def _draw_3d_box(
        self,
        panel: np.ndarray,
        obj: TrackedObject,
        y_offset: int,
        offset_x: int,
        offset_y: int,
        scale: float,
        color: tuple[int, int, int],
    ) -> bool:
        corners = self._project_3d_box(obj)
        if corners is None:
            return False
        points = np.empty_like(corners, dtype=np.int32)
        points[:, 0] = (offset_x + corners[:, 0] * scale).astype(np.int32)
        points[:, 1] = (y_offset + offset_y + corners[:, 1] * scale).astype(np.int32)
        front = points[:4]
        back = points[4:]
        cv2.polylines(panel, [front], True, color, 2)
        cv2.polylines(panel, [back], True, color, 1)
        for index in range(4):
            cv2.line(panel, tuple(front[index]), tuple(back[index]), color, 1)
        return True

    def _project_3d_box(self, obj: TrackedObject) -> np.ndarray | None:
        center = obj.get_attribute("p_cam_center")
        intrinsics = obj.get_attribute("camera_intrinsics")
        if (
            center is None
            or intrinsics is None
            or obj.real_width is None
            or obj.real_height is None
            or obj.real_depth is None
        ):
            return None
        center = np.asarray(center, dtype=float)
        if center[2] <= 0.1:
            return None
        half_w = obj.real_width / 2
        half_h = obj.real_height / 2
        depth = obj.real_depth
        x, y, z = center
        corners_3d = np.array(
            [
                [x - half_w, y - half_h, z],
                [x + half_w, y - half_h, z],
                [x + half_w, y + half_h, z],
                [x - half_w, y + half_h, z],
                [x - half_w, y - half_h, z + depth],
                [x + half_w, y - half_h, z + depth],
                [x + half_w, y + half_h, z + depth],
                [x - half_w, y + half_h, z + depth],
            ],
            dtype=float,
        )
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        ppx = intrinsics["ppx"]
        ppy = intrinsics["ppy"]
        projected = np.empty((8, 2), dtype=float)
        projected[:, 0] = fx * corners_3d[:, 0] / corners_3d[:, 2] + ppx
        projected[:, 1] = fy * corners_3d[:, 1] / corners_3d[:, 2] + ppy
        if not np.isfinite(projected).all():
            return None
        return projected

    def _bev_to_pixel(
        self, origin: np.ndarray, scale: float, x: float, z: float
    ) -> tuple[int, int]:
        return int(origin[0] + x * scale), int(origin[1] - z * scale)

    def _node_positions(
        self, ids: list[int], width: int, height: int
    ) -> dict[int, tuple[int, int]]:
        center = np.array([width // 2, height // 2 + 18])
        radius = max(50, min(width, height) // 2 - 42)
        if len(ids) == 1:
            return {ids[0]: tuple(center)}
        positions = {}
        for index, node_id in enumerate(ids):
            angle = -math.pi / 2 + 2 * math.pi * index / len(ids)
            point = center + np.array([math.cos(angle), math.sin(angle)]) * radius
            positions[node_id] = (int(point[0]), int(point[1]))
        return positions

    def close(self) -> None:
        if self.enabled:
            cv2.destroyWindow(self.window_name)
