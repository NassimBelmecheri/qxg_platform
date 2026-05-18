from __future__ import annotations

import cv2
import numpy as np

from qxg_platform.domain import TrackedObject


class Visualizer:
    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", True))
        self.window_name = str(config.get("window_name", "QXG Platform"))
        if self.enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def display(
        self,
        frame: np.ndarray,
        objects: list[TrackedObject],
        relevant_objects: list[TrackedObject],
        relations: dict[tuple[int, int], dict],
    ) -> int:
        if not self.enabled:
            return -1
        canvas = frame.copy()
        relevant_ids = {obj.tracking_id for obj in relevant_objects}
        for obj in objects:
            if obj.category == "camera":
                continue
            x, y, w, h = [int(v) for v in obj.bbox]
            color = (0, 215, 255) if obj.tracking_id in relevant_ids else (255, 191, 0)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                canvas,
                f"{obj.category} {obj.tracking_id}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.putText(
            canvas,
            f"Objects: {len(objects)}  Relations: {len(relations)}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (230, 230, 230),
            2,
        )
        cv2.imshow(self.window_name, canvas)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self.enabled:
            cv2.destroyWindow(self.window_name)
