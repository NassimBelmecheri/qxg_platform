from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Any

import numpy as np

ALLEN_ERROR = "Error"


def allen_relation(interval_a: np.ndarray, interval_b: np.ndarray, slack: float) -> str:
    a0, a1 = float(interval_a[0]), float(interval_a[1])
    b0, b1 = float(interval_b[0]), float(interval_b[1])
    if abs(a0 - b0) <= slack and abs(a1 - b1) <= slack:
        return "E"
    if abs(a0 - b0) <= slack and a1 < b1 - slack:
        return "S"
    if abs(a0 - b0) <= slack and a1 > b1 + slack:
        return "SI"
    if a0 > b0 + slack and abs(a1 - b1) <= slack:
        return "F"
    if a0 < b0 - slack and abs(a1 - b1) <= slack:
        return "FI"
    if abs(a1 - b0) <= slack:
        return "M"
    if abs(a0 - b1) <= slack:
        return "MI"
    if a1 < b0 - slack:
        return "B"
    if a0 > b1 + slack:
        return "BI"
    if a0 > b0 + slack and a1 < b1 - slack:
        return "D"
    if a0 < b0 - slack and a1 > b1 + slack:
        return "DI"
    if a0 < b0 - slack < a1 < b1 - slack:
        return "O"
    if b0 + slack < a0 < b1 + slack and a1 > b1 + slack:
        return "OI"
    return ALLEN_ERROR


def rectangle_relation(box_a: np.ndarray, box_b: np.ndarray, slack: float) -> tuple[str, str]:
    return (
        allen_relation(np.array([box_a[0], box_a[2]]), np.array([box_b[0], box_b[2]]), slack)
        + "x",
        allen_relation(np.array([box_a[1], box_a[3]]), np.array([box_b[1], box_b[3]]), slack)
        + "y",
    )


def qtc_component(current: float, previous: float, slack: float) -> str:
    if current < previous - slack:
        return "moving towards"
    if current > previous + slack:
        return "moving away"
    return "stationary"


def qtc_relation(history_a: np.ndarray, history_b: np.ndarray, slack: float) -> tuple[str, str]:
    if len(history_a) < 2 or len(history_b) < 2:
        return "unknown", "unknown"
    curr_x = abs(history_a[-1, 0] - history_b[-1, 0])
    prev_x = abs(history_a[-2, 0] - history_b[-2, 0])
    curr_z = abs(history_a[-1, 2] - history_b[-1, 2])
    prev_z = abs(history_a[-2, 2] - history_b[-2, 2])
    return qtc_component(curr_x, prev_x, slack), qtc_component(curr_z, prev_z, slack)


def distance_label(distance: float, thresholds: dict[str, float]) -> str:
    if distance <= float(thresholds.get("very_close", 0.5)):
        return "very close"
    if distance <= float(thresholds.get("close", 7.5)):
        return "close"
    if distance <= float(thresholds.get("normal", 10.0)):
        return "normal"
    return "far"


def build_spatiotemporal_graph(
    object_ids: np.ndarray,
    position_histories: list[np.ndarray],
    timestamp_histories: list[np.ndarray],
    qtc_slack: float,
    algebras: dict[str, bool],
    distance_thresholds: dict[str, float],
    bev_boxes: np.ndarray | None = None,
    bev_centers: np.ndarray | None = None,
    bboxes_2d: np.ndarray | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    del timestamp_histories
    relations_by_pair: dict[tuple[int, int], dict[str, Any]] = {}

    for left, right in combinations(range(len(object_ids)), 2):
        pair_relations: dict[str, Any] = {}
        pair_key = tuple(sorted((int(object_ids[left]), int(object_ids[right]))))

        if algebras.get("RA", False) and bev_boxes is not None:
            pair_relations["RA"] = rectangle_relation(bev_boxes[left], bev_boxes[right], 0.1)

        if algebras.get("distance", False):
            if bev_centers is not None:
                dx = float(bev_centers[left, 0] - bev_centers[right, 0])
                dz = float(bev_centers[left, 1] - bev_centers[right, 1])
                pair_relations["distance"] = distance_label(
                    sqrt(dx * dx + dz * dz), distance_thresholds
                )
            elif bboxes_2d is not None:
                c1 = np.array(
                    [
                        bboxes_2d[left, 0] + bboxes_2d[left, 2] / 2,
                        bboxes_2d[left, 1] + bboxes_2d[left, 3] / 2,
                    ]
                )
                c2 = np.array(
                    [
                        bboxes_2d[right, 0] + bboxes_2d[right, 2] / 2,
                        bboxes_2d[right, 1] + bboxes_2d[right, 3] / 2,
                    ]
                )
                pair_relations["distance"] = "close" if np.linalg.norm(c1 - c2) <= 300 else "far"

        if algebras.get("QTC", False):
            qtc_x, qtc_y = qtc_relation(
                position_histories[left], position_histories[right], qtc_slack
            )
            pair_relations["QTC_x"] = qtc_x
            pair_relations["QTC_y"] = qtc_y

        if pair_relations:
            relations_by_pair[pair_key] = pair_relations

    return relations_by_pair
