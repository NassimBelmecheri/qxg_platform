from __future__ import annotations

import numpy as np

from qxg_platform.detection import bbox_iou, is_duplicate_detection


def test_bbox_iou_for_overlapping_boxes() -> None:
    left = np.array([10.0, 10.0, 20.0, 20.0])
    right = np.array([12.0, 12.0, 20.0, 20.0])

    assert bbox_iou(left, right) > 0.65


def test_duplicate_detection_suppresses_same_class_high_iou() -> None:
    kept = [("person", np.array([10.0, 10.0, 20.0, 20.0]))]

    assert is_duplicate_detection(
        "person",
        np.array([11.0, 11.0, 20.0, 20.0]),
        kept,
        0.80,
    )


def test_duplicate_detection_keeps_different_class_or_low_iou() -> None:
    kept = [("person", np.array([10.0, 10.0, 20.0, 20.0]))]

    assert not is_duplicate_detection(
        "car",
        np.array([11.0, 11.0, 20.0, 20.0]),
        kept,
        0.80,
    )
    assert not is_duplicate_detection(
        "person",
        np.array([80.0, 80.0, 20.0, 20.0]),
        kept,
        0.80,
    )
