from __future__ import annotations

import numpy as np

from qxg_platform.domain import TrackedObject, create_camera_object
from qxg_platform.qualinet import QualiNetRelationConstructor
from qxg_platform.qxg_builder import QXGBuilder


def test_qualinet_geometry_fallback_builds_camera_object_ra_qdc() -> None:
    constructor = QualiNetRelationConstructor(
        {
            "enabled": True,
            "allow_geometry_fallback": True,
            "ra_model_path": "",
            "qdc_model_path": "",
        },
        camera_id=0,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    obj = TrackedObject(
        tracking_id=5,
        category="person",
        bbox=np.array([65.0, 65.0, 25.0, 25.0]),
    )

    relations = constructor.build(frame, [obj])

    assert relations[(0, 5)]["RA"] == ("right", "below")
    assert relations[(0, 5)]["QDC"] == "normal"


def test_builder_merges_qualinet_relations() -> None:
    camera = create_camera_object({"camera": {"include_camera_object": True, "camera_id": 0}})
    constructor = QualiNetRelationConstructor(
        {"enabled": True, "allow_geometry_fallback": True},
        camera_id=0,
    )
    builder = QXGBuilder(
        {
            "reasoning_mode": "2d",
            "algebras": {},
            "distance_thresholds": {},
        },
        camera,
        constructor,
    )
    obj = TrackedObject(
        tracking_id=9,
        category="car",
        bbox=np.array([5.0, 5.0, 20.0, 20.0]),
    )
    obj.update_history(1.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    relations, _objects = builder.build([obj], frame_idx=1, frame=frame)

    assert relations[(0, 9)]["RA"] == ("left", "above")
    assert "QDC" in relations[(0, 9)]
