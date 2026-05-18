import time

import numpy as np

from qxg_platform.domain import TrackedObject, create_camera_object
from qxg_platform.qxg_builder import QXGBuilder


def test_builder_includes_camera_and_relation_history() -> None:
    config = {
        "camera": {"include_camera_object": True, "camera_id": 0},
        "analysis": {},
    }
    camera = create_camera_object(config)
    obj = TrackedObject(
        tracking_id=7,
        category="person",
        bbox=np.array([10, 10, 20, 40], dtype=float),
        frame_id=1,
    )
    obj.set_world_coord(np.array([1.0, 0.0, 2.0]))
    obj.bev_center = np.array([1.0, 2.0])
    obj.bev_bbox = np.array([0.5, 2.0, 1.5, 2.5])
    obj.update_history(time.time())

    builder = QXGBuilder(
        {
            "reasoning_mode": "3d",
            "qtc_slack": 0.1,
            "algebras": {"distance": True, "RA": True},
            "distance_thresholds": {"very_close": 0.5, "close": 5.0, "normal": 10.0},
        },
        camera,
    )
    relations, objects = builder.build([obj], frame_idx=1)
    assert len(objects) == 2
    assert (0, 7) in relations
    assert builder.relations_graph[(0, 7)][0][0] == 1
