import numpy as np

from qxg_platform.qxg_core import allen_relation, build_spatiotemporal_graph, distance_label


def test_allen_before_relation() -> None:
    assert allen_relation(np.array([0.0, 1.0]), np.array([3.0, 4.0]), 0.01) == "B"


def test_distance_labels_thresholds() -> None:
    thresholds = {"very_close": 0.5, "close": 2.0, "normal": 5.0}
    assert distance_label(0.25, thresholds) == "very close"
    assert distance_label(1.0, thresholds) == "close"
    assert distance_label(4.0, thresholds) == "normal"
    assert distance_label(8.0, thresholds) == "far"


def test_build_graph_has_distance_and_qtc() -> None:
    object_ids = np.array([0, 1], dtype=np.int64)
    positions = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[3.0, 0.0, 4.0], [2.0, 0.0, 3.0]]),
    ]
    timestamps = [np.array([1.0, 2.0]), np.array([1.0, 2.0])]
    relations = build_spatiotemporal_graph(
        object_ids=object_ids,
        position_histories=positions,
        timestamp_histories=timestamps,
        qtc_slack=0.01,
        algebras={"distance": True, "QTC": True},
        distance_thresholds={"very_close": 0.5, "close": 5.0, "normal": 10.0},
        bev_centers=np.array([[0.0, 0.0], [2.0, 3.0]]),
    )
    assert relations[(0, 1)]["distance"] == "close"
    assert relations[(0, 1)]["QTC_x"] == "moving towards"
    assert relations[(0, 1)]["QTC_y"] == "moving towards"


def test_build_graph_has_ra_for_bev_boxes() -> None:
    relations = build_spatiotemporal_graph(
        object_ids=np.array([0, 1], dtype=np.int64),
        position_histories=[
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 2.0]]),
        ],
        timestamp_histories=[np.array([1.0]), np.array([1.0])],
        qtc_slack=0.01,
        algebras={"RA": True},
        distance_thresholds={},
        bev_boxes=np.array([[-0.25, 0.0, 0.25, 0.5], [0.5, 2.0, 1.5, 3.0]]),
    )
    assert relations[(0, 1)]["RA"] == ("Bx", "By")
