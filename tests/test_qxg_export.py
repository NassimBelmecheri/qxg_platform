from __future__ import annotations

import json

import numpy as np

from qxg_platform.domain import TrackedObject
from qxg_platform.qxg_export import QXGExporter


def test_qxg_exporter_writes_graph_json(tmp_path) -> None:
    exporter = QXGExporter({"enabled": True, "output_dir": str(tmp_path)}, "2d")
    obj = TrackedObject(
        tracking_id=7,
        category="person",
        bbox=np.array([1.0, 2.0, 3.0, 4.0]),
        confidence=0.8,
    )

    exporter.add_frame(
        3,
        [obj],
        [obj],
        {(0, 7): {"distance": "close", "qtc": ["0", "+"]}},
    )
    exporter.save()

    export_path = next(tmp_path.glob("*.json"))
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "qxg-platform.graph.v1"
    assert payload["reasoning_mode"] == "2d"
    assert payload["frames"][0]["objects"][0]["tracking_id"] == 7
    assert payload["frames"][0]["relations"]["0:7"]["distance"] == "close"
