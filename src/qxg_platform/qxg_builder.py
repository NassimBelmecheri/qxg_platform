from __future__ import annotations

import time
from typing import Any

import numpy as np

from qxg_platform.domain import TrackedObject
from qxg_platform.qualinet import QualiNetRelationConstructor
from qxg_platform.qxg_core import build_spatiotemporal_graph


class QXGBuilder:
    def __init__(
        self,
        analysis_config: dict[str, Any],
        camera_object: TrackedObject | None = None,
        qualinet_constructor: QualiNetRelationConstructor | None = None,
    ):
        self.config = analysis_config
        self.camera_object = camera_object
        self.qualinet_constructor = qualinet_constructor
        self.reasoning_mode = str(analysis_config.get("reasoning_mode", "3d"))
        self.relations_graph: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
        self.object_attributes_history: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self.current_frame_relations: dict[tuple[str, str], dict[str, Any]] = {}

    def build(
        self, tracked_objects: list[TrackedObject], frame_idx: int, frame: np.ndarray | None = None
    ) -> tuple[dict, list[TrackedObject]]:
        valid_objects = self._valid_objects(tracked_objects)
        if self.camera_object is not None:
            self.camera_object.update_history(time.time())
            valid_objects.insert(0, self.camera_object)

        self.current_frame_relations = {}
        if len(valid_objects) < 2:
            return {}, valid_objects

        object_ids = np.array([obj.tracking_id for obj in valid_objects], dtype=np.int64)
        position_histories = [
            np.asarray(obj.position_history, dtype=float) for obj in valid_objects
        ]
        timestamp_histories = [
            np.asarray(obj.timestamp_history, dtype=float) for obj in valid_objects
        ]
        kwargs: dict[str, np.ndarray] = {}

        if self.reasoning_mode == "3d":
            kwargs["bev_boxes"] = np.asarray([obj.bev_bbox for obj in valid_objects], dtype=float)
            kwargs["bev_centers"] = np.asarray(
                [obj.bev_center for obj in valid_objects], dtype=float
            )
        else:
            kwargs["bboxes_2d"] = np.asarray([obj.bbox for obj in valid_objects], dtype=float)

        relations = build_spatiotemporal_graph(
            object_ids=object_ids,
            position_histories=position_histories,
            timestamp_histories=timestamp_histories,
            qtc_slack=float(self.config.get("qtc_slack", 0.1)),
            algebras=self.config.get("algebras", {}),
            distance_thresholds=self.config.get("distance_thresholds", {}),
            **kwargs,
        )
        if self.qualinet_constructor is not None and frame is not None:
            relations.update(self.qualinet_constructor.build(frame, valid_objects))

        object_map = {obj.tracking_id: obj for obj in valid_objects}
        for pair, pair_relations in relations.items():
            self.relations_graph.setdefault(pair, []).append((frame_idx, pair_relations))
            left, right = pair
            left_key = f"{object_map[left].category}_{left}"
            right_key = f"{object_map[right].category}_{right}"
            self.current_frame_relations[tuple(sorted((left_key, right_key)))] = pair_relations
        return relations, valid_objects

    def _valid_objects(self, tracked_objects: list[TrackedObject]) -> list[TrackedObject]:
        if self.reasoning_mode == "3d":
            return [
                obj
                for obj in tracked_objects
                if obj.bev_bbox is not None
                and obj.bev_center is not None
                and len(obj.position_history) > 0
            ]
        return [obj for obj in tracked_objects if len(obj.position_history) > 0]
