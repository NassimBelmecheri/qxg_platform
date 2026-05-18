from __future__ import annotations

from qxg_platform.domain import TrackedObject


class RelevanceSelector:
    def __init__(self, config: dict):
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.mode = str(config.get("mode", "rules"))
        self.top_k = int(config.get("top_k", 2))
        self.threshold = float(config.get("relevance_threshold", 0.4))

    def select(
        self,
        objects: list[TrackedObject],
        relations: dict[tuple[int, int], dict],
        camera_id: int = 0,
    ) -> list[TrackedObject]:
        if not self.enabled:
            return objects
        if self.mode == "rules":
            return self._rule_select(objects, relations, camera_id)
        if self.mode == "sklearn":
            return self._sklearn_select(objects, relations, camera_id)
        return objects

    def _rule_select(
        self,
        objects: list[TrackedObject],
        relations: dict[tuple[int, int], dict],
        camera_id: int,
    ) -> list[TrackedObject]:
        relevant = []
        for obj in objects:
            if obj.tracking_id == camera_id:
                relevant.append(obj)
                continue
            relation = relations.get(tuple(sorted((camera_id, obj.tracking_id))), {})
            if relation.get("distance") in {"very close", "close"}:
                relevant.append(obj)
        return relevant or objects

    def _sklearn_select(
        self,
        objects: list[TrackedObject],
        relations: dict[tuple[int, int], dict],
        camera_id: int,
    ) -> list[TrackedObject]:
        del relations, camera_id
        # Production hook: keep the public contract stable while model-specific feature
        # engineering is added with tests and versioned artifacts.
        return objects[: self.top_k]
