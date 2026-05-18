from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from qxg_platform.serialization import object_to_dict, relation_keys_to_json

LOGGER = logging.getLogger(__name__)


class QXGExporter:
    def __init__(self, config: dict[str, Any], reasoning_mode: str):
        self.enabled = bool(config.get("enabled", False))
        self.reasoning_mode = reasoning_mode
        self.frames: list[dict[str, Any]] = []
        self.output_path: Path | None = None
        if not self.enabled:
            return

        output_dir = Path(str(config.get("output_dir", "qxg_exports"))).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str(config.get("filename", f"qxg_graph_{timestamp}.json"))
        self.output_path = (output_dir / filename).resolve()
        LOGGER.info("QXG graph export enabled: %s", self.output_path)

    def add_frame(
        self,
        frame_idx: int,
        objects: list[Any],
        relevant_objects: list[Any],
        relations: dict[tuple[int, int], dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return
        self.frames.append(
            {
                "frame_idx": int(frame_idx),
                "objects": [object_to_dict(obj) for obj in objects],
                "relevant_object_ids": [int(obj.tracking_id) for obj in relevant_objects],
                "relations": relation_keys_to_json(relations),
            }
        )

    def save(self) -> None:
        if not self.enabled or self.output_path is None:
            return
        payload = {
            "schema": "qxg-platform.graph.v1",
            "reasoning_mode": self.reasoning_mode,
            "frames": self.frames,
        }
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved QXG graph export to %s", self.output_path)
