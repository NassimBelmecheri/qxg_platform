from __future__ import annotations

import argparse

from flask import Flask, jsonify, request

from qxg_platform.config import load_config
from qxg_platform.detection import DetectionTracker
from qxg_platform.domain import create_camera_object
from qxg_platform.inputs import CameraIntrinsics, DepthFrame
from qxg_platform.qxg_builder import QXGBuilder
from qxg_platform.serialization import (
    decode_array,
    decode_image_jpeg,
    object_to_dict,
    relation_keys_to_json,
)


def create_app(config_path: str) -> Flask:
    platform_config = load_config(config_path)
    camera = create_camera_object(platform_config.raw)
    analysis = platform_config.section("analysis") | {
        "reasoning_mode": platform_config.reasoning_mode
    }
    detector = DetectionTracker(
        platform_config.section("detection"), platform_config.reasoning_mode
    )
    builder = QXGBuilder(analysis, camera)
    app = Flask(__name__)
    state = {"frame_idx": 0}

    @app.post("/process_frame")
    def process_frame():
        payload = request.get_json(force=True)
        frame = decode_image_jpeg(payload["color_frame"])
        world_info = None
        if "depth_map" in payload and "intrinsics" in payload:
            intr = payload["intrinsics"]
            world_info = DepthFrame(
                decode_array(payload["depth_map"]),
                CameraIntrinsics(
                    int(intr["width"]),
                    int(intr["height"]),
                    float(intr["fx"]),
                    float(intr["fy"]),
                    float(intr["ppx"]),
                    float(intr["ppy"]),
                ),
            )
        state["frame_idx"] += 1
        objects = detector.process_frame(frame, world_info)
        relations, all_objects = builder.build(objects, state["frame_idx"])
        return jsonify(
            {
                "objects": [object_to_dict(obj) for obj in all_objects],
                "relations": relation_keys_to_json(relations),
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="QXG processing server")
    parser.add_argument("--config", default="configs/realtime.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app(args.config).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
