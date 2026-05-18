from __future__ import annotations

import argparse

from qxg_platform.config import load_config
from qxg_platform.inputs import RealtimeInput, RecordingInput
from qxg_platform.platform import QXGPlatform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QXG Platform")
    parser.add_argument("--config", default="configs/video.yaml")
    parser.add_argument("--mode", choices=["local", "remote"], default="local")
    parser.add_argument("--input", choices=["recording", "realsense"], default="recording")
    parser.add_argument("--source", default="D:/nassim/qxg_artifacts/recordings/clip1")
    parser.add_argument("--server-url", default="http://127.0.0.1:5000")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.mode == "remote":
        raise NotImplementedError(
            "Remote client mode is intentionally separated from local inference."
        )
    if args.input == "realsense":
        input_handler = RealtimeInput(config.section("realsense"), config.reasoning_mode)
    else:
        input_handler = RecordingInput(args.source, config.reasoning_mode)
    QXGPlatform(config, input_handler).run()


if __name__ == "__main__":
    main()
