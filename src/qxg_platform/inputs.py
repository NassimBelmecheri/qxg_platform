from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class CameraIntrinsics:
    def __init__(self, width: int, height: int, fx: float, fy: float, ppx: float, ppy: float):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


class DepthFrame:
    def __init__(self, depth_map: np.ndarray, intrinsics: CameraIntrinsics):
        self._depth_map = depth_map
        self._intrinsics = intrinsics
        self.profile = self

    def get_data(self) -> np.ndarray:
        return self._depth_map

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def as_video_stream_profile(self) -> DepthFrame:
        return self


class InputHandler(ABC):
    @abstractmethod
    def frames(self) -> Iterator[tuple[np.ndarray, object | None]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class RecordingInput(InputHandler):
    def __init__(self, path: str | Path, reasoning_mode: str = "3d"):
        self.path = Path(path).expanduser().resolve()
        self.reasoning_mode = reasoning_mode
        self.color_dir = self.path / "color"
        self.depth_dir = self.path / "depth"
        self.config_path = self.path / "config.json"
        if not self.path.exists() or not self.color_dir.exists():
            raise FileNotFoundError(f"Invalid recording directory: {self.path}")
        self.color_files = sorted(self.color_dir.glob("*.jpg"))
        if not self.color_files:
            raise FileNotFoundError(f"No color frames found in {self.color_dir}")
        self.intrinsics = CameraIntrinsics(1280, 720, 900.0, 900.0, 640.0, 360.0)
        self.depth_scale = 1000.0
        self._load_recording_config()

    def _load_recording_config(self) -> None:
        if not self.config_path.exists():
            return
        import json

        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.depth_scale = float(data.get("depth_scale", self.depth_scale))
        matrix = data.get("cam_intr")
        if matrix:
            self.intrinsics = CameraIntrinsics(
                int(data.get("im_w", 1280)),
                int(data.get("im_h", 720)),
                float(matrix[0][0]),
                float(matrix[1][1]),
                float(matrix[0][2]),
                float(matrix[1][2]),
            )

    def frames(self) -> Iterator[tuple[np.ndarray, object | None]]:
        for color_path in self.color_files:
            frame = cv2.imread(str(color_path))
            if frame is None:
                continue
            world_info = None
            if self.reasoning_mode == "3d":
                depth_path = self.depth_dir / color_path.name.replace("-color.jpg", "-depth.png")
                if depth_path.exists():
                    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if depth_raw is not None:
                        world_info = DepthFrame(
                            depth_raw.astype(np.float32) / self.depth_scale,
                            self.intrinsics,
                        )
            yield frame, world_info


class RealtimeInput(InputHandler):
    def __init__(self, config: dict, reasoning_mode: str = "3d"):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("Install qxg-platform[realsense] for RealSense support") from exc
        self.rs = rs
        self.reasoning_mode = reasoning_mode
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        width = int(config.get("width", 1280))
        height = int(config.get("height", 720))
        fps = int(config.get("fps", 30))
        rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.align = None
        if reasoning_mode == "3d":
            rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.align = rs.align(rs.stream.color)
        self.pipeline.start(rs_config)
        self.is_running = True

    def frames(self) -> Iterator[tuple[np.ndarray, object | None]]:
        while self.is_running:
            frames = self.pipeline.wait_for_frames()
            world_info = None
            if self.align is not None:
                frames = self.align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                world_info = frames
            else:
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
            yield np.asarray(color_frame.get_data()), world_info

    def close(self) -> None:
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
