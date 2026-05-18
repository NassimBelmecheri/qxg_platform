from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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


class MonocularDepthEstimator:
    def __init__(self, model_name: str = "Intel/dpt-hybrid-midas"):
        try:
            import torch
            from transformers import DPTForDepthEstimation, DPTImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Install qxg-platform[ml] to use monocular depth estimation."
            ) from exc

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def estimate(self, frame_bgr: np.ndarray) -> DepthFrame:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            predicted = self.model(**inputs).predicted_depth
        prediction = self.torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = prediction.detach().cpu().numpy().astype(np.float32)
        depth = normalize_estimated_depth(depth)
        intrinsics = approximate_intrinsics(frame_bgr)
        return DepthFrame(depth, intrinsics)


def normalize_estimated_depth(
    depth: np.ndarray, min_m: float = 0.5, max_m: float = 12.0
) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.full_like(depth, min_m, dtype=np.float32)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.full_like(depth, (min_m + max_m) / 2, dtype=np.float32)
    clipped = np.clip(depth, low, high)
    normalized = (clipped - low) / (high - low)
    # DPT predicts inverse-ish relative depth. Flip so larger values are farther away.
    metric_like = max_m - normalized * (max_m - min_m)
    return metric_like.astype(np.float32)


def approximate_intrinsics(frame: np.ndarray) -> CameraIntrinsics:
    height, width = frame.shape[:2]
    focal = float(max(width, height) * 1.2)
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        ppx=width / 2,
        ppy=height / 2,
    )


class DepthEstimatedInputMixin:
    def _setup_depth(self, reasoning_mode: str, depth_config: dict[str, Any] | None = None) -> None:
        self.reasoning_mode = reasoning_mode
        self.depth_estimator = None
        if reasoning_mode == "3d":
            depth_config = depth_config or {}
            self.depth_estimator = MonocularDepthEstimator(
                str(depth_config.get("model_name", "Intel/dpt-hybrid-midas"))
            )

    def _world_info_for_frame(self, frame: np.ndarray) -> object | None:
        if self.depth_estimator is None:
            return None
        return self.depth_estimator.estimate(frame)


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


class VideoFileInput(DepthEstimatedInputMixin, InputHandler):
    def __init__(
        self,
        path: str | Path,
        reasoning_mode: str = "2d",
        depth_config: dict[str, Any] | None = None,
    ):
        self._setup_depth(reasoning_mode, depth_config)
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video file: {self.path}")

    def frames(self) -> Iterator[tuple[np.ndarray, object | None]]:
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame, self._world_info_for_frame(frame)

    def close(self) -> None:
        self.capture.release()


class WebcamInput(DepthEstimatedInputMixin, InputHandler):
    def __init__(
        self,
        camera_index: int = 0,
        reasoning_mode: str = "2d",
        depth_config: dict[str, Any] | None = None,
    ):
        self._setup_depth(reasoning_mode, depth_config)
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open camera index: {camera_index}")

    def frames(self) -> Iterator[tuple[np.ndarray, object | None]]:
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame, self._world_info_for_frame(frame)

    def close(self) -> None:
        self.capture.release()


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
