import numpy as np

from qxg_platform.inputs import approximate_intrinsics, normalize_estimated_depth


def test_normalize_estimated_depth_returns_metric_like_range() -> None:
    depth = np.linspace(0, 100, 100, dtype=np.float32).reshape(10, 10)
    normalized = normalize_estimated_depth(depth, min_m=0.5, max_m=12.0)
    assert normalized.shape == depth.shape
    assert normalized.min() >= 0.5
    assert normalized.max() <= 12.0


def test_approximate_intrinsics_uses_frame_size() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    intrinsics = approximate_intrinsics(frame)
    assert intrinsics.width == 640
    assert intrinsics.height == 480
    assert intrinsics.ppx == 320
    assert intrinsics.ppy == 240
