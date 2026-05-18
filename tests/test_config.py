from pathlib import Path

from qxg_platform.config import load_config


def test_load_video_config() -> None:
    config = load_config(Path("configs/video.yaml"))
    assert config.reasoning_mode == "3d"
    assert config.section("detection")["enabled"] is True
