from pathlib import Path

import yaml


def test_botsort_tracker_config_has_required_ultralytics_keys() -> None:
    config = yaml.safe_load(Path("configs/slow_tracker.yaml").read_text(encoding="utf-8"))
    assert config["tracker_type"] == "botsort"
    for key in ["gmc_method", "proximity_thresh", "appearance_thresh", "with_reid", "model"]:
        assert key in config
