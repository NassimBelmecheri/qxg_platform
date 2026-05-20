from __future__ import annotations

import pytest

from qxg_platform.yolo_models import ensure_yolo_model, yolo_download_proposal


def test_yolo_download_proposal_uses_configured_url(tmp_path) -> None:
    weights = tmp_path / "models" / "custom.pt"

    proposal = yolo_download_proposal(
        {
            "model_weights": str(weights),
            "model_urls": {"custom.pt": "https://example.test/custom.pt"},
        }
    )

    assert proposal is not None
    assert proposal.path == weights
    assert proposal.url == "https://example.test/custom.pt"


def test_yolo_download_proposal_returns_none_for_existing_file(tmp_path) -> None:
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"weights")

    assert yolo_download_proposal({"model_weights": str(weights)}) is None


def test_ensure_yolo_model_raises_when_missing_and_declined(tmp_path) -> None:
    weights = tmp_path / "yolov8n.pt"

    with pytest.raises(FileNotFoundError, match="download was declined"):
        ensure_yolo_model({"model_weights": str(weights)}, confirm=lambda _proposal: False)


def test_ensure_yolo_model_raises_without_known_url(tmp_path) -> None:
    weights = tmp_path / "unknown-model.pt"

    with pytest.raises(FileNotFoundError, match="configure a download URL"):
        ensure_yolo_model({"model_weights": str(weights)})
