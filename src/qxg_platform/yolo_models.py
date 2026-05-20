from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)

ULTRALYTICS_ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
KNOWN_YOLO_MODEL_URLS = {
    "yolov8n.pt": f"{ULTRALYTICS_ASSET_BASE}/yolov8n.pt",
    "yolov8s.pt": f"{ULTRALYTICS_ASSET_BASE}/yolov8s.pt",
    "yolov8m.pt": f"{ULTRALYTICS_ASSET_BASE}/yolov8m.pt",
    "yolov8l.pt": f"{ULTRALYTICS_ASSET_BASE}/yolov8l.pt",
    "yolov8x.pt": f"{ULTRALYTICS_ASSET_BASE}/yolov8x.pt",
    "yolo11n.pt": f"{ULTRALYTICS_ASSET_BASE}/yolo11n.pt",
    "yolo11s.pt": f"{ULTRALYTICS_ASSET_BASE}/yolo11s.pt",
    "yolo11m.pt": f"{ULTRALYTICS_ASSET_BASE}/yolo11m.pt",
    "yolo11l.pt": f"{ULTRALYTICS_ASSET_BASE}/yolo11l.pt",
    "yolo11x.pt": f"{ULTRALYTICS_ASSET_BASE}/yolo11x.pt",
}


@dataclass(frozen=True)
class ModelDownloadProposal:
    path: Path
    url: str | None

    @property
    def can_download(self) -> bool:
        return bool(self.url)


def yolo_download_proposal(config: dict[str, Any]) -> ModelDownloadProposal | None:
    weights = Path(str(config.get("model_weights", ""))).expanduser()
    if not weights or weights.exists():
        return None
    return ModelDownloadProposal(path=weights, url=_model_url(weights, config))


def ensure_yolo_model(
    config: dict[str, Any],
    confirm: Callable[[ModelDownloadProposal], bool] | None = None,
) -> Path:
    proposal = yolo_download_proposal(config)
    if proposal is None:
        return Path(str(config.get("model_weights", ""))).expanduser()
    if not proposal.can_download:
        raise FileNotFoundError(
            f"YOLO weights not found: {proposal.path}. "
            "Set detection.model_weights to an existing file or configure a download URL."
        )
    if confirm is not None and not confirm(proposal):
        raise FileNotFoundError(
            f"YOLO weights not found and download was declined: {proposal.path}"
        )
    download_yolo_model(proposal.path, str(proposal.url))
    return proposal.path


def download_yolo_model(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading YOLO model from %s to %s", url, path)
    with requests.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _model_url(weights: Path, config: dict[str, Any]) -> str | None:
    explicit_url = str(config.get("model_url", "")).strip()
    if explicit_url:
        return explicit_url
    configured_urls = config.get("model_urls", {})
    if isinstance(configured_urls, dict):
        configured_url = str(configured_urls.get(weights.name, "")).strip()
        if configured_url:
            return configured_url
    return KNOWN_YOLO_MODEL_URLS.get(weights.name)
