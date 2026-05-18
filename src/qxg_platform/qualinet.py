from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from qxg_platform.domain import TrackedObject

LOGGER = logging.getLogger(__name__)


class QualiNetRelationConstructor:
    def __init__(self, config: dict[str, Any], camera_id: int = 0):
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.camera_id = camera_id
        self.mode = str(config.get("mode", "geometry_fallback"))
        self.auto_download = bool(config.get("auto_download", True))
        self.allow_geometry_fallback = bool(config.get("allow_geometry_fallback", True))
        self.image_size = int(config.get("image_size", 224))
        self.ra_labels = list(config.get("ra_labels", []))
        self.qdc_labels = list(config.get("qdc_labels", []))
        self.device = None
        self.ra_model = None
        self.qdc_model = None
        if self.enabled:
            self._load_models_if_configured()

    def build(
        self, frame: np.ndarray, objects: list[TrackedObject]
    ) -> dict[tuple[int, int], dict[str, Any]]:
        if not self.enabled:
            return {}
        relations: dict[tuple[int, int], dict[str, Any]] = {}
        for obj in objects:
            if obj.tracking_id == self.camera_id or obj.category == "camera":
                continue
            relation: dict[str, Any] = {}
            if self.ra_model is not None:
                relation["RA"] = self._predict_label(frame, obj, self.ra_model, self.ra_labels)
            elif self.allow_geometry_fallback:
                relation["RA"] = self._fallback_ra(frame, obj)

            if self.qdc_model is not None:
                relation["QDC"] = self._predict_label(frame, obj, self.qdc_model, self.qdc_labels)
            elif self.allow_geometry_fallback:
                relation["QDC"] = self._fallback_qdc(frame, obj)

            if relation:
                relations[tuple(sorted((self.camera_id, int(obj.tracking_id))))] = relation
        return relations

    def _load_models_if_configured(self) -> None:
        ra_path = self._prepare_model_path("ra_model_path", "ra_model_url")
        qdc_path = self._prepare_model_path("qdc_model_path", "qdc_model_url")
        if ra_path is not None and ra_path.exists():
            self.ra_model = self._load_model(ra_path, len(self.ra_labels), "RA")
        if qdc_path is not None and qdc_path.exists():
            self.qdc_model = self._load_model(qdc_path, len(self.qdc_labels), "QDC")
        if self.ra_model is None and self.qdc_model is None and not self.allow_geometry_fallback:
            raise FileNotFoundError(
                "QualiNet is enabled but no RA/QDC model files are available."
            )

    def _prepare_model_path(self, path_key: str, url_key: str) -> Path | None:
        value = str(self.config.get(path_key, "")).strip()
        url = str(self.config.get(url_key, "")).strip()
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.exists() and url and self.auto_download:
            path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Downloading QualiNet model %s to %s", url, path)
            with requests.get(url, timeout=120, stream=True) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        return path

    def _load_model(self, path: Path, num_classes: int, relation_name: str) -> Any:
        if num_classes <= 0:
            raise ValueError(f"QualiNet {relation_name} labels must not be empty")
        try:
            import torch
            from torch import nn
            from torchvision import models
        except ImportError as exc:
            raise RuntimeError("Install qxg-platform[ml] to use QualiNet models") from exc

        class CustomResNet152(nn.Module):
            def __init__(self, classes: int):
                super().__init__()
                self.model = models.resnet152(weights=None)
                self.model.fc = nn.Sequential(
                    nn.Linear(self.model.fc.in_features, 1024),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(1024),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, classes),
                )

            def forward(self, x: Any) -> Any:
                return self.model(x)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=self.device)
        architecture = checkpoint.get("model_architecture", {})
        classes = int(architecture.get("num_classes", num_classes))
        model = CustomResNet152(classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        LOGGER.info("Loaded QualiNet %s model from %s", relation_name, path)
        return model

    def _predict_label(
        self, frame: np.ndarray, obj: TrackedObject, model: Any, labels: list[str]
    ) -> str:
        if self.device is None:
            raise RuntimeError("QualiNet model device is not initialized")
        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError("Install qxg-platform[ml] to use QualiNet models") from exc

        crop = self._masked_crop(frame, obj)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        tensor = transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)
            index = int(torch.argmax(probabilities, dim=1).item())
        if index >= len(labels):
            return str(index)
        return str(labels[index])

    def _masked_crop(self, frame: np.ndarray, obj: TrackedObject) -> np.ndarray:
        x, y, w, h = map(int, obj.bbox)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + max(1, w))
        y2 = min(frame.shape[0], y + max(1, h))
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(frame, frame, mask=mask)

    def _fallback_ra(self, frame: np.ndarray, obj: TrackedObject) -> tuple[str, str]:
        height, width = frame.shape[:2]
        x, y, w, h = obj.bbox
        center_x = x + w / 2
        center_y = y + h / 2
        if center_x < width * 0.4:
            horizontal = "left"
        elif center_x > width * 0.6:
            horizontal = "right"
        else:
            horizontal = "overlap"
        if center_y < height * 0.4:
            vertical = "above"
        elif center_y > height * 0.6:
            vertical = "below"
        else:
            vertical = "overlap"
        return horizontal, vertical

    def _fallback_qdc(self, frame: np.ndarray, obj: TrackedObject) -> str:
        frame_area = float(frame.shape[0] * frame.shape[1])
        bbox_area = float(max(obj.bbox[2], 1.0) * max(obj.bbox[3], 1.0))
        ratio = bbox_area / frame_area
        if ratio >= 0.18:
            return "very close"
        if ratio >= 0.08:
            return "close"
        if ratio >= 0.03:
            return "normal"
        return "far"
