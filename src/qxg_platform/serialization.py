from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np


def encode_image_jpeg(frame: np.ndarray, quality: int = 90) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to encode image as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_image_jpeg(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload.encode("ascii"))
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG payload")
    return frame


def encode_array(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    shape = tuple(int(v) for v in payload["shape"])
    raw = base64.b64decode(payload["data"].encode("ascii"))
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def object_to_dict(obj: Any) -> dict[str, Any]:
    return {
        "tracking_id": int(obj.tracking_id),
        "category": obj.category,
        "bbox": np.asarray(obj.bbox).tolist(),
        "confidence": float(obj.confidence),
        "bev_bbox": None if obj.bev_bbox is None else np.asarray(obj.bev_bbox).tolist(),
        "bev_center": None if obj.bev_center is None else np.asarray(obj.bev_center).tolist(),
        "world_coord": None if obj.world_coord is None else np.asarray(obj.world_coord).tolist(),
    }


def relation_keys_to_json(
    relations: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {f"{left}:{right}": value for (left, right), value in relations.items()}


def relation_keys_from_json(
    relations: dict[str, dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    parsed = {}
    for key, value in relations.items():
        left, right = key.split(":", maxsplit=1)
        parsed[(int(left), int(right))] = value
    return parsed
