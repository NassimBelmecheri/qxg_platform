from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PlatformConfig:
    raw: dict[str, Any]
    source_path: Path

    @property
    def reasoning_mode(self) -> str:
        return str(self.raw.get("runtime", {}).get("reasoning_mode", "3d"))

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section '{name}' must be a mapping")
        return value


def load_config(path: str | Path) -> PlatformConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return PlatformConfig(raw=raw, source_path=config_path)


def resolve_path(path: str | Path, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base.parent / candidate).resolve()
