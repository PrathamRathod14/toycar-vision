from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CarConfig:
    marker_id: int
    name: str
    heading_offset_deg: float = 0.0


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required = ("camera", "field", "cars", "network")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    if not config["cars"]:
        raise ValueError("At least one car must be configured")

    seen: set[int] = set()
    for car in config["cars"]:
        raw_id = car.get("car_id", car.get("marker_id"))
        if raw_id is None:
            raise ValueError("Every car needs a car_id (or marker_id for ArUco mode)")
        marker_id = int(raw_id)
        if marker_id in seen:
            raise ValueError(f"Duplicate car marker ID: {marker_id}")
        seen.add(marker_id)

    config["_base_dir"] = config_path.parent
    return config


def resolve_config_path(config: dict[str, Any], value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(config["_base_dir"]) / candidate
    return candidate.resolve()


def car_configs(config: dict[str, Any]) -> dict[int, CarConfig]:
    return {
        int(item.get("car_id", item.get("marker_id"))): CarConfig(
            marker_id=int(item.get("car_id", item.get("marker_id"))),
            name=str(item["name"]),
            heading_offset_deg=float(item.get("heading_offset_deg", 0.0)),
        )
        for item in config["cars"]
    }

