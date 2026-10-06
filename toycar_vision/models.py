from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    marker_id: int
    center_uv: tuple[float, float]
    heading_point_uv: tuple[float, float]
    corners: object
    quality: float


@dataclass(frozen=True)
class Measurement:
    car_id: int
    name: str
    x_mm: float
    y_mm: float
    theta_deg: float
    u: float
    v: float
    quality: float


@dataclass(frozen=True)
class Telemetry:
    timestamp_us: int
    car_id: int
    name: str
    x_mm: float
    y_mm: float
    theta_deg: float
    dx_mm_s: float
    dy_mm_s: float
    angular_velocity_deg_s: float
    u: int
    v: int
    detected: bool = True
    quality: float = 0.0

