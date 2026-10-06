from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Measurement, Telemetry


def wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def display_degrees(angle: float) -> float:
    return angle % 360.0


def _blend_factor(dt: float, time_constant: float) -> float:
    if time_constant <= 0.0:
        return 1.0
    return 1.0 - math.exp(-max(dt, 0.0) / time_constant)


@dataclass
class _Track:
    x: float
    y: float
    theta_unwrapped: float
    vx: float
    vy: float
    omega: float
    last_time_s: float


class MultiCarTracker:
    def __init__(
        self,
        position_time_constant_s: float = 0.010,
        velocity_time_constant_s: float = 0.080,
        max_gap_s: float = 0.25,
        max_speed_mm_s: float = 10000.0,
    ) -> None:
        self.position_tau = float(position_time_constant_s)
        self.velocity_tau = float(velocity_time_constant_s)
        self.max_gap_s = float(max_gap_s)
        self.max_speed_mm_s = float(max_speed_mm_s)
        self._tracks: dict[int, _Track] = {}

    def update(self, measurement: Measurement, timestamp_us: int) -> Telemetry:
        now_s = timestamp_us / 1_000_000.0
        track = self._tracks.get(measurement.car_id)
        if track is None or now_s <= track.last_time_s or now_s - track.last_time_s > self.max_gap_s:
            track = _Track(
                x=measurement.x_mm,
                y=measurement.y_mm,
                theta_unwrapped=measurement.theta_deg,
                vx=0.0,
                vy=0.0,
                omega=0.0,
                last_time_s=now_s,
            )
            self._tracks[measurement.car_id] = track
        else:
            dt = now_s - track.last_time_s
            jump_mm = math.hypot(measurement.x_mm - track.x, measurement.y_mm - track.y)
            if self.max_speed_mm_s > 0 and jump_mm / dt > self.max_speed_mm_s:
                return self.missing(timestamp_us, measurement.car_id, measurement.name)
            old_x, old_y, old_theta = track.x, track.y, track.theta_unwrapped
            position_alpha = _blend_factor(dt, self.position_tau)
            angle_delta = wrap_degrees(measurement.theta_deg - display_degrees(old_theta))
            track.x += position_alpha * (measurement.x_mm - track.x)
            track.y += position_alpha * (measurement.y_mm - track.y)
            track.theta_unwrapped += position_alpha * angle_delta

            raw_vx = (track.x - old_x) / dt
            raw_vy = (track.y - old_y) / dt
            raw_omega = (track.theta_unwrapped - old_theta) / dt
            velocity_alpha = _blend_factor(dt, self.velocity_tau)
            track.vx += velocity_alpha * (raw_vx - track.vx)
            track.vy += velocity_alpha * (raw_vy - track.vy)
            track.omega += velocity_alpha * (raw_omega - track.omega)
            track.last_time_s = now_s

        return Telemetry(
            timestamp_us=timestamp_us,
            car_id=measurement.car_id,
            name=measurement.name,
            x_mm=track.x,
            y_mm=track.y,
            theta_deg=display_degrees(track.theta_unwrapped),
            dx_mm_s=track.vx,
            dy_mm_s=track.vy,
            angular_velocity_deg_s=track.omega,
            u=int(round(measurement.u)),
            v=int(round(measurement.v)),
            detected=True,
            quality=measurement.quality,
        )

    @staticmethod
    def missing(timestamp_us: int, car_id: int, name: str) -> Telemetry:
        return Telemetry(
            timestamp_us=timestamp_us,
            car_id=car_id,
            name=name,
            x_mm=-1000.0,
            y_mm=-1000.0,
            theta_deg=0.0,
            dx_mm_s=0.0,
            dy_mm_s=0.0,
            angular_velocity_deg_s=0.0,
            u=-1,
            v=-1,
            detected=False,
            quality=0.0,
        )

