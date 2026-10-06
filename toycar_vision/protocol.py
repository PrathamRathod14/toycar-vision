from __future__ import annotations

from .models import Telemetry


def serialize(telemetry: Telemetry) -> bytes:
    """Encode one record in the assignment's timestamp:"name",... format."""
    safe_name = telemetry.name.replace('"', '""')
    line = (
        f'{telemetry.timestamp_us}:"{safe_name}",'
        f"{telemetry.x_mm:.3f},{telemetry.y_mm:.3f},{telemetry.theta_deg:.3f},"
        f"{telemetry.dx_mm_s:.3f},{telemetry.dy_mm_s:.3f},"
        f"{telemetry.angular_velocity_deg_s:.3f},{telemetry.u},{telemetry.v}\n"
    )
    return line.encode("utf-8")

