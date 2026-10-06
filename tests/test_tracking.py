import pytest

from toycar_vision.models import Measurement
from toycar_vision.tracking import MultiCarTracker, wrap_degrees


def measurement(x, theta=0.0):
    return Measurement(10, "Red Racer", x, 0.0, theta, 100.0, 200.0, 0.9)


def test_wrap_degrees_uses_short_direction():
    assert wrap_degrees(358.0) == -2.0
    assert wrap_degrees(-358.0) == 2.0


def test_tracker_calculates_velocity():
    tracker = MultiCarTracker(position_time_constant_s=0.0, velocity_time_constant_s=0.0)
    first = tracker.update(measurement(0.0), 0)
    second = tracker.update(measurement(100.0), 100_000)
    assert first.dx_mm_s == 0.0
    assert second.x_mm == 100.0
    assert second.dx_mm_s == pytest.approx(1000.0)


def test_heading_does_not_jump_at_360_boundary():
    tracker = MultiCarTracker(position_time_constant_s=0.0, velocity_time_constant_s=0.0)
    tracker.update(measurement(0.0, 359.0), 0)
    second = tracker.update(measurement(0.0, 1.0), 100_000)
    assert second.theta_deg == pytest.approx(1.0)
    assert second.angular_velocity_deg_s == pytest.approx(20.0)

