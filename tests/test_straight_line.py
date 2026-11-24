import pytest
from racelab.models.aerodynamics import drag
from racelab.models.tires import Tire
from racelab.models.vehicle import Vehicle
from racelab.physics.dynamics import (
    drive_force,
    longitudinal_acceleration,
    max_speed_power_limited,
)


def test_power_balances_drag_at_top_speed():
    """À v_max : puissance moteur == puissance dissipée par la traînée."""
    v = Vehicle()
    v_top = max_speed_power_limited(v)
    assert drag(v, v_top) * v_top == pytest.approx(v.power)


def test_zero_acceleration_at_top_speed():
    v, t = Vehicle(), Tire()
    v_top = max_speed_power_limited(v)
    assert longitudinal_acceleration(v, t, v_top) == pytest.approx(0, abs=1e-6)


def test_traction_limited_at_low_speed():
    v, t = Vehicle(), Tire()
    assert drive_force(v, t, 10) < v.power / 10


def test_power_limited_at_high_speed():
    v, t = Vehicle(), Tire()
    assert drive_force(v, t, 90) == pytest.approx(v.power / 90)