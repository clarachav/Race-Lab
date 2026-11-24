import math

import pytest
from racelab.models.aerodynamics import downforce
from racelab.models.tires import Tire
from racelab.models.vehicle import Vehicle
from racelab.simulation.cornering import (
    G,
    critical_radius,
    v_max_aero,
    v_max_no_aero,
)


def test_no_aero_formula():
    t = Tire(mu=1.5)
    assert v_max_no_aero(t, 100) == pytest.approx(math.sqrt(1.5 * G * 100))


def test_aero_model_reduces_to_no_aero_when_cl_is_zero():
    v, t = Vehicle(cl=0.0), Tire()
    assert v_max_aero(v, t, 80) == pytest.approx(v_max_no_aero(t, 80))


def test_downforce_increases_cornering_speed():
    v, t = Vehicle(), Tire()
    assert v_max_aero(v, t, 100) > v_max_no_aero(t, 100)


def test_grip_balance_at_vmax():
    """À v_max, la force latérale requise égale exactement mu * Fz total."""
    v, t, radius = Vehicle(), Tire(), 100
    vm = v_max_aero(v, t, radius)
    required = v.mass * vm**2 / radius
    available = t.max_force(v.mass * G + downforce(v, vm))
    assert required == pytest.approx(available)


def test_no_limit_beyond_critical_radius():
    v, t = Vehicle(), Tire()
    assert math.isinf(v_max_aero(v, t, critical_radius(v, t) * 1.1))