import pytest
from racelab.models.aerodynamics import downforce, drag
from racelab.models.vehicle import Vehicle


def test_forces_are_quadratic_in_speed():
    v = Vehicle()
    assert drag(v, 100) == pytest.approx(4 * drag(v, 50))
    assert downforce(v, 100) == pytest.approx(4 * downforce(v, 50))


def test_no_force_at_zero_speed():
    assert downforce(Vehicle(), 0) == 0


def test_invalid_vehicle_is_rejected():
    with pytest.raises(ValueError):
        Vehicle(mass=-1)
    with pytest.raises(ValueError):
        Vehicle(l_f=1.0)  # l_f + l_r != wheelbase