import pytest
from racelab.models.tires import Tire


def test_coulomb_max_force():
    assert Tire(mu=1.5).max_force(1000) == 1500


def test_friction_circle():
    t = Tire(mu=1.0)
    assert t.available_lateral(0, 1000) == pytest.approx(1000)
    assert t.available_lateral(600, 1000) == pytest.approx(800)  # 3-4-5
    assert t.available_lateral(1000, 1000) == 0
    assert t.available_lateral(1200, 1000) == 0