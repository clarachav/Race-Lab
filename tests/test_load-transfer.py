import pytest
from racelab.constants import G
from racelab.models.vehicle import Vehicle
from racelab.physics.load_transfer import axle_loads, static_axle_loads


def test_static_loads_sum_to_weight():
    v = Vehicle()
    front, rear = static_axle_loads(v)
    assert front + rear == pytest.approx(v.mass * G)


def test_braking_moves_load_to_front():
    v = Vehicle()
    s_front, s_rear = static_axle_loads(v)
    front, rear = axle_loads(v, ax=-10)
    assert front > s_front
    assert rear < s_rear


def test_transfer_conserves_total_load():
    v = Vehicle()
    front, rear = axle_loads(v, ax=-10)
    assert front + rear == pytest.approx(v.mass * G)