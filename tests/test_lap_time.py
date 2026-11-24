import pytest
from racelab.models.tires import Tire
from racelab.models.track import Segment, build_track
from racelab.models.vehicle import Vehicle
from racelab.simulation.cornering import v_max_aero
from racelab.simulation.lap_time import simulate_lap


def test_constant_corner_gives_length_over_speed():
    """Virage constant R=50 : vitesse constante => t = longueur / v_max."""
    v, t = Vehicle(), Tire()
    track = build_track([Segment(400, radius=50)], ds=1.0)
    result = simulate_lap(v, t, track, v_start=v_max_aero(v, t, 50))
    expected = track.length / v_max_aero(v, t, 50)
    assert result.lap_time == pytest.approx(expected, rel=0.01)


def test_never_exceeds_cornering_limit():
    v, t = Vehicle(), Tire()
    track = build_track([Segment(500), Segment(100, radius=50), Segment(300)])
    result = simulate_lap(v, t, track)
    limit = v_max_aero(v, t, 50)
    assert result.speed[500:600].max() <= limit + 1e-9


def test_car_brakes_before_the_corner():
    v, t = Vehicle(), Tire()
    track = build_track([Segment(500), Segment(100, radius=50), Segment(300)])
    result = simulate_lap(v, t, track)
    assert result.speed[499] > result.speed[500]   # ça freine avant d'entrer
    assert result.speed[450] > result.speed[499]


def test_lap_is_slower_in_a_corner_than_on_a_straight():
    v, t = Vehicle(), Tire()
    straight = build_track([Segment(600)])
    twisty = build_track([Segment(300), Segment(100, radius=50), Segment(200)])
    assert simulate_lap(v, t, twisty).lap_time > simulate_lap(v, t, straight).lap_time