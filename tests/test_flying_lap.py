import math

import numpy as np
import pytest
from racelab.models.tires import Tire
from racelab.models.track import track_from_xy
from racelab.models.vehicle import Vehicle
from racelab.simulation.cornering import v_max_aero
from racelab.simulation.lap_time import simulate_flying_lap


def test_flying_lap_on_a_circle_is_length_over_vmax():
    """Cercle R=100 m : vitesse constante = v_max(R) => t = longueur / v_max."""
    v, t = Vehicle(), Tire()
    th = np.linspace(0, 2 * math.pi, 300, endpoint=False)
    track = track_from_xy(100 * np.cos(th), 100 * np.sin(th), ds=5.0, closed=True)
    result = simulate_flying_lap(v, t, track)
    expected = 2 * math.pi * 100 / v_max_aero(v, t, 100)
    assert result.lap_time == pytest.approx(expected, rel=0.02)