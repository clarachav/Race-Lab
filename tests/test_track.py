import math

import numpy as np
import pytest
from racelab.models.track import Segment, build_track, track_from_xy, track_xy

def test_length_and_points():
    track = build_track([Segment(500), Segment(100, radius=50)], ds=1.0)
    assert track.length == 600
    assert len(track.s) == 600


def test_curvature_is_zero_on_straight_and_inverse_radius_in_corner():
    track = build_track([Segment(500), Segment(100, radius=50)], ds=1.0)
    assert track.curvature[0] == 0
    assert track.curvature[499] == 0
    assert track.curvature[500] == pytest.approx(1 / 50)
    assert track.curvature[599] == pytest.approx(1 / 50)


def test_invalid_segment_is_rejected():
    with pytest.raises(ValueError):
        Segment(-10)
    with pytest.raises(ValueError):
        Segment(100, radius=0)


def test_straight_line_coordinates():
    track = build_track([Segment(100)], ds=1.0)
    x, y = track_xy(track)
    assert x[-1] == pytest.approx(100)
    assert y[-1] == pytest.approx(0, abs=1e-9)


def test_full_circle_comes_back_to_start():
    track = build_track([Segment(2 * math.pi * 50, radius=50)], ds=0.5)
    x, y = track_xy(track)
    assert x[-1] == pytest.approx(0, abs=2.0)
    assert y[-1] == pytest.approx(0, abs=2.0)

def test_track_from_xy_circle():
    th = np.linspace(0, 2 * math.pi, 800, endpoint=False)
    track = track_from_xy(100 * np.cos(th), 100 * np.sin(th), ds=5.0)
    assert track.length == pytest.approx(2 * math.pi * 100, rel=0.01)
    assert np.median(track.curvature) == pytest.approx(1 / 100, rel=0.02)


def test_track_from_xy_straight_has_no_curvature():
    x = np.linspace(0, 1000, 400)
    track = track_from_xy(x, np.zeros_like(x), ds=5.0)
    assert track.curvature.max() == pytest.approx(0, abs=1e-9)