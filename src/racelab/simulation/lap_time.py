"""Profil de vitesse et temps au tour (quasi-statique, 3 passes).

1. Limite en virage : v <= v_max_aero(R), et v <= vitesse max en ligne droite.
2. Passe arrière : on freine à temps avant chaque virage.
3. Passe avant   : l'accélération est bornée par la dynamique longitudinale.

Relation cinématique entre deux points espacés de ds :
    v_suivant^2 = v^2 + 2 * a * ds
"""
import math
from dataclasses import dataclass

import numpy as np

from racelab.models.tires import Tire
from racelab.models.track import Track
from racelab.models.vehicle import Vehicle
from racelab.physics.dynamics import (
    braking_deceleration,
    longitudinal_acceleration,
    max_speed_power_limited,
)
from racelab.simulation.cornering import v_max_aero


@dataclass(frozen=True)
class LapResult:
    speed: np.ndarray   # vitesse en chaque point [m/s]
    lap_time: float     # temps total [s]


def speed_limit(vehicle: Vehicle, tire: Tire, track: Track) -> np.ndarray:
    """Étape 1 : vitesse maximale permise par la géométrie, point par point."""
    v_top = max_speed_power_limited(vehicle)
    limit = np.full(len(track.s), v_top)
    for i, kappa in enumerate(track.curvature):
        if kappa > 0:
            limit[i] = min(v_top, v_max_aero(vehicle, tire, 1.0 / kappa))
    return limit


def simulate_lap(vehicle: Vehicle, tire: Tire, track: Track,
                 v_start: float = 1.0) -> LapResult:
    v = speed_limit(vehicle, tire, track)
    ds = track.ds
    n = len(v)

    # Étape 2 : passe arrière (freinage)
    for i in range(n - 2, -1, -1):
        a_brake = braking_deceleration(vehicle, tire, v[i + 1])
        v_reachable = math.sqrt(v[i + 1] ** 2 + 2 * a_brake * ds)
        v[i] = min(v[i], v_reachable)

    # Étape 3 : passe avant (accélération), départ arrêté
    v[0] = min(v[0], v_start)
    for i in range(n - 1):
        a = longitudinal_acceleration(vehicle, tire, v[i])
        v_reachable = math.sqrt(max(v[i] ** 2 + 2 * a * ds, 0.0))
        v[i + 1] = min(v[i + 1], v_reachable)

    # Temps : somme des ds / vitesse moyenne entre deux points
    lap_time = float(np.sum(2 * ds / (v[:-1] + v[1:])))
    return LapResult(speed=v, lap_time=lap_time)
def simulate_flying_lap(vehicle: Vehicle, tire: Tire, track: Track) -> LapResult:
    """Tour lancé sur un circuit FERMÉ (la fin du tour rejoint le début).

    On simule 3 tours collés bout à bout, départ arrêté, et on garde le tour
    du milieu : il est libre de l'effet du départ arrêté, et le freinage de
    la fin du tour tient compte des virages du début du tour suivant.
    """
    n = len(track.s)
    tiled = Track(s=np.arange(3 * n) * track.ds,
                  curvature=np.tile(track.curvature, 3),
                  ds=track.ds, length=3 * n * track.ds)
    result = simulate_lap(vehicle, tire, tiled)
    v = result.speed[n:2 * n]
    lap_time = float(np.sum(2 * track.ds / (v + np.roll(v, -1))))
    return LapResult(speed=v, lap_time=lap_time)
