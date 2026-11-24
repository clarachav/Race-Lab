"""Piste : segments (droites/virages) ou coordonnées réelles (x, y).

On découpe la piste en points espacés de ds mètres. En chaque point on garde
la courbure kappa = 1 / R (positive, 0 sur une droite).
"""
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    length: float                 # longueur [m]
    radius: float | None = None   # rayon [m] ; None = ligne droite

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length doit être strictement positif")
        if self.radius is not None and self.radius <= 0:
            raise ValueError("radius doit être strictement positif")


@dataclass(frozen=True)
class Track:
    s: np.ndarray          # distance parcourue en chaque point [m]
    curvature: np.ndarray  # courbure en chaque point [1/m]
    ds: float              # pas entre deux points [m]
    length: float          # longueur totale [m]
    x: np.ndarray | None = None   # coordonnées [m] (si la piste en a)
    y: np.ndarray | None = None


def build_track(segments: list[Segment], ds: float = 1.0) -> Track:
    ends = np.cumsum([seg.length for seg in segments])
    s = np.arange(0.0, ends[-1], ds)
    seg_index = np.searchsorted(ends, s, side="right")
    kappa_per_segment = np.array(
        [0.0 if seg.radius is None else 1.0 / seg.radius for seg in segments]
    )
    return Track(s=s, curvature=kappa_per_segment[seg_index],
                 ds=ds, length=float(ends[-1]))


def _moving_average(a: np.ndarray, half: int, closed: bool) -> np.ndarray:
    """Moyenne glissante sur 2*half+1 points. Boucle fermée : on 'enroule'."""
    if half <= 0:
        return a
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    mode = "wrap" if closed else "edge"
    return np.convolve(np.pad(a, half, mode=mode), kernel, mode="valid")


def track_from_xy(x, y, ds: float = 5.0, smooth_m: float = 50.0,
                  closed: bool = False) -> Track:
    """Construit une piste à partir de coordonnées (x, y) en mètres.

    1. On rééchantillonne la trajectoire tous les ds mètres.
    2. Cap de la voiture : theta = atan2(dy, dx).
    3. Courbure : kappa = |d theta / d s|, lissée sur smooth_m mètres.

    Le lissage compte : une piste décrite par peu de points (un tous les
    ~50 m) ne donne pas des rayons de virage précis. closed=True pour un
    circuit : la boucle est refermée et le lissage 'enroule' le début et la fin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if closed:
        x, y = np.append(x, x[0]), np.append(y, y[0])
    d = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    keep = np.concatenate(([True], np.diff(d) > 1e-9))  # enlève les doublons
    x, y, d = x[keep], y[keep], d[keep]

    s = np.arange(0.0, d[-1], ds)
    xs = np.interp(s, d, x)
    ys = np.interp(s, d, y)

    if closed:
        dx, dy = np.roll(xs, -1) - xs, np.roll(ys, -1) - ys
        theta = np.arctan2(dy, dx)
        # différence d'angle ramenée dans ]-pi, pi] (évite le saut de 2*pi)
        dtheta = np.angle(np.exp(1j * (np.roll(theta, -1) - theta)))
        dtheta_ds = dtheta / ds
    else:
        theta = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
        dtheta_ds = np.gradient(theta, ds)

    half = int(round(smooth_m / (2 * ds)))
    curvature = np.abs(_moving_average(dtheta_ds, half, closed))
    return Track(s=s, curvature=curvature, ds=ds, length=float(d[-1]),
                 x=xs, y=ys)


def load_geojson_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Lit un circuit GeoJSON (longitude, latitude) et le convertit en mètres.

    Projection locale : x = R cos(lat0) * dlon, y = R * dlat (lon/lat en radians).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    coords = np.array(data["features"][0]["geometry"]["coordinates"], dtype=float)
    lon, lat = coords[:, 0], coords[:, 1]
    earth_radius = 6_371_000.0
    lat0 = np.radians(lat.mean())
    x = earth_radius * np.cos(lat0) * np.radians(lon - lon[0])
    y = earth_radius * np.radians(lat - lat[0])
    return x, y


def track_xy(track: Track) -> tuple[np.ndarray, np.ndarray]:
    """Coordonnées (x, y) [m] de la piste.

    Si la piste en a (données réelles), on les renvoie. Sinon on les
    reconstruit : le cap tourne de kappa * ds à chaque pas, puis on avance
    de ds dans la direction du cap.
    """
    if track.x is not None and track.y is not None:
        return track.x, track.y
    heading = np.cumsum(track.curvature) * track.ds
    x = np.cumsum(np.cos(heading)) * track.ds
    y = np.cumsum(np.sin(heading)) * track.ds
    return x, y