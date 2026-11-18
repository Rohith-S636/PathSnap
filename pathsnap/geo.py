"""Coordinate projection and planar geometry.  Owner: P1

Trips are spread worldwide (lon -100..137, lat -34..62), so ONE fixed UTM zone is wrong.
Use a per-trip local metric projection, e.g. pyproj Azimuthal Equidistant centred on the
track centroid (or the UTM zone of the centroid). Justify the choice in the report (section 2).
"""
from __future__ import annotations

import numpy as np


class LocalProjector:
    """Project lon/lat <-> local x/y metres for one trip."""

    def __init__(self, lon0: float, lat0: float):
        raise NotImplementedError

    def to_xy(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def to_lonlat(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def point_segment_distance(px, py, ax, ay, bx, by) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised distance from points P to segments AB (all in metres).

    Returns (dist_m, proj_frac) where proj_frac in [0, 1] is the position of the foot point along AB.
    """
    raise NotImplementedError


def bearing_deg(ax, ay, bx, by) -> np.ndarray:
    """Planar bearing of vector A->B in degrees [0, 360), 0 = north, clockwise."""
    raise NotImplementedError
