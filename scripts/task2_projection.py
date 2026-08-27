"""Local Cartesian <-> lat/lon projection for Task 2 siting work.

Task2 CC prompt (2026-07-13), Task E. Per the 2026-07-10 report (Task A.6):
no lat/lon<->local-metres conversion exists anywhere in the kit
(`wind_farm_simulator.py`/`optimization.py` pass layout x_m/y_m straight
into PyWake as bare Cartesian offsets around the farm centre; confirmed by
exhaustive grep of `phase_2/kit/phase_2/`). `scripts/task2_depth_sampler.py`
already introduced a working equirectangular projection inline
(`local_xy_to_latlon`) to make per-turbine depth queries possible; this
module is that same projection pulled out as the single source of truth,
so `depth_sampler`, the Task F scorer replica, and any future layout code
all use the identical transform instead of drifting copies.

**Standing assumption (undocumented in the scorer, ours to state):**
WGS84 lat/lon, local equirectangular (tangent-plane) projection centred on
the farm centre - `x_m` east, `y_m` north. Per the 2026-07-13 prompt: no
forum answer yet on projection/datum; proceeding with this documented
assumption. Distortion at the 15x15 km box scale is negligible (<0.1%),
and the Task F well-tie is the check that this assumption (plus every
other pinned-physics choice) reproduces the kit's own reported numbers -
if the well-tie fails, this projection is one of the things to
re-examine, but a projection choice this local is an unlikely culprit
for a CF/wake-loss-scale discrepancy.

Deterministic (pure geometry), no stochastic step.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def local_xy_to_latlon(
    centre_lat: float, centre_lon: float, x_m: np.ndarray, y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Local (x_m east, y_m north) of the centre -> (lat, lon) in degrees.

    Equirectangular tangent-plane approximation - not extracted from the
    scorer (none exists there), a standard documented choice.
    """
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    lat = centre_lat + np.degrees(y_m / EARTH_RADIUS_M)
    lon = centre_lon + np.degrees(x_m / (EARTH_RADIUS_M * np.cos(np.radians(centre_lat))))
    return lat, lon


def latlon_to_local_xy(
    centre_lat: float, centre_lon: float, lat: np.ndarray, lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of local_xy_to_latlon: (lat, lon) -> local (x_m east, y_m north).

    Exact algebraic inverse of the same tangent-plane approximation (not an
    independent projection) - round-trips local_xy_to_latlon to float
    precision by construction.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    y_m = np.radians(lat - centre_lat) * EARTH_RADIUS_M
    x_m = np.radians(lon - centre_lon) * EARTH_RADIUS_M * np.cos(np.radians(centre_lat))
    return x_m, y_m


if __name__ == "__main__":
    clat, clon = 53.5, 1.5
    x = np.array([0.0, 7500.0, -7500.0])
    y = np.array([0.0, 7500.0, -7500.0])
    lat, lon = local_xy_to_latlon(clat, clon, x, y)
    print("xy -> latlon:", list(zip(lat.round(5), lon.round(5))))
    xb, yb = latlon_to_local_xy(clat, clon, lat, lon)
    print("round-trip max abs error (m):", float(np.max(np.abs(xb - x))), float(np.max(np.abs(yb - y))))
