"""EMODnet water-depth sampler for Task 2 siting work.

Task2 CC prompt (2026-07-10), Task B.2.

Reads `static/bathymetry/emodnet_northsea_1km.nc` straight from the Phase-2
ship layout. NOT a wrapper around the kit's `part0_dataset_setup/bathymetry.py`:
that module's `EMODNET_CANDIDATES` are

    ROOT / "build" / "phase2_dataset" / "bathymetry" / "emodnet_northsea_1km.nc"
    _HERE.parents[1] / "data" / "bathymetry" / "emodnet_northsea_1km.nc"

(bathymetry.py:36-39) and neither path exists under the ship layout used here
(`phase_2/phase2_dataset_ship/static/bathymetry/...`) - confirmed empirically:
`bathymetry.available()` returns False when imported against
`PHASE2_DATA_ROOT=.../phase2_dataset_ship`, so `is_fixed_bottom`/`is_eligible`
silently return True (constraint inactive) rather than raising. See report
`reports/task2_scorer_bathymetry_ranking_20260710.md` Task A.5 for the full
finding. This module reads the file directly so depth queries are correct
regardless of that path-resolution gap.

Grid: EMODnet DTM 2022 resampled to 0.01 deg (~1.1 km), lat 50.00-56.50 N,
lon -5.00-9.50 E. depth_max_m=60.0 / dist_min_km=5.6 baked into the file's
own `eligible` layer (file attrs) - NOT the 50 m the siting notebooks
actually use (`zone.is_in_allowed_zone(..., max_depth_m=50)` in
3a/3b_farm_optimization*.ipynb). This module exposes raw depth; callers
choose their own threshold.

No projection is defined anywhere in the kit (`wind_farm_simulator.py`,
`optimization.py` pass layout x_m/y_m straight into PyWake as bare Cartesian
offsets; no lat/lon<->x_m/y_m conversion function exists in the shipped
code - confirmed by grep, see report Task A.6). `depth_at_xy` below uses
`task2_projection.local_xy_to_latlon` (2026-07-13, Task E) - the single
source of truth for this projection, so it can't drift from what the
Task F scorer replica / future layout code use.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from task2_projection import local_xy_to_latlon  # noqa: E402

EMODNET_PATH = (
    REPO_ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "bathymetry"
    / "emodnet_northsea_1km.nc"
)


@lru_cache(maxsize=1)
def _load(path: str = str(EMODNET_PATH)) -> xr.Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"EMODnet bathymetry not found at {p}. Expected the Phase-2 ship "
            "layout at phase_2/phase2_dataset_ship/static/bathymetry/."
        )
    return xr.load_dataset(p)


def depth_at(lat: float, lon: float) -> float:
    """Water depth in metres (positive down), nearest EMODnet 0.01 deg pixel.

    NaN on land or outside the file's lat 50-56.5N / lon -5-9.5E coverage.
    """
    ds = _load()
    d = float(ds["water_depth_m"].sel(lat=lat, lon=lon, method="nearest").values)
    return d if np.isfinite(d) else float("nan")


def depth_at_array(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vectorized depth_at over equal-length lat/lon arrays (nearest pixel each)."""
    ds = _load()
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    da = ds["water_depth_m"]
    sel = da.sel(
        lat=xr.DataArray(lat, dims="points"),
        lon=xr.DataArray(lon, dims="points"),
        method="nearest",
    ).values.astype(float)
    return np.where(np.isfinite(sel), sel, np.nan)


def dist_to_coast_at(lat: float, lon: float) -> float:
    ds = _load()
    d = float(ds["dist_coast_km"].sel(lat=lat, lon=lon, method="nearest").values)
    return d if np.isfinite(d) else float("nan")


def depth_at_xy(
    centre_lat: float, centre_lon: float, x_m: np.ndarray, y_m: np.ndarray,
) -> np.ndarray:
    """Vectorized depth lookup for layout positions (x_m, y_m) around a farm centre."""
    lat, lon = local_xy_to_latlon(centre_lat, centre_lon, x_m, y_m)
    return depth_at_array(lat, lon)


if __name__ == "__main__":
    print("EMODnet file:", EMODNET_PATH, "exists:", EMODNET_PATH.exists())
    print("depth at kit baseline centre (53.5N, 1.5E):", depth_at(53.5, 1.5))
    print("dist to coast at baseline centre:", dist_to_coast_at(53.5, 1.5))
