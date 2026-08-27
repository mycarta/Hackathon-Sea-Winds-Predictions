#!/usr/bin/env python3
"""TIER-2 ERA5 fetch from the public WeatherBench2 GCS mirror (anonymous, no CDS).

Opus-authorized Tier-2 build (Phase 0 confirmed 2026-07-17). Source recorded in
every committed artifact. Compliance Clause 1: only a single analysis time at or
before the issue time is ever read.

  fetch_global(date, hour) -> (upper (5,13,721,1440), surface (4,721,1440)) float32
      full Pangu initial state (Z,Q,T,U,V x 13 levels ; MSLP,U10,V10,T2M).
  fetch_box(date, hour)    -> dict of 8 (45,57) fields for the coupler (fast; F1).

WB2 already matches Pangu orientation (lat 90..-90, lon 0..359.75) and
'geopotential' is m^2/s^2 (no x9.80665). Verified in the smoke test.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import xarray as xr

WB2_ZARR = ("gs://weatherbench2/datasets/era5/"
            "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr")
PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
UPPER_VARS = ["geopotential", "specific_humidity", "temperature",
              "u_component_of_wind", "v_component_of_wind"]
SURF_VARS = ["mean_sea_level_pressure", "10m_u_component_of_wind",
             "10m_v_component_of_wind", "2m_temperature"]


@lru_cache(maxsize=1)
def _ds():
    return xr.open_zarr(WB2_ZARR, storage_options={"token": "anon"},
                        chunks=None, decode_timedelta=True)


def _snap(date, hour):
    import pandas as pd
    t = np.datetime64(f"{pd.Timestamp(date).strftime('%Y-%m-%d')}T{hour:02d}:00:00")
    return _ds().sel(time=t), ("level" if "level" in _ds().dims else "pressure_level")


def fetch_global(date, hour=0):
    """Full Pangu initial state at (date, hour). float32 arrays."""
    snap, lvl = _snap(date, hour)
    upper = np.stack([snap[v].sel({lvl: PANGU_LEVELS}).values.astype("float32")
                      for v in UPPER_VARS], axis=0)          # (5,13,721,1440)
    surface = np.stack([snap[v].values.astype("float32")
                        for v in SURF_VARS], axis=0)          # (4,721,1440)
    assert upper.shape == (5, 13, 721, 1440), upper.shape
    assert surface.shape == (4, 721, 1440), surface.shape
    return upper, surface


def fetch_box(date, hour=0):
    """The 8 coupler fields on the AROME coarse 45x57 grid (fast: box only)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tier2_pangu_couple import coarse_grid
    lat, lon = coarse_grid()
    snap, lvl = _snap(date, hour)
    wb_lon = xr.DataArray(lon % 360.0, dims="lon")
    wb_lat = xr.DataArray(lat, dims="lat")

    def sel(v, level=None):
        da = snap[v]
        if level is not None:
            da = da.sel({lvl: level})
        return da.sel(latitude=wb_lat, longitude=wb_lon,
                      method="nearest").values.astype("float64")

    return {
        "u10": sel("10m_u_component_of_wind"),
        "v10": sel("10m_v_component_of_wind"),
        "u1000": sel("u_component_of_wind", 1000),
        "v1000": sel("v_component_of_wind", 1000),
        "z1000": sel("geopotential", 1000) / 9.80665,
        "u925": sel("u_component_of_wind", 925),
        "v925": sel("v_component_of_wind", 925),
        "z925": sel("geopotential", 925) / 9.80665,
    }
