#!/usr/bin/env python3
"""TIER-2 coupling: Pangu-Weather (or ERA5) state -> 125 m u/v on the AROME
coarse 45x57 grid, the field the existing downscaler/interval machinery consumes.

Opus-authorized Tier-2 build (Phase 0 confirmed 2026-07-17). NEW module; touches
no pipeline file. Driver = Pangu-Weather ONNX (ERA5-trained, CC BY-NC-SA 4.0);
initial states = ERA5 (WeatherBench2 public GCS) at the issue time only (Clause 1).

Grid (verified):
  * AROME coarse target grid: lat 51.00..62.00 (45), lon -4.00..10.00 (57), 0.25 deg
    (config.coarse_root()/*/coarse_*.nc; vars u125c/v125c).
  * Pangu/ERA5 grid: lat 90..-90 (721), lon 0..359.75 (1440), 0.25 deg.
  Both on the same 0.25 deg offsets, so coarse cells map to EXACT Pangu cells
  (lon<0 -> +360); no horizontal interpolation, only selection.

Pangu tensor order (verified, tier2_smoke/pangu_README.md lines 80-82):
  upper   (5,13,721,1440): Z,Q,T,U,V x [1000,925,850,700,600,500,400,300,250,
                           200,150,100,50] hPa
  surface (4,721,1440):    MSLP,U10,V10,T2M

Vertical to 125 m (Phase 0 approved): per cell, power-law-in-height (alpha=0.11,
scorer physics, CLAUDE.md) interpolation of u and v COMPONENTS between the two
heights bracketing 125 m:
  * primary  bracket [10 m, 1000 hPa]  when z1000 > 125 m
  * fallback bracket [1000 hPa, 925 hPa] when z1000 <= 125 m
  z(level) = geopotential / 9.80665 (Z is geopotential, m^2/s^2). Interpolating
  components (not speed) preserves directional veer. Weight
  w = (125^a - z_lo^a)/(z_hi^a - z_lo^a); u125 = (1-w)*u_lo + w*u_hi.
"""
from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

import numpy as np

G = 9.80665
ALPHA = 0.11
HUB_M = 125.0
PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
L1000, L925 = 0, 1  # indices into the 13-level axis

_REPO = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def coarse_grid():
    """(lat[45], lon[57]) of the AROME coarse target grid, read from a shipped file."""
    import xarray as xr
    import sys
    sys.path.insert(0, str(_REPO / "phase_2" / "kit" / "phase_2"))
    import config
    f = sorted(glob.glob(str(config.coarse_root() / "*" / "coarse_*.nc")))[0]
    ds = xr.open_dataset(f)
    lat = ds["latitude"].values.astype(float)
    lon = ds["longitude"].values.astype(float)
    ds.close()
    return lat, lon


@lru_cache(maxsize=1)
def _global_indices():
    """(lat_idx[45], lon_idx[57]) selecting the coarse box out of the 721x1440 grid."""
    lat, lon = coarse_grid()
    lat_idx = np.rint((90.0 - lat) / 0.25).astype(int)      # 90..-90 descending
    lon_idx = np.rint((lon % 360.0) / 0.25).astype(int)     # 0..359.75 ascending
    assert lat_idx.min() >= 0 and lat_idx.max() < 721, lat_idx
    assert lon_idx.min() >= 0 and lon_idx.max() < 1440, lon_idx
    return lat_idx, lon_idx


def extract_box_from_global(upper: np.ndarray, surface: np.ndarray) -> dict:
    """Pangu/ERA5 global arrays -> the 8 (45,57) box fields the coupler needs."""
    assert upper.shape == (5, 13, 721, 1440), upper.shape
    assert surface.shape == (4, 721, 1440), surface.shape
    li, lj = _global_indices()
    sel = np.ix_(li, lj)
    return {
        "u10": surface[1][sel].astype("float64"),
        "v10": surface[2][sel].astype("float64"),
        "u1000": upper[3, L1000][sel].astype("float64"),
        "v1000": upper[4, L1000][sel].astype("float64"),
        "z1000": upper[0, L1000][sel].astype("float64") / G,
        "u925": upper[3, L925][sel].astype("float64"),
        "v925": upper[4, L925][sel].astype("float64"),
        "z925": upper[0, L925][sel].astype("float64") / G,
    }


def couple_box(box: dict, alpha: float = ALPHA, hub: float = HUB_M):
    """Power-law-in-height interpolation to `hub` m. Returns (u125, v125, fallback_mask).

    Pick the two anchor heights bracketing `hub` from {10 m, z1000, z925}:
      * z1000 > hub          -> [10 m,   1000 hPa]   (primary)
      * 10 m < z1000 <= hub  -> [1000 hPa, 925 hPa]  (925 fallback, hub above 1000 hPa)
      * z1000 <= 10 m        -> [10 m,   925 hPa]    (1000 hPa at/below the 10 m anchor;
                                                      happens when MSLP < ~1000 hPa so
                                                      z1000 <= 0 -- avoids a negative base
                                                      in z**alpha)
    fallback_mask[c] = True where z1000 <= hub (the 925 hPa level is an anchor).
    """
    z1000 = box["z1000"]
    z925 = box["z925"]
    primary = z1000 > hub                                   # [10 m, 1000 hPa]
    mid = (z1000 > 10.0) & ~primary                         # [1000 hPa, 925 hPa]
    fallback = ~primary                                     # 925 hPa is an anchor
    # low = z1000 <= 10 m -> [10 m, 925 hPa]

    z_lo = np.where(primary, 10.0, np.where(mid, z1000, 10.0))
    z_hi = np.where(primary, z1000, z925)
    u_lo = np.where(primary, box["u10"], np.where(mid, box["u1000"], box["u10"]))
    v_lo = np.where(primary, box["v10"], np.where(mid, box["v1000"], box["v10"]))
    u_hi = np.where(primary, box["u1000"], box["u925"])
    v_hi = np.where(primary, box["v1000"], box["v925"])

    zla, zha, ha = z_lo ** alpha, z_hi ** alpha, hub ** alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (ha - zla) / (zha - zla)
    w = np.clip(np.where(np.isfinite(w), w, 1.0), 0.0, 1.0)
    u125 = (1.0 - w) * u_lo + w * u_hi
    v125 = (1.0 - w) * v_lo + w * v_hi
    return u125.astype("float32"), v125.astype("float32"), fallback


def couple_global(upper: np.ndarray, surface: np.ndarray, alpha: float = ALPHA):
    """Convenience: global Pangu/ERA5 arrays -> (u125, v125, fallback_mask) on 45x57."""
    return couple_box(extract_box_from_global(upper, surface), alpha=alpha)
