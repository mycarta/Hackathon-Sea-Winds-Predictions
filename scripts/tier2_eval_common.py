#!/usr/bin/env python3
"""TIER-2 shared evaluation invariant + data loaders (blocks, exclusion, truth).

Opus-authorized Tier-2 build (Phase 0 confirmed 2026-07-17). The blocks, buffer,
and exclusion are IDENTICAL to the 2026-07-17 refit (scripts/ws_d7_block_refit.py,
commit a139fa3) so both arms are scored on the same split with the same invariant:
nothing within block+/-14 d enters any fitted component. Seeded where sampled.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "phase_2" / "kit" / "phase_2"))
sys.path.insert(0, str(_REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))

SEED = 42
HOURS = (0, 6, 12, 18)
BUFFER_DAYS = 14
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
BLOCKS = {                                             # confirmed 2026-07-16 (a139fa3)
    "DJF": ("2017-01-09", "2017-02-05"),
    "MAM": ("2018-04-09", "2018-05-06"),
    "JJA": ("2019-07-08", "2019-08-04"),
    "SON": ("2020-10-05", "2020-11-01"),
}
AROME_DIR = _REPO / "phase_2" / "phase2_dataset_ship" / "train" / "arome"


def block_days() -> dict:
    out = {}
    for s, (a, b) in BLOCKS.items():
        A, B = pd.Timestamp(a).date(), pd.Timestamp(b).date()
        out[s] = [A + timedelta(days=i) for i in range((B - A).days + 1)]
    return out


def block_valid_dates() -> list:
    out = []
    for s in SEASONS:
        out += block_days()[s]
    return out


def exclusion_set() -> set:
    excl = set()
    for s, (a, b) in BLOCKS.items():
        A = pd.Timestamp(a).date() - timedelta(days=BUFFER_DAYS)
        B = pd.Timestamp(b).date() + timedelta(days=BUFFER_DAYS)
        excl |= {A + timedelta(days=i) for i in range((B - A).days + 1)}
    return excl


def season_of(d) -> str:
    return SEASON_OF_MONTH[pd.Timestamp(d).month]


@lru_cache(maxsize=2048)
def load_u125c(date_iso: str, hour: int):
    """Shipped AROME coarse truth (u125c, v125c) as (45,57), or None if absent."""
    import xarray as xr
    import config
    d = pd.Timestamp(date_iso)
    p = config.coarse_root() / f"{d.year}" / f"coarse_{d:%Y%m%d}.nc"
    if not p.exists():
        return None
    ds = xr.open_dataset(p)
    try:
        sel = ds.sel(time=d + pd.Timedelta(hours=hour))
        return (sel["u125c"].values.astype("float64"),
                sel["v125c"].values.astype("float64"))
    finally:
        ds.close()


@lru_cache(maxsize=1)
def footprint_yx():
    import footprint as fp_mod
    ys, xs = np.where(fp_mod.footprint_mask())
    assert ys.size == 43715, ys.size
    return ys, xs


def arome_truth_speed(date_iso: str, hour: int):
    """AROME native-grid 125 m speed at the 43,715 footprint cells for one (date, hour)."""
    import xarray as xr
    d = pd.Timestamp(date_iso)
    path = AROME_DIR / f"{d.year}" / f"arome_{d:%Y%m%d}.nc"
    if not path.exists():
        return None
    ys, xs = footprint_yx()
    ds = xr.open_dataset(path)
    times = pd.to_datetime(ds["time"].values)
    m = times.hour == hour
    if not m.any():
        ds.close()
        return None
    i = np.where(m)[0][0]
    u = ds["u125m"].values[i][ys, xs]
    v = ds["v125m"].values[i][ys, xs]
    ds.close()
    return np.sqrt(u ** 2 + v ** 2)
