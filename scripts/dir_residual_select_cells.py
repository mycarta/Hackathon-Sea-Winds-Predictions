#!/usr/bin/env python3
"""
Step 1 (part A) of the d1/d7 direction-residual fix: select a seeded,
spatially-spread subsample of footprint fine cells for training data
construction (full 43,715-cell x ~1826-day extraction is intractable;
see the parameter-gate discussion, approved 2026-07-07).

Also resolves the open question in CLAUDE.md / phase2-repo-map memory:
"distinct 0.25 deg reanalysis cells over footprint_points.parquet == 159?"
by directly computing how many distinct coarse (0.25 deg) HRES/reanalysis
grid cells the full 43,715-point footprint touches.

Method: nearest-coarse-cell mapping for every footprint fine cell, then
np.random.default_rng(42).choice of 5000 fine-cell indices (uniform, no
replacement). If any distinct coarse cell touched by the full footprint has
zero representation in the 5000-cell sample, switch to stratified-by-
coarse-cell sampling (proportional allocation, same seed) to guarantee full
coverage - per Matteo's explicit fallback instruction, 2026-07-07.

Output: scripts/artifacts/dir_residual_cells.parquet
  columns: cell_idx (0..43714, matches footprint np.where(mask) order),
           y, x (native AROME grid indices), lat, lon (fine cell, float64),
           coarse_i, coarse_j (indices into the 45x57 reanalysis grid),
           coarse_lat, coarse_lon
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
STATIC_PATH = ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "arome_static.nc"
KIT_PART0 = ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"
KIT_PART1 = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
OUT_PATH = ARTIFACTS_DIR / "dir_residual_cells.parquet"

sys.path.insert(0, str(KIT_PART0))
sys.path.insert(0, str(KIT_PART1))
import footprint as fp_mod  # noqa: E402
import forecast_hres as fh  # noqa: E402

N_SAMPLE = 5000
SEED = 42


def main():
    static = xr.open_dataset(STATIC_PATH)
    lat2d = static["latitude"].values.astype("float64")
    lon2d = static["longitude"].values.astype("float64")

    ys, xs = np.where(fp_mod.footprint_mask())
    n_cells = ys.size
    assert n_cells == 43715, f"unexpected footprint size {n_cells}"
    lat = lat2d[ys, xs]
    lon = lon2d[ys, xs]

    lat1d, lon1d = fh._reanalysis_grid()  # (45,), (57,) - the 0.25 deg coarse grid
    lat1d = lat1d.astype("float64")
    lon1d = lon1d.astype("float64")
    print(f"Coarse grid: {lat1d.size} lat x {lon1d.size} lon = {lat1d.size * lon1d.size} cells "
          f"(lat {lat1d.min()}-{lat1d.max()}, lon {lon1d.min()}-{lon1d.max()})")

    coarse_i = np.abs(lat[:, None] - lat1d[None, :]).argmin(axis=1)
    coarse_j = np.abs(lon[:, None] - lon1d[None, :]).argmin(axis=1)
    coarse_key = coarse_i * lon1d.size + coarse_j

    distinct_coarse = np.unique(coarse_key)
    print(f"\nFull 43,715-point footprint touches {distinct_coarse.size} distinct "
          f"coarse (0.25 deg) cells.")
    print(f"CLAUDE.md open question ('== 159?'): "
          f"{'CONFIRMED, exactly 159' if distinct_coarse.size == 159 else f'NOT 159 (got {distinct_coarse.size})'}")

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(n_cells, size=N_SAMPLE, replace=False)
    sample_idx.sort()

    sample_coarse_key = coarse_key[sample_idx]
    sample_distinct = np.unique(sample_coarse_key)
    missing = np.setdiff1d(distinct_coarse, sample_distinct)
    print(f"\nUniform random sample ({N_SAMPLE} cells, seed={SEED}): "
          f"{sample_distinct.size} / {distinct_coarse.size} distinct coarse cells represented.")

    if missing.size > 0:
        print(f"Gap: {missing.size} coarse cells have ZERO representation - "
              f"switching to stratified-by-coarse-cell sampling.")
        df_all = pd.DataFrame({"cell_idx": np.arange(n_cells), "coarse_key": coarse_key})
        groups = {k: g["cell_idx"].to_numpy() for k, g in df_all.groupby("coarse_key")}
        n_groups = len(groups)
        base_per_group = N_SAMPLE // n_groups
        remainder = N_SAMPLE - base_per_group * n_groups
        rng2 = np.random.default_rng(SEED)
        keys_sorted = sorted(groups.keys())
        picks = []
        for gi, k in enumerate(keys_sorted):
            members = groups[k]
            n_take = base_per_group + (1 if gi < remainder else 0)
            n_take = min(n_take, members.size)
            picks.append(rng2.choice(members, size=n_take, replace=False))
        sample_idx = np.concatenate(picks)
        # top up to exactly N_SAMPLE from leftover cells if group caps left us short
        if sample_idx.size < N_SAMPLE:
            remaining_pool = np.setdiff1d(np.arange(n_cells), sample_idx)
            extra = rng2.choice(remaining_pool, size=N_SAMPLE - sample_idx.size, replace=False)
            sample_idx = np.concatenate([sample_idx, extra])
        sample_idx.sort()
        sample_coarse_key = coarse_key[sample_idx]
        sample_distinct = np.unique(sample_coarse_key)
        print(f"Stratified sample ({sample_idx.size} cells): "
              f"{sample_distinct.size} / {distinct_coarse.size} distinct coarse cells represented.")
        assert sample_distinct.size == distinct_coarse.size, "stratified sampling still has gaps"
    else:
        print("Full coverage achieved with plain uniform sampling - no stratification needed.")

    out = pd.DataFrame({
        "cell_idx": sample_idx.astype(np.int32),
        "y": ys[sample_idx].astype(np.int32),
        "x": xs[sample_idx].astype(np.int32),
        "lat": lat[sample_idx],
        "lon": lon[sample_idx],
        "coarse_i": coarse_i[sample_idx].astype(np.int32),
        "coarse_j": coarse_j[sample_idx].astype(np.int32),
        "coarse_lat": lat1d[coarse_i[sample_idx]],
        "coarse_lon": lon1d[coarse_j[sample_idx]],
    })
    assert len(out) == N_SAMPLE
    assert out["cell_idx"].is_unique

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(out)} rows)")

    static.close()


if __name__ == "__main__":
    main()
