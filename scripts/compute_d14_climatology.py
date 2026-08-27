#!/usr/bin/env python3
"""
Build the d14 climatology lookup table: per footprint cell, per 3-month
season (DJF/MAM/JJA/SON, approved 2026-07-06 per the Step-1 parameter gate
- bin sizes 452-460 samples/cell, spot-checked 0 NaN), per hour (0/6/12/18):
  - scalar speed q05/q50/q95, computed from sqrt(u125m^2 + v125m^2) per
    timestep (never from averaged u/v - that's the vector-mean bug this
    fix replaces).
  - circular direction dir_05/dir_50/dir_95: per-timestep direction via
    the kit's own convention degrees(atan2(-u,-v)) % 360 (forecast_pipeline.py
    :37-38), then dir_50 = circular mean of those per-timestep unit-vector
    directions (NOT vector-mean of u/v - averaging angle unit-vectors is
    unbiased by speed, unlike averaging raw wind vectors), and dir_05/dir_95
    = endpoints of the shortest circular arc containing 90% of the
    per-timestep directions.

Source: training AROME native-grid files (u125m/v125m), 2016-2020, at the
43,715-point forecast footprint (phase_2/kit/phase_2/part0_dataset_setup/
footprint.py::footprint_mask, reused directly for identical (y,x) ordering
so point_id in the output matches static/footprint_points.parquet exactly).

Fully deterministic - no stochastic step, no seed needed.

Output: scripts/artifacts/d14_climatology_season.parquet
  columns: point_id, season, hour, speed_q05, speed_q50, speed_q95,
           dir_05, dir_50, dir_95, n_samples
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
AROME_DIR = ROOT / "phase_2" / "phase2_dataset_ship" / "train" / "arome"
KIT_PART0 = ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
OUT_PATH = ARTIFACTS_DIR / "d14_climatology_season.parquet"

sys.path.insert(0, str(KIT_PART0))
import footprint as fp_mod  # noqa: E402

YEARS = (2016, 2017, 2018, 2019, 2020)
HOURS = (0, 6, 12, 18)
SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
FNAME_RE = re.compile(r"arome_(\d{4})(\d{2})(\d{2})\.nc$")


def find_daily_files():
    files = []
    for year in YEARS:
        for p in sorted((AROME_DIR / str(year)).glob("arome_*.nc")):
            m = FNAME_RE.match(p.name)
            if m:
                y, mo, d = map(int, m.groups())
                files.append((y, mo, SEASON_OF_MONTH[mo], p))
    return files


def circular_arc_quantile(sorted_deg, k):
    """sorted_deg: (n_cells, n) ascending per row, degrees in [0,360).
    Returns (lo, hi) each shape (n_cells,) = endpoints of the shortest arc
    containing k of the n points (vectorized sliding-window over the
    360-duplicated array, no python loop over samples)."""
    n = sorted_deg.shape[1]
    extended = np.concatenate([sorted_deg, sorted_deg + 360.0], axis=1)
    start_vals = extended[:, 0:n]
    end_vals = extended[:, k - 1:k - 1 + n]
    widths = end_vals - start_vals
    best_i = np.argmin(widths, axis=1)
    rows = np.arange(sorted_deg.shape[0])
    lo = start_vals[rows, best_i] % 360.0
    hi = end_vals[rows, best_i] % 360.0
    return lo, hi


def main():
    ys, xs = np.where(fp_mod.footprint_mask())
    n_cells = ys.size
    assert n_cells == 43715, f"unexpected footprint size {n_cells}"
    point_id = np.arange(n_cells, dtype=np.int32)

    files = find_daily_files()
    print(f"Found {len(files)} daily AROME files across {YEARS}")

    rows_out = []
    for season in ("DJF", "MAM", "JJA", "SON"):
        season_files = [p for (_y, _mo, s, p) in files if s == season]
        n_files = len(season_files)
        print(f"\nSeason {season}: {n_files} files")

        u_stack = np.empty((n_files, len(HOURS), n_cells), dtype=np.float32)
        v_stack = np.empty((n_files, len(HOURS), n_cells), dtype=np.float32)

        for i, path in enumerate(season_files):
            ds = xr.open_dataset(path)
            times = pd.to_datetime(ds["time"].values)
            hour_mask = times.hour.isin(HOURS)
            order = np.argsort(times.hour[hour_mask].values)  # ensure 0,6,12,18 order
            u = ds["u125m"].values[hour_mask][order][:, ys, xs]
            v = ds["v125m"].values[hour_mask][order][:, ys, xs]
            u_stack[i] = u
            v_stack[i] = v
            ds.close()

        speed_stack = np.sqrt(u_stack ** 2 + v_stack ** 2)

        for h_idx, hour in enumerate(HOURS):
            speed_h = speed_stack[:, h_idx, :]  # (n_files, n_cells)
            q05, q50, q95 = np.quantile(speed_h, [0.05, 0.5, 0.95], axis=0)

            u_h = u_stack[:, h_idx, :]
            v_h = v_stack[:, h_idx, :]
            dir_h = np.degrees(np.arctan2(-u_h, -v_h)) % 360.0  # (n_files, n_cells)

            theta = np.radians(dir_h)
            mean_sin = np.mean(np.sin(theta), axis=0)
            mean_cos = np.mean(np.cos(theta), axis=0)
            dir_mean = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0

            dir_h_T = dir_h.T  # (n_cells, n_files)
            sorted_dir = np.sort(dir_h_T, axis=1)
            k = int(np.ceil(0.9 * n_files))
            dir_05, dir_95 = circular_arc_quantile(sorted_dir, k)

            rows_out.append(pd.DataFrame({
                "point_id": point_id,
                "season": season,
                "hour": hour,
                "speed_q05": q05.astype(np.float32),
                "speed_q50": q50.astype(np.float32),
                "speed_q95": q95.astype(np.float32),
                "dir_05": dir_05.astype(np.float32),
                "dir_50": dir_mean.astype(np.float32),
                "dir_95": dir_95.astype(np.float32),
                "n_samples": n_files,
            }))
            print(f"  hour {hour:02d}: speed_q50 median-across-cells="
                  f"{np.median(q50):.3f} m/s, dir_50 median-across-cells="
                  f"{np.median(dir_mean):.1f} deg, n={n_files}")

        del u_stack, v_stack, speed_stack

    table = pd.concat(rows_out, ignore_index=True)
    assert len(table) == 43715 * 4 * 4, f"unexpected row count {len(table)}"
    assert table.isna().sum().sum() == 0, "NaN in climatology table"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(table)} rows)")


if __name__ == "__main__":
    main()
