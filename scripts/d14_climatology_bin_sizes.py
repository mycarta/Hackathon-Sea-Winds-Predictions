#!/usr/bin/env python3
"""
Step 1 parameter-gate check for the d14 climatological fix (CLAUDE.md
build discipline #6: propose parameters, get approval, then build).

Counts how many training-AROME days (2016-2020) fall into each
(3-month season, hour) bin and each (calendar month, hour) bin, so Matteo
can pick the stratification before any climatology table is built.

A "sample" for a given footprint cell/bin is one (day, hour) pair, so the
day count IS the per-cell sample count as long as every daily AROME file
has valid (non-NaN) u125m/v125m at all footprint cells for all four
hours — verified separately below on a stratified spot-check, not assumed.

Does not read any u/v data at native resolution (out of scope for a
bin-size count) and does not modify submission.csv or any kit file.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
AROME_DIR = ROOT / "phase_2" / "phase2_dataset_ship" / "train" / "arome"
STATIC_PATH = ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "arome_static.nc"
FOOTPRINT_PATH = ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "footprint_points.parquet"

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
                files.append((y, mo, d, p))
    return files


def spot_check_nan(files, n_files=8, seed_indices=None):
    """Verify daily files have complete (non-NaN) u125m/v125m at all footprint
    cells for all 4 target hours — confirms 'day count == per-cell sample
    count'. Checks a spread of files across years/seasons, not just one."""
    import pandas as pd

    fp = pd.read_parquet(FOOTPRINT_PATH)
    static = xr.open_dataset(STATIC_PATH)
    lat2d = static["latitude"].values
    lon2d = static["longitude"].values
    seamask = static["seamask"].values > 0.5
    ys, xs = np.where(seamask)  # placeholder; real mask below matches footprint.py logic
    # Use the shipped footprint parquet directly instead of re-deriving the mask,
    # by nearest-match by (lat,lon) rounded — but the footprint module's own
    # np.where(mask) order is what we need; re-derive exactly as footprint.py does.
    import sys
    sys.path.insert(0, str(ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))
    import footprint as fp_mod  # noqa: E402
    mask = fp_mod.footprint_mask()
    ys, xs = np.where(mask)
    assert ys.size == 43715, f"footprint mask size mismatch: {ys.size}"

    if seed_indices is None:
        idx = np.linspace(0, len(files) - 1, n_files, dtype=int)
    else:
        idx = seed_indices

    total_nan = 0
    total_checked = 0
    for i in idx:
        y, mo, d, path = files[i]
        ds = xr.open_dataset(path)
        times = pd.to_datetime(ds["time"].values)
        hour_mask = times.hour.isin(HOURS)
        u = ds["u125m"].values[hour_mask][:, ys, xs]
        v = ds["v125m"].values[hour_mask][:, ys, xs]
        n_nan = int(np.isnan(u).sum() + np.isnan(v).sum())
        total_nan += n_nan
        total_checked += u.size + v.size
        print(f"  spot-check {path.name}: hours_present={sorted(times.hour[hour_mask].unique())}, "
              f"nan_count={n_nan} / {u.size + v.size}")
        ds.close()
    static.close()
    return total_nan, total_checked


def main():
    files = find_daily_files()
    print(f"Total daily AROME files found (2016-2020): {len(files)}")

    expected_days = 0
    missing = []
    for year in YEARS:
        import calendar
        is_leap = calendar.isleap(year)
        expected_days += 366 if is_leap else 365
    print(f"Expected calendar days across 2016-2020 (incl. leap): {expected_days}")
    print(f"Actual files: {len(files)}  ->  missing days: {expected_days - len(files)}")

    # Which calendar dates are missing (report, don't guess)
    have = {(y, mo, d) for (y, mo, d, _) in files}
    for year in YEARS:
        import datetime
        start = datetime.date(year, 1, 1)
        end = datetime.date(year, 12, 31)
        cur = start
        while cur <= end:
            if (cur.year, cur.month, cur.day) not in have:
                missing.append(cur.isoformat())
            cur += datetime.timedelta(days=1)
    print(f"Missing dates: {missing}")

    season_counts = Counter()
    month_counts = Counter()
    for (y, mo, d, _p) in files:
        season_counts[SEASON_OF_MONTH[mo]] += 1
        month_counts[mo] += 1

    print("\n--- 3-month SEASON stratification: day-count per (season, hour) bin ---")
    print("(same day-count applies to every hour in HOURS, since each valid day")
    print(" contributes one (day,hour) sample per hour if the file has no gaps)")
    for season in ("DJF", "MAM", "JJA", "SON"):
        n_days = season_counts[season]
        print(f"  {season}: {n_days} days  -> samples per (cell, {season}, hour) bin = {n_days}"
              f"  (x{len(HOURS)} hours = {n_days * len(HOURS)} samples per cell across all hours in this season)")

    print("\n--- MONTHLY stratification: day-count per (month, hour) bin ---")
    for mo in range(1, 13):
        n_days = month_counts[mo]
        print(f"  month {mo:02d}: {n_days} days  -> samples per (cell, month, hour) bin = {n_days}")

    print("\n--- Spot-check for per-cell NaN gaps (confirms day-count == per-cell sample count) ---")
    total_nan, total_checked = spot_check_nan(files, n_files=10)
    print(f"\nTotal NaN found in spot-checked files: {total_nan} / {total_checked} values checked")

    print("\n--- Summary ---")
    print(f"Season bins (4 seasons x 4 hours = 16 bins): min samples/bin = "
          f"{min(season_counts.values())}, max = {max(season_counts.values())}")
    print(f"Month bins (12 months x 4 hours = 48 bins): min samples/bin = "
          f"{min(month_counts.values())}, max = {max(month_counts.values())}")


if __name__ == "__main__":
    main()
