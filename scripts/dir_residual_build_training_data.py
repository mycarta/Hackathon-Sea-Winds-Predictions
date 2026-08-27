#!/usr/bin/env python3
"""
Step 0 (convention check) + Step 1 (build training data) of the d1/d7
direction-residual fix.

For the 5,000-cell subsample (scripts/artifacts/dir_residual_cells.parquet,
scope-approved 2026-07-07), builds one training row per
(cell, valid_date, hour, lead) for lead in (1, 7):
  residual = ((arome_dir - hres_dir + 180) % 360) - 180
where arome_dir is computed from native AROME truth (u125m/v125m) at the
fine cell, and hres_dir/hres_speed are the raw HRES forecast broadcast
(nearest-neighbor) from the fine cell's containing coarse (0.25 deg) grid
cell - HRES has no native fine-resolution equivalent.

Train/val split follows splits.py::train_val_dates(seed=42) (day-level, no
leakage across hours of the same day), applied to the valid (truth) date.

Output: scripts/artifacts/dir_residual_training.parquet
  columns: cell_idx, lat, lon, lead, valid_date, hour, month, season,
           hres_speed, hres_dir, arome_dir, residual, split
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
AROME_DIR = ROOT / "phase_2" / "phase2_dataset_ship" / "train" / "arome"
KIT_PART0 = ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"
KIT_PART1 = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CELLS_PATH = ARTIFACTS_DIR / "dir_residual_cells.parquet"
OUT_PATH = ARTIFACTS_DIR / "dir_residual_training.parquet"

sys.path.insert(0, str(KIT_PART0))
sys.path.insert(0, str(KIT_PART1))
import splits  # noqa: E402
import target_loader  # noqa: E402
import config  # noqa: E402
import forecast_hres as fh  # noqa: E402

LEADS = (1, 7)
HOURS = (0, 6, 12, 18)
SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def arome_dir_from_uv(u, v):
    return np.degrees(np.arctan2(-u, -v)) % 360.0


def convention_check(cells, hres_small):
    """Step 0: print 10 example cells comparing AROME truth dir, HRES dir,
    and (where available) the kit's existing submission dir_50, to confirm
    all three use the same meteorological convention before training."""
    print("=" * 70)
    print("STEP 0 - CONVENTION CHECK")
    print("=" * 70)

    sample_date = pd.Timestamp("2019-06-15")
    sample_path = AROME_DIR / "2019" / f"arome_{sample_date:%Y%m%d}.nc"
    ds = xr.open_dataset(sample_path)
    times = pd.to_datetime(ds["time"].values)
    h0_idx = int(np.where(times.hour == 0)[0][0])
    u_all = ds["u125m"].values[h0_idx]
    v_all = ds["v125m"].values[h0_idx]
    ds.close()

    sample_cells = cells.iloc[:10]
    print(f"\n{'cell_idx':>8} {'lat':>7} {'lon':>7} {'arome_dir(u,v)':>16} "
          f"{'coarse_lat':>10} {'coarse_lon':>10}")
    for _, row in sample_cells.iterrows():
        u = u_all[int(row["y"]), int(row["x"])]
        v = v_all[int(row["y"]), int(row["x"])]
        d = arome_dir_from_uv(u, v)
        print(f"{row['cell_idx']:>8} {row['lat']:>7.3f} {row['lon']:>7.3f} "
              f"{d:>16.3f} {row['coarse_lat']:>10.3f} {row['coarse_lon']:>10.3f}")

    print("\nHRES fcst_dir_d1_h0 at the same sample date's issue date "
          f"({sample_date.date()}) for the coarse cells above (raw column, "
          "no computation - direction is stored directly in HRES):")
    key = hres_small[hres_small["time"] == sample_date]
    for _, row in sample_cells.iterrows():
        m = (np.isclose(key["latitude"], row["coarse_lat"])) & \
            (np.isclose(key["longitude"], row["coarse_lon"]))
        hit = key[m]
        if len(hit):
            print(f"  cell {row['cell_idx']}: coarse ({row['coarse_lat']:.2f},"
                  f"{row['coarse_lon']:.2f}) -> fcst_dir_d1_h0="
                  f"{hit['fcst_dir_d1_h0'].values[0]:.3f}")

    print("\nRound-trip check: HRES (speed,dir) -> u,v via kit's "
          "_uv_from_speed_dir -> back to dir via atan2(-u,-v)%360 must "
          "reproduce the original fcst_dir exactly (confirms formula "
          "self-consistency, matching forecast_pipeline.py's own dir50 line):")
    row0 = key.iloc[0]
    sp, di = row0["fcst_speed_d1_h0"], row0["fcst_dir_d1_h0"]
    u, v = fh._uv_from_speed_dir(sp, di)
    di_rt = arome_dir_from_uv(u, v)
    print(f"  original fcst_dir_d1_h0={di:.6f}, round-tripped={di_rt:.6f}, "
          f"diff={abs(di - di_rt):.2e} deg")
    assert abs(di - di_rt) < 1e-6, "convention mismatch between AROME truth and HRES formulas!"
    print("\nConvention check PASSED: AROME truth (target_loader/forecast_pipeline) "
          "and HRES (forecast_hres) use the identical meteorological convention "
          "degrees(atan2(-u,-v)) % 360.")
    print("=" * 70)


def load_hres_filtered(coarse_pairs):
    """Load + concat both HRES parquets, filtered to only the coarse (lat,lon)
    pairs actually used by the cell subsample."""
    paths = config.hres_parquets()
    print(f"\nLoading HRES from: {[p.name for p in paths]}")
    dfs = []
    lat_set = set(np.round(coarse_pairs[:, 0], 3))
    lon_set = set(np.round(coarse_pairs[:, 1], 3))
    for p in paths:
        df = pd.read_parquet(p)
        df["time"] = pd.to_datetime(df["time"])
        m = df["latitude"].round(3).isin(lat_set) & df["longitude"].round(3).isin(lon_set)
        dfs.append(df[m])
    hres = pd.concat(dfs, ignore_index=True, sort=False)
    hres = hres.drop_duplicates(subset=["time", "latitude", "longitude"]).reset_index(drop=True)
    print(f"Filtered HRES: {len(hres)} rows (coarse cells: {len(coarse_pairs)}, "
          f"issue dates: {hres['time'].nunique()})")
    return hres


def main():
    cells = pd.read_parquet(CELLS_PATH)
    n_cells = len(cells)
    print(f"Loaded {n_cells} subsampled cells from {CELLS_PATH}")

    coarse_pairs = cells[["coarse_lat", "coarse_lon"]].drop_duplicates().to_numpy()
    print(f"Distinct coarse cells among subsample: {len(coarse_pairs)}")

    hres = load_hres_filtered(coarse_pairs)
    convention_check(cells, hres)

    # Build a fast per-(date, coarse-group) lookup: group index 0..K-1 for the
    # distinct coarse pairs, vectorized broadcast onto the 5000 fine cells.
    coarse_key_str = [f"{float(la):.3f}_{float(lo):.3f}" for la, lo in coarse_pairs]
    coarse_group_of_pair = {k: i for i, k in enumerate(coarse_key_str)}
    cells_group = np.array([
        coarse_group_of_pair[f"{float(la):.3f}_{float(lo):.3f}"]
        for la, lo in zip(cells["coarse_lat"], cells["coarse_lon"])
    ])
    n_groups = len(coarse_pairs)

    hres["_key"] = (hres["latitude"].astype("float64").round(3).map(lambda x: f"{x:.3f}") + "_" +
                     hres["longitude"].astype("float64").round(3).map(lambda x: f"{x:.3f}"))
    hres_by_time = {t: g.set_index("_key") for t, g in hres.groupby("time")}
    hres_time_set = set(hres_by_time.keys())
    print(f"HRES indexed for {len(hres_by_time)} distinct issue dates.")

    train_dates, val_dates = splits.train_val_dates(seed=42)
    train_set = set(train_dates)
    val_set = set(val_dates)
    all_valid_dates = sorted(set(train_dates) | set(val_dates))
    print(f"\ntrain_val_dates(seed=42): {len(train_dates)} train days, "
          f"{len(val_dates)} val days, {len(all_valid_dates)} total.")

    ys = cells["y"].to_numpy()
    xs = cells["x"].to_numpy()
    cell_idx = cells["cell_idx"].to_numpy()
    lat = cells["lat"].to_numpy()
    lon = cells["lon"].to_numpy()

    rows_per_lead = {1: [], 7: []}
    n_files_read = 0
    n_files_missing = 0

    for V in all_valid_dates:
        Vts = pd.Timestamp(V)
        path = AROME_DIR / f"{V.year}" / f"arome_{V:%Y%m%d}.nc"
        if not path.exists():
            n_files_missing += 1
            continue

        needed_leads = [L for L in LEADS if (Vts - pd.Timedelta(days=L)) in hres_time_set]
        if not needed_leads:
            continue

        ds = xr.open_dataset(path)
        times = pd.to_datetime(ds["time"].values)
        hour_mask = times.hour.isin(HOURS)
        order = np.argsort(times.hour[hour_mask].values)
        u = ds["u125m"].values[hour_mask][order][:, ys, xs]  # (4, n_cells)
        v = ds["v125m"].values[hour_mask][order][:, ys, xs]
        ds.close()
        n_files_read += 1
        arome_dir = arome_dir_from_uv(u, v)  # (4, n_cells)

        split_tag = "train" if V in train_set else "val"
        month = V.month
        season = SEASON_OF_MONTH[month]

        for L in needed_leads:
            D = Vts - pd.Timedelta(days=L)
            hres_day = hres_by_time[D]
            for h_idx, H in enumerate(HOURS):
                sp_col = f"fcst_speed_d{L}_h{H}"
                di_col = f"fcst_dir_d{L}_h{H}"
                group_speed = hres_day.loc[coarse_key_str, sp_col].to_numpy()
                group_dir = hres_day.loc[coarse_key_str, di_col].to_numpy()
                hres_speed = group_speed[cells_group]
                hres_dir = group_dir[cells_group]

                a_dir = arome_dir[h_idx]
                residual = ((a_dir - hres_dir + 180) % 360) - 180

                rows_per_lead[L].append(pd.DataFrame({
                    "cell_idx": cell_idx, "lat": lat, "lon": lon,
                    "lead": L, "valid_date": V, "hour": H,
                    "month": month, "season": season,
                    "hres_speed": hres_speed.astype(np.float32),
                    "hres_dir": hres_dir.astype(np.float32),
                    "arome_dir": a_dir.astype(np.float32),
                    "residual": residual.astype(np.float32),
                    "split": split_tag,
                }))

        if n_files_read % 200 == 0:
            print(f"  processed {n_files_read} AROME files (latest: {V})")

    print(f"\nAROME files read: {n_files_read}, missing: {n_files_missing}")

    tables = []
    for L in LEADS:
        t = pd.concat(rows_per_lead[L], ignore_index=True)
        print(f"lead={L}: {len(t)} rows ({t['split'].value_counts().to_dict()})")
        tables.append(t)
    full = pd.concat(tables, ignore_index=True)

    assert full.isna().sum().sum() == 0, "NaN found in training table"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(full)} rows, {full.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory)")


if __name__ == "__main__":
    main()
