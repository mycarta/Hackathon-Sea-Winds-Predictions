#!/usr/bin/env python3
"""
Step 3 + 4 of the d1/d7 direction-residual fix: predict residual q05/q50/q95
for all 43,715 footprint cells across the 8 (2021) eval windows x leads
(1,7) x hours (0,6,12,18), reconstruct directions, and replace dir_05/
dir_50/dir_95 for horizon==1 and horizon==7 rows in submission.csv.
horizon==14 rows and ALL speed columns are left untouched.

Uses the kit's pre-sliced per-window inference HRES files
(phase_2/phase2_dataset_ship/inference/window_{1..8}/context_hres_north_sea.parquet)
rather than filtering the full training parquet.

Row <-> footprint-cell correspondence is positional (submission.csv is 96
contiguous 43,715-row blocks matching np.where(footprint_mask()) order) -
same validated approach as the d14 climatology fix
(scripts/apply_d14_climatology.py).

Checkpoints the pre-fix submission.csv before any write (build discipline
#5) - but does NOT overwrite scripts/artifacts/submission_pre_d14fix_backup.csv
(that remains the original kit-CSV-fix checkpoint); this script's own
checkpoint is a new, separate named file.
"""

import shutil
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
STATIC_PATH = ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "arome_static.nc"
INFERENCE_DIR = ROOT / "phase_2" / "phase2_dataset_ship" / "inference"
SUB_DIR = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
CSV_PATH = SUB_DIR / "submission.csv"
ZIP_PATH = SUB_DIR / "submission.zip"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
BACKUP_PATH = ARTIFACTS_DIR / "submission_pre_dir_residual_fix_backup.csv"

KIT_PART0 = ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"
KIT_PART1 = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
sys.path.insert(0, str(KIT_PART0))
sys.path.insert(0, str(KIT_PART1))
import footprint as fp_mod  # noqa: E402
import splits  # noqa: E402
import forecast_hres as fh  # noqa: E402

# Runbook errata 2026-08-19. SEAWINDS_EVAL_YEAR is retained as a manual override
# but is REDUNDANT as of kit 9edd92b: splits.eval_windows() with no argument now
# auto-detects the year from the INSTALLED inference windows
# (splits._installed_windows_year reads window_1/metadata.json), so replacing
# inference/ in place re-dates everything. The calls below pass no argument and
# inherit that auto-detect; EVAL_YEAR is kept only for logging and assertions.
EVAL_YEAR = int(os.environ.get("SEAWINDS_EVAL_YEAR", 2021))  # runbook errata 2026-08-19,
# blocker 3: the swap re-run scores a different year than the frozen default. Unset -> 2021,
# so the 2021 rehearsal still reproduces byte-identically; SEAWINDS_EVAL_YEAR=2022 for the
# final-evaluation run. splits.eval_windows() re-dates _BASE_WINDOWS (defined in 2022) to the
# requested year, and eval_windows(2022) was verified equal to the organizers' shipped
# window_1..8 metadata on all four date fields plus score_days (2026-08-19).
LEADS = (1, 7)
HOURS = (0, 6, 12, 18)
N_CELLS = 43715
SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASON_CODE = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


def build_footprint_coarse_mapping():
    static = xr.open_dataset(STATIC_PATH)
    lat2d = static["latitude"].values.astype("float64")
    lon2d = static["longitude"].values.astype("float64")
    ys, xs = np.where(fp_mod.footprint_mask())
    lat = lat2d[ys, xs]
    lon = lon2d[ys, xs]
    static.close()

    lat1d, lon1d = fh._reanalysis_grid()
    lat1d = lat1d.astype("float64")
    lon1d = lon1d.astype("float64")
    coarse_i = np.abs(lat[:, None] - lat1d[None, :]).argmin(axis=1)
    coarse_j = np.abs(lon[:, None] - lon1d[None, :]).argmin(axis=1)
    coarse_lat = lat1d[coarse_i]
    coarse_lon = lon1d[coarse_j]
    return lat, lon, coarse_lat, coarse_lon


def load_models():
    models = {}
    for L in LEADS:
        for q in (5, 50, 95):
            p = MODELS_DIR / f"dir_residual_d{L}_q{q:02d}.lgb"
            models[(L, q)] = lgb.Booster(model_file=str(p))
    return models


def predict_directions_for_window(window_idx, issue_date, lat, lon, coarse_lat, coarse_lon,
                                    models, hres_window_df):
    """Returns dict[(lead,hour)] -> (dir05, dir50, dir95) arrays, len N_CELLS each."""
    key = (hres_window_df["latitude"].astype("float64").round(3).map(lambda x: f"{x:.3f}") + "_" +
           hres_window_df["longitude"].astype("float64").round(3).map(lambda x: f"{x:.3f}"))
    hres_indexed = hres_window_df.assign(_key=key).set_index("_key")
    cell_key = [f"{float(la):.3f}_{float(lo):.3f}" for la, lo in zip(coarse_lat, coarse_lon)]

    out = {}
    for L in LEADS:
        valid_date = pd.Timestamp(issue_date) + pd.Timedelta(days=L)
        month = valid_date.month
        season = SEASON_OF_MONTH[month]
        season_code = SEASON_CODE[season]
        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)

        for H in HOURS:
            sp_col = f"fcst_speed_d{L}_h{H}"
            di_col = f"fcst_dir_d{L}_h{H}"
            hres_speed = hres_indexed.loc[cell_key, sp_col].to_numpy()
            hres_dir = hres_indexed.loc[cell_key, di_col].to_numpy()

            hour_sin = np.sin(2 * np.pi * H / 24.0)
            hour_cos = np.cos(2 * np.pi * H / 24.0)

            X = pd.DataFrame({
                "hres_dir_sin": np.sin(np.radians(hres_dir)),
                "hres_dir_cos": np.cos(np.radians(hres_dir)),
                "hres_speed": hres_speed,
                "hour_sin": np.full(N_CELLS, hour_sin),
                "hour_cos": np.full(N_CELLS, hour_cos),
                "season_code": np.full(N_CELLS, season_code),
                "month_sin": np.full(N_CELLS, month_sin),
                "month_cos": np.full(N_CELLS, month_cos),
                "lat": lat,
                "lon": lon,
            })

            r05 = models[(L, 5)].predict(X)
            r50 = models[(L, 50)].predict(X)
            r95 = models[(L, 95)].predict(X)
            r_stack = np.sort(np.vstack([r05, r50, r95]), axis=0)  # enforce non-crossing
            r05, r50, r95 = r_stack[0], r_stack[1], r_stack[2]

            dir05 = (hres_dir + r05) % 360
            dir50 = (hres_dir + r50) % 360
            dir95 = (hres_dir + r95) % 360

            fwd = (dir95 - dir05) % 360
            assert (fwd < 360).all(), "constructed arc must span < 360 deg"

            out[(L, H)] = (dir05, dir50, dir95)
    return out


def main():
    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if BACKUP_PATH.exists():
        print(f"Pre-fix checkpoint already exists at {BACKUP_PATH} - not overwriting.")
    else:
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print(f"Checkpointed pre-fix submission.csv -> {BACKUP_PATH}")

    lat, lon, coarse_lat, coarse_lon = build_footprint_coarse_mapping()
    print(f"Footprint: {len(lat)} cells mapped to coarse grid.")

    models = load_models()
    print(f"Loaded {len(models)} models: {sorted(models.keys())}")

    windows = splits.eval_windows()

    df = pd.read_csv(CSV_PATH)
    df_before = df.copy()

    grp_sizes = df.groupby(["window", "horizon", "hour"], sort=False).size()
    assert len(grp_sizes) == 96 and (grp_sizes == N_CELLS).all()
    keys = list(zip(df["window"], df["horizon"], df["hour"]))
    boundaries = [i for i in range(len(keys)) if i == 0 or keys[i] != keys[i - 1]]
    assert len(boundaries) == 96
    block_starts = {keys[i]: i for i in boundaries}
    print("Verified 96 contiguous 43,715-row blocks.")

    for w_idx, w in enumerate(windows):
        issue_date = date.fromisoformat(w["context_end"])
        window_path = INFERENCE_DIR / f"window_{w['id']}" / "context_hres_north_sea.parquet"
        hres_window_df = pd.read_parquet(window_path)
        print(f"\nWindow {w_idx} (id={w['id']}), issue_date={issue_date}: "
              f"HRES rows={len(hres_window_df)}")

        preds = predict_directions_for_window(
            w_idx, issue_date, lat, lon, coarse_lat, coarse_lon, models, hres_window_df)

        for L in LEADS:
            for H in HOURS:
                dir05, dir50, dir95 = preds[(L, H)]
                key = (w_idx, L, H)
                assert key in block_starts, f"missing block {key}"
                start = block_starts[key]
                end = start + N_CELLS
                assert (df.loc[start:end - 1, "window"] == w_idx).all()
                assert (df.loc[start:end - 1, "horizon"] == L).all()
                assert (df.loc[start:end - 1, "hour"] == H).all()

                def wrap360(v):
                    r = np.round(v, 3)
                    return np.where(r >= 360.0, 0.0, r)

                df.loc[start:end - 1, "dir_05"] = wrap360(dir05)
                df.loc[start:end - 1, "dir_50"] = wrap360(dir50)
                df.loc[start:end - 1, "dir_95"] = wrap360(dir95)

    d1d7_mask = df["horizon"].isin([1, 7])
    other_mask = ~d1d7_mask
    unchanged = df.loc[other_mask].equals(df_before.loc[other_mask])
    print(f"\nhorizon==14 rows + all non-direction columns unchanged: {unchanged}")
    if not unchanged:
        diff_cols = [c for c in df.columns
                     if not df.loc[other_mask, c].equals(df_before.loc[other_mask, c])]
        print(f"  DIFFERING columns outside scope: {diff_cols}")
    assert unchanged, "rows/columns outside scope (d14, speed) were modified - aborting write"

    speed_cols = ["q05", "q50", "q95"]
    speed_unchanged = df[speed_cols].equals(df_before[speed_cols])
    print(f"ALL speed columns unchanged (every horizon): {speed_unchanged}")
    assert speed_unchanged, "speed columns were modified - aborting write"

    df.to_csv(CSV_PATH, index=False)
    print(f"\nWrote updated {CSV_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(CSV_PATH, "submission.csv")
    print(f"Re-zipped {ZIP_PATH}")

    def arc_half_width(sub):
        return (((sub["dir_95"] - sub["dir_05"]) % 360) / 2.0)

    print("\n--- Before/after direction half-width (mean, forward-arc metric) ---")
    for L in (1, 7, 14):
        b = df_before[df_before["horizon"] == L]
        a = df[df["horizon"] == L]
        print(f"  d{L}: before={arc_half_width(b).mean():.3f} deg  "
              f"after={arc_half_width(a).mean():.3f} deg")


if __name__ == "__main__":
    main()
