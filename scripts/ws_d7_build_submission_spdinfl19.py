#!/usr/bin/env python3
"""
Build a full Task 1 submission with d7's spd_infl recalibrated to 1.9 -
a single, isolated change relative to the current best (submission
835026's lineage: kit base -> d14 climatological fix -> direction
residual fix -> d7 bias correction -> d7 alpha=0.90). Everything else
held fixed, per Matteo's explicit instruction.

Value of 1.9 comes from scripts/ws_d7_spdinfl_recal_and_asym_conformal.py's
Fix #1 sweep (holdout Winkler-optimal was k=1.7; Matteo chose the more
conservative 1.9, mirroring the earlier alpha decision precedent).

Fidelity to "everything else unchanged":
  - mos/qmos/adj/offs (coarse MOS + quantile-MOS + conformal_adjust +
    direction offsets) and the 8 windows' precomputed coarse fields are
    loaded VERBATIM from `part1_forecast/cache/coarse_forecasts.pkl` -
    the exact same cache underlying the currently-shipped submission
    (written by notebook 1, confirmed via an Explore-agent audit before
    writing this script). Not refit.
  - The downscaler `dwn` is retrained fresh (`dn.train_downscaler`) -
    this is a PRE-EXISTING, unavoidable characteristic of the kit as
    shipped (the notebooks never cache `dwn`; confirmed via source read,
    not something this script introduces). LightGBM's internal
    subsampling has no explicit random_state anywhere in the kit, so
    minor run-to-run jitter in the downscaled center is inherent, not
    new. In practice `spd_infl[1]`/`spd_infl[14]` auto-derived by
    `calibrate_intervals()` have been empirically stable at 1.067/12.745
    across >=5 independent refits this session.
  - d1's spd_infl and d14's climatology are untouched (only `spd_infl[7]`
    is overridden, between `calibrate_intervals()` and `downscale_window()`
    - exactly where the kit's own notebook 2 would apply it).
  - d14 climatological replacement, direction residual model, and d7 bias
    correction are re-applied using the SAME artifacts as the current
    submission (`d14_climatology_season.parquet`, `dir_residual_d{1,7}_
    q{05,50,95}.lgb`, `ws_d7_bias_shrunk_table.parquet`) - none of these
    depend on spd_infl (q50/center is independent of spd_infl's k-scaling
    by construction - only q05/q95 are touched by k - so the bias table,
    which corrects q50 and shifts q05/q95 by the same amount, remains
    exactly valid for a different k).
  - d7 alpha stays at 0.90 (matching submission 835026, unchanged, not
    re-optimized for the new k - this is deliberately an isolated,
    single-variable test, not a joint re-optimization).

Checkpoints the pre-rebuild submission.csv (current alpha=1.0 experimental
state) before overwriting (build discipline #5).

Diagnostic note: this DOES modify submission.csv/zip (this is the actual
build step, not a diagnostic) - but does NOT submit to Codabench. Matteo
submits and reports back the submission ID + per-dimension Winkler scores
once scored.
"""

import pickle
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
CSV_PATH = SUB_DIR / "submission.csv"
ZIP_PATH = SUB_DIR / "submission.zip"
CACHE_PATH = SUB_DIR / "cache" / "coarse_forecasts.pkl"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
D14_CLIM_PATH = ARTIFACTS_DIR / "d14_climatology_season.parquet"
BIAS_TABLE_PATH = ARTIFACTS_DIR / "ws_d7_bias_shrunk_table.parquet"
MODELS_DIR = ROOT / "models"
BACKUP_PATH = ARTIFACTS_DIR / "submission_pre_spdinfl19_rebuild_backup.csv"
STATIC_PATH = ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "arome_static.nc"
INFERENCE_DIR = ROOT / "phase_2" / "phase2_dataset_ship" / "inference"

KIT_PART0 = ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"
KIT_PART1 = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
sys.path.insert(0, str(KIT_PART0))
sys.path.insert(0, str(KIT_PART1))
import footprint as fp_mod  # noqa: E402
import splits  # noqa: E402
import target_loader  # noqa: E402
import config  # noqa: E402
import forecast_hres as fh  # noqa: E402
import forecast_pipeline as P  # noqa: E402
import downscaling as dn  # noqa: E402
import build_forecast_submission as bfs  # noqa: E402

N_CELLS = 43715
HOURS = (0, 6, 12, 18)
NEW_SPD_INFL_D7 = 1.9
CHOSEN_ALPHA = 0.90
SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASON_CODE = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


def round3_wrap360(values):
    """sub's dir_* columns are float32 (from field_to_rows's explicit cast,
    since sub is built fresh in-memory here rather than CSV-round-tripped
    like the other apply scripts, which default to float64) - cast to
    float32 explicitly to avoid pandas' LossySetitemError on assignment."""
    rounded = np.round(values, 3)
    return np.where(rounded >= 360.0, 0.0, rounded).astype(np.float32)


def round3(values):
    """Same float32 rationale as round3_wrap360, for the q05/q50/q95 columns."""
    return np.round(values, 3).astype(np.float32)


def main():
    # ================= Step A: kit base, spd_infl[7] overridden =================
    print("Step A: building kit base with spd_infl[7] overridden...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    offs = cache["offs"]
    mos, qmos, adj = cache["models"]
    windows = splits.eval_windows()
    assert len(windows) == 8

    d2020 = [d for d in target_loader.list_dates(config.target_root()) if d.year == 2020][::5]
    dwn = dn.train_downscaler(d2020, hours=HOURS)
    print(f"  downscaler trained on {len(d2020)} days (freshly retrained - kit never caches this)")

    spd_infl, dir_off = P.calibrate_intervals(mos, qmos, adj, dwn, offs)
    print(f"  auto-derived spd_infl: {spd_infl}")
    old_d7_k = spd_infl[7]
    spd_infl[7] = NEW_SPD_INFL_D7
    print(f"  OVERRIDE: spd_infl[7] {old_d7_k} -> {NEW_SPD_INFL_D7} (only change from auto)")

    blocks = []
    for wi in range(len(windows)):
        blocks += P.downscale_window(dwn, cache[wi], offs, wi, spd_infl=spd_infl, dir_off=offs)
    sub = bfs.assemble(blocks)
    print(f"  built {len(sub)} rows")
    assert len(sub) == 4_196_640

    grp_sizes = sub.groupby(["window", "horizon", "hour"], sort=False).size()
    assert len(grp_sizes) == 96 and (grp_sizes == N_CELLS).all()
    keys = list(zip(sub["window"], sub["horizon"], sub["hour"]))
    boundaries = [i for i in range(len(keys)) if i == 0 or keys[i] != keys[i - 1]]
    assert len(boundaries) == 96
    block_starts = {keys[i]: i for i in boundaries}
    print("  verified 96 contiguous 43,715-row blocks")

    # ================= Step B: d14 climatological replacement =================
    print("Step B: applying d14 climatological replacement...")
    clim = pd.read_parquet(D14_CLIM_PATH)
    clim_by_bin = {(season, hour): g.sort_values("point_id").reset_index(drop=True)
                   for (season, hour), g in clim.groupby(["season", "hour"])}

    win_season_d14 = {}
    for i, w in enumerate(windows):
        d14_date = date.fromisoformat(w["score_days"]["d14"])
        win_season_d14[i] = SEASON_OF_MONTH[d14_date.month]

    value_cols = ["q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]
    clim_cols = ["speed_q05", "speed_q50", "speed_q95", "dir_05", "dir_50", "dir_95"]
    dir_dst_cols = {"dir_05", "dir_50", "dir_95"}
    for window in range(8):
        season = win_season_d14[window]
        for hour in HOURS:
            key = (window, 14, hour)
            start = block_starts[key]
            end = start + N_CELLS
            bin_table = clim_by_bin[(season, hour)]
            assert len(bin_table) == N_CELLS
            for dst, src in zip(value_cols, clim_cols):
                vals = bin_table[src].values
                sub.loc[start:end - 1, dst] = (
                    round3_wrap360(vals) if dst in dir_dst_cols else round3(vals))
    print(f"  d14 replaced ({8 * len(HOURS)} blocks)")

    # ================= Step C: direction residual model (d1 + d7) =================
    print("Step C: applying direction residual model...")
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
    cell_key = [f"{float(la):.3f}_{float(lo):.3f}" for la, lo in zip(coarse_lat, coarse_lon)]

    dr_models = {}
    for L in (1, 7):
        for q in (5, 50, 95):
            p = MODELS_DIR / f"dir_residual_d{L}_q{q:02d}.lgb"
            dr_models[(L, q)] = lgb.Booster(model_file=str(p))

    for w_idx, w in enumerate(windows):
        issue_date = date.fromisoformat(w["context_end"])
        window_path = INFERENCE_DIR / f"window_{w['id']}" / "context_hres_north_sea.parquet"
        hres_window_df = pd.read_parquet(window_path)
        key = (hres_window_df["latitude"].astype("float64").round(3).map(lambda x: f"{x:.3f}") + "_" +
               hres_window_df["longitude"].astype("float64").round(3).map(lambda x: f"{x:.3f}"))
        hres_indexed = hres_window_df.assign(_key=key).set_index("_key")

        for L in (1, 7):
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
                    "lat": lat, "lon": lon,
                })
                r05 = dr_models[(L, 5)].predict(X)
                r50 = dr_models[(L, 50)].predict(X)
                r95 = dr_models[(L, 95)].predict(X)
                r_stack = np.sort(np.vstack([r05, r50, r95]), axis=0)
                r05, r50, r95 = r_stack[0], r_stack[1], r_stack[2]
                dir05 = (hres_dir + r05) % 360
                dir50 = (hres_dir + r50) % 360
                dir95 = (hres_dir + r95) % 360

                key2 = (w_idx, L, H)
                start = block_starts[key2]
                end = start + N_CELLS
                sub.loc[start:end - 1, "dir_05"] = round3_wrap360(dir05)
                sub.loc[start:end - 1, "dir_50"] = round3_wrap360(dir50)
                sub.loc[start:end - 1, "dir_95"] = round3_wrap360(dir95)
    print(f"  direction residual applied ({8 * 2 * len(HOURS)} blocks)")

    # ================= Step D: d7 bias correction + alpha=0.90 =================
    print("Step D: applying d7 bias correction + alpha=0.90...")
    bias_table = pd.read_parquet(BIAS_TABLE_PATH)
    bias_lookup = {}
    for season, grp in bias_table.groupby("season"):
        grp = grp.sort_values("cell_idx")
        assert (grp["cell_idx"].to_numpy() == np.arange(N_CELLS)).all()
        bias_lookup[season] = grp["bias_shrunk"].to_numpy(dtype="float64")

    win_season_d7 = {}
    for i, w in enumerate(windows):
        issue_date = date.fromisoformat(w["context_end"])
        valid_date = issue_date + pd.Timedelta(days=7)
        win_season_d7[i] = SEASON_OF_MONTH[valid_date.month]

    for w_idx in range(8):
        season = win_season_d7[w_idx]
        bias_shrunk = bias_lookup[season]
        for H in HOURS:
            key = (w_idx, 7, H)
            start = block_starts[key]
            end = start + N_CELLS
            q05 = sub.loc[start:end - 1, "q05"].to_numpy(dtype="float64")
            q50 = sub.loc[start:end - 1, "q50"].to_numpy(dtype="float64")
            q95 = sub.loc[start:end - 1, "q95"].to_numpy(dtype="float64")

            q05_new = np.maximum(0.0, q05 - bias_shrunk)
            q50_new = q50 - bias_shrunk
            q95_new = q95 - bias_shrunk

            q95_final = q50_new + CHOSEN_ALPHA * (q95_new - q50_new)
            q05_final = np.maximum(0.0, q50_new - CHOSEN_ALPHA * (q50_new - q05_new))
            q50_final = q50_new

            sub.loc[start:end - 1, "q05"] = round3(q05_final)
            sub.loc[start:end - 1, "q50"] = round3(q50_final)
            sub.loc[start:end - 1, "q95"] = round3(q95_final)
    print(f"  bias correction + alpha={CHOSEN_ALPHA} applied ({8 * len(HOURS)} blocks)")

    # ================= Step E: validate =================
    print("Step E: validating...")
    assert len(sub) == 4_196_640
    assert sub[["q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]].notna().all().all()
    assert (sub["q05"] >= 0).all(), "negative q05"
    assert (sub["q05"] <= sub["q50"]).all() and (sub["q50"] <= sub["q95"]).all(), "quantile ordering violated"
    for c in ("dir_05", "dir_50", "dir_95"):
        assert (sub[c] >= 0).all() and (sub[c] < 360).all(), f"{c} out of [0,360)"
    print("  PASS: 4,196,640 rows, 0 NaN, q05<=q50<=q95, speeds>=0, directions in [0,360)")

    d1 = sub[sub["horizon"] == 1]
    d7 = sub[sub["horizon"] == 7]
    d14 = sub[sub["horizon"] == 14]
    print(f"  d1  q50 median: {d1['q50'].median():.3f} m/s, width median: {(d1['q95']-d1['q05']).median():.3f}")
    print(f"  d7  q50 median: {d7['q50'].median():.3f} m/s, width median: {(d7['q95']-d7['q05']).median():.3f}")
    print(f"  d14 q50 median: {d14['q50'].median():.3f} m/s, width median: {(d14['q95']-d14['q05']).median():.3f}")

    # ================= Step F: checkpoint current submission =================
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if BACKUP_PATH.exists():
        print(f"Pre-rebuild checkpoint already exists at {BACKUP_PATH} - not overwriting.")
    else:
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print(f"Checkpointed pre-rebuild submission.csv (alpha=1.0 experimental state) -> {BACKUP_PATH}")

    # ================= Step G: write final =================
    bfs.write_submission(sub, str(CSV_PATH))
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(CSV_PATH, "submission.csv")
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {ZIP_PATH}")
    print("\nNOT submitted to Codabench - Matteo submits. Do not select on the "
          "leaderboard until reviewed at the WS d7 sketch gate.")


if __name__ == "__main__":
    main()
