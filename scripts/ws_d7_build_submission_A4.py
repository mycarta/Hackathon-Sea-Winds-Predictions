#!/usr/bin/env python3
"""TASK 4 — build the A4 candidate Task-1 submission (NO submission to board).

Config (from the WS d7 dispatch, 2026-07-16): **A4** = static per-cell features
(depth, dist_coast) + residual-both target, with the **Task-3 alpha = 0.912**
(the minimal increase that restores A4 holdout d7 coverage to the baseline's
0.860 operating point). Direction pipeline and d14 climatology are UNTOUCHED.
Nothing is submitted — Matteo submits after review.

RELATION TO THE BOARD SUBMISSION (835026) — confirmed against
`scripts/ws_d7_build_submission_spdinfl19.py` (the 835026-lineage build):
  * The board build loads mos/qmos/adj VERBATIM from the notebook-1 cache
    (`part1_forecast/cache/coarse_forecasts.pkl`) and the precomputed per-window
    coarse fields. A4 changes those models, so this script REGENERATES them:
    `apply_arm('A4')` (identical monkeypatches to the holdout run) + a fresh
    `P.fit_forecast(P.train_dates('6D'))` + `P.coarse_fields(...)` per window.
  * ** FLAG (dispatch asked to confirm/flag): the dispatch's premise "holdout
    arms trained on the train split; the submission model uses everything" does
    NOT match this pipeline. The holdout arms already fit on
    `P.train_dates('6D')` (the full 2016-2020 6-day grid); the train/val split
    is used ONLY to select which dates are *scored*, not to reduce the fit set.
    So the A4 submission model == the A4 holdout-arm model (same fit, same
    seed=42). No extra "retrain on more data" step exists or is needed. The one
    intentional difference from the board's cached models is A4's
    features/target + the explicit seed (the kit cache is unseeded). Training
    DATA (train_dates('6D')), downscaler retrain, d14/direction pipelines, and
    the per-cell x season K=30 bias methodology are all unchanged. **
  * spd_infl: A4 uses its OWN auto-calibrated spd_infl (like 835026 used auto);
    the 1.9 override was a separate, unrelated experiment and is NOT applied.
  * Bias table: the board table (`ws_d7_bias_shrunk_table.parquet`) is the
    BASELINE model's; A4 needs its own. Reconstructed here from the A4 holdout
    checkpoint (`ws_d7_featexp_A4_stacked.npz`) with the identical K=30 shrink.
    (q50/center is spd_infl-independent, so the bias table is valid regardless
    of A4's k — same argument as the board build's docstring lines 36-39.)

Steps B (d14 climatology) and C (direction residual) are copied VERBATIM from
`ws_d7_build_submission_spdinfl19.py:153-256` so the direction columns are
produced by the identical frozen pipeline; a byte-identical check vs the on-disk
835026-lineage submission is Step F (any diff => STOP, do not write).

Output: `scripts/artifacts/submission_A4_alpha0912.csv` (+ .zip). Never
overwrites the board `submission.csv`.
"""
from __future__ import annotations

import sys
import zipfile
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
BOARD_SUB = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast" / "submission.csv"
OUT_CSV = ARTIFACTS_DIR / "submission_A4_alpha0912.csv"
OUT_ZIP = ARTIFACTS_DIR / "submission_A4_alpha0912.zip"
D14_CLIM_PATH = ARTIFACTS_DIR / "d14_climatology_season.parquet"
A4_STACKED = ARTIFACTS_DIR / "ws_d7_featexp_A4_stacked.npz"
A4_BIAS_OUT = ARTIFACTS_DIR / "ws_d7_A4_bias_shrunk_table.parquet"
MODELS_DIR = ROOT / "models"
STATIC_PATH = ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "arome_static.nc"
INFERENCE_DIR = ROOT / "phase_2" / "phase2_dataset_ship" / "inference"

KIT_PART0 = ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"
KIT_PART1 = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
sys.path.insert(0, str(KIT_PART0))
sys.path.insert(0, str(KIT_PART1))
sys.path.insert(0, str(ROOT / "scripts"))
import footprint as fp_mod          # noqa: E402
import splits                       # noqa: E402
import target_loader                # noqa: E402
import config                       # noqa: E402
import forecast_hres as fh          # noqa: E402
import forecast_pipeline as P       # noqa: E402
import downscaling as dn            # noqa: E402
import build_forecast_submission as bfs  # noqa: E402
from ws_d7_feature_experiments import apply_arm  # A4 monkeypatches (identical to arm)

N_CELLS = 43715
HOURS = (0, 6, 12, 18)
CHOSEN_ALPHA = 0.912                 # Task-3 minimal coverage retune
K_SHRINK = 30.0
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_CODE = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
SEASONS = ("DJF", "MAM", "JJA", "SON")


def round3_wrap360(values):
    rounded = np.round(values, 3)
    return np.where(rounded >= 360.0, 0.0, rounded).astype(np.float32)


def round3(values):
    return np.round(values, 3).astype(np.float32)


def a4_bias_table():
    """Reconstruct A4's per-cell x season shrunk bias (K=30) from the A4 holdout
    stacked arrays — identical formula to ws_d7_feature_experiments.py."""
    z = np.load(A4_STACKED)
    ys, xs = np.where(fp_mod.footprint_mask())
    rows, lookup = [], {}
    for si, s in enumerate(SEASONS):
        if f"{s}_q50" not in z.files:
            continue
        q50 = z[f"{s}_q50"]; tr = z[f"{s}_truth"]
        bias = q50 - tr
        raw = bias.mean(axis=0)
        smean = float(bias.mean())
        nb = bias.shape[0]
        w = nb / (nb + K_SHRINK)
        shrunk = w * raw + (1.0 - w) * smean
        lookup[s] = shrunk.astype("float64")
        rows.append(pd.DataFrame({"cell_idx": np.arange(N_CELLS), "y": ys, "x": xs,
                                  "season": s, "bias_shrunk": shrunk,
                                  "n_samples": nb, "w": w}))
    pd.concat(rows, ignore_index=True).to_parquet(A4_BIAS_OUT, index=False)
    print(f"  reconstructed A4 bias table -> {A4_BIAS_OUT}")
    return lookup


def main():
    # ============ Step A: A4 model regen + per-window coarse fields ============
    print("Step A: A4 regen (apply_arm A4 + fit_forecast + coarse_fields per window)...")
    prov = apply_arm("A4")
    print(f"  A4 features={prov['features_used']} target={prov['target']} seed=42")
    mos, qmos, adj, offs = P.fit_forecast(P.train_dates("6D"))
    d2020 = [d for d in target_loader.list_dates(config.target_root()) if d.year == 2020][::5]
    dwn = dn.train_downscaler(d2020, hours=HOURS)
    spd_infl, dir_off = P.calibrate_intervals(mos, qmos, adj, dwn, offs)
    print(f"  A4 auto spd_infl={spd_infl} (no override)")

    windows = splits.eval_windows()
    assert len(windows) == 8
    blocks = []
    for wi, w in enumerate(windows):
        fields = P.coarse_fields(mos, qmos, adj, pd.Timestamp(w["context_end"]))
        blocks += P.downscale_window(dwn, fields, offs, wi, spd_infl=spd_infl, dir_off=offs)
    sub = bfs.assemble(blocks)
    assert len(sub) == 4_196_640, len(sub)
    grp = sub.groupby(["window", "horizon", "hour"], sort=False).size()
    assert len(grp) == 96 and (grp == N_CELLS).all()
    keys = list(zip(sub["window"], sub["horizon"], sub["hour"]))
    boundaries = [i for i in range(len(keys)) if i == 0 or keys[i] != keys[i - 1]]
    block_starts = {keys[i]: i for i in boundaries}
    print(f"  built {len(sub)} rows, 96 blocks")

    # ============ Step B: d14 climatological replacement (VERBATIM) ============
    print("Step B: d14 climatological replacement...")
    clim = pd.read_parquet(D14_CLIM_PATH)
    clim_by_bin = {(season, hour): g.sort_values("point_id").reset_index(drop=True)
                   for (season, hour), g in clim.groupby(["season", "hour"])}
    win_season_d14 = {i: SEASON_OF_MONTH[date.fromisoformat(w["score_days"]["d14"]).month]
                      for i, w in enumerate(windows)}
    value_cols = ["q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]
    clim_cols = ["speed_q05", "speed_q50", "speed_q95", "dir_05", "dir_50", "dir_95"]
    dir_dst_cols = {"dir_05", "dir_50", "dir_95"}
    for window in range(8):
        season = win_season_d14[window]
        for hour in HOURS:
            start = block_starts[(window, 14, hour)]; end = start + N_CELLS
            bin_table = clim_by_bin[(season, hour)]
            assert len(bin_table) == N_CELLS
            for dst, src in zip(value_cols, clim_cols):
                vals = bin_table[src].values
                sub.loc[start:end - 1, dst] = (round3_wrap360(vals) if dst in dir_dst_cols
                                               else round3(vals))
    print("  d14 replaced")

    # ============ Step C: direction residual (VERBATIM) ============
    print("Step C: direction residual (d1 + d7)...")
    static = xr.open_dataset(STATIC_PATH)
    lat2d = static["latitude"].values.astype("float64")
    lon2d = static["longitude"].values.astype("float64")
    ys, xs = np.where(fp_mod.footprint_mask())
    lat = lat2d[ys, xs]; lon = lon2d[ys, xs]
    static.close()
    lat1d, lon1d = fh._reanalysis_grid()
    lat1d = lat1d.astype("float64"); lon1d = lon1d.astype("float64")
    coarse_i = np.abs(lat[:, None] - lat1d[None, :]).argmin(axis=1)
    coarse_j = np.abs(lon[:, None] - lon1d[None, :]).argmin(axis=1)
    coarse_lat = lat1d[coarse_i]; coarse_lon = lon1d[coarse_j]
    cell_key = [f"{float(la):.3f}_{float(lo):.3f}" for la, lo in zip(coarse_lat, coarse_lon)]
    dr_models = {}
    for L in (1, 7):
        for q in (5, 50, 95):
            dr_models[(L, q)] = lgb.Booster(model_file=str(MODELS_DIR / f"dir_residual_d{L}_q{q:02d}.lgb"))
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
            season_code = SEASON_CODE[SEASON_OF_MONTH[month]]
            month_sin = np.sin(2 * np.pi * month / 12.0); month_cos = np.cos(2 * np.pi * month / 12.0)
            for H in HOURS:
                hres_speed = hres_indexed.loc[cell_key, f"fcst_speed_d{L}_h{H}"].to_numpy()
                hres_dir = hres_indexed.loc[cell_key, f"fcst_dir_d{L}_h{H}"].to_numpy()
                hour_sin = np.sin(2 * np.pi * H / 24.0); hour_cos = np.cos(2 * np.pi * H / 24.0)
                X = pd.DataFrame({
                    "hres_dir_sin": np.sin(np.radians(hres_dir)),
                    "hres_dir_cos": np.cos(np.radians(hres_dir)),
                    "hres_speed": hres_speed,
                    "hour_sin": np.full(N_CELLS, hour_sin), "hour_cos": np.full(N_CELLS, hour_cos),
                    "season_code": np.full(N_CELLS, season_code),
                    "month_sin": np.full(N_CELLS, month_sin), "month_cos": np.full(N_CELLS, month_cos),
                    "lat": lat, "lon": lon,
                })
                r05 = dr_models[(L, 5)].predict(X); r50 = dr_models[(L, 50)].predict(X)
                r95 = dr_models[(L, 95)].predict(X)
                r_stack = np.sort(np.vstack([r05, r50, r95]), axis=0)
                r05, r50, r95 = r_stack[0], r_stack[1], r_stack[2]
                start = block_starts[(w_idx, L, H)]; end = start + N_CELLS
                sub.loc[start:end - 1, "dir_05"] = round3_wrap360((hres_dir + r05) % 360)
                sub.loc[start:end - 1, "dir_50"] = round3_wrap360((hres_dir + r50) % 360)
                sub.loc[start:end - 1, "dir_95"] = round3_wrap360((hres_dir + r95) % 360)
    print("  direction residual applied")

    # ============ Step D: d7 bias correction (A4 table) + alpha=0.912 ============
    print(f"Step D: d7 bias correction (A4 table) + alpha={CHOSEN_ALPHA}...")
    bias_lookup = a4_bias_table()
    win_season_d7 = {i: SEASON_OF_MONTH[(date.fromisoformat(w["context_end"]) + pd.Timedelta(days=7)).month]
                     for i, w in enumerate(windows)}
    for w_idx in range(8):
        bias_shrunk = bias_lookup[win_season_d7[w_idx]]
        for H in HOURS:
            start = block_starts[(w_idx, 7, H)]; end = start + N_CELLS
            q05 = sub.loc[start:end - 1, "q05"].to_numpy(dtype="float64")
            q50 = sub.loc[start:end - 1, "q50"].to_numpy(dtype="float64")
            q95 = sub.loc[start:end - 1, "q95"].to_numpy(dtype="float64")
            q05_new = np.maximum(0.0, q05 - bias_shrunk)
            q50_new = q50 - bias_shrunk
            q95_new = q95 - bias_shrunk
            q95_final = q50_new + CHOSEN_ALPHA * (q95_new - q50_new)
            q05_final = np.maximum(0.0, q50_new - CHOSEN_ALPHA * (q50_new - q05_new))
            sub.loc[start:end - 1, "q05"] = round3(q05_final)
            sub.loc[start:end - 1, "q50"] = round3(q50_new)
            sub.loc[start:end - 1, "q95"] = round3(q95_final)
    print("  bias + alpha applied")

    # ============ Step E: format validation ============
    print("Step E: format validation...")
    assert len(sub) == 4_196_640
    assert sub[["q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]].notna().all().all(), "NaN present"
    assert (sub["q05"] >= 0).all(), "negative q05"
    assert (sub["q05"] <= sub["q50"]).all() and (sub["q50"] <= sub["q95"]).all(), "quantile order"
    for c in ("dir_05", "dir_50", "dir_95"):
        assert (sub[c] >= 0).all() and (sub[c] < 360).all(), f"{c} out of range"
    exp_cols = ["type", "window", "region", "latitude", "longitude", "horizon",
                "hour", "level", "q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]
    assert list(sub.columns) == exp_cols, f"schema mismatch: {list(sub.columns)}"
    print("  PASS: 4,196,640 rows; schema ok; 0 NaN; q05<=q50<=q95>=0; dir in [0,360)")

    # ============ Step F: direction check vs board (835026-lineage) ============
    # The submission is a CSV rounded to 3 decimals (write_submission: .round(3)
    # -> to_csv). "byte-identical" is therefore a claim about the WRITTEN CSV
    # text, NOT about in-memory float32-vs-float64 bit patterns. We check both:
    #  (1) numeric gate: max CIRCULAR abs diff over all direction cells; on the
    #      shared 3-decimal grid, equal values differ by <~1e-4 (float repr), a
    #      genuine 3rd-decimal difference is >=1e-3 -> threshold 5e-4 separates.
    #  (2) confirmation: after writing, compare the two CSVs' direction columns
    #      as raw strings (the true byte-level test).
    print("Step F: direction check vs board submission...")
    if not BOARD_SUB.exists():
        print(f"  ** BLOCKER: board submission not found at {BOARD_SUB} — cannot "
              f"verify direction. STOP (not writing A4 file). **")
        sys.exit(2)
    board = pd.read_csv(BOARD_SUB)
    assert len(board) == len(sub), f"row count differs: board {len(board)} vs A4 {len(sub)}"
    for kc in ("window", "horizon", "hour"):
        if not (board[kc].to_numpy() == sub[kc].to_numpy()).all():
            print(f"  ** BLOCKER: key column '{kc}' not aligned board vs A4 — STOP. **")
            sys.exit(2)
    if not (np.isclose(board["latitude"].to_numpy(), sub["latitude"].to_numpy()).all()
            and np.isclose(board["longitude"].to_numpy(), sub["longitude"].to_numpy()).all()):
        print("  ** BLOCKER: lat/lon not aligned board vs A4 — STOP. **")
        sys.exit(2)
    maxdiff = {}
    for c in ("dir_05", "dir_50", "dir_95"):
        a = board[c].to_numpy(dtype="float64")
        b = sub[c].to_numpy(dtype="float64")
        circ = np.abs((a - b + 180.0) % 360.0 - 180.0)      # circular abs diff (deg)
        maxdiff[c] = float(circ.max())
    print(f"  max circular abs diff (deg) vs board: {maxdiff}")
    if any(v >= 5e-4 for v in maxdiff.values()):
        for c in ("dir_05", "dir_50", "dir_95"):
            a = board[c].to_numpy(dtype="float64"); b = sub[c].to_numpy(dtype="float64")
            circ = np.abs((a - b + 180.0) % 360.0 - 180.0)
            idx = np.where(circ >= 5e-4)[0][:5]
            if idx.size:
                print(f"    {c} genuine diffs at rows {idx.tolist()}: "
                      f"board={a[idx].tolist()} A4={b[idx].tolist()}")
        print("  ** DIRECTION GENUINELY DIFFERS from board — STOP, not writing A4 file. **")
        sys.exit(3)
    print("  numeric gate PASS: direction identical to board within 5e-4 deg "
          "(i.e. identical at the 3-decimal CSV write precision).")

    # ============ Step G: write A4 submission (NOT submitted) ============
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bfs.write_submission(sub, str(OUT_CSV))
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_CSV, "submission.csv")
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_ZIP}")

    # ============ Step H: written-CSV byte-level direction confirmation =========
    print("Step H: written-CSV direction byte-level confirmation...")
    dir_cols = ["dir_05", "dir_50", "dir_95"]
    a4_str = pd.read_csv(OUT_CSV, usecols=dir_cols, dtype=str, keep_default_na=False)
    bd_str = pd.read_csv(BOARD_SUB, usecols=dir_cols, dtype=str, keep_default_na=False)
    strdiff = {c: int((a4_str[c].to_numpy() != bd_str[c].to_numpy()).sum()) for c in dir_cols}
    print(f"  written-CSV direction string diffs vs board: {strdiff}")
    if any(v != 0 for v in strdiff.values()):
        print("  ** WRITTEN direction text DIFFERS from board — STOP-AND-REPORT. **")
        sys.exit(4)
    print("  PASS: dir_05/dir_50/dir_95 columns BYTE-IDENTICAL to board CSV "
          "(835026-lineage: on-disk file's direction == 835026 by the frozen Step C).")
    d1 = sub[sub.horizon == 1]; d7 = sub[sub.horizon == 7]; d14 = sub[sub.horizon == 14]
    for nm, d in (("d1", d1), ("d7", d7), ("d14", d14)):
        print(f"  {nm} q50 median={d['q50'].median():.3f} width median="
              f"{(d['q95']-d['q05']).median():.3f}")
    print("\nNOT submitted — Matteo submits after review. spd_infl(A4)=%s alpha=%.3f"
          % (spd_infl, CHOSEN_ALPHA))


if __name__ == "__main__":
    main()
