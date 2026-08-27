#!/usr/bin/env python3
"""
WS d7 speed bias correction, Step 1 (+ Step 3 alpha re-sweep) - 2026-07-08.

Builds a per-cell x per-season d7 speed bias correction table (shrunk
toward the season's global mean bias) from the same shipped-pipeline
holdout evaluation as scripts/ws_d7_diagnostic_coverage_bias.py, then
re-sweeps the interval-width alpha multiplier on the bias-corrected
predictions to find the post-correction Winkler-optimal alpha.

Per-cell x per-season (not just per-cell) because
reports/ws_d7_diagnostics_20260708.md found a seasonal spread of 0.934
m/s (> the 0.5 m/s threshold in the task prompt), which explicitly calls
for per-cell x per-season.

Shrinkage note (implementation detail not fully specified in the task
prompt, documented here for review): the shrinkage target for
bias_shrunk(cell, season) is the SEASON's global mean bias (pooled over
all cells for that season), not the single overall-global mean bias
across all seasons - this keeps the shrinkage target consistent with the
population the per-cell estimate is drawn from. The overall global mean
bias (pooled over everything) is also reported separately for reference.

Sample-count note: truth for a given (issue_date, hour) comes from a
single full-grid AROME snapshot, so every footprint cell gets exactly the
same number of holdout samples within a season (no per-cell sparsity, in
contrast to the direction-residual fix's subsampled-cell training set).
Consequently the shrinkage weight w = n/(n+k) is CONSTANT across cells
within a season (only varies season to season, by holdout-date count) -
the per-cell bias VALUES still vary spatially and are preserved; only the
shrinkage weight is season-uniform. This is reported explicitly below
rather than silently assumed.

No model files are modified or saved; the speed pipeline is refit
in-memory only, exactly reproducing scripts/ws_d7_diagnostic_coverage_bias.py's
methodology. submission.csv is never touched by this script (that's
scripts/ws_d7_apply_bias_correction.py).

Outputs:
  scripts/artifacts/ws_d7_bias_shrunk_table.parquet
    (cell_idx, y, x, season, bias_raw, bias_shrunk, n_samples, w)
  scripts/artifacts/ws_d7_bias_correction_params.json
    (global_mean_bias, k, w stats, chosen alpha, before/after Winkler)
  reports/ws_d7_bias_correction_20260708.md (new report, Step 1 + Step 3 sections)
"""

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
AROME_DIR = ROOT / "phase_2" / "phase2_dataset_ship" / "train" / "arome"
STATIC_PATH = ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "arome_static.nc"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REPORT_PATH = ROOT / "reports" / "ws_d7_bias_correction_20260708.md"
TABLE_OUT_PATH = ARTIFACTS_DIR / "ws_d7_bias_shrunk_table.parquet"
PARAMS_OUT_PATH = ARTIFACTS_DIR / "ws_d7_bias_correction_params.json"

KIT_PART0 = ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"
KIT_PART1 = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
sys.path.insert(0, str(KIT_PART0))
sys.path.insert(0, str(KIT_PART1))
import footprint as fp_mod  # noqa: E402
import splits  # noqa: E402
import target_loader  # noqa: E402
import config  # noqa: E402
import forecast_pipeline as P  # noqa: E402
import downscaling as dn  # noqa: E402

LEAD = 7
HOURS = (0, 6, 12, 18)
K_SHRINK = 30.0
ALPHA_LEVEL = 0.10
ALPHA_RESWEEP = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                  0.7, 0.75, 0.8, 0.85, 0.9)
SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
SEASONS = ("DJF", "MAM", "JJA", "SON")


def winkler_score(y, lo, hi, alpha_level=ALPHA_LEVEL):
    width = hi - lo
    below = np.maximum(0.0, lo - y) * (2.0 / alpha_level)
    above = np.maximum(0.0, y - hi) * (2.0 / alpha_level)
    return width + below + above


def truth_cache_get(cache, V, ys, xs):
    if V in cache:
        return cache[V]
    path = AROME_DIR / f"{V.year}" / f"arome_{V:%Y%m%d}.nc"
    if not path.exists():
        cache[V] = None
        return None
    ds = xr.open_dataset(path)
    times = pd.to_datetime(ds["time"].values)
    hour_mask = times.hour.isin(HOURS)
    order = np.argsort(times.hour[hour_mask].values)
    u = ds["u125m"].values[hour_mask][order][:, ys, xs]
    v = ds["v125m"].values[hour_mask][order][:, ys, xs]
    ds.close()
    speed = np.sqrt(u ** 2 + v ** 2)
    cache[V] = speed
    return speed


def main():
    t0 = time.time()

    static = xr.open_dataset(STATIC_PATH)
    ys, xs = np.where(fp_mod.footprint_mask())
    n_cells = ys.size
    assert n_cells == 43715
    static.close()

    print("Fitting shipped speed pipeline (P.fit_forecast(P.train_dates('6D')))...")
    train = P.train_dates("6D")
    mos, qmos, adj, offs = P.fit_forecast(train)

    print("Training downscaler (2020[::5], matches shipped notebook)...")
    d2020 = [d for d in target_loader.list_dates(config.target_root()) if d.year == 2020][::5]
    dwn = dn.train_downscaler(d2020, hours=HOURS)

    print("Calibrating intervals (spd_infl / dir_off)...")
    spd_infl, dir_off = P.calibrate_intervals(mos, qmos, adj, dwn, offs)
    print(f"  spd_infl: {spd_infl}")

    train_dates_seed, val_dates_seed = splits.train_val_dates(seed=42)
    val_set = set(val_dates_seed)
    print(f"train_val_dates(seed=42): {len(train_dates_seed)} train, {len(val_dates_seed)} val")

    all_dates = sorted(target_loader.list_dates(config.target_root()))
    candidate_issue_dates = [d for d in all_dates if d.year in (2016, 2017, 2018, 2019, 2020)]
    print(f"Candidate issue dates: {len(candidate_issue_dates)}")

    # per-season lists of (q05, q50, q95, truth) blocks, each block length n_cells
    season_blocks = {s: {"q05": [], "q50": [], "q95": [], "truth": []} for s in SEASONS}
    global_bias_sum = 0.0
    global_bias_count = 0

    truth_cache = {}
    n_processed = 0

    for D in candidate_issue_dates:
        V = D + timedelta(days=LEAD)
        if V not in val_set:
            continue

        try:
            fields = P.coarse_fields(mos, qmos, adj, pd.Timestamp(D))
        except Exception as e:
            print(f"  skip {D}: coarse_fields failed ({e})")
            continue

        blocks = P.downscale_window(dwn, fields, offs, window=0,
                                     spd_infl=spd_infl, dir_off=dir_off)
        # blocks order: LEADS x HOURS, LEADS=(1,7,14) -> lead=7 starts at index 4
        lead_idx = P.LEADS.index(LEAD)
        base = lead_idx * len(HOURS)

        speed_truth_all_hours = truth_cache_get(truth_cache, V, ys, xs)
        if speed_truth_all_hours is None:
            continue

        season = SEASON_OF_MONTH[V.month]
        contributed = False
        for h_idx, H in enumerate(HOURS):
            blk = blocks[base + h_idx]
            assert int(blk["horizon"].iloc[0]) == LEAD and int(blk["hour"].iloc[0]) == H
            q05 = blk["q05"].to_numpy(dtype="float32")
            q50 = blk["q50"].to_numpy(dtype="float32")
            q95 = blk["q95"].to_numpy(dtype="float32")
            truth = speed_truth_all_hours[h_idx].astype("float32")

            season_blocks[season]["q05"].append(q05)
            season_blocks[season]["q50"].append(q50)
            season_blocks[season]["q95"].append(q95)
            season_blocks[season]["truth"].append(truth)

            global_bias_sum += float((q50 - truth).sum())
            global_bias_count += n_cells
            contributed = True

        if contributed:
            n_processed += 1
        if n_processed % 200 == 0 and contributed:
            print(f"  processed {n_processed} d7 issue dates (latest D={D}), "
                  f"elapsed={time.time() - t0:.0f}s")

    print(f"\nTotal d7 issue dates processed: {n_processed}")
    global_mean_bias = global_bias_sum / global_bias_count
    print(f"Overall global mean d7 bias: {global_mean_bias:.4f} m/s "
          f"(n={global_bias_count})")

    # ---- stack per season, compute bias table + shrinkage ----
    per_cell_season_bias_raw = np.full((n_cells, len(SEASONS)), np.nan, dtype="float64")
    per_cell_season_bias_shrunk = np.full((n_cells, len(SEASONS)), np.nan, dtype="float64")
    n_blocks_per_season = {}
    w_per_season = {}
    season_global_mean = {}

    stacked = {}  # season -> dict of 2D arrays (n_blocks, n_cells), kept for the alpha resweep
    for si, s in enumerate(SEASONS):
        sb = season_blocks[s]
        n_blocks = len(sb["q50"])
        n_blocks_per_season[s] = n_blocks
        if n_blocks == 0:
            print(f"  WARNING: season {s} has 0 holdout blocks - skipping")
            continue
        q05_arr = np.stack(sb["q05"])  # (n_blocks, n_cells)
        q50_arr = np.stack(sb["q50"])
        q95_arr = np.stack(sb["q95"])
        truth_arr = np.stack(sb["truth"])
        stacked[s] = {"q05": q05_arr, "q50": q50_arr, "q95": q95_arr, "truth": truth_arr}

        bias_arr = q50_arr - truth_arr  # (n_blocks, n_cells)
        bias_raw_cell = bias_arr.mean(axis=0)  # (n_cells,)
        season_mean = float(bias_arr.mean())
        season_global_mean[s] = season_mean

        w = n_blocks / (n_blocks + K_SHRINK)
        w_per_season[s] = w
        bias_shrunk_cell = w * bias_raw_cell + (1 - w) * season_mean

        per_cell_season_bias_raw[:, si] = bias_raw_cell
        per_cell_season_bias_shrunk[:, si] = bias_shrunk_cell

        print(f"  {s}: n_blocks={n_blocks}, w={w:.4f}, "
              f"season_mean_bias={season_mean:.4f}, "
              f"per-cell bias_raw range=[{bias_raw_cell.min():.3f}, {bias_raw_cell.max():.3f}]")

    # checkpoint the raw stacked arrays to disk (build discipline #5 - before the
    # alpha resweep, which is cheap to redo/extend from this checkpoint without
    # re-running the ~30 min forecast+downscale pipeline again)
    stacked_npz = {}
    for s in SEASONS:
        if s not in stacked:
            continue
        for k, v in stacked[s].items():
            stacked_npz[f"{s}_{k}"] = v
    STACKED_OUT_PATH = ARTIFACTS_DIR / "ws_d7_bias_correction_stacked.npz"
    np.savez_compressed(STACKED_OUT_PATH, **stacked_npz)
    print(f"Checkpointed raw stacked holdout arrays -> {STACKED_OUT_PATH} "
          f"(enables re-sweeping alpha without re-running the pipeline)")

    w_values = list(w_per_season.values())
    print(f"\nShrinkage w distribution (season-level, uniform within season - see docstring): "
          f"min={min(w_values):.4f}, median={float(np.median(w_values)):.4f}, "
          f"max={max(w_values):.4f}")

    # ---- alpha resweep on bias-corrected holdout predictions ----
    print("\nRe-sweeping alpha on bias-corrected predictions...")
    resweep_rows = []
    before_rows = []
    for a in ALPHA_RESWEEP:
        n_tot, cov_tot, width_sum, wink_sum = 0, 0, 0.0, 0.0
        for si, s in enumerate(SEASONS):
            if s not in stacked:
                continue
            bias_shrunk_cell = per_cell_season_bias_shrunk[:, si]  # (n_cells,)
            q05 = stacked[s]["q05"] - bias_shrunk_cell[None, :]
            q50 = stacked[s]["q50"] - bias_shrunk_cell[None, :]
            q95 = stacked[s]["q95"] - bias_shrunk_cell[None, :]
            q05 = np.maximum(0.0, q05)
            truth = stacked[s]["truth"]

            lo = q50 - a * (q50 - q05)
            hi = q50 + a * (q95 - q50)
            lo = np.maximum(0.0, lo)
            covered = (truth >= lo) & (truth <= hi)

            n_tot += truth.size
            cov_tot += int(covered.sum())
            width_sum += float((hi - lo).sum())
            wink_sum += float(winkler_score(truth, lo, hi).sum())

        resweep_rows.append({
            "alpha": a, "coverage": cov_tot / n_tot,
            "mean_width": width_sum / n_tot, "mean_winkler": wink_sum / n_tot,
        })

    # before (pre-bias-correction, alpha=1.0) baseline, same holdout samples, for comparison
    n_tot, cov_tot, width_sum, wink_sum = 0, 0, 0.0, 0.0
    for s in SEASONS:
        if s not in stacked:
            continue
        q05, q50, q95, truth = (stacked[s]["q05"], stacked[s]["q50"],
                                 stacked[s]["q95"], stacked[s]["truth"])
        covered = (truth >= q05) & (truth <= q95)
        n_tot += truth.size
        cov_tot += int(covered.sum())
        width_sum += float((q95 - q05).sum())
        wink_sum += float(winkler_score(truth, q05, q95).sum())
    before_summary = {"coverage": cov_tot / n_tot, "mean_width": width_sum / n_tot,
                       "mean_winkler": wink_sum / n_tot}

    best = min(resweep_rows, key=lambda r: r["mean_winkler"])
    print(f"\nBEFORE (no bias correction, alpha=1.0): coverage={before_summary['coverage']*100:.1f}%, "
          f"width={before_summary['mean_width']:.3f}, winkler={before_summary['mean_winkler']:.3f}")
    for r in resweep_rows:
        print(f"  AFTER bias-correction, alpha={r['alpha']}: coverage={r['coverage']*100:.1f}%, "
              f"width={r['mean_width']:.3f}, winkler={r['mean_winkler']:.3f}")
    print(f"Chosen alpha (min Winkler post-correction): {best['alpha']}")

    # ---- write bias table parquet ----
    rows = []
    for si, s in enumerate(SEASONS):
        if s not in stacked:
            continue
        rows.append(pd.DataFrame({
            "cell_idx": np.arange(n_cells),
            "y": ys, "x": xs,
            "season": s,
            "bias_raw": per_cell_season_bias_raw[:, si],
            "bias_shrunk": per_cell_season_bias_shrunk[:, si],
            "n_samples": n_blocks_per_season[s],
            "w": w_per_season[s],
        }))
    table_df = pd.concat(rows, ignore_index=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table_df.to_parquet(TABLE_OUT_PATH, index=False)
    print(f"\nWrote {TABLE_OUT_PATH} ({len(table_df)} rows)")

    params = {
        "global_mean_bias": global_mean_bias,
        "k_shrink": K_SHRINK,
        "w_min": min(w_values), "w_median": float(np.median(w_values)), "w_max": max(w_values),
        "w_per_season": w_per_season,
        "season_global_mean_bias": season_global_mean,
        "n_blocks_per_season": n_blocks_per_season,
        "alpha_resweep": resweep_rows,
        "before_correction": before_summary,
        "chosen_alpha": best["alpha"],
        "chosen_alpha_winkler": best["mean_winkler"],
    }
    with open(PARAMS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Wrote {PARAMS_OUT_PATH}")

    # ---- report ----
    lines = [
        "# WS d7 speed bias correction — 2026-07-08\n",
        "New submission fix (not yet applied here - see "
        "`scripts/ws_d7_apply_bias_correction.py` for the submission.csv write). "
        "Driven by `reports/ws_d7_diagnostics_20260708.md` (Diagnostics 1 and 3).\n",
        "## Step 1: per-cell x per-season bias table\n",
        "Per-cell x per-season (not just per-cell) because the diagnostics report found "
        "a seasonal spread of 0.934 m/s (> the 0.5 m/s threshold that calls for "
        "per-season granularity).\n",
        f"Overall global mean d7 bias (pooled all seasons/cells): "
        f"**{global_mean_bias:.4f} m/s** (n={global_bias_count}).\n",
        "| Season | n holdout blocks | shrinkage w | season global mean bias (m/s) | "
        "per-cell bias_raw range (m/s) |",
        "|---|---|---|---|---|",
    ]
    for si, s in enumerate(SEASONS):
        if s not in stacked:
            continue
        b = per_cell_season_bias_raw[:, si]
        lines.append(f"| {s} | {n_blocks_per_season[s]} | {w_per_season[s]:.4f} | "
                      f"{season_global_mean[s]:.4f} | [{b.min():.3f}, {b.max():.3f}] |")
    lines.append(f"\nShrinkage w distribution: min={min(w_values):.4f}, "
                  f"median={float(np.median(w_values)):.4f}, max={max(w_values):.4f}\n")
    lines.append(
        "**Note:** truth for each (issue_date, hour) sample is a single full-grid AROME "
        "snapshot, so every footprint cell gets the same holdout sample count within a "
        "season (no per-cell sparsity as in the direction-residual fix's subsampled "
        "training set). Consequently w is constant across cells within a season (varies "
        "only season to season); the per-cell bias VALUES still vary spatially and are "
        "preserved by this table - only the shrinkage weight is season-uniform. With "
        "hundreds of holdout dates per season, w is close to 1 in all seasons (shrinkage "
        "has a small effect here, but is applied as specified).\n")
    lines.append(
        "**Shrinkage-target interpretation:** the task prompt's formula was written for "
        "a plain per-cell table; extended here to shrink `bias(cell,season)` toward that "
        "season's own global mean bias (not the single overall-global mean across all "
        "seasons), to keep the shrinkage target consistent with the estimate's population. "
        "Flagged for review.\n")

    lines.append("## Step 3: alpha re-sweep on bias-corrected holdout predictions\n")
    lines.append(f"BEFORE (no bias correction, alpha=1.0): coverage="
                  f"{before_summary['coverage']*100:.1f}%, width={before_summary['mean_width']:.3f} m/s, "
                  f"Winkler={before_summary['mean_winkler']:.3f}\n")
    lines.append("| alpha (post-correction) | coverage | mean width (m/s) | mean Winkler |")
    lines.append("|---|---|---|---|")
    for r in resweep_rows:
        marker = " **<- chosen**" if r["alpha"] == best["alpha"] else ""
        lines.append(f"| {r['alpha']} | {r['coverage']*100:.1f}% | {r['mean_width']:.3f} | "
                      f"{r['mean_winkler']:.3f}{marker} |")
    lines.append(f"\n**Chosen alpha: {best['alpha']}** (min Winkler post-correction = "
                  f"{best['mean_winkler']:.3f}, vs {before_summary['mean_winkler']:.3f} "
                  f"before any correction - "
                  f"{(1 - best['mean_winkler']/before_summary['mean_winkler'])*100:.1f}% improvement).\n")

    lines.append(f"\nWall-clock (Step 1 + Step 3 combined): {time.time() - t0:.0f}s")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
