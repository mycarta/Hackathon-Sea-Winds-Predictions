#!/usr/bin/env python3
"""
WS d1 speed bias diagnostic (2026-07-08) - same per-cell bias methodology
as Diagnostic 3 in scripts/ws_d7_diagnostic_coverage_bias.py, applied to
d1 instead of d7, on the same holdout split.

Reproduces the SHIPPED speed forecast pipeline exactly as
1_predict_target.ipynb / 2_downscale_to_target.ipynb build it
(P.fit_forecast(P.train_dates('6D')), dn.train_downscaler on a 2020
subsample, P.calibrate_intervals for the widening factors already baked
into the real submission), evaluated on splits.train_val_dates(seed=42)'s
VAL split - same holdout as every other WS diagnostic this session.

For d1 only, accumulates per-cell bias = mean(q50 - truth) across all
held-out (valid_date, hour) samples, for all 43,715 footprint cells, plus
a per-season breakdown - same methodology, same report format as
Diagnostic 3 (global mean bias, std, p05/p95, per-season table, one-line
recommendation).

No model files are modified or saved; models/mos/downscaler are refit
in-memory only. submission.csv is never touched - diagnostic numbers
only, no submission changes.

Outputs:
  reports/ws_d1_bias_diagnostic_20260708.md
  scripts/artifacts/ws_d1_percell_bias.parquet (per-cell bias, gitignored)
"""

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
REPORT_PATH = ROOT / "reports" / "ws_d1_bias_diagnostic_20260708.md"
BIAS_OUT_PATH = ARTIFACTS_DIR / "ws_d1_percell_bias.parquet"

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

LEAD = 1
HOURS = (0, 6, 12, 18)
SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


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

    bias_sum = np.zeros(n_cells, dtype=np.float64)
    bias_count = np.zeros(n_cells, dtype=np.int64)
    bias_by_season = {s: {"sum": 0.0, "count": 0} for s in ("DJF", "MAM", "JJA", "SON")}

    truth_cache = {}
    n_processed = 0
    lead_idx = P.LEADS.index(LEAD)
    base = lead_idx * len(HOURS)

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

        speed_truth_all_hours = truth_cache_get(truth_cache, V, ys, xs)
        if speed_truth_all_hours is None:
            continue

        season = SEASON_OF_MONTH[V.month]
        for h_idx, H in enumerate(HOURS):
            blk = blocks[base + h_idx]
            assert int(blk["horizon"].iloc[0]) == LEAD and int(blk["hour"].iloc[0]) == H
            q50 = blk["q50"].to_numpy()
            truth = speed_truth_all_hours[h_idx]

            bias = q50 - truth
            bias_sum += bias
            bias_count += 1
            bias_by_season[season]["sum"] += float(bias.sum())
            bias_by_season[season]["count"] += n_cells

        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  processed {n_processed} d1 issue dates (latest D={D}), "
                  f"elapsed={time.time() - t0:.0f}s")

    print(f"\nTotal d1 issue dates processed: {n_processed}")

    valid_mask = bias_count > 0
    per_cell_bias = np.full(n_cells, np.nan)
    per_cell_bias[valid_mask] = bias_sum[valid_mask] / bias_count[valid_mask]
    b = per_cell_bias[valid_mask]

    lines = [
        "# WS d1 speed bias diagnostic — 2026-07-08\n",
        "No submission or model changes. Same per-cell bias methodology as "
        "Diagnostic 3 in `reports/ws_d7_diagnostics_20260708.md`, applied to d1, "
        "same `splits.train_val_dates(seed=42)` holdout, same shipped-pipeline "
        "reproduction.\n",
        f"Cells with >=1 sample: {valid_mask.sum()} / {n_cells}",
        f"- mean bias: {b.mean():.4f} m/s",
        f"- std bias: {b.std():.4f} m/s",
        f"- p05 / p95: {np.percentile(b, 5):.4f} / {np.percentile(b, 95):.4f} m/s",
    ]
    verdict_cell = (abs(b.mean()) > 0.3) or (b.std() > 0.5)
    lines.append(f"- Correctable structure (|mean|>0.3 or std>0.5): "
                  f"{'YES' if verdict_cell else 'NO'}")

    lines.append("\n## Per-season d1 bias\n")
    lines.append("| Season | mean bias (m/s) | n samples |")
    lines.append("|---|---|---|")
    season_means = {}
    for s in ("DJF", "MAM", "JJA", "SON"):
        c = bias_by_season[s]
        m = c["sum"] / c["count"] if c["count"] else float("nan")
        season_means[s] = m
        lines.append(f"| {s} | {m:.4f} | {c['count']} |")
    season_spread = max(season_means.values()) - min(season_means.values())
    lines.append(f"\nSeasonal spread (max-min mean bias): {season_spread:.4f} m/s")

    lines.append("\n## Recommendation\n")
    lines.append(f"{'Per-cell bias correction worth pursuing for d1' if verdict_cell else 'd1 MOS already well-centered; bias correction unlikely to help'} "
                  f"(mean={b.mean():.4f}, std={b.std():.4f} m/s).")

    lines.append(f"\nWall-clock: {time.time() - t0:.0f}s")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")

    bias_df = pd.DataFrame({
        "cell_idx": np.arange(n_cells)[valid_mask],
        "y": ys[valid_mask], "x": xs[valid_mask],
        "bias_mean": per_cell_bias[valid_mask],
        "n_samples": bias_count[valid_mask],
    })
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bias_df.to_parquet(BIAS_OUT_PATH, index=False)
    print(f"Wrote {BIAS_OUT_PATH} ({len(bias_df)} cells)")


if __name__ == "__main__":
    main()
