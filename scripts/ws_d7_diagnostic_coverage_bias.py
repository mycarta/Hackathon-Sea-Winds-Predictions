#!/usr/bin/env python3
"""
WS d7 diagnostics 1 + 3 (2026-07-08): per-horizon speed coverage/alpha-sweep
+ per-cell d7 speed bias. Combined into one script because both need the
same expensive per-issue-date forecast+downscale pipeline.

Diagnostic 1: reproduces the SHIPPED speed forecast pipeline exactly as
1_predict_target.ipynb / 2_downscale_to_target.ipynb build it
(fh.fit_forecast(P.train_dates('6D')), dn.train_downscaler on a 2020
subsample, P.calibrate_intervals for the widening factors already baked
into the real submission), then evaluates fine-grid coverage on
splits.train_val_dates(seed=42)'s VAL split (an independent held-out set
from the model's own training/calib dates - not the same split the kit
itself uses internally). For each held-out (valid_date, hour, cell), sweeps
an additional alpha multiplier on TOP of the already-shipped interval:
  q95_new = q50 + alpha*(q95-q50);  q05_new = q50 - alpha*(q50-q05)
and reports coverage / mean width / estimated Winkler score at each alpha.

Diagnostic 3: for d7 only, accumulates per-cell bias = mean(q50 - truth)
across all held-out (valid_date, hour) samples, for all 43,715 footprint
cells (not subsampled - downscale() already produces full-grid output at
no extra cost), plus a per-season breakdown.

CAVEAT (reported prominently, not hidden): d14's climatology pools ALL of
TRAIN_YEARS_CLIM=(2016..2020) per (week-of-year,hour) bin, so evaluating it
on any TRAIN_YEARS valid_date has unavoidable self-leakage (the date being
scored contributes to its own climatology bin). d14's coverage numbers
here are optimistically biased for that reason; d1/d7 don't have this
issue in the same way (MOS trained on a different, disjoint-ish date
scheme - see the small-overlap note printed at runtime).

No model files are modified or saved; models/mos/downscaler are refit
in-memory only, exactly reproducing the shipped notebook's methodology.
submission.csv is never touched.

Outputs:
  reports/ws_d7_diagnostics_20260708.md (diagnostic 1 + 3 sections)
  scripts/artifacts/ws_d7_percell_bias.parquet (per-cell bias, gitignored)
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
AROME_DIR = ROOT / "phase_2" / "phase2_dataset_ship" / "train" / "arome"
STATIC_PATH = ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "arome_static.nc"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REPORT_PATH = ROOT / "reports" / "ws_d7_diagnostics_20260708.md"
BIAS_OUT_PATH = ARTIFACTS_DIR / "ws_d7_percell_bias.parquet"

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

LEADS = (1, 7, 14)
HOURS = (0, 6, 12, 18)
ALPHAS = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1)
ALPHA_LEVEL = 0.10  # nominal miscoverage rate for the Winkler score (90% PI)
SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


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
    u = ds["u125m"].values[hour_mask][order][:, ys, xs]  # (4, n_cells)
    v = ds["v125m"].values[hour_mask][order][:, ys, xs]
    ds.close()
    speed = np.sqrt(u ** 2 + v ** 2)  # (4, n_cells), hour order matches HOURS
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
    print(f"  offs (dir half-width, coarse-cal): {offs}")

    print("Training downscaler (matches notebook: 2020 dates, every 5th)...")
    d2020 = [d for d in target_loader.list_dates(config.target_root()) if d.year == 2020][::5]
    dwn = dn.train_downscaler(d2020, hours=HOURS)
    print(f"  downscaler trained on {len(d2020)} days")

    print("Calibrating intervals (spd_infl / dir_off - matches shipped widening)...")
    spd_infl, dir_off = P.calibrate_intervals(mos, qmos, adj, dwn, offs)
    print(f"  spd_infl: {spd_infl}")

    train_dates_seed, val_dates_seed = splits.train_val_dates(seed=42)
    val_set = set(val_dates_seed)
    print(f"\ntrain_val_dates(seed=42): {len(train_dates_seed)} train, {len(val_dates_seed)} val")

    # transparency: quantify any overlap between this eval set and the
    # model's own training/calib date populations
    mos_train_dates = set(pd.Timestamp(d).date() for d in train)
    calib_dates = set(pd.Timestamp(d).date() for d in train[pd.DatetimeIndex(train).year == 2020])
    downscaler_dates = set(d2020)
    overlap_mos = val_set & mos_train_dates
    overlap_calib = val_set & calib_dates
    overlap_dwn = val_set & downscaler_dates
    print(f"Overlap check: val_dates ∩ mos-train-dates = {len(overlap_mos)}, "
          f"∩ conformal-calib-dates = {len(overlap_calib)}, "
          f"∩ downscaler-train-dates = {len(overlap_dwn)} "
          f"(all should be small relative to {len(val_set)} val dates)")

    # candidate issue dates: every day 2016-01-01..2020-12-31 with an AROME file
    all_dates = sorted(target_loader.list_dates(config.target_root()))
    candidate_issue_dates = [d for d in all_dates if d.year in (2016, 2017, 2018, 2019, 2020)]
    print(f"\nCandidate issue dates: {len(candidate_issue_dates)}")

    cov_acc = {L: {a: {"n": 0, "covered": 0, "width_sum": 0.0, "winkler_sum": 0.0}
                    for a in ALPHAS} for L in LEADS}
    bias_sum = np.zeros(n_cells, dtype=np.float64)
    bias_count = np.zeros(n_cells, dtype=np.int64)
    bias_by_season = {s: {"sum": 0.0, "count": 0} for s in ("DJF", "MAM", "JJA", "SON")}

    truth_cache = {}
    n_processed = 0

    for D in candidate_issue_dates:
        # skip issue dates where none of the 3 leads' valid dates are in val_set
        valid_dates = {L: D + timedelta(days=L) for L in LEADS}
        if not any(V in val_set for V in valid_dates.values()):
            continue

        try:
            fields = P.coarse_fields(mos, qmos, adj, pd.Timestamp(D))
        except Exception as e:
            print(f"  skip {D}: coarse_fields failed ({e})")
            continue

        blocks = P.downscale_window(dwn, fields, offs, window=0,
                                     spd_infl=spd_infl, dir_off=dir_off)
        # blocks: one df per (lead,hour) in LEADS x HOURS order, footprint-ordered rows
        bi = 0
        for L in P.LEADS:
            V = valid_dates.get(L)
            for H in HOURS:
                blk = blocks[bi]
                bi += 1
                if V is None or V not in val_set:
                    continue
                speed_truth_all_hours = truth_cache_get(truth_cache, V, ys, xs)
                if speed_truth_all_hours is None:
                    continue
                h_idx = HOURS.index(H)
                truth = speed_truth_all_hours[h_idx]  # (n_cells,)

                q05 = blk["q05"].to_numpy()
                q50 = blk["q50"].to_numpy()
                q95 = blk["q95"].to_numpy()

                for a in ALPHAS:
                    lo = q50 - a * (q50 - q05)
                    hi = q50 + a * (q95 - q50)
                    covered = (truth >= lo) & (truth <= hi)
                    acc = cov_acc[L][a]
                    acc["n"] += n_cells
                    acc["covered"] += int(covered.sum())
                    acc["width_sum"] += float((hi - lo).sum())
                    acc["winkler_sum"] += float(winkler_score(truth, lo, hi).sum())

                if L == 7:
                    bias = q50 - truth
                    bias_sum += bias
                    bias_count += 1
                    season = SEASON_OF_MONTH[V.month]
                    bias_by_season[season]["sum"] += float(bias.sum())
                    bias_by_season[season]["count"] += n_cells

        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  processed {n_processed} issue dates (latest D={D}), "
                  f"elapsed={time.time() - t0:.0f}s")

    print(f"\nTotal issue dates processed (contributed >=1 eval sample): {n_processed}")

    # ---- Diagnostic 1 report ----
    lines = ["# WS d7 diagnostics — 2026-07-08\n",
             "No submission or model changes. Diagnostic numbers only.\n",
             "## Diagnostic 1: per-horizon coverage / alpha-sweep\n",
             f"Evaluated on `splits.train_val_dates(seed=42)` val split "
             f"({len(val_dates_seed)} dates), using the SHIPPED speed pipeline "
             f"(`fh.fit_forecast(P.train_dates('6D'))` + `dn.train_downscaler` "
             f"on 2020[::5] + `P.calibrate_intervals`), refit in-memory only "
             f"(no models modified/saved).\n",
             f"Overlap with model's own training/calib dates: "
             f"mos-train={len(overlap_mos)}, conformal-calib={len(overlap_calib)}, "
             f"downscaler-train={len(overlap_dwn)} (out of {len(val_set)} val dates) "
             f"- small, noted for transparency, not excluded.\n",
             "**CAVEAT: d14 uses climatology pooled over ALL of 2016-2020 per "
             "(week-of-year, hour) - any TRAIN_YEARS valid_date scored here "
             "contributes to its own climatology bin (unavoidable self-leakage). "
             "d14 numbers below are optimistically biased; treat d1/d7 as the "
             "reliable comparison.**\n",
             "| Horizon | alpha | coverage | mean width (m/s) | mean Winkler (est., alpha_level=0.10) |",
             "|---|---|---|---|---|"]

    winkler_optimal = {}
    for L in LEADS:
        best_a, best_w = None, np.inf
        for a in ALPHAS:
            acc = cov_acc[L][a]
            if acc["n"] == 0:
                continue
            cov = acc["covered"] / acc["n"]
            width = acc["width_sum"] / acc["n"]
            wink = acc["winkler_sum"] / acc["n"]
            lines.append(f"| d{L} | {a} | {cov * 100:.1f}% | {width:.3f} | {wink:.3f} |")
            if wink < best_w:
                best_w, best_a = wink, a
        winkler_optimal[L] = (best_a, best_w)

    lines.append("\n### Winkler-optimal alpha per horizon\n")
    for L in LEADS:
        a, w = winkler_optimal[L]
        base_cov = cov_acc[L][1.0]["covered"] / max(cov_acc[L][1.0]["n"], 1)
        lines.append(f"- d{L}: alpha*={a} (est. Winkler={w:.3f}), "
                      f"current (alpha=1.0) coverage={base_cov * 100:.1f}%")

    # ---- Diagnostic 3 report ----
    lines.append("\n## Diagnostic 3: per-cell d7 speed bias (q50 - truth)\n")
    valid_mask = bias_count > 0
    per_cell_bias = np.full(n_cells, np.nan)
    per_cell_bias[valid_mask] = bias_sum[valid_mask] / bias_count[valid_mask]
    b = per_cell_bias[valid_mask]
    lines.append(f"Cells with >=1 sample: {valid_mask.sum()} / {n_cells}")
    lines.append(f"- mean bias: {b.mean():.4f} m/s")
    lines.append(f"- std bias: {b.std():.4f} m/s")
    lines.append(f"- p05 / p95: {np.percentile(b, 5):.4f} / {np.percentile(b, 95):.4f} m/s")
    verdict_cell = (abs(b.mean()) > 0.3) or (b.std() > 0.5)
    lines.append(f"- Correctable structure (|mean|>0.3 or std>0.5): "
                  f"{'YES' if verdict_cell else 'NO'}")

    lines.append("\n### Per-season d7 bias\n")
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

    lines.append("\n### Recommendations\n")
    lines.append(f"- Diagnostic 1: {'Recalibration (alpha != 1.0) may help' if any(winkler_optimal[L][0] != 1.0 for L in (1,7)) else 'Current widening (alpha=1.0) already near-optimal for d1/d7'} "
                  f"per the Winkler-optimal alpha values above.")
    lines.append(f"- Diagnostic 3: {'Per-cell bias correction worth pursuing for d7' if verdict_cell else 'd7 MOS already well-centered; bias correction unlikely to help'} "
                  f"(mean={b.mean():.4f}, std={b.std():.4f} m/s).")

    lines.append(f"\nTotal wall-clock: {time.time() - t0:.0f}s")

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
