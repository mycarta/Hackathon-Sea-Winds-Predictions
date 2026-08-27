#!/usr/bin/env python3
"""f-factor width-calibration sweep for the Pangu d7 arm (dev-year four-block data).

Opus/Matteo 2026-07-19. Board 854984 (Pangu d7 swap) scored WS d7 = 25.32 with
coverage_d7 = 74.6% -- intervals too tight on the withheld year. Phase-1 calibration
study: the Winkler optimum sits near coverage 0.88. This computes a scalar width
factor f (post-processing, NOT a pipeline recalibration; nominal alpha stays 0.90)
on the SAME dev data that produced the +31% four-block result:

  q95_new = q50 + f*(q95 - q50);  q05_new = max(0, q50 - f*(q50 - q05))

Reproduces the Pangu d7 arm's FINAL (q05,q50,q95) via the validated scorer functions
(frozen bias seed 42, k=1.846, alpha 0.90) over the 112 four-block valid dates at
00 UTC, then sweeps f in [1.0, 1.6] step 0.05 (extends if the optimum is on the
boundary). Winkler = ws_d7_feature_experiments.winkler_score (ALPHA_LEVEL 0.10, the
board metric). Deterministic. Writes tier2_d7_widthcal_sweep.json. No build here --
report + hold for confirm (stop point a).

Hour note: the four-block data is 00 UTC ONLY (no dev Pangu d7 at 06/12/18), so a
per-hour sweep is not supported -> one f for all hours, with the transfer caveat.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
sys.path.insert(0, str(_HERE.parent / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))
import tier2_eval_common as ec                        # noqa: E402
import tier2_d7_score_blocks as S                     # noqa: E402
from ws_d7_feature_experiments import winkler_score    # noqa: E402

K_PANGU = 1.846
ALPHA = S.ALPHA                                        # 0.90 nominal (unchanged)
HOUR = 0
ARTIFACTS = _HERE / "artifacts"


def collect():
    S._prep()
    ys, xs = S._YS, S._XS
    dm = json.load(open(ARTIFACTS / "tier2_d7_datemap.json"))
    bias = S.build_bias(S.pg_ctx, S.pg_coarse, dm["bias_issue_to_validV"])
    dates = S.scorable_dates()
    Q05, Q50, Q95, T = [], [], [], []
    nd = 0
    for s in ec.SEASONS:
        bb = bias[s]
        for W, issue in dates[s]:
            truth = ec.arome_truth_speed(W, HOUR)
            if truth is None:
                continue
            ctx = S.pg_ctx(issue)
            spd_fine = S.downscaled(S.pg_coarse(ctx))
            q05f, q95f = S.pg_interval(ctx, spd_fine, K_PANGU)
            spd = spd_fine[ys, xs]; q05 = q05f[ys, xs]; q95 = q95f[ys, xs]
            q50c = spd - bb; q05c = q05 - bb; q95c = q95 - bb
            lo = np.maximum(0.0, q50c - ALPHA * (q50c - q05c))
            hi = q50c + ALPHA * (q95c - q50c)
            m = np.isfinite(spd) & np.isfinite(truth) & np.isfinite(lo) & np.isfinite(hi)
            Q05.append(lo[m]); Q50.append(q50c[m]); Q95.append(hi[m]); T.append(truth[m])
            nd += 1
    return (np.concatenate(Q05), np.concatenate(Q50),
            np.concatenate(Q95), np.concatenate(T), nd)


def sweep(q05, q50, q95, t, fs):
    rows = []
    for f in fs:
        hi = q50 + f * (q95 - q50)
        lo = np.maximum(0.0, q50 - f * (q50 - q05))
        cov = float(np.mean((t >= lo) & (t <= hi)))
        wink = float(np.mean(winkler_score(t, lo, hi)))
        rows.append((round(float(f), 2), round(cov, 4), round(wink, 4)))
    return rows


def main():
    q05, q50, q95, t, nd = collect()
    print(f"[sweep] dev dates {nd}, finite cells {t.size}")
    rows = sweep(q05, q50, q95, t, np.arange(0.80, 1.601, 0.05))
    best = min(rows, key=lambda r: r[2])
    if best[0] >= 1.60:                                # extend up if optimum on upper boundary
        rows += sweep(q05, q50, q95, t, np.arange(1.65, 2.001, 0.05))
    if best[0] <= 0.80:                                # extend down if optimum on lower boundary
        rows = sweep(q05, q50, q95, t, np.arange(0.50, 0.801, 0.05)) + rows
    rows.sort(key=lambda r: r[0])
    best = min(rows, key=lambda r: r[2])
    print(f"{'f':>6}{'coverage':>10}{'Winkler':>10}")
    for r in rows:
        mark = "  <== f* (min Winkler)" if r == best else ""
        print(f"{r[0]:>6.2f}{r[1]:>10.4f}{r[2]:>10.4f}{mark}")
    base = next(r for r in rows if r[0] == 1.00)
    print(f"\nbaseline f=1.00: coverage {base[1]}  Winkler {base[2]} "
          f"(self-check vs four-block 0.840 / 22.57)")
    print(f"f* = {best[0]}  coverage {best[1]}  Winkler {best[2]}  "
          f"(Winkler {100*(base[2]-best[2])/base[2]:+.2f}% vs f=1.0)")
    out = {"n_dates": nd, "n_cells": int(t.size), "alpha_nominal": ALPHA,
           "k_pangu": K_PANGU, "hour": HOUR,
           "sweep": [{"f": f, "coverage": c, "winkler": w} for f, c, w in rows],
           "f_star": best[0], "coverage_at_fstar": best[1], "winkler_at_fstar": best[2],
           "baseline_f1_coverage": base[1], "baseline_f1_winkler": base[2]}
    json.dump(out, open(ARTIFACTS / "tier2_d7_widthcal_sweep.json", "w"), indent=2)
    print("wrote tier2_d7_widthcal_sweep.json")


if __name__ == "__main__":
    main()
