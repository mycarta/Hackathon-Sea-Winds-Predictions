#!/usr/bin/env python3
"""TIER-2 ARM-D7 pre-check: Pangu-d7 vs A0 (HRES-MOS d7), DJF, 00 UTC.

Opus-authorized (2026-07-18, after ARM D14 was killed). Decision 2: re-score the
A0 baseline (HRES-MOS d7) at 00 UTC on DJF and compare Pangu-d7 from the in-hand
DJF extracts, with the center/interval diagnostic split (decision 5).

d7 is INSIDE the useful-skill window (unlike the killed d14 arm). The go/no-go
read (decision 2): if Pangu-d7 CENTER (de-biased q50 RMSE) is decorrelation-grade
worse than A0, kill both arms; if competitive, propose the reduced remaining-blocks
budget (7-step rollouts only).

Pangu-d7 for a valid date W comes from extract_(W-7).npz (issue D=W-7, d7 harvest
valid at D+7=W). The in-hand DJF extracts (issue 2016-12-26..2017-01-22 -> d7 valid
2017-01-02..2017-01-29) cover 21 of the 28 DJF block days (2017-01-09..01-29); the
tail 7 need I7-only rollouts not yet run. Both arms scored on the SAME 21 dates.

CENTER is primary (decision 2). Intervals are a secondary diagnostic and are NOT
the fully-calibrated configs (A0 here uses its quantile-MOS interval; Pangu the raw
0.55/1.60 fallback) -- calibration would come only if we proceed. Deterministic.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
sys.path.insert(0, str(_HERE.parent / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))
import tier2_eval_common as ec                       # noqa: E402
import downscaling as dn                             # noqa: E402
import forecast_pipeline as P                        # noqa: E402
from ws_d7_feature_experiments import winkler_score  # noqa: E402
from tier2_f2_d14_precheck import get_downscaler, accum, finish  # noqa: E402
from _publication_paths import ppath  # noqa: E402  (publication tree)

EXTRACTS = ppath("<NETWORK_SHARE>/Matteo/large downloads/tier2_smoke/"
                 "arm_extracts")
ARTIFACTS = _HERE / "artifacts"
HOUR = 0
LEAD = 7
D7_LO, D7_HI = 0.55, 1.60          # same raw fallback spread as d14 (uncalibrated)


def main():
    t0 = time.time()
    dwn = get_downscaler()
    ys, xs = ec.footprint_yx()
    excl = ec.exclusion_set()

    # --- fit A0 (HRES-MOS), block-excluded train (same protocol as the refit) ---
    train_all = [d.date() for d in P.train_dates("6D")]
    train_red = [d for d in train_all if d not in excl]
    print(f"[D7] fit_forecast on {len(train_red)} block-excluded train dates ...")
    mos, qmos, adj, offs = P.fit_forecast(pd.to_datetime(train_red))
    print(f"[D7] fit done ({time.time()-t0:.0f}s)")

    # DJF block valid dates whose Pangu-d7 extract (issue = V-7) is in hand
    djf = ec.block_days()["DJF"]
    dates = []
    for V in djf:
        D = pd.Timestamp(V) - timedelta(days=LEAD)
        if (EXTRACTS / f"extract_{D:%Y%m%d}.npz").exists():
            dates.append(V)
    print(f"[D7] scoring {len(dates)}/{len(djf)} DJF dates with in-hand Pangu-d7 extracts")

    A0, PG = accum(), accum()
    for V in dates:
        truth = ec.arome_truth_speed(str(V), HOUR)
        if truth is None:
            continue
        t = truth
        Dp = pd.Timestamp(V) - timedelta(days=LEAD)
        # --- Pangu-d7 ---
        z = np.load(EXTRACTS / f"extract_{Dp:%Y%m%d}.npz", allow_pickle=True)
        pfu, pfv = dn.downscale(dwn, z["d7_u"], z["d7_v"])
        pspd = np.sqrt(pfu ** 2 + pfv ** 2)[ys, xs]
        mp = np.isfinite(pspd) & np.isfinite(t)
        # --- A0 (HRES-MOS d7): coarse_fields det -> downscale ---
        flds = P.coarse_fields(mos, qmos, adj, Dp)
        U, Vv = flds[(LEAD, HOUR, "det")]
        afu, afv = dn.downscale(dwn, U, Vv)
        aspd = np.sqrt(afu ** 2 + afv ** 2)[ys, xs]
        # A0 interval from its quantile-MOS spd grids (raw, no bias/spd_infl)
        alo, ahi = P._speed_interval(flds, LEAD, HOUR,
                                     np.sqrt(afu ** 2 + afv ** 2), k=1.0)
        alo = alo[ys, xs]; ahi = ahi[ys, xs]
        ma = np.isfinite(aspd) & np.isfinite(t) & np.isfinite(alo) & np.isfinite(ahi)
        m = mp & ma                                        # same cells for both
        # accumulate Pangu
        q = pspd[m]; tt = t[m]; lo = q * D7_LO; hi = q * D7_HI
        PG["se2"] += float(np.sum((q - tt) ** 2)); PG["sbias"] += float(np.sum(q - tt))
        PG["wink"] += float(winkler_score(tt, lo, hi).sum())
        PG["cov"] += int(((tt >= lo) & (tt <= hi)).sum()); PG["width"] += float((hi - lo).sum())
        PG["n"] += tt.size; PG["n_dates"] += 1
        # accumulate A0
        q = aspd[m]; alo_, ahi_ = alo[m], ahi[m]
        A0["se2"] += float(np.sum((q - tt) ** 2)); A0["sbias"] += float(np.sum(q - tt))
        A0["wink"] += float(winkler_score(tt, alo_, ahi_).sum())
        A0["cov"] += int(((tt >= alo_) & (tt <= ahi_)).sum()); A0["width"] += float((ahi_ - alo_).sum())
        A0["n"] += tt.size; A0["n_dates"] += 1

    a, p = finish(A0), finish(PG)
    center_gap = 100 * (p["debiased_rmse"] - a["debiased_rmse"]) / a["debiased_rmse"]
    print(f"\n[D7] DJF (00 UTC), {p['n_dates']} dates")
    print(f"  {'':<12}{'q50_RMSE':>10}{'bias':>9}{'debiasRMSE':>12}{'Winkler':>10}{'cov':>7}{'width':>8}")
    print(f"  {'A0-d7(HRES)':<12}{a['q50_rmse']:>10}{a['q50_bias']:>9}{a['debiased_rmse']:>12}"
          f"{a['winkler']:>10}{a['coverage']:>7}{a['mean_width']:>8}")
    print(f"  {'Pangu-d7':<12}{p['q50_rmse']:>10}{p['q50_bias']:>9}{p['debiased_rmse']:>12}"
          f"{p['winkler']:>10}{p['coverage']:>7}{p['mean_width']:>8}")
    print(f"  CENTER: Pangu-d7 de-biased RMSE vs A0 = {center_gap:+.1f}% "
          f"(sqrt2/decorrelation would be ~+41%)")

    out = {"gate": "F2b_d7_block_precheck", "block": "DJF", "hour": HOUR,
           "n_dates": p["n_dates"], "note": "intervals uncalibrated; center is the signal",
           "a0_d7": a, "pangu_d7": p, "center_debiased_rmse_gap_pct": round(center_gap, 2),
           "elapsed_sec": round(time.time() - t0, 1)}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "tier2_f2b_d7_DJF.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[D7] wrote {ARTIFACTS / 'tier2_f2b_d7_DJF.json'}")


if __name__ == "__main__":
    main()
