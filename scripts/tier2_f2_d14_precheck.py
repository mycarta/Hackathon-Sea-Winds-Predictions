#!/usr/bin/env python3
"""TIER-2 Night-1 gate F2: 1-block d14 pre-check, Pangu vs climatology (00 UTC).

Phase 0 F2 (confirmed 2026-07-17). Scores ARM-D14 (Pangu -> couple -> block-excluded
downscaler -> raw d14 interval) against the CURRENT d14 baseline (fine seasonal
climatology, scripts/artifacts/d14_climatology_season.parquet) on ONE block, at
00 UTC only, on the AROME footprint truth. Baseline is RE-SCORED at 00 UTC here
(decision 1), NOT the 4-hour pooled number.

Reports (decision 5 -- center vs interval split, de-confounds attribution):
  * CENTER : q50 speed RMSE, mean bias vs truth, implied de-biased RMSE
             (= sqrt(RMSE^2 - bias^2); estimates post-bias-table center skill)
  * INTERVAL: coverage, mean width, pooled Winkler
Pangu intervals here are UNCALIBRATED (raw d14 spread 0.55/1.60, no bias table,
no spd_infl) -- a floor; center skill is the primary d14 signal. Deterministic.
NO pipeline file modified; downscaler is the block-excluded refit fit, cached.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
import config                                        # noqa: E402
import target_loader                                 # noqa: E402
from ws_d7_feature_experiments import winkler_score  # noqa: E402

EXTRACTS = _HERE.parent / "data" / "arm_extracts_20260821"   # R1 rebuild, committed
DWN_CACHE = _HERE.parent / "data" / "downscaler_blockexcl_20260822.pkl"   # R2 refit
ARTIFACTS = _HERE / "artifacts"
CLIM_PARQUET = ARTIFACTS / "d14_climatology_season.parquet"
HOUR = 0
D14_LO, D14_HI = 0.55, 1.60                          # kit d14 climatology spread


def get_downscaler():
    """Block-excluded downscaler (2020[::5] minus block+/-buffer), cached to disk."""
    if DWN_CACHE.exists():
        with open(DWN_CACHE, "rb") as f:
            return pickle.load(f)
    excl = ec.exclusion_set()
    d2020 = [d for d in target_loader.list_dates(config.target_root())
             if d.year == 2020][::5]
    d2020_red = [d for d in d2020 if d not in excl]
    print(f"[F2] training downscaler on {len(d2020_red)} block-excluded 2020 days ...")
    t0 = time.time()
    dwn = dn.train_downscaler(d2020_red, hours=ec.HOURS)
    print(f"[F2] downscaler trained in {time.time()-t0:.0f}s")
    DWN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(DWN_CACHE, "wb") as f:
        pickle.dump(dwn, f)
    return dwn


def accum():
    return {"se2": 0.0, "sbias": 0.0, "n": 0, "wink": 0.0, "cov": 0, "width": 0.0,
            "n_dates": 0}


def finish(a):
    return {"q50_rmse": round(float(np.sqrt(a["se2"] / a["n"])), 4),
            "q50_bias": round(a["sbias"] / a["n"], 4),
            "debiased_rmse": round(float(np.sqrt(max(0.0, a["se2"] / a["n"]
                                    - (a["sbias"] / a["n"]) ** 2))), 4),
            "winkler": round(a["wink"] / a["n"], 4),
            "coverage": round(a["cov"] / a["n"], 4),
            "mean_width": round(a["width"] / a["n"], 4),
            "n_dates": a["n_dates"], "n": a["n"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="DJF", choices=list(ec.SEASONS))
    args = ap.parse_args()
    s = args.block
    t0 = time.time()
    dwn = get_downscaler()

    # climatology d14 baseline rows for this season/hour (point_id order == footprint)
    clim = pd.read_parquet(CLIM_PARQUET)
    cc = clim[(clim.season == s) & (clim.hour == HOUR)].sort_values("point_id")
    c_q05 = cc.speed_q05.to_numpy(float); c_q50 = cc.speed_q50.to_numpy(float)
    c_q95 = cc.speed_q95.to_numpy(float)

    bdays = ec.block_days()[s]
    ys, xs = ec.footprint_yx()                                 # (479,433) -> 43715
    pangu, climb = accum(), accum()
    n_missing = 0
    for V in bdays:
        truth = ec.arome_truth_speed(str(V), HOUR)
        if truth is None:
            continue
        D = pd.Timestamp(V) - timedelta(days=14)
        ex = EXTRACTS / f"extract_{D:%Y%m%d}.npz"
        if not ex.exists():
            n_missing += 1
            continue
        z = np.load(ex, allow_pickle=True)
        fu, fv = dn.downscale(dwn, z["d14_u"], z["d14_v"])      # coarse -> fine (479,433)
        spd = np.sqrt(fu ** 2 + fv ** 2)[ys, xs]               # footprint, canonical order
        m = np.isfinite(spd) & np.isfinite(truth)
        # --- Pangu-d14 ---
        q50 = spd[m]; t = truth[m]
        lo = q50 * D14_LO; hi = q50 * D14_HI
        pangu["se2"] += float(np.sum((q50 - t) ** 2)); pangu["sbias"] += float(np.sum(q50 - t))
        pangu["wink"] += float(winkler_score(t, lo, hi).sum())
        pangu["cov"] += int(((t >= lo) & (t <= hi)).sum()); pangu["width"] += float((hi - lo).sum())
        pangu["n"] += t.size; pangu["n_dates"] += 1
        # --- climatology-d14 (same masked cells) ---
        cq50 = c_q50[m]; clo = c_q05[m]; chi = c_q95[m]
        climb["se2"] += float(np.sum((cq50 - t) ** 2)); climb["sbias"] += float(np.sum(cq50 - t))
        climb["wink"] += float(winkler_score(t, clo, chi).sum())
        climb["cov"] += int(((t >= clo) & (t <= chi)).sum()); climb["width"] += float((chi - clo).sum())
        climb["n"] += t.size; climb["n_dates"] += 1

    P, C = finish(pangu), finish(climb)
    rmse_gain = 100 * (C["q50_rmse"] - P["debiased_rmse"]) / C["q50_rmse"]
    wink_gain = 100 * (C["winkler"] - P["winkler"]) / C["winkler"]
    print(f"\n[F2] block {s} (00 UTC), {P['n_dates']} dates, {n_missing} missing extracts")
    print(f"  {'':<14}{'q50_RMSE':>10}{'bias':>9}{'debiasRMSE':>12}{'Winkler':>10}{'cov':>7}{'width':>8}")
    print(f"  {'Pangu-d14':<14}{P['q50_rmse']:>10}{P['q50_bias']:>9}{P['debiased_rmse']:>12}"
          f"{P['winkler']:>10}{P['coverage']:>7}{P['mean_width']:>8}")
    print(f"  {'clim-d14':<14}{C['q50_rmse']:>10}{C['q50_bias']:>9}{'-':>12}"
          f"{C['winkler']:>10}{C['coverage']:>7}{C['mean_width']:>8}")
    print(f"  CENTER: Pangu de-biased RMSE vs clim RMSE = {rmse_gain:+.1f}%  "
          f"(raw Winkler {wink_gain:+.1f}%, UNCALIBRATED)")

    out = {"gate": "F2_d14_block_precheck", "block": s, "hour": HOUR,
           "n_dates": P["n_dates"], "n_missing_extracts": n_missing,
           "pangu_d14": P, "clim_d14": C,
           "center_debiased_rmse_gain_pct": round(rmse_gain, 2),
           "raw_winkler_gain_pct_uncalibrated": round(wink_gain, 2),
           "note": "Pangu intervals uncalibrated (no bias table/spd_infl); center is the d14 signal",
           "elapsed_sec": round(time.time() - t0, 1)}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / f"tier2_f2_d14_{s}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[F2] wrote {ARTIFACTS / f'tier2_f2_d14_{s}.json'}")


if __name__ == "__main__":
    main()
