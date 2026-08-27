#!/usr/bin/env python3
"""TIER-2 ARM-D7 four-block scorer: Pangu-d7 vs A0 (HRES-MOS d7), 00 UTC.

Opus-authorized, Matteo GO 2026-07-18. Both arms run through IDENTICAL machinery
-- block-excluded downscaler, per-cell x season bias table (N=20/season seeded 42,
shrink K=30) on the SAME sampled dates, spd_infl calibrated to 0.90 coverage, alpha
0.90 -- differing ONLY in the driver (HRES-MOS `det` vs Pangu-d7 coupled extract).
A0 is re-scored at 00 UTC on all four blocks (decision 1). Reports pooled + per-block
with the center (bias-corrected q50 RMSE) vs interval (Winkler/coverage/width) split
(decision 5). Reports numbers; the >=4% ship/no-ship bar is human review's.

Exclusion invariant (a139fa3): nothing within block+/-14 d enters any fitted
component (downscaler, bias table, spd_infl) -- the sampled bias/calib dates in
tier2_d7_datemap.json are all non-excluded and seeded. Deterministic.
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
from tier2_f2_d14_precheck import get_downscaler     # noqa: E402

EXTRACTS = _HERE.parent / "data" / "arm_extracts_20260821"   # R1 rebuild, committed
ARTIFACTS = _HERE / "artifacts"
HOUR, LEAD, ALPHA, K_SHRINK = 0, 7, 0.90, 30.0
PG_LO, PG_HI = 0.55, 1.60

_DWN = None
_YS = _XS = None


def _prep():
    global _DWN, _YS, _XS
    _DWN = get_downscaler()
    _YS, _XS = ec.footprint_yx()


# ---- driver-specific coarse (U,V) on the 45x57 grid for one issue date ----
def a0_ctx(issue, mos, qmos, adj):
    flds = P.coarse_fields(mos, qmos, adj, pd.Timestamp(issue))
    return flds

def a0_coarse(ctx):
    return ctx[(LEAD, HOUR, "det")]

def a0_interval(ctx, spd50_fine, k):                 # uses quantile-MOS spd grids
    return P._speed_interval(ctx, LEAD, HOUR, spd50_fine, k=k)

def pg_ctx(issue):
    return np.load(EXTRACTS / f"extract_{pd.Timestamp(issue):%Y%m%d}.npz", allow_pickle=True)

def pg_coarse(ctx):
    return ctx["d7_u"], ctx["d7_v"]

def pg_interval(ctx, spd50_fine, k):                 # raw fallback spread widened by k
    q05 = np.maximum(0.0, spd50_fine - k * (spd50_fine - PG_LO * spd50_fine))
    q95 = spd50_fine + k * (PG_HI * spd50_fine - spd50_fine)
    return q05, q95


def downscaled(coarse_uv):
    fu, fv = dn.downscale(_DWN, coarse_uv[0], coarse_uv[1])
    return np.sqrt(fu ** 2 + fv ** 2)                # fine (479,433)


# ---- bias table: per season, per cell, q50 - truth on sampled non-excluded V ----
def build_bias(get_ctx, get_coarse, bias_map):
    acc = {s: (np.zeros(_YS.size), 0) for s in ec.SEASONS}
    for issue, V in bias_map.items():
        truth = ec.arome_truth_speed(str(V), HOUR)
        if truth is None:
            continue
        spd = downscaled(get_coarse(get_ctx(issue)))[_YS, _XS]
        m = np.isfinite(spd) & np.isfinite(truth)
        s = ec.season_of(V)
        tot, n = acc[s]
        d = np.where(m, spd - truth, 0.0)
        acc[s] = (tot + d, n + 1)                     # note: per-cell mean over dates
    bias = {}
    for s in ec.SEASONS:
        tot, n = acc[s]
        if n == 0:
            bias[s] = np.zeros(_YS.size); continue
        raw = tot / n
        smean = float(raw.mean())
        w = n / (n + K_SHRINK)
        bias[s] = w * raw + (1 - w) * smean
    return bias


# ---- spd_infl: bisect k so pooled coverage over calib dates == 0.90 ----
def calib_k(get_ctx, get_coarse, get_interval, calib_issue, target=0.90):
    mids, los, his, ts = [], [], [], []
    for issue in calib_issue:
        V = (pd.Timestamp(issue) + timedelta(days=LEAD)).date()
        truth = ec.arome_truth_speed(str(V), HOUR)
        if truth is None:
            continue
        ctx = get_ctx(issue)
        spd_fine = downscaled(get_coarse(ctx))
        q05f, q95f = get_interval(ctx, spd_fine, 1.0)
        spd = spd_fine[_YS, _XS]; q05 = q05f[_YS, _XS]; q95 = q95f[_YS, _XS]
        m = np.isfinite(spd) & np.isfinite(truth) & np.isfinite(q05) & np.isfinite(q95)
        mids.append(spd[m]); los.append(q05[m]); his.append(q95[m]); ts.append(truth[m])
    mid = np.concatenate(mids); lo = np.concatenate(los)
    hi = np.concatenate(his); t = np.concatenate(ts)

    def cov(k):
        a = np.maximum(0.0, mid - k * (mid - lo)); b = mid + k * (hi - mid)
        return float(np.mean((t >= a) & (t <= b)))
    klo, khi = 0.5, 1.0
    while cov(khi) < target and khi < 12:
        khi *= 1.5
    for _ in range(30):
        km = 0.5 * (klo + khi)
        if cov(km) < target:
            klo = km
        else:
            khi = km
    return round(0.5 * (klo + khi), 3)


# ---- dates scorable by BOTH arms (Pangu extract present + AROME truth present) ----
def scorable_dates():
    out = {}
    for s in ec.SEASONS:
        days = []
        for W in ec.block_days()[s]:
            issue = (pd.Timestamp(W) - timedelta(days=LEAD)).date()
            if (EXTRACTS / f"extract_{issue:%Y%m%d}.npz").exists() \
                    and ec.arome_truth_speed(str(W), HOUR) is not None:
                days.append((str(W), str(issue)))
        out[s] = days
    return out


# ---- score the four blocks at 00 UTC on the shared date set ----
def score(get_ctx, get_coarse, get_interval, bias, k, dates):
    per = {}
    for s in ec.SEASONS:
        acc = {"se2": 0.0, "sb": 0.0, "wink": 0.0, "cov": 0, "wid": 0.0, "n": 0, "nd": 0}
        for W, issue in dates[s]:
            truth = ec.arome_truth_speed(W, HOUR)
            ctx = get_ctx(issue)
            spd_fine = downscaled(get_coarse(ctx))
            q05f, q95f = get_interval(ctx, spd_fine, k)
            spd = spd_fine[_YS, _XS]; q05 = q05f[_YS, _XS]; q95 = q95f[_YS, _XS]
            bb = bias[s]
            q50c = spd - bb; q05c = q05 - bb; q95c = q95 - bb
            lo = np.maximum(0.0, q50c - ALPHA * (q50c - q05c))
            hi = q50c + ALPHA * (q95c - q50c)
            m = np.isfinite(spd) & np.isfinite(truth) & np.isfinite(lo) & np.isfinite(hi)
            t = truth[m]; q = q50c[m]; a = lo[m]; b = hi[m]
            acc["se2"] += float(np.sum((q - t) ** 2)); acc["sb"] += float(np.sum(q - t))
            acc["wink"] += float(winkler_score(t, a, b).sum())
            acc["cov"] += int(((t >= a) & (t <= b)).sum()); acc["wid"] += float((b - a).sum())
            acc["n"] += t.size; acc["nd"] += 1
        per[s] = acc
    return per


def summarize(per):
    tot_n = sum(per[s]["n"] for s in ec.SEASONS)
    out = {"per_block": {}, "pooled": {}}
    swink = scov = swid = sse2 = ssb = 0.0
    for s in ec.SEASONS:
        a = per[s]
        if a["n"] == 0:
            out["per_block"][s] = None; continue
        out["per_block"][s] = {
            "winkler": round(a["wink"] / a["n"], 4), "coverage": round(a["cov"] / a["n"], 4),
            "mean_width": round(a["wid"] / a["n"], 4),
            "center_rmse": round(float(np.sqrt(a["se2"] / a["n"])), 4),
            "center_bias": round(a["sb"] / a["n"], 4), "n_dates": a["nd"], "n": a["n"]}
        swink += a["wink"]; scov += a["cov"]; swid += a["wid"]; sse2 += a["se2"]; ssb += a["sb"]
    out["pooled"] = {"winkler": round(swink / tot_n, 4), "coverage": round(scov / tot_n, 4),
                     "mean_width": round(swid / tot_n, 4),
                     "center_rmse": round(float(np.sqrt(sse2 / tot_n)), 4),
                     "center_bias": round(ssb / tot_n, 4), "n": tot_n}
    return out


def main():
    t0 = time.time()
    _prep()
    dm = json.load(open(ARTIFACTS / "tier2_d7_datemap.json"))
    bias_map, calib_issue = dm["bias_issue_to_validV"], dm["calib_issue"]
    excl = ec.exclusion_set()
    train_red = [d.date() for d in P.train_dates("6D") if d.date() not in excl]
    print(f"[score] fit A0 on {len(train_red)} dates ...")
    mos, qmos, adj, offs = P.fit_forecast(pd.to_datetime(train_red))

    dates = scorable_dates()
    nd = {s: len(dates[s]) for s in ec.SEASONS}
    print(f"[score] scorable dates/block (both arms): {nd} (total {sum(nd.values())})")

    a0_get_ctx = lambda i: a0_ctx(i, mos, qmos, adj)
    print("[score] A0 bias table ..."); bias_a0 = build_bias(a0_get_ctx, a0_coarse, bias_map)
    print("[score] A0 spd_infl ..."); k_a0 = calib_k(a0_get_ctx, a0_coarse, a0_interval, calib_issue)
    print(f"[score] A0 k={k_a0}; scoring blocks ...")
    res_a0 = summarize(score(a0_get_ctx, a0_coarse, a0_interval, bias_a0, k_a0, dates))

    print("[score] Pangu bias table ..."); bias_pg = build_bias(pg_ctx, pg_coarse, bias_map)
    print("[score] Pangu spd_infl ..."); k_pg = calib_k(pg_ctx, pg_coarse, pg_interval, calib_issue)
    print(f"[score] Pangu k={k_pg}; scoring blocks ...")
    res_pg = summarize(score(pg_ctx, pg_coarse, pg_interval, bias_pg, k_pg, dates))

    pa, pp = res_a0["pooled"], res_pg["pooled"]
    wgain = 100 * (pa["winkler"] - pp["winkler"]) / pa["winkler"]
    cgain = 100 * (pa["center_rmse"] - pp["center_rmse"]) / pa["center_rmse"]
    print(f"\n=== ARM D7 four-block table (00 UTC) ===")
    print(f"{'':<10}{'Winkler':>9}{'cov':>7}{'width':>8}{'ctrRMSE':>9}{'ctrBias':>9}")
    for name, r in [("A0", pa), ("Pangu", pp)]:
        print(f"{name:<10}{r['winkler']:>9}{r['coverage']:>7}{r['mean_width']:>8}"
              f"{r['center_rmse']:>9}{r['center_bias']:>9}")
    print(f"Pangu vs A0: Winkler {wgain:+.1f}%  center RMSE {cgain:+.1f}%  "
          f"cov d {pp['coverage']-pa['coverage']:+.3f}  (>=4% bar = human review)")
    for s in ec.SEASONS:
        ra, rp = res_a0["per_block"][s], res_pg["per_block"][s]
        if ra and rp:
            g = 100 * (ra["winkler"] - rp["winkler"]) / ra["winkler"]
            print(f"  {s}: A0 W={ra['winkler']} Pangu W={rp['winkler']} ({g:+.1f}%) "
                  f"nd={rp['n_dates']}")

    out = {"arm": "D7", "hour": HOUR, "alpha": ALPHA, "bias_per_season": 20, "K": K_SHRINK,
           "k_a0": k_a0, "k_pangu": k_pg, "A0": res_a0, "Pangu": res_pg,
           "pangu_vs_a0_winkler_pct": round(wgain, 2),
           "pangu_vs_a0_center_rmse_pct": round(cgain, 2),
           "elapsed_sec": round(time.time() - t0, 1)}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "tier2_d7_fourblock.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[score] wrote {ARTIFACTS / 'tier2_d7_fourblock.json'} ({out['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
