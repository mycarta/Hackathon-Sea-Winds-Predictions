#!/usr/bin/env python3
"""WS d7 BLOCK REFIT — clean-split go/no-go gate (one arm per invocation).

Opus-authorized dispatch 2026-07-16 (blocks confirmed by Matteo). This REPLACES
the invalid `train_val_dates(seed=42)` protocol: that split was seeded-random
scattered (100% of val dates within 7 days of a training date) AND the arms were
fit on the full `train_dates('6D')`, so the fit saw the val neighbourhoods. A
re-SCORE is therefore insufficient — this script re-FITS FROM SCRATCH with block
exclusion.

DESIGN (confirmed):
  * VALIDATION BLOCKS: 4 contiguous 28-day blocks, one per season, spread across
    four different years (2016 untouched). Verified not to overlap any inference
    window: all 8 windows span 2021-01-01..2021-11-18 (context_start..predict_end),
    outside the 2016-2020 training years.
  * BUFFER: 14 days each side. Exclusion window = block +/- 14 d (56 d each,
    224 d total). Excluding FIT ISSUE DATES over that window provably removes
    every training sample whose target (lead 1/7/14) lands in a block: for a
    block valid date V, the issue date V-L lies in [block_start-14, block_end+14]
    for all L <= 14.
  * "Nothing within block+buffer enters training" is applied to EVERY fitted
    component, not just fit_forecast:
        - fit_forecast train dates          305 -> 267  (38 excluded)
            (its internals: quantile-MOS train year<2020 244->215; conformal
             calib year==2020 61->52)
        - downscaler train days 2020[::5]    74 ->  63  (11 excluded)
        - calibrate_intervals calib_dates    15 ->  13  ( 2 excluded)
      The downscaler matters because it trains on 2020 target truth and the SON
      block is in 2020; without this it would leak.
  * KNOWN IMMATERIAL EXCEPTION: the d14 climatology (`_climatology`,
    TRAIN_YEARS_CLIM, cached npz) pools all coarse files across train years and
    therefore includes block dates. d14 is NOT scored here (d7 primary, d1
    side-check) and the climatology never feeds d7/d1 (`coarse_fields` uses it
    for lead 14 only), so it changes no scored number. Flagged, not silently
    ignored.
  * BIAS TABLE: a fitted artifact, so by the same principle it may NOT be built
    from the eval blocks (that would reintroduce the leak this run removes). Each
    arm's per-cell x season K=30 table is built from a seeded (42), season-
    balanced sample of NON-EXCLUDED dates and APPLIED to the blocks — the
    faithful analogue of the real submission flow (bias estimated on seen data,
    applied to unseen 2021).
  * ARMS / alpha as configured, FROZEN (no retuning on the new split):
        A0 alpha=0.90, A2 alpha=0.90, A4 alpha=0.912.
  * SCORING: fine-primary d7 Winkler (bias + alpha) pooled over the 4 blocks and
    per-block; d7 coverage; d1 side-check (native downscaled interval, no bias/
    alpha — same treatment as the 2026-07-16 run).

The decision rule is pre-registered and NOT applied here: this script reports.
Nothing is submitted.
"""
from __future__ import annotations

import argparse
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
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
KIT_PART0 = ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"
KIT_PART1 = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
sys.path.insert(0, str(KIT_PART0))
sys.path.insert(0, str(KIT_PART1))
sys.path.insert(0, str(ROOT / "scripts"))

import footprint as fp_mod          # noqa: E402
import target_loader                # noqa: E402
import config                       # noqa: E402
import forecast_pipeline as P       # noqa: E402
import downscaling as dn            # noqa: E402
from ws_d7_feature_experiments import apply_arm, winkler_score  # noqa: E402

SEED = 42
HOURS = (0, 6, 12, 18)
K_SHRINK = 30.0
ALPHA_LEVEL = 0.10
ARM_ALPHA = {"A0": 0.90, "A2": 0.90, "A4": 0.912}     # as configured, FROZEN
BUFFER_DAYS = 14
BIAS_PER_SEASON = 40                                   # seeded sample size
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

BLOCKS = {                                             # confirmed 2026-07-16
    "DJF": ("2017-01-09", "2017-02-05"),
    "MAM": ("2018-04-09", "2018-05-06"),
    "JJA": ("2019-07-08", "2019-08-04"),
    "SON": ("2020-10-05", "2020-11-01"),
}


def block_days():
    out = {}
    for s, (a, b) in BLOCKS.items():
        A, B = pd.Timestamp(a).date(), pd.Timestamp(b).date()
        out[s] = [A + timedelta(days=i) for i in range((B - A).days + 1)]
    return out


def exclusion_set():
    excl = set()
    for s, (a, b) in BLOCKS.items():
        A = pd.Timestamp(a).date() - timedelta(days=BUFFER_DAYS)
        B = pd.Timestamp(b).date() + timedelta(days=BUFFER_DAYS)
        excl |= {A + timedelta(days=i) for i in range((B - A).days + 1)}
    return excl


def truth_cache_get(cache, V, ys, xs):
    """AROME truth speed at the footprint cells (identical to the 07-16 run)."""
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
    cache[V] = np.sqrt(u ** 2 + v ** 2)
    return cache[V]


def lead_blocks(dwn, mos, qmos, adj, offs, spd_infl, D):
    """Downscaled submission blocks for one issue date (all leads/hours)."""
    fields = P.coarse_fields(mos, qmos, adj, pd.Timestamp(D))
    return P.downscale_window(dwn, fields, offs, window=0,
                              spd_infl=spd_infl, dir_off=offs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A0", "A2", "A4"])
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check")
    args = ap.parse_args()
    arm = args.arm
    alpha = ARM_ALPHA[arm]
    t0 = time.time()

    excl = exclusion_set()
    bdays = block_days()

    prov = apply_arm(arm)
    print(f"[{arm}] features={prov['features_used']} target={prov['target']} "
          f"seed={SEED} alpha={alpha} (FROZEN)")

    # ---------- reduced fit sets (exclusion applied to every fitted component) ----------
    train_all = [d.date() for d in P.train_dates("6D")]
    train_red = [d for d in train_all if d not in excl]
    d2020_all = [d for d in target_loader.list_dates(config.target_root())
                 if d.year == 2020][::5]
    d2020_red = [d for d in d2020_all if d not in excl]
    calib_all = [pd.Timestamp(f"{y}-{m:02d}-15").date()
                 for y in (2016, 2017, 2018, 2019, 2020) for m in (2, 6, 10)]
    calib_red = [d for d in calib_all if d not in excl]
    print(f"[{arm}] fit train: {len(train_all)} -> {len(train_red)} "
          f"(excluded {len(train_all)-len(train_red)})")
    print(f"[{arm}] downscaler days: {len(d2020_all)} -> {len(d2020_red)}")
    print(f"[{arm}] calib_dates: {len(calib_all)} -> {len(calib_red)}")

    print(f"[{arm}] fit_forecast on reduced train ...")
    mos, qmos, adj, offs = P.fit_forecast(pd.to_datetime(train_red))
    print(f"[{arm}] train downscaler on reduced 2020 days ...")
    dwn = dn.train_downscaler(d2020_red, hours=HOURS)
    print(f"[{arm}] calibrate_intervals on reduced calib_dates ...")
    spd_infl, dir_off = P.calibrate_intervals(mos, qmos, adj, dwn, offs,
                                              calib_dates=pd.to_datetime(calib_red))
    print(f"[{arm}] spd_infl={spd_infl}")

    ys, xs = np.where(fp_mod.footprint_mask())
    n_cells = ys.size
    assert n_cells == 43715, n_cells
    truth_cache = {}

    # ---------- bias table from NON-EXCLUDED dates (never from the blocks) ----------
    all_targets = sorted(target_loader.list_dates(config.target_root()))
    target_set = set(all_targets)
    pool = {s: [] for s in SEASONS}
    for V in all_targets:
        if V.year not in (2016, 2017, 2018, 2019, 2020):
            continue
        D = V - timedelta(days=7)
        if V in excl or D in excl:          # bias never sees block+buffer
            continue
        pool[SEASON_OF_MONTH[V.month]].append(V)
    rng = np.random.default_rng(SEED)
    n_per = 2 if args.smoke else BIAS_PER_SEASON
    bias_dates = {}
    for s in SEASONS:
        cand = pool[s]
        pick = rng.choice(len(cand), size=min(n_per, len(cand)), replace=False)
        bias_dates[s] = sorted(cand[i] for i in pick)
    print(f"[{arm}] bias-table sample (non-excluded): "
          f"{ {s: len(v) for s, v in bias_dates.items()} }")

    bias_sum = {s: np.zeros(n_cells) for s in SEASONS}
    bias_n = {s: 0 for s in SEASONS}
    for s in SEASONS:
        for V in bias_dates[s]:
            D = V - timedelta(days=7)
            truth_all = truth_cache_get(truth_cache, V, ys, xs)
            if truth_all is None:
                continue
            blks = lead_blocks(dwn, mos, qmos, adj, offs, spd_infl, D)
            base = P.LEADS.index(7) * len(HOURS)
            for h_idx in range(len(HOURS)):
                blk = blks[base + h_idx]
                q50 = blk["q50"].to_numpy(dtype="float64")
                bias_sum[s] += (q50 - truth_all[h_idx].astype("float64"))
                bias_n[s] += 1
    bias_shrunk = {}
    bias_meta = {}
    for s in SEASONS:
        if bias_n[s] == 0:
            bias_shrunk[s] = np.zeros(n_cells)
            bias_meta[s] = {"n_blocks": 0}
            continue
        raw = bias_sum[s] / bias_n[s]
        smean = float(raw.mean())
        w = bias_n[s] / (bias_n[s] + K_SHRINK)
        bias_shrunk[s] = w * raw + (1 - w) * smean
        bias_meta[s] = {"n_blocks": bias_n[s], "w": w, "season_mean_bias": smean}
        print(f"[{arm}]   bias {s}: n={bias_n[s]} w={w:.4f} mean={smean:.4f}")

    # ---------- evaluate on the blocks ----------
    per_block = {}
    stacked = {}
    d1_acc = {"wink": 0.0, "cov": 0, "width": 0.0, "n": 0}
    for s in SEASONS:
        days = [V for V in bdays[s] if V in target_set]
        if args.smoke:
            days = days[:2]
        b = {"wink": 0.0, "cov": 0, "width": 0.0, "n": 0, "n_dates": 0}
        q05L, q50L, q95L, trL = [], [], [], []
        for V in days:
            truth_all = truth_cache_get(truth_cache, V, ys, xs)
            if truth_all is None:
                continue
            # --- d7 primary (bias + alpha) ---
            blks = lead_blocks(dwn, mos, qmos, adj, offs, spd_infl, V - timedelta(days=7))
            base = P.LEADS.index(7) * len(HOURS)
            bb = bias_shrunk[s]
            for h_idx in range(len(HOURS)):
                blk = blks[base + h_idx]
                q05 = np.maximum(0.0, blk["q05"].to_numpy(dtype="float64") - bb)
                q50 = blk["q50"].to_numpy(dtype="float64") - bb
                q95 = blk["q95"].to_numpy(dtype="float64") - bb
                t = truth_all[h_idx].astype("float64")
                lo = np.maximum(0.0, q50 - alpha * (q50 - q05))
                hi = q50 + alpha * (q95 - q50)
                b["wink"] += float(winkler_score(t, lo, hi).sum())
                b["width"] += float((hi - lo).sum())
                b["cov"] += int(((t >= lo) & (t <= hi)).sum())
                b["n"] += t.size
                q05L.append(q05.astype("float32")); q50L.append(q50.astype("float32"))
                q95L.append(q95.astype("float32")); trL.append(t.astype("float32"))
            b["n_dates"] += 1
            # --- d1 side-check (native interval: spd_infl, no bias/alpha) ---
            blks1 = lead_blocks(dwn, mos, qmos, adj, offs, spd_infl, V - timedelta(days=1))
            base1 = P.LEADS.index(1) * len(HOURS)
            for h_idx in range(len(HOURS)):
                blk = blks1[base1 + h_idx]
                q05 = blk["q05"].to_numpy(dtype="float64")
                q95 = blk["q95"].to_numpy(dtype="float64")
                t = truth_all[h_idx].astype("float64")
                d1_acc["wink"] += float(winkler_score(t, q05, q95).sum())
                d1_acc["cov"] += int(((t >= q05) & (t <= q95)).sum())
                d1_acc["width"] += float((q95 - q05).sum())
                d1_acc["n"] += t.size
        per_block[s] = {
            "winkler": b["wink"] / b["n"] if b["n"] else None,
            "coverage": b["cov"] / b["n"] if b["n"] else None,
            "mean_width": b["width"] / b["n"] if b["n"] else None,
            "n": b["n"], "n_dates": b["n_dates"],
            "block": [BLOCKS[s][0], BLOCKS[s][1]],
        }
        if q50L and not args.smoke:
            stacked[f"{s}_q05"] = np.stack(q05L); stacked[f"{s}_q50"] = np.stack(q50L)
            stacked[f"{s}_q95"] = np.stack(q95L); stacked[f"{s}_truth"] = np.stack(trL)
        print(f"[{arm}] block {s}: winkler={per_block[s]['winkler']:.4f} "
              f"cov={per_block[s]['coverage']:.4f} n_dates={b['n_dates']}")

    # pooled over the 4 blocks (sample-weighted = simple pooling of all cells)
    tot_n = sum(per_block[s]["n"] for s in SEASONS)
    pooled_w = sum(per_block[s]["winkler"] * per_block[s]["n"] for s in SEASONS) / tot_n
    pooled_c = sum(per_block[s]["coverage"] * per_block[s]["n"] for s in SEASONS) / tot_n
    pooled_wid = sum(per_block[s]["mean_width"] * per_block[s]["n"] for s in SEASONS) / tot_n
    d1 = ({"winkler": d1_acc["wink"] / d1_acc["n"], "coverage": d1_acc["cov"] / d1_acc["n"],
           "mean_width": d1_acc["width"] / d1_acc["n"], "n": d1_acc["n"]}
          if d1_acc["n"] else None)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if stacked:
        np.savez_compressed(ARTIFACTS_DIR / f"ws_d7_blockrefit_{arm}_stacked.npz", **stacked)
    result = {
        "arm": arm, "alpha": alpha, "seed": SEED, "smoke": bool(args.smoke),
        "provenance": prov,
        "blocks": {s: list(BLOCKS[s]) for s in SEASONS},
        "buffer_days": BUFFER_DAYS,
        "fit_counts": {"train_total": len(train_all), "train_kept": len(train_red),
                       "downscaler_total": len(d2020_all), "downscaler_kept": len(d2020_red),
                       "calib_total": len(calib_all), "calib_kept": len(calib_red)},
        "spd_infl": {str(k): v for k, v in spd_infl.items()},
        "bias_source": "non-excluded dates, seeded season-balanced sample",
        "bias_meta": bias_meta,
        "d7_pooled": {"winkler": pooled_w, "coverage": pooled_c,
                      "mean_width": pooled_wid, "n": tot_n},
        "d7_per_block": per_block,
        "d1_native": d1,
        "elapsed_sec": time.time() - t0,
    }
    out = ARTIFACTS_DIR / (f"ws_d7_blockrefit_{arm}" + ("_smoke" if args.smoke else "") + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[{arm}] wrote {out}")
    print(f"[{arm}] DONE {result['elapsed_sec']:.0f}s | d7 pooled Winkler={pooled_w:.4f} "
          f"cov={pooled_c:.4f} | d1={d1['winkler']:.4f}")


if __name__ == "__main__":
    main()
