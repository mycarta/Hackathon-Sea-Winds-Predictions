#!/usr/bin/env python3
"""WS d7 speed — Tier-1 feature experiments (single arm per invocation).

Opus-authorized dispatch, 2026-07-16 (Part A). Runs ONE experiment arm and
writes its holdout metrics + a stacked-array checkpoint. The driver
``ws_d7_feature_experiments_driver.py`` orchestrates all arms and collates the
results table.

WHAT "THE d7 SPEED MODEL" ACTUALLY IS (verified against the kit, not memory):
  The board 27.14 d7-speed result is produced by the HRES-MOS pipeline in
  phase_2/kit/phase_2/part1_forecast, NOT a single CatBoost model. It is TWO
  LightGBM models feeding one downscaled interval:
    * point forecast q50 = |downscale(deterministic u/v MOS)|
        forecast_pipeline.py:112-116, forecast_hres.train_mos:250-261
    * interval q05/q95 = quantile-speed MOS, carried to the fine grid only as
      RATIOS r=cq05/cq50, cq95/cq50   (forecast_pipeline._speed_interval:81-96;
      forecast_hres.train_quantile_mos:272-288)
  Both models read the module-global forecast_hres.FEATURES (single-height HRES
  forecast winds): FEATURES = [fcst_u, fcst_v, fcst_speed, lat, lon, woy_sin,
  woy_cos]  (forecast_hres.py:166-167).

ARMS (A1 shear is a documented SKIP — the HRES forecast parquet is single-height,
  cols fcst_speed_d{L}_h{H}/fcst_dir_d{L}_h{H} only; the two-height winds
  u10/v10/u100/v100 are reanalysis ANALYSIS, unavailable at valid time D+L, so a
  shear feature is not computable at inference — A1 instruction: report & skip):
    A0  baseline (kit FEATURES), SEEDED. The comparator for every delta.
    A2  + static per-cell features: water depth, distance-to-coast
        (scripts/task2_depth_sampler.py, reused not rewritten).
    A3  residual-target reformulation: retarget BOTH models against the
        interpolated coarse forecast — det -> (u125c-fcst_u),(v125c-fcst_v);
        quantile-speed -> (speed-fcst_speed) — reconstruct absolute at inference.
    A3b det-only residual (decomposition; run by driver only if A3 wins).
    A4  combination of A2/A3 components that individually beat A0 (driver-chosen).

PROTOCOL (mirrors the established WS-d7 holdout scripts, e.g.
  scripts/ws_d7_bias_correction_build_table.py):
    * split = splits.train_val_dates(seed=42) — the same 80/20 split every
      prior d7 experiment used. Model fit on P.train_dates('6D') (2016-2020, the
      established fit set); holdout = issue dates whose valid date lands in the
      val set. NOTE the fit set and the holdout share years (the established
      protocol trains on the 6-day grid over all of 2016-2020) — this is a weak
      (leaky) holdout by construction; it is IDENTICAL for every arm, so
      arm-vs-A0 deltas are clean. Holdout gains do NOT guarantee eval gains
      (documented calibration lesson) — board 27.14 is context, not baseline.
    * alpha FIXED at 0.90 (no retune) so deltas isolate the feature effect.
    * SEEDED: random_state=42 on every LightGBM fit (the kit is unseeded,
      forecast_hres.py:278-279). A0 here is the seeded baseline; its absolute
      number may differ marginally from an unseeded board rerun.

TWO SCORING ALTITUDES (both reported):
    COARSE  (clean feature signal): the quantile-MOS's own coarse-grid interval
            (with conformal adj), native alpha. Directly measures the feature's
            effect on the model being modified.
    FINE    (board-faithful): the full downscale + spd_infl + per-cell x season
            bias (K=30, refit on this arm's holdout) + alpha=0.90 recipe, over
            the 43,715-point footprint. d7 primary; d1 side-check; d14 is the
            +14 climatology path (no FEATURES dependency: forecast_pipeline
            coarse_fields:65-71) => feature-invariant, computed in A0 only.

No kit file is edited; features are injected by monkeypatching forecast_hres
attributes inside this script. Nothing is submitted. Outputs go to
scripts/artifacts/ws_d7_featexp_<arm>.json and ..._<arm>_stacked.npz.
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
import splits                       # noqa: E402
import target_loader                # noqa: E402
import config                       # noqa: E402
import forecast_hres as fh          # noqa: E402
import forecast_pipeline as P       # noqa: E402
import downscaling as dn            # noqa: E402
import task2_depth_sampler as depth_sampler  # noqa: E402

SEED = 42
HOURS = (0, 6, 12, 18)
LEAD_PRIMARY = 7
K_SHRINK = 30.0
ALPHA_LEVEL = 0.10           # Winkler interval level (fixed by the metric)
ALPHA_FIX = 0.90             # interval-width multiplier, FIXED per dispatch
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASONS = ("DJF", "MAM", "JJA", "SON")

_LGBM_BASE = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                  subsample=0.8, colsample_bytree=0.8, verbose=-1,
                  random_state=SEED)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def winkler_score(y, lo, hi, alpha_level=ALPHA_LEVEL):
    width = hi - lo
    below = np.maximum(0.0, lo - y) * (2.0 / alpha_level)
    above = np.maximum(0.0, y - hi) * (2.0 / alpha_level)
    return width + below + above


# ---------------------------------------------------------------------------
# static (per-cell) features for A2 / A4  — reuse task2_depth_sampler, no rewrite
# ---------------------------------------------------------------------------
def static_features(lat, lon):
    """(water_depth_m, dist_coast_km) at nearest EMODnet pixel; NaN off-coverage.

    depth uses the sampler's vectorized depth_at_array; dist reuses the sampler's
    cached dataset (_load) with the identical nearest-pixel .sel pattern.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    depth = depth_sampler.depth_at_array(lat, lon)
    ds = depth_sampler._load()
    dist = ds["dist_coast_km"].sel(
        lat=xr.DataArray(lat, dims="points"),
        lon=xr.DataArray(lon, dims="points"),
        method="nearest",
    ).values.astype(float)
    dist = np.where(np.isfinite(dist), dist, np.nan)
    return depth, dist


# ---------------------------------------------------------------------------
# arm setup: monkeypatch forecast_hres (kit files untouched)
# ---------------------------------------------------------------------------
def apply_arm(arm: str) -> dict:
    """Install the arm's feature/target modifications onto the fh module.

    Returns a provenance dict describing exactly what was changed.
    """
    orig_build = fh.build_hres_table
    orig_features = list(fh.FEATURES)
    prov = {"arm": arm, "base_features": orig_features, "seed": SEED,
            "target": "absolute", "extra_features": []}

    static = arm in ("A2", "A4")
    residual = arm in ("A3", "A4")
    residual_det_only = arm == "A3b"

    # ---- static features: extend FEATURES + wrap build_hres_table ----
    if static:
        new_features = orig_features + ["depth", "dist_coast"]
        prov["extra_features"] = ["depth", "dist_coast"]

        def build_with_static(issue_dates, hours=fh.HOURS, leads=fh.HRES_LEADS,
                              with_truth=True):
            df = orig_build(issue_dates, hours=hours, leads=leads,
                            with_truth=with_truth)
            if len(df):
                d, c = static_features(df["lat"].to_numpy(), df["lon"].to_numpy())
                df["depth"] = d
                df["dist_coast"] = c
            return df

        fh.build_hres_table = build_with_static
        fh.FEATURES = new_features

    feats = list(fh.FEATURES)

    # ---- seeded / residual training + prediction ----
    import lightgbm as lgb

    if residual or residual_det_only:
        prov["target"] = "residual_both" if residual else "residual_det_only"

    def train_mos(train_df, leads=fh.HRES_LEADS, **kw):
        params = dict(_LGBM_BASE, **kw)
        models = {}
        for L in leads:
            sub = train_df[train_df["lead"] == L]
            X = sub[feats]
            yu, yv = sub["u125c"].to_numpy(), sub["v125c"].to_numpy()
            if residual or residual_det_only:                # det residual vs coarse fcst
                yu = yu - sub["fcst_u"].to_numpy()
                yv = yv - sub["fcst_v"].to_numpy()
            models[(L, "u")] = lgb.LGBMRegressor(**params).fit(X, yu)
            models[(L, "v")] = lgb.LGBMRegressor(**params).fit(X, yv)
        return models

    def predict_mos(models, df):
        out = df.copy()
        out["u_pred"] = np.nan
        out["v_pred"] = np.nan
        for L in sorted(df["lead"].unique()):
            m = (df["lead"] == L).to_numpy()
            pu = models[(L, "u")].predict(df.loc[m, feats])
            pv = models[(L, "v")].predict(df.loc[m, feats])
            if residual or residual_det_only:
                pu = pu + df.loc[m, "fcst_u"].to_numpy()
                pv = pv + df.loc[m, "fcst_v"].to_numpy()
            out.loc[m, "u_pred"] = pu
            out.loc[m, "v_pred"] = pv
        return out

    def train_quantile_mos(train_df, leads=fh.HRES_LEADS, quantiles=fh.QUANTILES, **kw):
        params = dict(_LGBM_BASE, **kw)
        models = {}
        for L in leads:
            sub = train_df[train_df["lead"] == L]
            X = sub[feats]
            y = np.hypot(sub["u125c"].to_numpy(), sub["v125c"].to_numpy())
            if residual:                                     # speed residual vs coarse fcst
                y = y - sub["fcst_speed"].to_numpy()         # (A3b keeps quantile absolute)
            for q in quantiles:
                models[(L, q)] = lgb.LGBMRegressor(
                    objective="quantile", alpha=q, **params).fit(X, y)
        return models

    def predict_quantile_mos(models, df, quantiles=fh.QUANTILES, adjust=None):
        out = df.copy()
        qs = sorted(quantiles)
        cols = [fh._qcol(q) for q in qs]
        for c in cols:
            out[c] = np.nan
        for L in sorted(df["lead"].unique()):
            m = (df["lead"] == L).to_numpy()
            if not m.any():
                continue
            X = df.loc[m, feats]
            base = df.loc[m, "fcst_speed"].to_numpy() if residual else 0.0
            for q, c in zip(qs, cols):
                out.loc[m, c] = models[(L, q)].predict(X) + base
            if adjust and L in adjust:
                out.loc[m, cols[0]] = out.loc[m, cols[0]] - adjust[L]
                out.loc[m, cols[-1]] = out.loc[m, cols[-1]] + adjust[L]
        vals = np.sort(out[cols].to_numpy(float), axis=1)
        out[cols] = np.clip(vals, 0.0, None)
        return out

    fh.train_mos = train_mos
    fh.predict_mos = predict_mos
    fh.train_quantile_mos = train_quantile_mos
    fh.predict_quantile_mos = predict_quantile_mos
    prov["features_used"] = feats
    return prov


# ---------------------------------------------------------------------------
# AROME truth loader (identical to ws_d7_bias_correction_build_table.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# COARSE-altitude eval (clean feature signal): quantile-MOS's own coarse interval
# ---------------------------------------------------------------------------
def coarse_eval(qmos, adj, val_set, leads=(1, 7)):
    out = {}
    for L in leads:
        issue = sorted({V - timedelta(days=L) for V in val_set})
        tbl = fh.build_hres_table(issue, leads=(L,), with_truth=True)
        if not len(tbl):
            out[L] = None
            continue
        pred = fh.predict_quantile_mos(qmos, tbl, adjust=adj)
        truth = np.hypot(tbl["u125c"].to_numpy(), tbl["v125c"].to_numpy())
        lo = pred["spd_q05"].to_numpy()
        hi = pred["spd_q95"].to_numpy()
        cov = float(np.mean((truth >= lo) & (truth <= hi)))
        out[L] = {"winkler": float(winkler_score(truth, lo, hi).mean()),
                  "coverage": cov, "mean_width": float((hi - lo).mean()),
                  "n_rows": int(len(tbl))}
    return out


# ---------------------------------------------------------------------------
# main (one arm)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["A0", "A2", "A3", "A3b", "A4"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run for plumbing validation (NOT a deliverable)")
    ap.add_argument("--smoke-n", type=int, default=6)
    args = ap.parse_args()
    arm = args.arm
    t0 = time.time()

    prov = apply_arm(arm)
    print(f"[{arm}] features={prov['features_used']} target={prov['target']} seed={SEED}")

    # ---- fit the arm's pipeline on the established fit set ----
    print(f"[{arm}] fit_forecast(train_dates('6D')) ...")
    train = P.train_dates("6D")
    mos, qmos, adj, offs = P.fit_forecast(train)
    print(f"[{arm}] train downscaler (2020[::5]) ...")
    d2020 = [d for d in target_loader.list_dates(config.target_root())
             if d.year == 2020][::5]
    dwn = dn.train_downscaler(d2020, hours=HOURS)
    print(f"[{arm}] calibrate intervals ...")
    spd_infl, dir_off = P.calibrate_intervals(mos, qmos, adj, dwn, offs)
    print(f"[{arm}] spd_infl={spd_infl}")

    # ---- split ----
    _, val_dates = splits.train_val_dates(seed=SEED)
    val_set = set(val_dates)
    print(f"[{arm}] val dates: {len(val_set)}")

    # ---- COARSE eval ----
    print(f"[{arm}] coarse-altitude eval ...")
    coarse = coarse_eval(qmos, adj, val_set, leads=(1, 7))
    for L in (1, 7):
        if coarse[L]:
            print(f"[{arm}]   coarse d{L}: winkler={coarse[L]['winkler']:.4f} "
                  f"cov={coarse[L]['coverage']*100:.1f}% n={coarse[L]['n_rows']}")

    # ---- FINE eval (downscale holdout loop) ----
    ys, xs = np.where(fp_mod.footprint_mask())
    n_cells = ys.size
    assert n_cells == 43715, n_cells

    leads_eval = (1, 7, 14) if arm == "A0" else (1, 7)   # d14 invariant -> A0 only
    cand = sorted(d for d in target_loader.list_dates(config.target_root())
                  if d.year in (2016, 2017, 2018, 2019, 2020))
    if args.smoke:
        # keep only enough issue dates to exercise every lead a few times
        keep = set()
        for L in leads_eval:
            got = [D for D in cand if (D + timedelta(days=L)) in val_set]
            keep.update(got[:args.smoke_n])
        cand = sorted(keep)
        print(f"[{arm}] SMOKE: {len(cand)} issue dates")

    # d7 season blocks (for bias table); d1/d14 running Winkler sums
    season_blocks = {s: {"q05": [], "q50": [], "q95": [], "truth": []} for s in SEASONS}
    side = {L: {"wink_sum": 0.0, "cov": 0, "width_sum": 0.0, "n": 0}
            for L in (1, 14)}
    truth_cache = {}
    n_dates = {L: 0 for L in leads_eval}

    for D in cand:
        needed = [L for L in leads_eval if (D + timedelta(days=L)) in val_set]
        if not needed:
            continue
        try:
            fields = P.coarse_fields(mos, qmos, adj, pd.Timestamp(D))
        except Exception as e:
            print(f"[{arm}]   skip {D}: coarse_fields failed ({e})")
            continue
        blocks = P.downscale_window(dwn, fields, offs, window=0,
                                    spd_infl=spd_infl, dir_off=dir_off)
        for L in needed:
            V = D + timedelta(days=L)
            truth_all = truth_cache_get(truth_cache, V, ys, xs)
            if truth_all is None:
                continue
            base = P.LEADS.index(L) * len(HOURS)
            n_dates[L] += 1
            for h_idx, H in enumerate(HOURS):
                blk = blocks[base + h_idx]
                assert int(blk["horizon"].iloc[0]) == L and int(blk["hour"].iloc[0]) == H
                q05 = blk["q05"].to_numpy(dtype="float32")
                q50 = blk["q50"].to_numpy(dtype="float32")
                q95 = blk["q95"].to_numpy(dtype="float32")
                truth = truth_all[h_idx].astype("float32")
                if L == LEAD_PRIMARY:
                    s = SEASON_OF_MONTH[V.month]
                    season_blocks[s]["q05"].append(q05)
                    season_blocks[s]["q50"].append(q50)
                    season_blocks[s]["q95"].append(q95)
                    season_blocks[s]["truth"].append(truth)
                else:  # d1 / d14 side-check: native downscaled interval (spd_infl, no bias)
                    acc = side[L]
                    acc["wink_sum"] += float(winkler_score(truth, q05, q95).sum())
                    acc["cov"] += int(((truth >= q05) & (truth <= q95)).sum())
                    acc["width_sum"] += float((q95 - q05).sum())
                    acc["n"] += truth.size

    # ---- d7 fine: build per-cell x season bias (K=30), apply, then alpha=0.90 ----
    stacked = {}
    per_cell_season_shrunk = np.full((n_cells, len(SEASONS)), np.nan)
    season_meta = {}
    gbias_sum, gbias_n = 0.0, 0
    for si, s in enumerate(SEASONS):
        sb = season_blocks[s]
        nb = len(sb["q50"])
        if nb == 0:
            season_meta[s] = {"n_blocks": 0}
            continue
        q05a = np.stack(sb["q05"]); q50a = np.stack(sb["q50"])
        q95a = np.stack(sb["q95"]); tra = np.stack(sb["truth"])
        stacked[s] = {"q05": q05a, "q50": q50a, "q95": q95a, "truth": tra}
        bias = q50a - tra
        bias_raw_cell = bias.mean(axis=0)
        season_mean = float(bias.mean())
        w = nb / (nb + K_SHRINK)
        per_cell_season_shrunk[:, si] = w * bias_raw_cell + (1 - w) * season_mean
        season_meta[s] = {"n_blocks": nb, "w": w, "season_mean_bias": season_mean}
        gbias_sum += float(bias.sum()); gbias_n += bias.size

    # d7 primary Winkler at alpha=0.90 (bias-corrected) + a pre-bias/native-alpha reference
    n_tot = cov_p = 0
    wink_p = width_p = 0.0
    n_raw = cov_raw = 0
    wink_raw = 0.0
    for si, s in enumerate(SEASONS):
        if s not in stacked:
            continue
        b = per_cell_season_shrunk[:, si][None, :]
        q05 = np.maximum(0.0, stacked[s]["q05"] - b)
        q50 = stacked[s]["q50"] - b
        q95 = stacked[s]["q95"] - b
        truth = stacked[s]["truth"]
        lo = np.maximum(0.0, q50 - ALPHA_FIX * (q50 - q05))
        hi = q50 + ALPHA_FIX * (q95 - q50)
        wink_p += float(winkler_score(truth, lo, hi).sum())
        width_p += float((hi - lo).sum())
        cov_p += int(((truth >= lo) & (truth <= hi)).sum())
        n_tot += truth.size
        # raw reference: no bias, native alpha (=1.0)
        r05, r95 = stacked[s]["q05"], stacked[s]["q95"]
        wink_raw += float(winkler_score(truth, r05, r95).sum())
        cov_raw += int(((truth >= r05) & (truth <= r95)).sum())
        n_raw += truth.size

    fine = {
        "d7_primary_bias_alpha090": {
            "winkler": wink_p / n_tot if n_tot else None,
            "coverage": cov_p / n_tot if n_tot else None,
            "mean_width": width_p / n_tot if n_tot else None,
            "n": n_tot, "n_issue_dates": n_dates.get(7, 0)},
        "d7_raw_prebias_nativealpha": {
            "winkler": wink_raw / n_raw if n_raw else None,
            "coverage": cov_raw / n_raw if n_raw else None, "n": n_raw},
        "d1_native": None,
        "d14_native": None,
    }
    for L, key in ((1, "d1_native"), (14, "d14_native")):
        acc = side[L]
        if acc["n"]:
            fine[key] = {"winkler": acc["wink_sum"] / acc["n"],
                         "coverage": acc["cov"] / acc["n"],
                         "mean_width": acc["width_sum"] / acc["n"],
                         "n": acc["n"], "n_issue_dates": n_dates.get(L, 0)}

    # ---- checkpoint stacked d7 arrays (build discipline #5) ----
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    npz = {}
    for s in stacked:
        for k, v in stacked[s].items():
            npz[f"{s}_{k}"] = v
    npz_path = ARTIFACTS_DIR / f"ws_d7_featexp_{arm}_stacked.npz"
    if not args.smoke:
        np.savez_compressed(npz_path, **npz)

    result = {
        "arm": arm,
        "provenance": prov,
        "smoke": bool(args.smoke),
        "spd_infl": {str(k): v for k, v in spd_infl.items()},
        "dir_off": {str(k): v for k, v in dir_off.items()},
        "n_val_dates": len(val_set),
        "global_mean_d7_bias": gbias_sum / gbias_n if gbias_n else None,
        "season_meta": season_meta,
        "alpha_fixed": ALPHA_FIX,
        "coarse": {f"d{L}": coarse[L] for L in (1, 7)},
        "fine": fine,
        "elapsed_sec": time.time() - t0,
    }
    out_path = ARTIFACTS_DIR / (f"ws_d7_featexp_{arm}"
                                + ("_smoke" if args.smoke else "") + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[{arm}] wrote {out_path}")
    d7 = fine["d7_primary_bias_alpha090"]["winkler"]
    print(f"[{arm}] DONE in {result['elapsed_sec']:.0f}s | "
          f"fine d7 (bias,a=0.90) Winkler = {d7:.4f} | "
          f"coarse d7 Winkler = "
          f"{coarse[7]['winkler'] if coarse[7] else float('nan'):.4f}")


if __name__ == "__main__":
    main()
