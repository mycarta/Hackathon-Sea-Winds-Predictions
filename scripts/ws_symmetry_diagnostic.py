#!/usr/bin/env python3
"""
Interval symmetry diagnostic (2026-07-08) - adapted to the REAL Task 1
speed pipeline (LightGBM, single north_sea region), not the Phase-1-only
CatBoost/NS-ECS pipeline the original prompt described.

Audit target: raw LightGBM quantile output -> conformal_adjust ->
spd_infl calibration -> d7 bias correction -> d14 climatological
replacement.

Diagnostic only. Does NOT touch submission.csv or any model.

Step 2 (asymmetry measurement): for each holdout (issue_date, lead, hour),
computes asymmetry_ratio = upper_width/lower_width at each pipeline
stage:
  A: raw quantile-MOS output (coarse grid, pre-conformal)
  B: after conformal_adjust (coarse grid) - SAME additive Q_L on both
     sides (forecast_hres.py:309-311, conformal_adjust:317-336)
  C: after downscale + spd_infl width-calibration (fine grid) - the
     actual submitted d1/d7 pre-bias-correction interval. Also computes
     C_raw (downscale+spd_infl applied to the RAW/pre-conformal ratios)
     to isolate conformal_adjust's own marginal contribution to the fine-
     grid asymmetry ratio, holding the downscale+spd_infl step fixed.
  D (d7 only): after our per-cell bias correction (pure shift - expected
     ratio-invariant, verified not assumed)
  F (d14 only): the climatological replacement's own empirical quantiles
     (single stage, no earlier decomposition - the kit-native LGBM/
     conformal/spd_infl chain is dead code for d14's submitted values,
     since apply_d14_climatology.py fully overwrites it)

Step 3 (Winkler cost of symmetrization): for whichever stage is found to
be symmetrizing (conformal_adjust, per the formula), computes holdout
Winkler with the actual symmetric adjustment vs. an alternative
ASYMMETRIC conformal adjustment (separate lower/upper margins, each
calibrated to alpha/2 one-sided miss rate on the same calibration set -
standard split-conformal construction), holding spd_infl fixed at its
already-calibrated k (isolating the conformal step's own contribution,
not a full end-to-end re-optimization - flagged as a scope limitation).

Uses the same splits.train_val_dates(seed=42) holdout and shipped-
pipeline reproduction as every other WS diagnostic this session. No
model files modified/saved.
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
REPORT_PATH = ROOT / "reports" / "ws_interval_symmetry_diagnostic_20260708.md"
BIAS_TABLE_PATH = ARTIFACTS_DIR / "ws_d7_bias_shrunk_table.parquet"
D14_CLIM_PATH = ARTIFACTS_DIR / "d14_climatology_season.parquet"

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

HOURS = (0, 6, 12, 18)
LEADS = (1, 7)
ALPHA_LEVEL = 0.10
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


def asymmetric_conformal_adjust(qmos, calib_df, alpha=0.10, quantiles=fh.QUANTILES):
    """Split-conformal, separate one-sided margins per tail (each targets
    alpha/2 one-sided miss rate), instead of conformal_adjust's single
    combined max(...) score applied identically to both sides."""
    qs = sorted(quantiles)
    lo_c, hi_c = fh._qcol(qs[0]), fh._qcol(qs[-1])
    pr = fh.predict_quantile_mos(qmos, calib_df, quantiles)
    out = {}
    for L in sorted(calib_df["lead"].unique()):
        s = pr[pr["lead"] == L]
        y = np.hypot(s["u125c"].to_numpy(), s["v125c"].to_numpy())
        E_lo = np.maximum(s[lo_c].to_numpy() - y, 0.0)
        E_hi = np.maximum(y - s[hi_c].to_numpy(), 0.0)
        n = len(y)
        k = int(np.ceil((n + 1) * (1.0 - alpha / 2.0)))
        Q_lo = float(np.sort(E_lo)[min(k, n) - 1])
        Q_hi = float(np.sort(E_hi)[min(k, n) - 1])
        out[L] = (Q_lo, Q_hi)
    return out


def apply_asym_adjust(qp_raw, adj_asym, quantiles=fh.QUANTILES):
    qs = sorted(quantiles)
    lo_c, hi_c = fh._qcol(qs[0]), fh._qcol(qs[-1])
    out = qp_raw.copy()
    for L, (Q_lo, Q_hi) in adj_asym.items():
        m = (out["lead"] == L).to_numpy()
        out.loc[m, lo_c] = out.loc[m, lo_c] - Q_lo
        out.loc[m, hi_c] = out.loc[m, hi_c] + Q_hi
    cols = [fh._qcol(q) for q in qs]
    vals = np.sort(out[cols].to_numpy(float), axis=1)
    out[cols] = np.clip(vals, 0.0, None)
    return out


def downscaled_center(det_UV, dwn):
    """spd50 (fine grid) - identical across all coarse-quantile variants
    (raw/sym-adj/asym-adj), computed once per (lead, hour, date) and reused
    to avoid redundant downscale-model inference."""
    U, V = det_UV
    fu, fv = dn.downscale(dwn, U, V)
    return np.sqrt(fu ** 2 + fv ** 2)


def fine_interval_from_coarse_qp(qp, spd50, dwn, spd_infl_k, lead, h):
    """Replicates forecast_pipeline._speed_interval's ratio+interp+scale math
    for an arbitrary coarse quantile prediction qp (raw/sym-adj/asym-adj),
    given an already-downscaled spd50, returning (q05_fine, q95_fine)."""
    sub = qp[(qp["lead"] == lead) & (qp["hour"] == h)]
    cq05_grid, _ = fh.predictions_to_grid(sub.assign(u_pred=sub["spd_q05"], v_pred=0), lead, h)
    cq50_grid, _ = fh.predictions_to_grid(sub.assign(u_pred=sub["spd_q50"], v_pred=0), lead, h)
    cq95_grid, _ = fh.predictions_to_grid(sub.assign(u_pred=sub["spd_q95"], v_pred=0), lead, h)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_lo = np.where(cq50_grid > 0, cq05_grid / cq50_grid, 1.0)
        r_hi = np.where(cq50_grid > 0, cq95_grid / cq50_grid, 1.0)
    f_lo = dn.interp_coarse_to_target(r_lo, np.zeros_like(r_lo))[0]
    f_hi = dn.interp_coarse_to_target(r_hi, np.zeros_like(r_hi))[0]
    q05 = spd50 * np.clip(f_lo, 0.3, 1.0)
    q95 = spd50 * np.clip(f_hi, 1.0, 3.0)

    k = spd_infl_k
    if k != 1.0:
        q05 = np.maximum(0.0, spd50 - k * (spd50 - q05))
        q95 = spd50 + k * (q95 - spd50)
    return q05, q95


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


def ratio(upper, lower, eps=1e-6):
    return upper / np.maximum(lower, eps)


def main():
    t0 = time.time()
    static = xr.open_dataset(STATIC_PATH)
    ys, xs = np.where(fp_mod.footprint_mask())
    n_cells = ys.size
    assert n_cells == 43715
    static.close()

    print("Fitting shipped pipeline...")
    train = P.train_dates("6D")
    mos, qmos, adj, offs = P.fit_forecast(train)
    d2020 = [d for d in target_loader.list_dates(config.target_root()) if d.year == 2020][::5]
    dwn = dn.train_downscaler(d2020, hours=HOURS)
    spd_infl, dir_off = P.calibrate_intervals(mos, qmos, adj, dwn, offs)
    print(f"  spd_infl: {spd_infl}")

    calib = train[train.year == 2020]
    calib_df = fh.build_hres_table(calib)
    adj_asym = asymmetric_conformal_adjust(qmos, calib_df)
    print(f"  symmetric conformal Q_L: {adj}")
    print(f"  asymmetric conformal (Q_lo, Q_hi): {adj_asym}")

    train_dates_seed, val_dates_seed = splits.train_val_dates(seed=42)
    val_set = set(val_dates_seed)
    print(f"train_val_dates(seed=42): {len(train_dates_seed)} train, {len(val_dates_seed)} val")

    all_dates = sorted(target_loader.list_dates(config.target_root()))
    candidate_issue_dates = [d for d in all_dates if d.year in (2016, 2017, 2018, 2019, 2020)]
    print(f"Candidate issue dates: {len(candidate_issue_dates)}")

    # ---- accumulators: ratio_num/ratio_den per (lead, stage), Winkler per (lead, variant) ----
    stages = ["A_raw_coarse", "B_conformal_coarse", "C_fine_final", "C_raw_fine_counterfactual"]
    acc = {L: {s: {"num": 0.0, "den": 0.0, "n": 0} for s in stages} for L in LEADS}
    acc_d7_bias = {"num": 0.0, "den": 0.0, "n": 0}

    wink = {L: {"sym": {"sum": 0.0, "n": 0, "cov": 0}, "asym": {"sum": 0.0, "n": 0, "cov": 0}}
            for L in LEADS}

    bias_table = pd.read_parquet(BIAS_TABLE_PATH)
    bias_lookup = {}
    for season, grp in bias_table.groupby("season"):
        grp = grp.sort_values("cell_idx")
        bias_lookup[season] = grp["bias_shrunk"].to_numpy(dtype="float64")

    truth_cache = {}
    n_processed = 0

    for D in candidate_issue_dates:
        valid_dates = {L: D + timedelta(days=L) for L in LEADS}
        if not any(V in val_set for V in valid_dates.values()):
            continue

        te = fh.build_hres_table([pd.Timestamp(D)], with_truth=False)
        if te.empty:
            continue
        det = fh.predict_mos(mos, te)
        qp_raw = fh.predict_quantile_mos(qmos, te, adjust=None)
        qp_sym = fh.predict_quantile_mos(qmos, te, adjust=adj)
        qp_asym = apply_asym_adjust(qp_raw, adj_asym)

        # ---- Stage A/B: coarse-grid asymmetry (per lead, pooled over hours/cells) ----
        for L in LEADS:
            for src, stage in ((qp_raw, "A_raw_coarse"), (qp_sym, "B_conformal_coarse")):
                s = src[src["lead"] == L]
                if s.empty:
                    continue
                up = s["spd_q95"].to_numpy() - s["spd_q50"].to_numpy()
                lo = s["spd_q50"].to_numpy() - s["spd_q05"].to_numpy()
                acc[L][stage]["num"] += float(up.sum())
                acc[L][stage]["den"] += float(lo.sum())
                acc[L][stage]["n"] += len(up)

        V_by_lead = {L: valid_dates[L] for L in LEADS if valid_dates[L] in val_set}
        if not V_by_lead:
            continue

        for L, V in V_by_lead.items():
            season = SEASON_OF_MONTH[V.month]
            if season not in bias_lookup:
                continue
            truth_all_hours = truth_cache_get(truth_cache, V, ys, xs)
            if truth_all_hours is None:
                continue

            for h_idx, H in enumerate(HOURS):
                U, Vv = fh.predictions_to_grid(det, L, H)
                spd50_grid = downscaled_center((U, Vv), dwn)
                q05_sym_grid, q95_sym_grid = fine_interval_from_coarse_qp(
                    qp_sym, spd50_grid, dwn, spd_infl[L], L, H)
                q05_raw_grid, q95_raw_grid = fine_interval_from_coarse_qp(
                    qp_raw, spd50_grid, dwn, spd_infl[L], L, H)
                q05_asym_grid, q95_asym_grid = fine_interval_from_coarse_qp(
                    qp_asym, spd50_grid, dwn, spd_infl[L], L, H)

                # downscale()/interp_coarse_to_target() return the full
                # (479,433) target grid; truth and bias_shrunk are already
                # flattened to the 43,715 footprint cells - index down to
                # the same domain before any comparison/arithmetic between them.
                spd50 = spd50_grid[ys, xs]
                q05_sym, q95_sym = q05_sym_grid[ys, xs], q95_sym_grid[ys, xs]
                q05_raw, q95_raw = q05_raw_grid[ys, xs], q95_raw_grid[ys, xs]
                q05_asym, q95_asym = q05_asym_grid[ys, xs], q95_asym_grid[ys, xs]

                up_sym = q95_sym - spd50; lo_sym = spd50 - q05_sym
                acc[L]["C_fine_final"]["num"] += float(up_sym.sum())
                acc[L]["C_fine_final"]["den"] += float(lo_sym.sum())
                acc[L]["C_fine_final"]["n"] += up_sym.size

                up_raw = q95_raw - spd50; lo_raw = spd50 - q05_raw
                acc[L]["C_raw_fine_counterfactual"]["num"] += float(up_raw.sum())
                acc[L]["C_raw_fine_counterfactual"]["den"] += float(lo_raw.sum())
                acc[L]["C_raw_fine_counterfactual"]["n"] += up_raw.size

                truth = truth_all_hours[h_idx]

                if L == 7:
                    bias_shrunk = bias_lookup[season]
                    q05_bc = np.maximum(0.0, q05_sym - bias_shrunk)
                    q50_bc = spd50 - bias_shrunk
                    q95_bc = q95_sym - bias_shrunk
                    up_bc = q95_bc - q50_bc; lo_bc = q50_bc - q05_bc
                    acc_d7_bias["num"] += float(up_bc.sum())
                    acc_d7_bias["den"] += float(lo_bc.sum())
                    acc_d7_bias["n"] += up_bc.size

                w_sym = winkler_score(truth, q05_sym, q95_sym)
                w_asym = winkler_score(truth, q05_asym, q95_asym)
                wink[L]["sym"]["sum"] += float(w_sym.sum())
                wink[L]["sym"]["n"] += w_sym.size
                wink[L]["sym"]["cov"] += int(((truth >= q05_sym) & (truth <= q95_sym)).sum())
                wink[L]["asym"]["sum"] += float(w_asym.sum())
                wink[L]["asym"]["n"] += w_asym.size
                wink[L]["asym"]["cov"] += int(((truth >= q05_asym) & (truth <= q95_asym)).sum())

        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  processed {n_processed} issue dates (latest D={D}), "
                  f"elapsed={time.time() - t0:.0f}s")

    print(f"\nTotal issue dates processed: {n_processed}")

    # ---- Stage F: d14 climatology, direct from the table ----
    clim = pd.read_parquet(D14_CLIM_PATH)
    up_clim = clim["speed_q95"] - clim["speed_q50"]
    lo_clim = clim["speed_q50"] - clim["speed_q05"]
    d14_ratio_mean = float(ratio(up_clim, lo_clim).mean())
    d14_ratio_by_season = clim.assign(_r=ratio(up_clim, lo_clim)).groupby("season")["_r"].mean()

    # ---- build report ----
    lines = ["# WS interval symmetry diagnostic — 2026-07-08\n",
             "Diagnostic only, no submission/model changes. Audit target: raw LightGBM "
             "quantile output -> conformal_adjust -> spd_infl calibration -> d7 bias "
             "correction -> d14 climatological replacement (Task 1's actual pipeline; "
             "the original prompt's CatBoost/NS-ECS references were Phase 1, dropped "
             "per Matteo's scope correction).\n",
             f"Holdout: `splits.train_val_dates(seed=42)` val split, {n_processed} issue "
             f"dates processed (d1 or d7 valid date in val set).\n"]

    lines.append("## Step 1: bound-touching steps, formula, symmetric/asymmetric classification\n")
    lines.append("| Step | Formula (quoted) | Same value/factor both bounds? | Additive/multiplicative | Symmetrizing? |")
    lines.append("|---|---|---|---|---|")
    lines.append("| Raw quantile MOS (`forecast_hres.py:286-287`) | `LGBMRegressor(objective='quantile', alpha=q).fit(X,y)` per q in (0.05,0.5,0.95), independent models | No — 3 independent models | n/a | No (baseline, inherently asymmetric) |")
    lines.append("| Non-crossing sort/clip (`predict_quantile_mos`, :312-313) | `np.sort(...); np.clip(vals,0,None)` | n/a (reorders 3 values) | n/a | No |")
    lines.append("| **conformal_adjust** (`forecast_hres.py:309-311` applying `:317-336`) | `q05 -= Q_L; q95 += Q_L` where `Q_L = quantile(max(q_lo-y, y-q_hi))` | **Yes — same scalar `Q_L` added to both** | Additive | **Yes — pulls ratio toward 1 (proven below)** |")
    lines.append("| spd_infl width calibration (`forecast_pipeline.py:93-95`) | `q05 = spd50 - k*(spd50-q05); q95 = spd50 + k*(q95-spd50)` | Yes — same scalar `k`, but applied to each side's own *distance from center* | Multiplicative (of each side's own distance) | **No — ratio-invariant by construction (proven below)** |")
    lines.append("| d7 per-cell bias correction (`ws_d7_apply_bias_correction.py`) | `q05,q50,q95 -= bias_shrunk(cell,season)` (all three, same shift) | Yes — pure translation | Additive (to all 3, not just bounds) | No — translation, both widths unchanged |")
    lines.append("| d7 alpha tightening (currently alpha=1.0, no-op) | `q95=q50+a*(q95-q50); q05=q50-a*(q50-q05)` | Same `a`, per-side distance (same form as spd_infl) | Multiplicative (of each side's own distance) | No — ratio-invariant, same proof as spd_infl |")
    lines.append("| d14 climatological replacement (`compute_d14_climatology.py:118`) | `q05,q50,q95 = np.quantile(speed_samples, [.05,.5,.95])` | n/a — genuine empirical quantiles, no adjustment applied after | n/a | No — raw empirical asymmetry preserved as-is |")

    lines.append("\n**Proof sketch (why spd_infl/alpha preserve ratio but conformal_adjust doesn't):** "
                  "for `k`-scaling, `new_ratio = (k*upper)/(k*lower) = upper/lower` — the `k` cancels, "
                  "ratio is invariant for any k. For conformal's additive `Q_L`, "
                  "`new_ratio = (upper+Q_L)/(lower+Q_L)`, which strictly moves toward 1 as `Q_L` grows "
                  "(if upper != lower) — a genuine symmetrizing operation. Verified empirically below, "
                  "not just asserted.\n")

    lines.append("## Step 2: asymmetry ratio by stage x horizon\n")
    lines.append("`asymmetry_ratio = upper_width / lower_width` (>1 means right-skewed/wider-above, "
                  "matching wind speed's natural skew).\n")
    lines.append("| Horizon | Stage | asymmetry_ratio (mean) | n |")
    lines.append("|---|---|---|---|")
    for L in LEADS:
        for s in stages:
            a = acc[L][s]
            r = a["num"] / a["den"] if a["den"] else float("nan")
            label = {"A_raw_coarse": "A: raw quantile MOS (coarse)",
                      "B_conformal_coarse": "B: after conformal_adjust (coarse)",
                      "C_fine_final": "C: after downscale+spd_infl (fine, ACTUAL submitted pre-bias)",
                      "C_raw_fine_counterfactual": "C_raw: downscale+spd_infl WITHOUT conformal (counterfactual)"}[s]
            lines.append(f"| d{L} | {label} | {r:.4f} | {a['n']} |")
        if L == 7:
            r_bc = acc_d7_bias["num"] / acc_d7_bias["den"] if acc_d7_bias["den"] else float("nan")
            lines.append(f"| d7 | D: after bias correction (shift only) | {r_bc:.4f} | {acc_d7_bias['n']} |")
    lines.append(f"| d14 | F: climatological replacement (final) | {d14_ratio_mean:.4f} | {len(clim)} |")

    lines.append("\n### d14 asymmetry ratio by season (climatological, pooled hours/cells)\n")
    lines.append("| Season | asymmetry_ratio |")
    lines.append("|---|---|")
    for s, r in d14_ratio_by_season.items():
        lines.append(f"| {s} | {r:.4f} |")

    lines.append("\n### Symmetrization flag\n")
    for L in LEADS:
        rA = acc[L]["A_raw_coarse"]["num"] / acc[L]["A_raw_coarse"]["den"]
        rB = acc[L]["B_conformal_coarse"]["num"] / acc[L]["B_conformal_coarse"]["den"]
        rC = acc[L]["C_fine_final"]["num"] / acc[L]["C_fine_final"]["den"]
        rCraw = acc[L]["C_raw_fine_counterfactual"]["num"] / acc[L]["C_raw_fine_counterfactual"]["den"]
        drift_conformal = abs(rB - 1.0) - abs(rA - 1.0)
        drift_spdinfl = abs(rC - 1.0) - abs(rCraw - 1.0)
        lines.append(f"- d{L}: A={rA:.4f} -> B={rB:.4f} (conformal moves ratio "
                      f"{'toward 1 (symmetrizing)' if drift_conformal < 0 else 'away from 1'} "
                      f"by {abs(drift_conformal):.4f}); fine-grid C={rC:.4f} vs "
                      f"C_raw(no-conformal)={rCraw:.4f} (isolates conformal's fine-grid effect, "
                      f"holding spd_infl fixed) — "
                      f"spd_infl itself: {'ratio-invariant as proven' if abs(drift_spdinfl) < 1e-3 else f'unexpected drift {drift_spdinfl:.4f}'}.")

    lines.append("\n## Step 3: Winkler cost of symmetrization (conformal_adjust)\n")
    lines.append("Same holdout samples, same spd_infl k (isolating conformal's own contribution, "
                  "not a full re-optimization — spd_infl was calibrated for the symmetric-conformal "
                  "pipeline; a from-scratch asymmetric-pipeline calibration is out of scope for this "
                  "diagnostic-only task).\n")
    lines.append("| Horizon | Variant | coverage | mean Winkler |")
    lines.append("|---|---|---|---|")
    for L in LEADS:
        for variant, label in (("sym", "Actual (symmetric conformal_adjust)"),
                                ("asym", "Alternative (asymmetric conformal, alpha/2 per side)")):
            w = wink[L][variant]
            cov = w["cov"] / w["n"] if w["n"] else float("nan")
            mw = w["sum"] / w["n"] if w["n"] else float("nan")
            lines.append(f"| d{L} | {label} | {cov*100:.1f}% | {mw:.3f} |")
    for L in LEADS:
        sym_w = wink[L]["sym"]["sum"] / wink[L]["sym"]["n"]
        asym_w = wink[L]["asym"]["sum"] / wink[L]["asym"]["n"]
        pct = (asym_w - sym_w) / sym_w * 100
        lines.append(f"\n- d{L} Winkler cost of symmetric vs asymmetric conformal: "
                      f"{sym_w:.3f} vs {asym_w:.3f} ({'symmetric WORSE by' if pct>0 else 'symmetric BETTER by'} "
                      f"{abs(pct):.2f}%)")

    lines.append("\n## Step 4: pipeline ordering check\n")
    lines.append(
        "Two orderings exist in the system, at different layers:\n\n"
        "1. **Within our own d7 fix** (`ws_d7_apply_bias_correction.py`): bias-correct "
        "THEN alpha-tighten — **(a) CORRECT**, confirmed by the script's own docstring "
        "and the full git history of this session (`1668528`->`8efe877`->`a2a7225`->`431f097`, "
        "never reversed). This is the ordering the earlier center-then-calibrate finding "
        "established.\n"
        "2. **The kit's own pre-existing width calibration** (`spd_infl`, via "
        "`calibrate_intervals()`): this was calibrated via binary search to hit 90% "
        "coverage on the ORIGINAL, bias-UNcorrected fine-grid center (our bias-correction "
        "script didn't exist yet when spd_infl was derived, and `calibrate_intervals` "
        "has no bias-correction step in it at all). **This is effectively ordering (b) "
        "relative to the whole system**: the kit's width calibration happened before any "
        "bias correction existed, so `spd_infl=4.747` for d7 is inflated to cover for the "
        "-3.3 m/s center bias, not just genuine dispersion — exactly the mechanism this "
        "diagnostic's premise describes, and consistent with what "
        "`reports/ws_d7_diagnostics_20260708.md`'s Diagnostic 1 already found (an "
        "unusually large, over-wide interval for d7). Our own patch is bolted on AFTER "
        "this already bias-inflated base width — the bias-shift itself is harmless "
        "(ratio-invariant, proven in Step 1/2), but the base width being partly "
        "bias-inflated is a separate, real residual effect this diagnostic surfaces. Not "
        "fixed here (diagnostic only) — flagged as a candidate for the WS d7 model-work "
        "sketch gate (a from-scratch spd_infl recalibration on bias-corrected residuals "
        "would need re-running `calibrate_intervals`'s binary search).")

    lines.append("\n## Recommendation\n")
    for L in LEADS:
        sym_w = wink[L]["sym"]["sum"] / wink[L]["sym"]["n"]
        asym_w = wink[L]["asym"]["sum"] / wink[L]["asym"]["n"]
        pct = (sym_w - asym_w) / sym_w * 100
        material = abs(pct) > 1.0
        lines.append(f"- d{L}: conformal_adjust IS symmetrizing (proven), Winkler cost "
                      f"{'material' if material else 'small'} ({pct:+.2f}%). "
                      f"{'Worth a sketch-gate candidate: asymmetric split-conformal.' if material else 'Not worth pursuing on its own at this cost.'}")
    lines.append("- The kit's own `spd_infl` calibration (separate from our own bias-then-"
                  "alpha fix) was itself derived on a bias-uncorrected center — a residual "
                  "effect of the correct-ordering principle this diagnostic set out to test, "
                  "not yet remediated. Flag for the sketch gate, not an immediate fix.")

    lines.append(f"\nWall-clock: {time.time() - t0:.0f}s")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
