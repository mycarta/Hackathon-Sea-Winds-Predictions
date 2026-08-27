#!/usr/bin/env python3
"""
Fix #1 + Fix #2 (sequential), 2026-07-08: recalibrate spd_infl on
bias-corrected residuals, then test asymmetric split-conformal for d7
using the recalibrated spd_infl.

Driven by reports/ws_interval_symmetry_diagnostic_20260708.md's critical
finding: spd_infl=4.747 (d7) was calibrated BEFORE bias correction
existed, so it clamps q05 to exactly 0.000 for 100% of pre-fix d7 rows.
Fix #1 re-derives spd_infl with the bias-corrected center already applied
(center -> decouple -> calibrate, done properly this time, inside the
calibration itself rather than as a downstream patch). Fix #2 (d7 only,
using Fix #1's new spd_infl) tests Matteo's asymmetric split-conformal
formula against the shipped symmetric conformal_adjust.

Sequencing (per task prompt: "#1 must complete before #2 starts, because
#1 changes the residuals #2 operates on"): implemented as ONE combined
script for efficiency (a single expensive per-date forecast+downscale
pass captures every PRE-WIDENING quantity both fixes need - re-running
the ~30 min pipeline twice would be wasteful), but the LOGICAL dependency
is honored exactly as specified: Fix #1's sweep completes and picks
k_new BEFORE Fix #2's Winkler comparison ever runs, and Fix #2's
Winkler comparison uses k_new (not the stale 4.747) throughout. Fix #1's
full table is printed/written before any Fix #2 computation begins.

Method for both fixes: capture PRE-WIDENING (spd_infl=1.0-equivalent)
fine-grid q05/q95 once (both under the shipped SYMMETRIC conformal and,
for d7 only, under Matteo's ASYMMETRIC conformal formula), plus the raw
downscaled center and truth, for every (date, hour) in the same
splits.train_val_dates(seed=42) holdout used throughout this session.
Everything downstream (spd_infl sweep, bias correction, conformal
variant comparison) is then vectorized numpy over these cached arrays -
no further pipeline reruns needed, including for the "finer grid around
the optimum" step the task explicitly asks for.

Matteo's asymmetric conformal formula (Fix #2, distinct from the
alpha/2-per-side construction used in the interval symmetry diagnostic -
implemented exactly as specified in the task prompt, not reused):
  score_upper = truth - q95   (rows where truth > q95, i.e. violates high)
  score_lower = q05 - truth   (rows where truth < q05, i.e. violates low)
  correction_upper = (1-alpha) quantile of score_upper's sample
  correction_lower = (1-alpha) quantile of score_lower's sample
  q95_new = q95 + correction_upper ; q05_new = q05 - correction_lower
computed on the RAW (unadjusted) coarse quantile prediction on the same
calibration set (train[train.year==2020]) the shipped conformal_adjust
uses - same population, different score formula. The task didn't specify
whether to use conformal_adjust's own finite-sample rank adjustment
(ceil((n+1)(1-alpha))-th order statistic) or a plain quantile; used the
same finite-sample approach as the shipped conformal_adjust for
consistency, documented here since it wasn't fully specified.

Diagnostic + holdout validation only. Does NOT touch submission.csv or
any model. New spd_infl/conformal choices are reported, not applied.

Outputs:
  reports/ws_spdinfl_recal_asym_conformal_20260708.md
  scripts/artifacts/ws_spdinfl_recal_stacked.npz (raw holdout arrays,
    gitignored, enables further re-sweeps without a pipeline rerun)
  scripts/artifacts/ws_spdinfl_recal_params.json
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
REPORT_PATH = ROOT / "reports" / "ws_spdinfl_recal_asym_conformal_20260708.md"
BIAS_TABLE_PATH = ARTIFACTS_DIR / "ws_d7_bias_shrunk_table.parquet"
STACKED_OUT_PATH = ARTIFACTS_DIR / "ws_spdinfl_recal_stacked.npz"
PARAMS_OUT_PATH = ARTIFACTS_DIR / "ws_spdinfl_recal_params.json"

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
K_SWEEP_COARSE = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.747)
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


def asymmetric_conformal_matteo(qmos, calib_df, alpha=ALPHA_LEVEL, quantiles=fh.QUANTILES):
    """Matteo's asymmetric conformal formula (see module docstring) -
    separate corrections from violated-rows-only score sets, using the
    same finite-sample rank adjustment as the shipped conformal_adjust."""
    qs = sorted(quantiles)
    lo_c, hi_c = fh._qcol(qs[0]), fh._qcol(qs[-1])
    pr = fh.predict_quantile_mos(qmos, calib_df, quantiles)  # raw, unadjusted
    out = {}
    for L in sorted(calib_df["lead"].unique()):
        s = pr[pr["lead"] == L]
        y = np.hypot(s["u125c"].to_numpy(), s["v125c"].to_numpy())
        q05, q95 = s[lo_c].to_numpy(), s[hi_c].to_numpy()

        mask_lo = y < q05
        mask_hi = y > q95
        score_lower = q05[mask_lo] - y[mask_lo]
        score_upper = y[mask_hi] - q95[mask_hi]

        def _order_stat_quantile(scores, alpha_):
            n = len(scores)
            if n == 0:
                return 0.0
            k = int(np.ceil((n + 1) * (1.0 - alpha_)))
            return float(np.sort(scores)[min(k, n) - 1])

        corr_lower = _order_stat_quantile(score_lower, alpha)
        corr_upper = _order_stat_quantile(score_upper, alpha)
        out[L] = (corr_lower, corr_upper)
        print(f"    lead={L}: n_calib={len(y)}, n_viol_lo={mask_lo.sum()}, "
              f"n_viol_hi={mask_hi.sum()}, corr_lower={corr_lower:.4f}, "
              f"corr_upper={corr_upper:.4f}")
    return out


def apply_asym_matteo(qp_raw, adj_asym, quantiles=fh.QUANTILES):
    qs = sorted(quantiles)
    lo_c, hi_c = fh._qcol(qs[0]), fh._qcol(qs[-1])
    out = qp_raw.copy()
    for L, (corr_lower, corr_upper) in adj_asym.items():
        m = (out["lead"] == L).to_numpy()
        out.loc[m, lo_c] = out.loc[m, lo_c] - corr_lower
        out.loc[m, hi_c] = out.loc[m, hi_c] + corr_upper
    cols = [fh._qcol(q) for q in qs]
    vals = np.sort(out[cols].to_numpy(float), axis=1)
    out[cols] = np.clip(vals, 0.0, None)
    return out


def downscaled_center(det_UV, dwn):
    U, V = det_UV
    fu, fv = dn.downscale(dwn, U, V)
    return np.sqrt(fu ** 2 + fv ** 2)


def fine_interval_pre_k(qp, spd50, lead, h):
    """PRE-WIDENING (spd_infl=1.0-equivalent) fine-grid q05/q95, i.e. the
    ratio-based base interval before any k-scaling - matches
    _speed_interval(...,k=1.0), which skips the widening branch entirely."""
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


def eval_k(k, spd50_bc, spd50_raw, q05_pre, q95_pre, truth):
    q05 = np.maximum(0.0, spd50_bc - k * (spd50_raw - q05_pre))
    q95 = spd50_bc + k * (q95_pre - spd50_raw)
    covered = (truth >= q05) & (truth <= q95)
    width = q95 - q05
    wink = winkler_score(truth, q05, q95)
    clamped = (q05 <= 1e-9)
    upper_w = q95 - spd50_bc
    lower_w = np.maximum(spd50_bc - q05, 1e-6)
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(width.mean()),
        "mean_winkler": float(wink.mean()),
        "pct_clamped": float(clamped.mean() * 100),
        "asymmetry_ratio": float((upper_w / lower_w).mean()),
    }


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
    spd_infl_orig, dir_off = P.calibrate_intervals(mos, qmos, adj, dwn, offs)
    print(f"  original shipped spd_infl (stale, pre-bias-correction): {spd_infl_orig}")

    calib = train[train.year == 2020]
    calib_df = fh.build_hres_table(calib)
    print("  computing Matteo's asymmetric conformal margins on calib set...")
    adj_asym = asymmetric_conformal_matteo(qmos, calib_df)
    print(f"  symmetric conformal Q_L (shipped): {adj}")
    print(f"  asymmetric conformal (corr_lower, corr_upper): {adj_asym}")

    bias_table = pd.read_parquet(BIAS_TABLE_PATH)
    bias_lookup = {}
    for season, grp in bias_table.groupby("season"):
        grp = grp.sort_values("cell_idx")
        bias_lookup[season] = grp["bias_shrunk"].to_numpy(dtype="float64")

    train_dates_seed, val_dates_seed = splits.train_val_dates(seed=42)
    val_set = set(val_dates_seed)
    print(f"train_val_dates(seed=42): {len(train_dates_seed)} train, {len(val_dates_seed)} val")

    all_dates = sorted(target_loader.list_dates(config.target_root()))
    candidate_issue_dates = [d for d in all_dates if d.year in (2016, 2017, 2018, 2019, 2020)]
    print(f"Candidate issue dates: {len(candidate_issue_dates)}")

    # per-lead lists of blocks (each block length n_cells)
    store = {L: {"q05_sym": [], "q95_sym": [], "spd50": [], "truth": [], "season": []}
             for L in LEADS}
    store[7].update({"q05_asym": [], "q95_asym": []})

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
        qp_sym = fh.predict_quantile_mos(qmos, te, adjust=adj)
        qp_raw = fh.predict_quantile_mos(qmos, te, adjust=None)
        qp_asym = apply_asym_matteo(qp_raw, adj_asym)

        contributed = False
        for L in LEADS:
            V = valid_dates[L]
            if V not in val_set:
                continue
            season = SEASON_OF_MONTH[V.month]
            truth_all_hours = truth_cache_get(truth_cache, V, ys, xs)
            if truth_all_hours is None:
                continue

            for h_idx, H in enumerate(HOURS):
                U, Vv = fh.predictions_to_grid(det, L, H)
                spd50_grid = downscaled_center((U, Vv), dwn)
                q05s_grid, q95s_grid = fine_interval_pre_k(qp_sym, spd50_grid, L, H)

                spd50 = spd50_grid[ys, xs]
                q05_sym, q95_sym = q05s_grid[ys, xs], q95s_grid[ys, xs]
                truth = truth_all_hours[h_idx]

                store[L]["q05_sym"].append(q05_sym.astype("float32"))
                store[L]["q95_sym"].append(q95_sym.astype("float32"))
                store[L]["spd50"].append(spd50.astype("float32"))
                store[L]["truth"].append(truth.astype("float32"))
                store[L]["season"].append(season)

                if L == 7:
                    q05a_grid, q95a_grid = fine_interval_pre_k(qp_asym, spd50_grid, L, H)
                    store[L]["q05_asym"].append(q05a_grid[ys, xs].astype("float32"))
                    store[L]["q95_asym"].append(q95a_grid[ys, xs].astype("float32"))

                contributed = True

        if contributed:
            n_processed += 1
        if n_processed % 200 == 0 and contributed:
            print(f"  processed {n_processed} issue dates (latest D={D}), "
                  f"elapsed={time.time() - t0:.0f}s")

    print(f"\nTotal issue dates processed: {n_processed}")

    # ---- stack + checkpoint ----
    stacked = {}
    for L in LEADS:
        for key in ("q05_sym", "q95_sym", "spd50", "truth"):
            stacked[f"L{L}_{key}"] = np.stack(store[L][key])
        stacked[f"L{L}_season"] = np.array(store[L]["season"])
    stacked["L7_q05_asym"] = np.stack(store[7]["q05_asym"])
    stacked["L7_q95_asym"] = np.stack(store[7]["q95_asym"])
    np.savez_compressed(STACKED_OUT_PATH, **stacked)
    print(f"Checkpointed raw stacked arrays -> {STACKED_OUT_PATH}")

    def bias_for_lead(L):
        if L != 7:
            return np.zeros((stacked[f"L{L}_spd50"].shape[0], n_cells), dtype="float64")
        seasons = stacked["L7_season"]
        return np.stack([bias_lookup[s] for s in seasons])

    # ================= Fix #1: spd_infl sweep =================
    print("\n" + "=" * 70)
    print("FIX #1: spd_infl recalibration on bias-corrected residuals")
    print("=" * 70)

    fix1_results = {}
    for L in LEADS:
        spd50_raw = stacked[f"L{L}_spd50"]
        q05_pre = stacked[f"L{L}_q05_sym"]
        q95_pre = stacked[f"L{L}_q95_sym"]
        truth = stacked[f"L{L}_truth"]
        bias = bias_for_lead(L)
        spd50_bc = spd50_raw - bias

        coarse_rows = []
        for k in K_SWEEP_COARSE:
            m = eval_k(k, spd50_bc, spd50_raw, q05_pre, q95_pre, truth)
            m["k"] = k
            coarse_rows.append(m)
            print(f"  d{L} k={k}: coverage={m['coverage']*100:.1f}%, "
                  f"winkler={m['mean_winkler']:.3f}, clamped={m['pct_clamped']:.1f}%, "
                  f"asym_ratio={m['asymmetry_ratio']:.3f}")

        best_coarse = min(coarse_rows, key=lambda r: r["mean_winkler"])
        fine_grid = sorted(set(
            round(max(0.1, best_coarse["k"] + d), 3)
            for d in np.arange(-0.4, 0.41, 0.05)
        ) - set(K_SWEEP_COARSE))
        fine_rows = []
        for k in fine_grid:
            m = eval_k(k, spd50_bc, spd50_raw, q05_pre, q95_pre, truth)
            m["k"] = k
            fine_rows.append(m)
            print(f"  d{L} k={k} (fine): coverage={m['coverage']*100:.1f}%, "
                  f"winkler={m['mean_winkler']:.3f}, clamped={m['pct_clamped']:.1f}%, "
                  f"asym_ratio={m['asymmetry_ratio']:.3f}")

        all_rows = sorted(coarse_rows + fine_rows, key=lambda r: r["k"])
        best_overall = min(all_rows, key=lambda r: r["mean_winkler"])
        fix1_results[L] = {"coarse": coarse_rows, "fine": fine_rows, "all": all_rows,
                            "best": best_overall,
                            "old_spd_infl": spd_infl_orig.get(L, 1.0)}
        print(f"  >>> d{L} recommended new spd_infl: {best_overall['k']} "
              f"(Winkler {best_overall['mean_winkler']:.3f}, "
              f"vs old k={spd_infl_orig.get(L,1.0)})")

    print(f"\nFix #1 complete. Proceeding to Fix #2 using d7's new spd_infl="
          f"{fix1_results[7]['best']['k']}.\n")

    # ================= Fix #2: asymmetric conformal (d7 only) =================
    print("=" * 70)
    print("FIX #2: asymmetric split-conformal for d7, using Fix #1's new spd_infl")
    print("=" * 70)

    k_new = fix1_results[7]["best"]["k"]
    spd50_raw7 = stacked["L7_spd50"]
    truth7 = stacked["L7_truth"]
    bias7 = bias_for_lead(7)
    spd50_bc7 = spd50_raw7 - bias7

    sym_at_new_k = eval_k(k_new, spd50_bc7, spd50_raw7,
                           stacked["L7_q05_sym"], stacked["L7_q95_sym"], truth7)
    asym_at_new_k = eval_k(k_new, spd50_bc7, spd50_raw7,
                            stacked["L7_q05_asym"], stacked["L7_q95_asym"], truth7)
    # baseline for context: old stale spd_infl, symmetric conformal (matches
    # the original diagnostic's numbers, sanity cross-check)
    baseline_old = eval_k(spd_infl_orig.get(7, 1.0), spd50_bc7, spd50_raw7,
                           stacked["L7_q05_sym"], stacked["L7_q95_sym"], truth7)

    print(f"  baseline (old k={spd_infl_orig.get(7,1.0)}, symmetric, bias-corrected center): "
          f"winkler={baseline_old['mean_winkler']:.3f}")
    print(f"  new k={k_new}, symmetric: winkler={sym_at_new_k['mean_winkler']:.3f}")
    print(f"  new k={k_new}, asymmetric: winkler={asym_at_new_k['mean_winkler']:.3f}")

    # ---- report ----
    lines = [
        "# WS d7 spd_infl recalibration + asymmetric conformal — 2026-07-08\n",
        "Diagnostic + holdout validation only. Does NOT touch submission.csv or "
        "any model. Sequential: Fix #1 (spd_infl recalibration) completes and "
        "picks `k_new` before Fix #2 (asymmetric conformal) runs, which uses "
        "`k_new` throughout - implemented as one combined script for efficiency "
        "(single expensive pipeline pass), logical sequencing honored via "
        "computation order.\n",
        f"Holdout: `splits.train_val_dates(seed=42)` val split, {n_processed} "
        f"issue dates processed.\n",
        "## Fix #1: spd_infl sweep results\n",
    ]

    for L in LEADS:
        r = fix1_results[L]
        lines.append(f"### d{L}\n")
        lines.append("| k | coverage | mean width (m/s) | mean Winkler | % q05 clamped | asymmetry_ratio |")
        lines.append("|---|---|---|---|---|---|")
        for m in r["all"]:
            marker = " **<- recommended**" if m["k"] == r["best"]["k"] else ""
            lines.append(f"| {m['k']} | {m['coverage']*100:.1f}% | {m['mean_width']:.3f} | "
                          f"{m['mean_winkler']:.3f} | {m['pct_clamped']:.1f}% | "
                          f"{m['asymmetry_ratio']:.3f}{marker} |")
        lines.append(f"\nOld shipped spd_infl: {r['old_spd_infl']}. "
                      f"**Recommended new spd_infl: {r['best']['k']}** "
                      f"(Winkler {r['best']['mean_winkler']:.3f} vs old-k-at-bias-corrected-center "
                      f"comparison below).\n")

    lines.append("\n### d14 (completeness check)\n")
    lines.append("d14 uses climatological empirical quantiles "
                  "(`compute_d14_climatology.py`), not `spd_infl`/`_speed_interval` at all "
                  "- confirmed in `reports/ws_interval_symmetry_diagnostic_20260708.md`. "
                  "spd_infl recalibration is not applicable to d14's actual submitted values.\n")

    lines.append("## Fix #2: asymmetric split-conformal (d7 only, at new spd_infl)\n")
    lines.append(f"Matteo's asymmetric conformal margins (calib set, alpha={ALPHA_LEVEL}): "
                 f"{adj_asym}\n")
    lines.append("| Variant | k used | coverage | mean width (m/s) | mean Winkler | asymmetry_ratio |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| Baseline (old k={spd_infl_orig.get(7,1.0)}, symmetric conformal) | "
                 f"{spd_infl_orig.get(7,1.0)} | {baseline_old['coverage']*100:.1f}% | "
                 f"{baseline_old['mean_width']:.3f} | {baseline_old['mean_winkler']:.3f} | "
                 f"{baseline_old['asymmetry_ratio']:.3f} |")
    lines.append(f"| New k, symmetric conformal (shipped) | {k_new} | "
                 f"{sym_at_new_k['coverage']*100:.1f}% | {sym_at_new_k['mean_width']:.3f} | "
                 f"{sym_at_new_k['mean_winkler']:.3f} | {sym_at_new_k['asymmetry_ratio']:.3f} |")
    lines.append(f"| New k, asymmetric conformal (Matteo's formula) | {k_new} | "
                 f"{asym_at_new_k['coverage']*100:.1f}% | {asym_at_new_k['mean_width']:.3f} | "
                 f"{asym_at_new_k['mean_winkler']:.3f} | {asym_at_new_k['asymmetry_ratio']:.3f} |")

    pct_vs_baseline = (baseline_old["mean_winkler"] - sym_at_new_k["mean_winkler"]) / baseline_old["mean_winkler"] * 100
    pct_asym_vs_sym = (sym_at_new_k["mean_winkler"] - asym_at_new_k["mean_winkler"]) / sym_at_new_k["mean_winkler"] * 100
    lines.append(f"\n- Fix #1 alone (new k, still symmetric conformal) vs baseline: "
                 f"{baseline_old['mean_winkler']:.3f} -> {sym_at_new_k['mean_winkler']:.3f} "
                 f"({pct_vs_baseline:+.2f}% Winkler change).")
    lines.append(f"- Fix #2 on top of Fix #1 (asymmetric vs symmetric, both at new k): "
                 f"{sym_at_new_k['mean_winkler']:.3f} -> {asym_at_new_k['mean_winkler']:.3f} "
                 f"({pct_asym_vs_sym:+.2f}% Winkler change, positive = asymmetric better).")

    lines.append("\n## Validation\n")
    lines.append(f"- Floor-clamp rate at new k: see Fix #1 table above "
                 f"(`% q05 clamped` column at k={k_new}) vs 100.0% at the old k=4.747.")
    lines.append(f"- Coverage at recommended configuration: "
                 f"{asym_at_new_k['coverage']*100:.1f}% (asymmetric) / "
                 f"{sym_at_new_k['coverage']*100:.1f}% (symmetric), both at new k={k_new}.")
    lines.append(f"- Asymmetry ratio: symmetric={sym_at_new_k['asymmetry_ratio']:.3f}, "
                 f"asymmetric={asym_at_new_k['asymmetry_ratio']:.3f}.")

    lines.append(f"\nWall-clock: {time.time() - t0:.0f}s")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")

    params = {
        "old_spd_infl": {str(k): v for k, v in spd_infl_orig.items()},
        "fix1_recommended_spd_infl": {str(L): fix1_results[L]["best"]["k"] for L in LEADS},
        "fix1_recommended_winkler": {str(L): fix1_results[L]["best"]["mean_winkler"] for L in LEADS},
        "fix2_baseline_winkler": baseline_old["mean_winkler"],
        "fix2_sym_at_new_k_winkler": sym_at_new_k["mean_winkler"],
        "fix2_asym_at_new_k_winkler": asym_at_new_k["mean_winkler"],
        "asym_conformal_margins": {str(k): v for k, v in adj_asym.items()},
    }
    with open(PARAMS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Wrote {PARAMS_OUT_PATH}")


if __name__ == "__main__":
    main()
