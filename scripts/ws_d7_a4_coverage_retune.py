#!/usr/bin/env python3
"""TASK 3 — minimal coverage retune of A4 (WS d7), 2026-07-16.

PRINCIPLE (not a search): the board-validated operating point had holdout d7
coverage 0.860 at alpha=0.90. A4 sits at 0.856. Increase alpha by the MINIMUM
amount that restores A4 holdout coverage to >= 0.860 (the baseline's operating
point, NOT nominal 0.90). One-dimensional and monotone (wider interval ->
higher coverage), so a single upward scan finds the threshold; no grid search,
no per-lead tuning.

Reads the A4 d7 holdout arrays checkpointed by ws_d7_feature_experiments.py
(scripts/artifacts/ws_d7_featexp_A4_stacked.npz — regenerable cold from the
committed harness) and reconstructs the SAME per-cell x season bias (K=30) and
alpha-tightening the harness used, so alpha=0.90 here reproduces the arm's
reported coverage/Winkler exactly (validation printed).

alpha is the d7 interval-width multiplier ONLY (harness applies it solely to
lead 7: ws_d7_feature_experiments.py fine-scoring block). d1 uses its own native
downscaled interval and is therefore invariant to this retune — reported as a
side-effect check with that structural note.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"
SEASONS = ("DJF", "MAM", "JJA", "SON")
K_SHRINK = 30.0
ALPHA_LEVEL = 0.10
ALPHA_OLD = 0.90
COV_TARGET = 0.860           # the baseline's holdout operating point (NOT 0.90 nominal)
A2_WINKLER_D7 = None         # filled from A2 json (the "untouched A2" comparison)


def winkler(y, lo, hi, a=ALPHA_LEVEL):
    return (hi - lo) + np.maximum(0.0, lo - y) * (2.0 / a) + np.maximum(0.0, y - hi) * (2.0 / a)


def load_stacked(arm):
    z = np.load(ART / f"ws_d7_featexp_{arm}_stacked.npz")
    out = {}
    for s in SEASONS:
        if f"{s}_q50" in z.files:
            out[s] = {k: z[f"{s}_{k}"] for k in ("q05", "q50", "q95", "truth")}
    return out


def bias_shrunk(stacked):
    """Per-cell x season shrunk bias (K=30) — identical to the harness."""
    b = {}
    for s, d in stacked.items():
        bias = d["q50"] - d["truth"]                 # (n_blocks, n_cells)
        raw = bias.mean(axis=0)
        smean = float(bias.mean())
        nb = bias.shape[0]
        w = nb / (nb + K_SHRINK)
        b[s] = w * raw + (1.0 - w) * smean
    return b


def score(stacked, bias, alpha):
    n = cov = 0
    wsum = wid = 0.0
    for s, d in stacked.items():
        bb = bias[s][None, :]
        q05 = np.maximum(0.0, d["q05"] - bb)
        q50 = d["q50"] - bb
        q95 = d["q95"] - bb
        t = d["truth"]
        lo = np.maximum(0.0, q50 - alpha * (q50 - q05))
        hi = q50 + alpha * (q95 - q50)
        wsum += float(winkler(t, lo, hi).sum())
        wid += float((hi - lo).sum())
        cov += int(((t >= lo) & (t <= hi)).sum())
        n += t.size
    return {"alpha": alpha, "winkler": wsum / n, "coverage": cov / n,
            "mean_width": wid / n, "n": n}


def main():
    st = load_stacked("A4")
    bias = bias_shrunk(st)

    old = score(st, bias, ALPHA_OLD)
    # validation against the arm's reported numbers
    a4 = json.load(open(ART / "ws_d7_featexp_A4.json"))["fine"]["d7_primary_bias_alpha090"]
    print(f"[validate] alpha=0.90 reconstructed: winkler={old['winkler']:.4f} "
          f"cov={old['coverage']:.4f}  vs reported winkler={a4['winkler']:.4f} "
          f"cov={a4['coverage']:.4f}")

    # minimal upward scan (monotone): first alpha with coverage >= target
    new = None
    for a in np.round(np.arange(ALPHA_OLD, 1.501, 0.001), 3):
        r = score(st, bias, float(a))
        if r["coverage"] >= COV_TARGET:
            new = r
            break
    if new is None:
        print("no alpha in [0.90, 1.50] reaches coverage target — reporting and stopping")
        return

    a2 = json.load(open(ART / "ws_d7_featexp_A2.json"))
    a2_w = a2["fine"]["d7_primary_bias_alpha090"]["winkler"]
    a4d1 = json.load(open(ART / "ws_d7_featexp_A4.json"))["fine"]["d1_native"]

    print("\n=== TASK 3 — A4 minimal coverage retune (d7 holdout) ===")
    print(f"target coverage (baseline operating point): {COV_TARGET}")
    print(f"OLD alpha={old['alpha']:.3f}: coverage={old['coverage']:.4f} "
          f"winkler_d7={old['winkler']:.4f} width={old['mean_width']:.3f}")
    print(f"NEW alpha={new['alpha']:.3f}: coverage={new['coverage']:.4f} "
          f"winkler_d7={new['winkler']:.4f} width={new['mean_width']:.3f}")
    print(f"alpha increase = {new['alpha']-old['alpha']:.3f}; "
          f"winkler_d7 cost = {new['winkler']-old['winkler']:+.4f} "
          f"({100*(new['winkler']-old['winkler'])/old['winkler']:+.2f}%)")

    print(f"\nd1 side-effect check: alpha is the d7 interval multiplier ONLY "
          f"(harness applies it solely to lead 7); d1 uses its own native "
          f"downscaled interval -> INVARIANT to this retune.")
    print(f"  A4 d1 (native, both alphas): winkler={a4d1['winkler']:.4f} "
          f"coverage={a4d1['coverage']:.4f} (unchanged)")

    print(f"\nGuard: retuned A4 d7 winkler={new['winkler']:.4f} vs untouched A2 "
          f"d7 winkler={a2_w:.4f}")
    if new["winkler"] > a2_w:
        print("  ** retuned A4 LOSES to untouched A2 — reporting both, choice "
              "returns to review (per dispatch). **")
    else:
        print("  retuned A4 still beats untouched A2 — A4@new-alpha stands.")

    out = {"target_coverage": COV_TARGET, "old": old, "new": new,
           "a2_d7_winkler": a2_w, "a4_d1_native": a4d1,
           "retuned_a4_beats_a2": bool(new["winkler"] <= a2_w)}
    with open(ART / "ws_d7_a4_coverage_retune.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {ART / 'ws_d7_a4_coverage_retune.json'}")


if __name__ == "__main__":
    main()
