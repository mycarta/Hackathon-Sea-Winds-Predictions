#!/usr/bin/env python3
"""Leg B R4b gates G1, G2, G3: is the rebuild the same submission as 855076?

Matteo adopted this gate on 2026-08-22 in place of the 2021 local Winkler score,
which cannot be computed: 27.14 and 24.7543 are Codabench BOARD numbers and
there is no 2021 truth on this machine (target years are 2016-2020 only).

The substitute asks a question that IS measurable. 855076 is the July 2021 Leg B
submission that scored 24.7543. If the rebuilt submission differs from it only
in d7 speed, and by little enough that the Winkler score cannot have moved by
the 2.39 that separates 24.7543 from Leg A's 27.14, then Leg B's advantage
survives the rebuild by argument rather than by measurement.

    G1  Only the d7 speed quantiles may differ. ANY difference in d1, d14, any
        direction field, or any metadata column is a stop-and-report: it would
        mean the rebuild changed something the refit had no business touching.

    G2  Worst-case bound on |change in mean d7 Winkler| against the 2.39 margin.
        Pass if the bound is below it.

    G3  On bound failure: the delta distribution (mean, p95, max, by season)
        plus the bound value, then halt for Matteo. No decision is taken here.

THE TWO ALPHAS, verified against the local scorer code before computing G2, as
instructed. They are different quantities and the shared name invites error:

  ALPHA_LEVEL = 0.10   `ws_d7_feature_experiments.py:99`
      The Winkler interval LEVEL, fixed by the competition metric. The interval
      is nominally central 1 - 0.10 = 90 percent. This is the alpha that sets
      the exceedance penalty coefficient 2/alpha = 20.

  ALPHA = ALPHA_FIX = chosen_alpha = 0.90
      `ws_d7_feature_experiments.py:100`, `tier2_d7_score_blocks.py:39`,
      `tier2_d7_build_submission.py:57`. NOT a probability. It is an interval
      WIDTH MULTIPLIER applied as lo = q50 - 0.90*(q50 - q05) and
      hi = q50 + 0.90*(q95 - q50), tightening the interval before scoring.

EXACT WINKLER FORMULA, `ws_d7_feature_experiments.py:113-117`, quoted verbatim:

    def winkler_score(y, lo, hi, alpha_level=ALPHA_LEVEL):
        width = hi - lo
        below = np.maximum(0.0, lo - y) * (2.0 / alpha_level)
        above = np.maximum(0.0, y - hi) * (2.0 / alpha_level)
        return width + below + above

so  W(y, lo, hi) = (hi - lo) + 20 * max(0, lo - y) + 20 * max(0, y - hi).

This script does not take that on trust: `verify_metric()` imports the real
function and checks it numerically at interior, below and above points, and
asserts the penalty coefficient is exactly 20.

THE BOUND, and why it is worst-case rather than an estimate. Truth is unknown,
so the penalty terms cannot be evaluated. But W is 1-Lipschitz in each of lo and
hi through the width term and 20-Lipschitz through the penalty terms, giving

    |dW| <= 21 * |d_lo| + 21 * |d_hi|

pointwise, hence for the mean over scored rows

    |d Winkler_mean| <= 21 * (mean|d_lo| + mean|d_hi|).

That is attained only if every single cell sits exactly at an interval edge and
every delta pushes it the wrong way, so it is loose by construction. A pass is
therefore conclusive; a failure is NOT evidence the score moved, only that this
bound cannot rule it out. G3 exists for exactly that case.

The exact, signed width component `mean(d_hi - d_lo)` IS computable without
truth and is reported alongside, because it is the part of the change that does
not depend on where truth falls.

Run:
    python scripts/legB_R4b_gates_20260822.py --candidate <rebuilt_f125.csv>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))

ARTIFACTS = _HERE / "artifacts"
BASELINE = ARTIFACTS / "submission_pangu_d7_f125_20260720.csv"
BASELINE_SHA = "169524f9ffa5c3e53f66a4d7f686299b1344c9f303040a68e610ed0816b33079"
DATEMAP_2021 = ARTIFACTS / "tier2_sub_datemap.json"
REPORT = REPO / "reports" / "legB_R4b_gates_20260822.md"

MARGIN = 27.14 - 24.7543          # 2.3857, Leg A board minus Leg B board, 2021
ALPHA_WIDTH = 0.90                # interval width multiplier, NOT a probability
LIPSCHITZ = 21.0                  # 1 (width) + 20 (penalty), see module docstring

COLS = ["type", "window", "region", "latitude", "longitude", "horizon", "hour",
        "level", "q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]
QCOLS = ["q05", "q50", "q95"]
DIRCOLS = ["dir_05", "dir_50", "dir_95"]
N_TOTAL = 4_196_640
N_D7 = 1_398_880


def verify_metric():
    """Check the alpha and the formula against the real scorer, numerically."""
    import ws_d7_feature_experiments as W

    level = float(W.ALPHA_LEVEL)
    coef = 2.0 / level
    checks = []

    # interior point: score is exactly the width, no penalty
    v = float(W.winkler_score(np.array([5.0]), np.array([4.0]), np.array([6.0]))[0])
    checks.append(("interior y=5 in [4,6]", v, 2.0))
    # below the interval: width + coef * (lo - y)
    v = float(W.winkler_score(np.array([3.0]), np.array([4.0]), np.array([6.0]))[0])
    checks.append(("below y=3 < lo=4", v, 2.0 + coef * 1.0))
    # above the interval: width + coef * (y - hi)
    v = float(W.winkler_score(np.array([9.0]), np.array([4.0]), np.array([6.0]))[0])
    checks.append(("above y=9 > hi=6", v, 2.0 + coef * 3.0))

    for name, got, want in checks:
        assert abs(got - want) < 1e-12, "%s: got %r, expected %r" % (name, got, want)

    assert abs(level - 0.10) < 1e-12, "ALPHA_LEVEL is %r, expected 0.10" % level
    assert abs(coef - 20.0) < 1e-12, "penalty coefficient is %r, expected 20" % coef
    assert abs(float(W.ALPHA_FIX) - ALPHA_WIDTH) < 1e-12, (
        "ALPHA_FIX is %r, expected %r" % (W.ALPHA_FIX, ALPHA_WIDTH))
    return level, coef, checks


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="the rebuilt 2021 Leg B submission, f=1.25 applied")
    ap.add_argument("--baseline", default=str(BASELINE))
    a = ap.parse_args()

    print("=" * 74)
    print("Leg B R4b gates G1/G2/G3")
    print("=" * 74)

    level, coef, checks = verify_metric()
    print("\nMETRIC VERIFICATION (before anything is computed against it)")
    print("  ALPHA_LEVEL (Winkler interval level) : %.2f  -> nominal %.0f%% interval"
          % (level, 100 * (1 - level)))
    print("  penalty coefficient 2/alpha_level    : %.1f" % coef)
    print("  ALPHA_FIX (interval WIDTH multiplier): %.2f  (not a probability)"
          % ALPHA_WIDTH)
    print("  W(y,lo,hi) = (hi-lo) + %.0f*max(0,lo-y) + %.0f*max(0,y-hi)" % (coef, coef))
    for name, got, want in checks:
        print("    %-24s -> %.4f  (expected %.4f) OK" % (name, got, want))

    base_p, cand_p = Path(a.baseline), Path(a.candidate)
    got = sha256(base_p)
    assert got == BASELINE_SHA, "baseline SHA mismatch: %s" % got
    print("\nbaseline pinned: %s" % base_p.name)
    print("  SHA-256 %s (855076, board WS d7 24.7543)" % got)
    print("candidate: %s" % cand_p.name)
    print("  SHA-256 %s" % sha256(cand_p))

    b = pd.read_csv(base_p)
    c = pd.read_csv(cand_p)
    assert list(b.columns) == COLS and list(c.columns) == COLS, "schema mismatch"
    assert len(b) == len(c) == N_TOTAL, "row count %d / %d" % (len(b), len(c))

    # ---------------- G1: only d7 speed may move -------------------------
    print("\n" + "-" * 74)
    print("G1: only the d7 speed quantiles may differ")
    print("-" * 74)
    g1_rows = []
    for col in COLS:
        if col in QCOLS:
            continue
        if b[col].dtype.kind in "fc" or c[col].dtype.kind in "fc":
            same = np.array_equal(b[col].to_numpy(), c[col].to_numpy(), equal_nan=True)
        else:
            same = b[col].equals(c[col])
        n_diff = 0 if same else int((b[col].to_numpy() != c[col].to_numpy()).sum())
        g1_rows.append((col, "all horizons", n_diff))
    for hor in (1, 14):
        m = b.horizon == hor
        for col in QCOLS:
            d = np.abs(b.loc[m, col].to_numpy(float) - c.loc[m, col].to_numpy(float))
            g1_rows.append(("%s (d%d)" % (col, hor), "horizon %d" % hor,
                            int((d > 0).sum())))

    for col, scope, n in g1_rows:
        flag = "OK" if n == 0 else "*** %d ROWS DIFFER ***" % n
        print("  %-16s %-14s %s" % (col, scope, flag))

    g1_violations = [(col, scope, n) for col, scope, n in g1_rows if n > 0]
    g1_pass = not g1_violations
    print("\nG1: %s" % ("PASS" if g1_pass else "FAIL"))

    m7 = (b.horizon == 7).to_numpy()
    assert int(m7.sum()) == N_D7, "d7 row count %d" % int(m7.sum())

    if not g1_pass:
        print("\nSTOP. G1 failed: the rebuild changed something outside d7 speed.")
        print("Nothing further is computed. Reporting and halting per the gate.")

    # ---------------- deltas on d7 speed ---------------------------------
    d_lo = c.loc[m7, "q05"].to_numpy(float) - b.loc[m7, "q05"].to_numpy(float)
    d_mid = c.loc[m7, "q50"].to_numpy(float) - b.loc[m7, "q50"].to_numpy(float)
    d_hi = c.loc[m7, "q95"].to_numpy(float) - b.loc[m7, "q95"].to_numpy(float)

    mean_abs_lo = float(np.mean(np.abs(d_lo)))
    mean_abs_hi = float(np.mean(np.abs(d_hi)))
    bound = LIPSCHITZ * (mean_abs_lo + mean_abs_hi)
    width_shift = float(np.mean(d_hi - d_lo))       # exact, signed, truth-free

    print("\n" + "-" * 74)
    print("G2: worst-case bound on |change in mean d7 Winkler|")
    print("-" * 74)
    print("  mean|d q05|            %.6f" % mean_abs_lo)
    print("  mean|d q95|            %.6f" % mean_abs_hi)
    print("  mean|d q50|            %.6f" % float(np.mean(np.abs(d_mid))))
    print("  bound = 21*(sum)       %.6f" % bound)
    print("  margin (27.14-24.7543) %.6f" % MARGIN)
    print("  exact signed width shift mean(d_hi - d_lo) = %+.6f" % width_shift)
    g2_pass = bound < MARGIN
    print("\nG2: %s" % ("PASS" if g2_pass else "FAIL (bound does not clear the margin)"))

    # ---------------- G3: distribution, only on G2 failure ---------------
    season_rows = []
    if not g2_pass:
        print("\n" + "-" * 74)
        print("G3: delta distribution by season (bound failed)")
        print("-" * 74)
        dm = json.loads(DATEMAP_2021.read_text(encoding="utf-8"))
        # The CSV `window` column is 0-INDEXED: the build writes
        # wcol = window_id - 1 (tier2_d7_build_submission.py:126). Keying the
        # season map on window_id therefore drops window 0 and shifts every
        # other label by one. That is exactly the bug the first version of this
        # script shipped, and it silently produced a seasonal table that summed
        # to 1,224,020 instead of 1,398,880. Asserted below so it cannot recur.
        wcol_season = {w["window_id"] - 1: w["season"] for w in dm["windows"]}
        win = b.loc[m7, "window"].to_numpy()
        seasons = np.array([wcol_season.get(int(w), "?") for w in win])
        assert not (seasons == "?").any(), (
            "%d d7 rows have no season: the window column and the datemap "
            "disagree" % int((seasons == "?").sum()))
        combined = np.abs(d_lo) + np.abs(d_hi)
        print("  %-8s %8s %10s %10s %10s %10s"
              % ("season", "n", "mean", "p95", "max", "bound"))
        for s in ["DJF", "MAM", "JJA", "SON"]:
            k = seasons == s
            if not k.any():
                continue
            row = (s, int(k.sum()), float(np.mean(combined[k])),
                   float(np.percentile(combined[k], 95)),
                   float(np.max(combined[k])),
                   LIPSCHITZ * float(np.mean(combined[k])))
            season_rows.append(row)
            print("  %-8s %8d %10.6f %10.6f %10.6f %10.4f" % row)
        covered = sum(r[1] for r in season_rows)
        assert covered == int(combined.size), (
            "seasonal rows cover %d of %d d7 rows" % (covered, combined.size))
        allrow = ("ALL", int(combined.size), float(np.mean(combined)),
                  float(np.percentile(combined, 95)), float(np.max(combined)),
                  bound)
        season_rows.append(allrow)
        print("  %-8s %8d %10.6f %10.6f %10.6f %10.4f" % allrow)

    verdict = ("G1 FAIL, STOP" if not g1_pass else
               ("PASS" if g2_pass else "G2 BOUND FAIL, HALT FOR MATTEO"))

    # ---------------- report ---------------------------------------------
    L = []
    L.append("# Leg B R4b gates G1/G2/G3, 2026-08-22")
    L.append("")
    L.append("Produced by `scripts/legB_R4b_gates_20260822.py`.")
    L.append("")
    L.append("## Verdict: **%s**" % verdict)
    L.append("")
    L.append("## 0. The two alphas, verified against the scorer before use")
    L.append("")
    L.append("They are different quantities and the shared name invites error.")
    L.append("")
    L.append("| Name | Value | What it is |")
    L.append("|---|---|---|")
    L.append("| `ALPHA_LEVEL` | %.2f | Winkler interval LEVEL, fixed by the metric. Nominal central %.0f%% interval. Sets the penalty coefficient. |"
             % (level, 100 * (1 - level)))
    L.append("| `ALPHA_FIX` / `chosen_alpha` | %.2f | Interval WIDTH MULTIPLIER, not a probability. `lo = q50 - 0.90*(q50-q05)`. |"
             % ALPHA_WIDTH)
    L.append("")
    L.append("Exact formula, `ws_d7_feature_experiments.py:113-117`:")
    L.append("")
    L.append("```")
    L.append("W(y, lo, hi) = (hi - lo) + %.0f * max(0, lo - y) + %.0f * max(0, y - hi)"
             % (coef, coef))
    L.append("```")
    L.append("")
    L.append("Not taken on trust: `verify_metric()` imports the real function and")
    L.append("checks it at an interior, a below and an above point, and asserts the")
    L.append("coefficient is exactly %.0f." % coef)
    L.append("")
    for name, got, want in checks:
        L.append("- %s -> %.4f, expected %.4f" % (name, got, want))
    L.append("")
    L.append("## 1. G1: only d7 speed may differ")
    L.append("")
    L.append("| Field | Scope | Rows differing |")
    L.append("|---|---|---|")
    for col, scope, n in g1_rows:
        L.append("| `%s` | %s | %s |" % (col, scope, n if n else "0"))
    L.append("")
    L.append("**G1: %s**" % ("PASS" if g1_pass else "FAIL"))
    L.append("")
    if not g1_pass:
        L.append("The rebuild changed something outside d7 speed. That is a")
        L.append("stop-and-report: the refit had no business touching it. G2 and G3")
        L.append("are not computed.")
        L.append("")
    L.append("## 2. G2: worst-case bound")
    L.append("")
    L.append("| Quantity | Value |")
    L.append("|---|---|")
    L.append("| mean abs delta q05 | %.6f |" % mean_abs_lo)
    L.append("| mean abs delta q95 | %.6f |" % mean_abs_hi)
    L.append("| mean abs delta q50 | %.6f |" % float(np.mean(np.abs(d_mid))))
    L.append("| **Bound** = 21 x (q05 + q95 terms) | **%.6f** |" % bound)
    L.append("| Margin, board 27.14 - 24.7543 | %.6f |" % MARGIN)
    L.append("| Exact signed width shift | %+.6f |" % width_shift)
    L.append("")
    L.append("**G2: %s**" % ("PASS" if g2_pass else "FAIL"))
    L.append("")
    L.append("The bound is worst-case, attained only if every cell sits exactly at an")
    L.append("interval edge and every delta pushes it the wrong way. **A pass is")
    L.append("conclusive; a failure is not evidence the score moved**, only that this")
    L.append("bound cannot rule it out. The signed width shift is exact and")
    L.append("truth-free: it is the part of the change that does not depend on where")
    L.append("truth falls.")
    L.append("")
    if season_rows:
        L.append("## 3. G3: delta distribution by season")
        L.append("")
        L.append("`|d q05| + |d q95|` per row, the quantity the bound multiplies by 21.")
        L.append("")
        L.append("| Season | n | mean | p95 | max | seasonal bound |")
        L.append("|---|---|---|---|---|---|")
        for s, n, mu, p95, mx, bd in season_rows:
            L.append("| %s | %d | %.6f | %.6f | %.6f | %.4f |" % (s, n, mu, p95, mx, bd))
        L.append("")
        L.append("**Halting for Matteo.** No decision is taken here.")
        L.append("")

    text = "\n".join(L) + "\n"
    bad = sorted(set(ch for ch in text if ord(ch) > 126))
    assert not bad, "non-ASCII in report: %r" % bad
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    print("\nwrote %s" % REPORT.relative_to(REPO).as_posix())
    print("\nVERDICT: %s" % verdict)

    return 0 if (g1_pass and g2_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
