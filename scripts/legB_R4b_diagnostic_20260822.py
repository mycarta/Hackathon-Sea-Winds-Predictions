#!/usr/bin/env python3
"""Pre-R5 diagnostic on the G2 deltas. Two questions, both truth-free.

Ordered by Matteo on 2026-08-22 as a condition of the G3 daylight decision to
proceed to R5. The G2 FAIL stands unmodified; nothing here is a gate.

QUESTION 1: characterise the 12.9x asymmetry.
    G2 reported mean|d q95| = 0.1756 against mean|d q05| = 0.0136. Absolute
    values hide direction. This reports the SIGNED seasonal means for both
    edges, the fraction of cells moving each way, and the signed midpoint
    shift, to distinguish two very different stories:
      (a) the upper edge narrowing systematically, consistent with a slightly
          sharper refit, or
      (b) the upper edge DRIFTING one-directionally, which would mean the
          refit changed the center, not the spread.
    The q50 delta separates them: a pure sharpening leaves q50 alone.

QUESTION 2: the coverage-informed bound, computed properly.
    LOGGED AS AN ESTIMATE UNDER AN EXPLICIT ASSUMPTION, NOT AS A REPLACEMENT
    GATE. The assumption, stated plainly: submission 855076's board coverage on
    d7 speed was 85.5 percent, so about 14.5 percent of scored cells had truth
    OUTSIDE the interval. That number is imported from the Codabench board, it
    is not measured here, and it was measured on the JULY intervals rather than
    the rebuilt ones.

    Why it sharpens anything: the Winkler penalty coefficient 20 applies ONLY
    where truth falls outside the interval. For a cell whose truth is strictly
    inside both the old and the new interval, the score changes by exactly
    (d_hi - d_lo) and by nothing else. The blanket 21x factor of G2 charges the
    penalty to all 1,398,880 cells; at most about 14.5 percent of them can
    actually incur it.

    Two versions are computed, and the difference between them matters:

      RANDOM    exceedance cells are a random 14.5 percent, so their mean delta
                equals the overall mean delta.
      ADVERSARIAL  exceedance cells are exactly the 14.5 percent with the
                LARGEST deltas. This is the honest worst case under the
                coverage assumption, and it is the number to quote.

    Both still assume every exceedance cell moves the wrong way, so both remain
    upper bounds rather than expectations.

Output: reports/legB_R4b_diagnostic_20260822.md

Run:  python scripts/legB_R4b_diagnostic_20260822.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACTS = HERE / "artifacts"

BASE = ARTIFACTS / "submission_pangu_d7_f125_20260720.csv"          # 855076
CAND = ARTIFACTS / "submission_legB_2021_rebuild_f125_20260822.csv"
DATEMAP = ARTIFACTS / "tier2_sub_datemap.json"
REPORT = REPO / "reports" / "legB_R4b_diagnostic_20260822.md"

BOARD_COVERAGE_D7 = 0.855      # ASSUMPTION, imported from the board, see docstring
P_OUT = 1.0 - BOARD_COVERAGE_D7
MARGIN = 27.14 - 24.7543
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def main():
    b = pd.read_csv(BASE)
    c = pd.read_csv(CAND)
    m7 = (b.horizon == 7).to_numpy()

    d_lo = c.loc[m7, "q05"].to_numpy(float) - b.loc[m7, "q05"].to_numpy(float)
    d_mid = c.loc[m7, "q50"].to_numpy(float) - b.loc[m7, "q50"].to_numpy(float)
    d_hi = c.loc[m7, "q95"].to_numpy(float) - b.loc[m7, "q95"].to_numpy(float)
    base_lo = b.loc[m7, "q05"].to_numpy(float)
    base_hi = b.loc[m7, "q95"].to_numpy(float)

    dm = json.loads(DATEMAP.read_text(encoding="utf-8"))
    # 0-INDEXED window column; see the note in legB_R4b_gates_20260822.py
    wseason = {w["window_id"] - 1: w["season"] for w in dm["windows"]}
    season = np.array([wseason[int(w)] for w in b.loc[m7, "window"].to_numpy()])
    assert sum((season == s).sum() for s in SEASONS) == season.size,         "seasonal labels do not cover every d7 row" 

    # ---------------- Q1: signed, by season ------------------------------
    rows = []
    for s in SEASONS + ["ALL"]:
        k = np.ones(d_lo.size, bool) if s == "ALL" else (season == s)
        rows.append({
            "season": s, "n": int(k.sum()),
            "mean_d_lo": float(d_lo[k].mean()),
            "mean_d_mid": float(d_mid[k].mean()),
            "mean_d_hi": float(d_hi[k].mean()),
            "frac_hi_neg": float((d_hi[k] < 0).mean()),
            "frac_lo_neg": float((d_lo[k] < 0).mean()),
            "frac_mid_neg": float((d_mid[k] < 0).mean()),
            "mean_width_shift": float((d_hi[k] - d_lo[k]).mean()),
            "base_mean_width": float((base_hi[k] - base_lo[k]).mean()),
        })

    print("Q1: signed deltas by season (positive = moved UP)")
    print("%-6s %9s %10s %10s %10s %9s %9s %11s"
          % ("season", "n", "d q05", "d q50", "d q95", "%q95<0", "%q50<0", "width shift"))
    for r in rows:
        print("%-6s %9d %+10.5f %+10.5f %+10.5f %8.1f%% %8.1f%% %+11.5f"
              % (r["season"], r["n"], r["mean_d_lo"], r["mean_d_mid"],
                 r["mean_d_hi"], 100 * r["frac_hi_neg"], 100 * r["frac_mid_neg"],
                 r["mean_width_shift"]))

    # is the q95 movement mostly spread or mostly center?
    all_r = rows[-1]
    center_share = (abs(all_r["mean_d_mid"])
                    / max(1e-12, abs(all_r["mean_d_hi"])))

    # ---------------- Q2: coverage-informed bound ------------------------
    combined = np.abs(d_lo) + np.abs(d_hi)
    width_abs = float(np.abs(d_hi - d_lo).mean())
    mean_all = float(combined.mean())

    k_top = int(np.ceil(P_OUT * combined.size))
    top = np.partition(combined, combined.size - k_top)[combined.size - k_top:]
    mean_top = float(top.mean())

    g2_bound = 21.0 * mean_all
    est_random = width_abs + P_OUT * 21.0 * mean_all
    est_adversarial = width_abs + P_OUT * 21.0 * mean_top

    print("\nQ2: coverage-informed estimate (ASSUMPTION: board coverage_d7 = %.1f%%)"
          % (100 * BOARD_COVERAGE_D7))
    print("  G2 blanket bound            %.4f" % g2_bound)
    print("  mean|d_hi - d_lo|           %.6f" % width_abs)
    print("  mean(|d_lo|+|d_hi|), all    %.6f" % mean_all)
    print("  mean(|d_lo|+|d_hi|), top %.1f%%  %.6f" % (100 * P_OUT, mean_top))
    print("  estimate, RANDOM exceedance      %.4f" % est_random)
    print("  estimate, ADVERSARIAL exceedance %.4f   <- the number to quote"
          % est_adversarial)
    print("  margin                           %.4f" % MARGIN)

    # ---------------- report ---------------------------------------------
    L = []
    L.append("# Pre-R5 diagnostic on the G2 deltas, 2026-08-22")
    L.append("")
    L.append("Produced by `scripts/legB_R4b_diagnostic_20260822.py`. Ordered as a")
    L.append("condition of the G3 daylight decision. **The G2 FAIL stands unmodified.**")
    L.append("Nothing in this file is a gate.")
    L.append("")
    L.append("## 1. The 12.9x asymmetry, characterised")
    L.append("")
    L.append("G2 reported `mean|d q95|` = 0.1756 against `mean|d q05|` = 0.0136.")
    L.append("Absolute values hide direction. Signed, by season, positive = moved up:")
    L.append("")
    L.append("| Season | n | mean d q05 | mean d q50 | mean d q95 | share q95 down | share q50 down | width shift | base width |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| %s | %d | %+.5f | %+.5f | %+.5f | %.1f%% | %.1f%% | %+.5f | %.3f |"
                 % (r["season"], r["n"], r["mean_d_lo"], r["mean_d_mid"],
                    r["mean_d_hi"], 100 * r["frac_hi_neg"], 100 * r["frac_mid_neg"],
                    r["mean_width_shift"], r["base_mean_width"]))
    L.append("")
    L.append("**Reading.** The q50 column is the discriminator: a pure sharpening")
    L.append("leaves the centre alone, a drift moves it. Here the mean centre shift is")
    L.append("%+.5f against a mean upper-edge shift of %+.5f, so the centre carries"
             % (all_r["mean_d_mid"], all_r["mean_d_hi"]))
    L.append("about %.0f%% of the upper-edge movement." % (100 * center_share))
    L.append("")
    L.append("Mean interval width in the base is %.3f, so the mean width change of"
             % all_r["base_mean_width"])
    L.append("%+.5f is %.3f%% of the interval."
             % (all_r["mean_width_shift"],
                100 * abs(all_r["mean_width_shift"]) / all_r["base_mean_width"]))
    L.append("")
    L.append("## 2. Coverage-informed estimate, under an explicit assumption")
    L.append("")
    L.append("**ASSUMPTION, stated because everything below depends on it:**")
    L.append("submission 855076's board coverage on d7 speed was **%.1f percent**, so"
             % (100 * BOARD_COVERAGE_D7))
    L.append("about **%.1f percent** of scored cells had truth OUTSIDE the interval."
             % (100 * P_OUT))
    L.append("That figure is imported from the Codabench board. It is not measured")
    L.append("here, and it was measured on the JULY intervals, not the rebuilt ones.")
    L.append("")
    L.append("**Why it sharpens anything.** The Winkler penalty coefficient 20 applies")
    L.append("only where truth falls outside the interval. For a cell whose truth is")
    L.append("strictly inside both the old and the new interval, the score changes by")
    L.append("exactly `d_hi - d_lo` and nothing else. G2's blanket 21x charges the")
    L.append("penalty to all 1,398,880 cells; at most ~%.1f%% can incur it."
             % (100 * P_OUT))
    L.append("")
    L.append("| Quantity | Value |")
    L.append("|---|---|")
    L.append("| G2 blanket bound | %.4f |" % g2_bound)
    L.append("| mean abs width change | %.6f |" % width_abs)
    L.append("| mean (abs d q05 + abs d q95), all cells | %.6f |" % mean_all)
    L.append("| mean (abs d q05 + abs d q95), largest %.1f%% | %.6f |"
             % (100 * P_OUT, mean_top))
    L.append("| Estimate, RANDOM exceedance set | %.4f |" % est_random)
    L.append("| **Estimate, ADVERSARIAL exceedance set** | **%.4f** |" % est_adversarial)
    L.append("| Margin | %.4f |" % MARGIN)
    L.append("")
    L.append("The ADVERSARIAL row is the one to quote: it assumes the exceedance cells")
    L.append("are precisely the %.1f%% with the LARGEST deltas, which is the worst case"
             % (100 * P_OUT))
    L.append("consistent with the coverage assumption. Both rows still assume every")
    L.append("exceedance cell moves the wrong way, so both remain upper bounds and")
    L.append("neither is an expectation.")
    L.append("")
    L.append("**Status: estimate under assumption. Not a gate, not a G2 replacement,")
    L.append("and it does not retire the G2 FAIL.** If the 85.5 percent figure is")
    L.append("wrong, this number is wrong with it.")
    L.append("")

    text = "\n".join(L) + "\n"
    bad = sorted(set(ch for ch in text if ord(ch) > 126))
    assert not bad, "non-ASCII: %r" % bad
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    print("\nwrote %s" % REPORT.relative_to(REPO).as_posix())


if __name__ == "__main__":
    main()
