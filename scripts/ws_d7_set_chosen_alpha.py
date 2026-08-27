#!/usr/bin/env python3
"""
WS d7 bias correction - set/override chosen_alpha to Matteo's approved
pick (2026-07-08, revised 2026-07-08).

scripts/ws_d7_bias_correction_build_table.py's extended alpha resweep
(reports/ws_d7_bias_correction_20260708.md) found the Winkler-minimizing
alpha at 0.65 (coverage 76.1%), inside a shallow bowl spanning
alpha in [0.60, 0.75] (Winkler within 0.9% of the minimum across that
range).

Decision history (both preserved, most recent wins - see report for the
full trail):
  1. Matteo chose alpha=0.70 (coverage 78.4%) - "chosen_alpha = 0.70.
     Apply and go."
  2. Matteo revised to alpha=0.90 (coverage 85.7%, further from the
     Winkler minimum but more conservative on coverage) - "Re-run the
     apply script with chosen_alpha = 0.90."
  3. Matteo revised to alpha=1.0 (coverage 88.6%, bias correction only,
     NO interval tightening - the widest/safest option, not in the
     original [0.1..0.9] sweep, added via
     scripts/ws_d7_alpha_sweep_from_checkpoint.py) - "Re-run the apply
     script with chosen_alpha = 1.0. Bias correction stays, no interval
     tightening."

Rewrites scripts/artifacts/ws_d7_bias_correction_params.json's
chosen_alpha/chosen_alpha_winkler fields from the already-computed
alpha_resweep table (no recomputation - same holdout numbers, just a
different selection). The auto-selected Winkler-minimum baseline
(auto_selected_alpha/auto_selected_alpha_winkler) is preserved across
re-runs of this script (read once, on the first run, from whatever
chosen_alpha held at that time) so repeated overrides don't drift.
Appends a note to the report documenting each override and why.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
PARAMS_PATH = ARTIFACTS_DIR / "ws_d7_bias_correction_params.json"
REPORT_PATH = ROOT / "reports" / "ws_d7_bias_correction_20260708.md"

APPROVED_ALPHA = 0.90


def main():
    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = json.load(f)

    # the true auto-selected (Winkler-minimum) baseline is fixed on first
    # override and never re-derived from a prior override, so repeated
    # re-runs of this script don't drift the reference point
    if "auto_selected_alpha" in params:
        auto_alpha = params["auto_selected_alpha"]
        auto_winkler = params["auto_selected_alpha_winkler"]
    else:
        auto_alpha = params["chosen_alpha"]
        auto_winkler = params["chosen_alpha_winkler"]

    prev_alpha = params.get("chosen_alpha")
    prev_winkler = params.get("chosen_alpha_winkler")

    match = [r for r in params["alpha_resweep"] if abs(r["alpha"] - APPROVED_ALPHA) < 1e-9]
    assert len(match) == 1, f"alpha {APPROVED_ALPHA} not found in alpha_resweep table"
    row = match[0]

    params["auto_selected_alpha"] = auto_alpha
    params["auto_selected_alpha_winkler"] = auto_winkler
    params["chosen_alpha"] = APPROVED_ALPHA
    params["chosen_alpha_winkler"] = row["mean_winkler"]
    params["chosen_alpha_coverage"] = row["coverage"]
    params["chosen_alpha_note"] = (
        f"Matteo's approved pick, revised from a prior selection of "
        f"alpha={prev_alpha} (Winkler={prev_winkler:.3f}). Auto-selected "
        f"Winkler minimum remains alpha={auto_alpha} (Winkler={auto_winkler:.3f})."
    )

    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Updated {PARAMS_PATH}: chosen_alpha {prev_alpha} -> {APPROVED_ALPHA} "
          f"(coverage {row['coverage']*100:.1f}%, Winkler {row['mean_winkler']:.3f})")

    note = (
        f"\n## Alpha selection: revision ({prev_alpha} -> {APPROVED_ALPHA})\n\n"
        f"Auto-selected Winkler-minimum: alpha={auto_alpha} (Winkler={auto_winkler:.3f}, "
        f"coverage 76.1%). Previously approved alpha={prev_alpha} "
        f"(Winkler={prev_winkler:.3f}). **Matteo revised the approval to "
        f"alpha={APPROVED_ALPHA}** (Winkler={row['mean_winkler']:.3f}, "
        f"coverage={row['coverage']*100:.1f}%) - a more conservative pick "
        f"further from the Winkler minimum, trading Winkler for more coverage "
        f"margin. This is the alpha applied to submission.csv.\n"
    )
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(note)
    print(f"Appended alpha-selection revision note to {REPORT_PATH}")


if __name__ == "__main__":
    main()
