#!/usr/bin/env python3
"""
WS d7 bias correction - extend the alpha resweep table from the persisted
holdout checkpoint (2026-07-08), no pipeline rerun needed.

scripts/ws_d7_bias_correction_build_table.py checkpoints the raw
per-season stacked holdout arrays (q05/q50/q95/truth, pre-bias-correction)
to scripts/artifacts/ws_d7_bias_correction_stacked.npz specifically so
alpha values outside the originally-swept [0.1..0.9] grid can be
evaluated in seconds instead of re-running the ~30 min forecast+downscale
pipeline.

Matteo asked for alpha=1.0 (bias correction only, NO interval tightening
- the "before" width, correctly re-centered). Computes its coverage/
width/Winkler from the checkpoint, inserts it into
ws_d7_bias_correction_params.json's alpha_resweep table (sorted, no
duplicates), and appends a note to the report.

Does NOT change chosen_alpha - run scripts/ws_d7_set_chosen_alpha.py
afterward (with APPROVED_ALPHA set to the new value) once its row exists
in the table.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
STACKED_PATH = ARTIFACTS_DIR / "ws_d7_bias_correction_stacked.npz"
TABLE_PATH = ARTIFACTS_DIR / "ws_d7_bias_shrunk_table.parquet"
PARAMS_PATH = ARTIFACTS_DIR / "ws_d7_bias_correction_params.json"
REPORT_PATH = ROOT / "reports" / "ws_d7_bias_correction_20260708.md"

N_CELLS = 43715
ALPHA_LEVEL = 0.10
SEASONS = ("DJF", "MAM", "JJA", "SON")
NEW_ALPHAS = (1.0,)


def winkler_score(y, lo, hi, alpha_level=ALPHA_LEVEL):
    width = hi - lo
    below = np.maximum(0.0, lo - y) * (2.0 / alpha_level)
    above = np.maximum(0.0, y - hi) * (2.0 / alpha_level)
    return width + below + above


def main():
    npz = np.load(STACKED_PATH)
    table_df = pd.read_parquet(TABLE_PATH)

    bias_lookup = {}
    for season, grp in table_df.groupby("season"):
        grp = grp.sort_values("cell_idx")
        assert (grp["cell_idx"].to_numpy() == np.arange(N_CELLS)).all()
        bias_lookup[season] = grp["bias_shrunk"].to_numpy(dtype="float64")

    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = json.load(f)

    existing_alphas = {round(r["alpha"], 6) for r in params["alpha_resweep"]}
    new_rows = []
    for a in NEW_ALPHAS:
        if round(a, 6) in existing_alphas:
            print(f"alpha={a} already in the table - skipping recompute.")
            continue
        n_tot, cov_tot, width_sum, wink_sum = 0, 0, 0.0, 0.0
        for season in SEASONS:
            key_q05, key_q50, key_q95, key_truth = (
                f"{season}_q05", f"{season}_q50", f"{season}_q95", f"{season}_truth")
            if key_q05 not in npz:
                continue
            bias_shrunk_cell = bias_lookup[season]
            q05 = npz[key_q05] - bias_shrunk_cell[None, :]
            q50 = npz[key_q50] - bias_shrunk_cell[None, :]
            q95 = npz[key_q95] - bias_shrunk_cell[None, :]
            q05 = np.maximum(0.0, q05)
            truth = npz[key_truth]

            lo = q50 - a * (q50 - q05)
            hi = q50 + a * (q95 - q50)
            lo = np.maximum(0.0, lo)
            covered = (truth >= lo) & (truth <= hi)

            n_tot += truth.size
            cov_tot += int(covered.sum())
            width_sum += float((hi - lo).sum())
            wink_sum += float(winkler_score(truth, lo, hi).sum())

        row = {"alpha": a, "coverage": cov_tot / n_tot,
               "mean_width": width_sum / n_tot, "mean_winkler": wink_sum / n_tot}
        new_rows.append(row)
        print(f"alpha={a}: coverage={row['coverage']*100:.1f}%, "
              f"width={row['mean_width']:.3f}, winkler={row['mean_winkler']:.3f}")

    if not new_rows:
        print("Nothing new to add.")
        return

    params["alpha_resweep"].extend(new_rows)
    params["alpha_resweep"].sort(key=lambda r: r["alpha"])
    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Updated {PARAMS_PATH} with {len(new_rows)} new alpha row(s).")

    lines = ["\n## Alpha resweep extension (from checkpoint, no pipeline rerun)\n",
             "| alpha | coverage | mean width (m/s) | mean Winkler |",
             "|---|---|---|---|"]
    for r in new_rows:
        lines.append(f"| {r['alpha']} | {r['coverage']*100:.1f}% | {r['mean_width']:.3f} | "
                      f"{r['mean_winkler']:.3f} |")
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Appended extension note to {REPORT_PATH}")


if __name__ == "__main__":
    main()
