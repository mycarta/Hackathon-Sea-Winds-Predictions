#!/usr/bin/env python3
"""
WS d7 speed bias correction, Steps 2-4 - 2026-07-08.

Applies the per-cell x per-season bias-shrunk correction table built by
scripts/ws_d7_bias_correction_build_table.py to submission.csv's d7 speed
columns (q05/q50/q95), then re-tightens the intervals using the
post-correction Winkler-optimal alpha found by that same script.

Order (per task prompt): bias correction first (move the center), then
alpha recalibration (tighten around the corrected center).

  speed_q{05,50,95}_new = speed_q{05,50,95} - bias_shrunk(cell, season)
  speed_q05_new = max(0, speed_q05_new)

  speed_q95_final = speed_q50_new + alpha * (speed_q95_new - speed_q50_new)
  speed_q05_final = speed_q50_new - alpha * (speed_q50_new - speed_q05_new)
  speed_q05_final = max(0, speed_q05_final)
  speed_q50_final = speed_q50_new (unchanged by the alpha step)

Season per window: each of the 8 (2021) eval windows has a single d7
valid date (issue_date + 7 days), so every row in a window's horizon==7
block shares one season - looked up via splits.eval_windows(2021), same
convention as scripts/dir_residual_apply.py.

Row <-> footprint-cell correspondence is positional (submission.csv is 96
contiguous 43,715-row blocks matching np.where(footprint_mask()) order) -
same validated approach as the d14 climatology fix and direction-residual
fix. Only the 32 horizon==7 blocks (8 windows x 4 hours) are touched; the
other 64 blocks (horizon 1 and 14) and ALL direction columns are left
byte-identical.

Checkpoints the pre-fix submission.csv before any write (build discipline
#5) to its own named backup, separate from prior fixes' checkpoints.

IMPORTANT (fixed 2026-07-08, after a re-run with a revised alpha):
this script always reads its SOURCE from BACKUP_PATH (the pristine,
pre-bias-correction submission), never from the live submission.csv. The
backup is created once on the first-ever run and never overwritten after
that. This makes the script safe to re-run with a different chosen_alpha
(e.g. after scripts/ws_d7_set_chosen_alpha.py updates the params file) -
every run applies bias-correction + alpha in ONE pass from the same
untouched baseline, so the bias can never be double-applied.
"""

import os
import json
import shutil
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
CSV_PATH = SUB_DIR / "submission.csv"
ZIP_PATH = SUB_DIR / "submission.zip"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
BACKUP_PATH = ARTIFACTS_DIR / "submission_pre_d7_bias_correction_backup.csv"
TABLE_PATH = ARTIFACTS_DIR / "ws_d7_bias_shrunk_table.parquet"
PARAMS_PATH = ARTIFACTS_DIR / "ws_d7_bias_correction_params.json"
REPORT_PATH = ROOT / "reports" / "ws_d7_bias_correction_20260708.md"

KIT_PART0 = ROOT / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"
sys.path.insert(0, str(KIT_PART0))
import splits  # noqa: E402

# Runbook errata 2026-08-19. SEAWINDS_EVAL_YEAR is retained as a manual override
# but is REDUNDANT as of kit 9edd92b: splits.eval_windows() with no argument now
# auto-detects the year from the INSTALLED inference windows
# (splits._installed_windows_year reads window_1/metadata.json), so replacing
# inference/ in place re-dates everything. The calls below pass no argument and
# inherit that auto-detect; EVAL_YEAR is kept only for logging and assertions.
EVAL_YEAR = int(os.environ.get("SEAWINDS_EVAL_YEAR", 2021))  # runbook errata 2026-08-19,
# blocker 3: the swap re-run scores a different year than the frozen default. Unset -> 2021,
# so the 2021 rehearsal still reproduces byte-identically; SEAWINDS_EVAL_YEAR=2022 for the
# final-evaluation run. splits.eval_windows() re-dates _BASE_WINDOWS (defined in 2022) to the
# requested year, and eval_windows(2022) was verified equal to the organizers' shipped
# window_1..8 metadata on all four date fields plus score_days (2026-08-19).
LEAD = 7
HOURS = (0, 6, 12, 18)
N_CELLS = 43715
SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def main():
    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if BACKUP_PATH.exists():
        print(f"Pre-fix checkpoint already exists at {BACKUP_PATH} - reading FROM this "
              f"pristine baseline (not the live submission.csv), so re-running with a "
              f"different alpha applies bias+alpha in one pass and never double-applies.")
    else:
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print(f"Checkpointed pre-fix submission.csv -> {BACKUP_PATH}")

    table_df = pd.read_parquet(TABLE_PATH)
    with open(PARAMS_PATH, encoding="utf-8") as f:
        params = json.load(f)
    alpha = params["chosen_alpha"]
    # Fail-loud alpha guard (swap runbook amendment 2026-07-21): the frozen base
    # (835026 / 855076) is alpha=0.90, but ws_d7_set_chosen_alpha.py currently pins
    # APPROVED_ALPHA=1.0. Abort rather than silently build a 1.0 base on swap day.
    assert abs(alpha - 0.90) < 1e-9, (
        f"chosen_alpha={alpha} != 0.90 -- the swap base must be alpha=0.90. Run "
        f"ws_d7_set_chosen_alpha.py with 0.90 (swap runbook Step 0) before this step.")
    print(f"Loaded bias table ({len(table_df)} rows) and params. "
          f"Chosen alpha = {alpha} (holdout Winkler {params['chosen_alpha_winkler']:.3f}, "
          f"vs {params['before_correction']['mean_winkler']:.3f} pre-correction).")

    # per-season, cell-ordered (0..N_CELLS-1) bias_shrunk lookup arrays
    bias_lookup = {}
    for season, grp in table_df.groupby("season"):
        grp = grp.sort_values("cell_idx")
        assert (grp["cell_idx"].to_numpy() == np.arange(N_CELLS)).all()
        bias_lookup[season] = grp["bias_shrunk"].to_numpy(dtype="float64")
    print(f"Seasons in table: {sorted(bias_lookup.keys())}")

    windows = splits.eval_windows()
    window_season = {}
    for w_idx, w in enumerate(windows):
        issue_date = date.fromisoformat(w["context_end"])
        valid_date = issue_date + timedelta(days=LEAD)
        season = SEASON_OF_MONTH[valid_date.month]
        window_season[w_idx] = season
        print(f"  window {w_idx} (id={w['id']}): issue={issue_date}, "
              f"d7 valid={valid_date}, season={season}")

    df = pd.read_csv(BACKUP_PATH)
    df_before = df.copy()

    grp_sizes = df.groupby(["window", "horizon", "hour"], sort=False).size()
    assert len(grp_sizes) == 96 and (grp_sizes == N_CELLS).all()
    keys = list(zip(df["window"], df["horizon"], df["hour"]))
    boundaries = [i for i in range(len(keys)) if i == 0 or keys[i] != keys[i - 1]]
    assert len(boundaries) == 96
    block_starts = {keys[i]: i for i in boundaries}
    print("Verified 96 contiguous 43,715-row blocks.")

    for w_idx in range(8):
        season = window_season[w_idx]
        bias_shrunk = bias_lookup[season]  # (N_CELLS,), cell-position ordered
        for H in HOURS:
            key = (w_idx, LEAD, H)
            assert key in block_starts, f"missing block {key}"
            start = block_starts[key]
            end = start + N_CELLS
            assert (df.loc[start:end - 1, "window"] == w_idx).all()
            assert (df.loc[start:end - 1, "horizon"] == LEAD).all()
            assert (df.loc[start:end - 1, "hour"] == H).all()

            q05 = df.loc[start:end - 1, "q05"].to_numpy(dtype="float64")
            q50 = df.loc[start:end - 1, "q50"].to_numpy(dtype="float64")
            q95 = df.loc[start:end - 1, "q95"].to_numpy(dtype="float64")

            # Step 2: bias correction
            q05_new = np.maximum(0.0, q05 - bias_shrunk)
            q50_new = q50 - bias_shrunk
            q95_new = q95 - bias_shrunk

            # Step 3: alpha recalibration around the corrected center
            q95_final = q50_new + alpha * (q95_new - q50_new)
            q05_final = np.maximum(0.0, q50_new - alpha * (q50_new - q05_new))
            q50_final = q50_new

            df.loc[start:end - 1, "q05"] = np.round(q05_final, 3)
            df.loc[start:end - 1, "q50"] = np.round(q50_final, 3)
            df.loc[start:end - 1, "q95"] = np.round(q95_final, 3)

    # ---- scope guards ----
    d7_mask = df["horizon"] == LEAD
    other_mask = ~d7_mask
    unchanged = df.loc[other_mask].equals(df_before.loc[other_mask])
    print(f"\nhorizon in (1,14) rows fully unchanged: {unchanged}")
    if not unchanged:
        diff_cols = [c for c in df.columns
                     if not df.loc[other_mask, c].equals(df_before.loc[other_mask, c])]
        print(f"  DIFFERING columns outside scope: {diff_cols}")
    assert unchanged, "rows outside horizon==7 were modified - aborting write"

    dir_cols = ["dir_05", "dir_50", "dir_95"]
    dir_unchanged = df[dir_cols].equals(df_before[dir_cols])
    print(f"ALL direction columns unchanged (every horizon): {dir_unchanged}")
    assert dir_unchanged, "direction columns were modified - aborting write"

    other_speed_cols_unchanged = df.loc[d7_mask, ["type", "window", "region", "latitude",
                                                    "longitude", "horizon", "hour", "level"]].equals(
        df_before.loc[d7_mask, ["type", "window", "region", "latitude", "longitude",
                                  "horizon", "hour", "level"]])
    print(f"Non-value columns on d7 rows unchanged: {other_speed_cols_unchanged}")
    assert other_speed_cols_unchanged

    assert (df["q05"] >= 0).all(), "negative q05 after correction"
    assert (df["q05"] <= df["q50"]).all() and (df["q50"] <= df["q95"]).all(), \
        "quantile ordering violated after correction"
    assert df[["q05", "q50", "q95"]].notna().all().all()
    assert len(df) == 4_196_640

    df.to_csv(CSV_PATH, index=False)
    print(f"\nWrote updated {CSV_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(CSV_PATH, "submission.csv")
    print(f"Re-zipped {ZIP_PATH}")

    # ---- before/after metrics ----
    b7 = df_before[df_before["horizon"] == LEAD]
    a7 = df[df["horizon"] == LEAD]
    metrics = {
        "q50_median_before": float(b7["q50"].median()),
        "q50_median_after": float(a7["q50"].median()),
        "width_median_before": float((b7["q95"] - b7["q05"]).median()),
        "width_median_after": float((a7["q95"] - a7["q05"]).median()),
    }
    print("\n--- Before/after d7 speed metrics (full submission, all windows/hours) ---")
    print(f"  q50 median: {metrics['q50_median_before']:.3f} -> {metrics['q50_median_after']:.3f} m/s")
    print(f"  q95-q05 width median: {metrics['width_median_before']:.3f} -> "
          f"{metrics['width_median_after']:.3f} m/s "
          f"({(1 - metrics['width_median_after']/metrics['width_median_before'])*100:.1f}% narrower)")

    report_lines = [
        "\n## Step 4: validation gate + apply results\n",
        f"Applied to submission.csv: {len(df)} rows, 0 NaN in q05/q50/q95, "
        f"q05<=q50<=q95 verified, all q05>=0. horizon in (1,14) rows and ALL direction "
        f"columns (every horizon) verified byte-identical pre/post.\n",
        "| Metric | Before | After |",
        "|---|---|---|",
        f"| d7 q50 median (m/s) | {metrics['q50_median_before']:.3f} | {metrics['q50_median_after']:.3f} |",
        f"| d7 q95-q05 width median (m/s) | {metrics['width_median_before']:.3f} | "
        f"{metrics['width_median_after']:.3f} |",
        f"\nWidth narrowed by "
        f"{(1 - metrics['width_median_after']/metrics['width_median_before'])*100:.1f}% "
        f"(alpha={alpha} applied post-bias-correction).\n",
        f"\nOutput: `{ZIP_PATH.relative_to(ROOT)}` (updated). Not submitted to Codabench - "
        f"Matteo submits.\n",
    ]
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nAppended Step 4 section to {REPORT_PATH}")


if __name__ == "__main__":
    main()
