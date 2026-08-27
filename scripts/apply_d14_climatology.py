#!/usr/bin/env python3
"""
Apply the d14 climatology lookup table (scripts/artifacts/
d14_climatology_season.parquet, built by compute_d14_climatology.py) to
submission.csv: replace speed (q05/q50/q95) and direction (dir_05/dir_50/
dir_95) for horizon==14 rows only. horizon 1 and 7 rows are left untouched
(and byte-identical-checked below).

Row <-> footprint-cell correspondence is positional, not a (lat,lon) value
join: submission.csv is verified (debug session, 2026-07-06) to consist of
96 perfectly contiguous 43,715-row blocks (8 windows x 3 horizons x 4
hours), and every block's row order matches
np.where(footprint.footprint_mask()) exactly. A value-based (lat,lon) join
was tried first and rejected: rounding 11/43,715 cells' latitude to 2dp
lands exactly on a .xx5 boundary, where the kit's float32 round-then-cast
and this script's independent float64 round pick opposite round-half-to-
even directions - a real but harmless artifact that breaks dict/merge
joins on rounded floats. Positional indexing sidesteps it entirely.

Season per window is looked up from the kit's own splits.py::eval_windows()
(single source of truth for window -> d14 valid date), not hardcoded here,
so it stays correct if the eval year or window dates ever change.

Checkpoints to disk BEFORE the risky in-place overwrite (build discipline
#5): a full pre-fix copy of submission.csv is saved before any row is
touched.

Rounding matches the kit's own CSV convention (build_forecast_submission.py
:78-85): lat/lon 2dp (unchanged here, only value columns touched),
q05/q50/q95/dir_05/dir_50/dir_95 3dp.
"""

import shutil
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
CSV_PATH = SUB_DIR / "submission.csv"
ZIP_PATH = SUB_DIR / "submission.zip"
CLIM_PATH = Path(__file__).resolve().parent / "artifacts" / "d14_climatology_season.parquet"
BACKUP_PATH = Path(__file__).resolve().parent / "artifacts" / "submission_pre_d14fix_backup.csv"

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
N_CELLS = 43715

SEASON_OF_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def window_to_season():
    """window (0-indexed, matches submission.csv's 'window' column) -> season,
    derived from splits.eval_windows()'s score_days['d14']."""
    windows = splits.eval_windows()
    mapping = {}
    for i, w in enumerate(windows):
        d14_date = date.fromisoformat(w["score_days"]["d14"])
        mapping[i] = SEASON_OF_MONTH[d14_date.month]
        print(f"  window {i} (id={w['id']}): d14 valid date = {d14_date} -> season {mapping[i]}")
    return mapping


def main():
    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if BACKUP_PATH.exists():
        print(f"Pre-fix checkpoint already exists at {BACKUP_PATH} - not overwriting "
              f"(a rerun of this script must not clobber the one true pre-fix snapshot).")
    else:
        shutil.copy2(CSV_PATH, BACKUP_PATH)
        print(f"Checkpointed pre-fix submission.csv -> {BACKUP_PATH}")

    df = pd.read_csv(CSV_PATH)
    df_before = df.copy()

    # Verify the block structure this script's positional-indexing approach
    # depends on, before touching anything.
    grp_sizes = df.groupby(["window", "horizon", "hour"], sort=False).size()
    assert len(grp_sizes) == 8 * 3 * 4, f"expected 96 blocks, got {len(grp_sizes)}"
    assert (grp_sizes == N_CELLS).all(), "not every (window,horizon,hour) block has 43,715 rows"
    keys = list(zip(df["window"], df["horizon"], df["hour"]))
    boundaries = [i for i in range(len(keys)) if i == 0 or keys[i] != keys[i - 1]]
    assert len(boundaries) == 96, f"blocks are not contiguous: found {len(boundaries)} boundaries"
    block_starts = {keys[i]: i for i in boundaries}
    print(f"Verified 96 contiguous {N_CELLS}-row blocks, positional footprint-order alignment holds.")

    clim = pd.read_parquet(CLIM_PATH)
    clim_by_bin = {
        (season, hour): g.sort_values("point_id").reset_index(drop=True)
        for (season, hour), g in clim.groupby(["season", "hour"])
    }

    print("Window -> season mapping (from splits.eval_windows):")
    win_season = window_to_season()

    d14_mask = df["horizon"] == 14
    n_d14 = int(d14_mask.sum())
    print(f"\n{n_d14} horizon==14 rows to update across {n_d14 // N_CELLS} (window,hour) blocks")

    value_cols = ["q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]
    clim_cols = ["speed_q05", "speed_q50", "speed_q95", "dir_05", "dir_50", "dir_95"]
    dir_dst_cols = {"dir_05", "dir_50", "dir_95"}

    def round3_wrap360(values):
        # round(x,3) can push an angle like 359.9997 up to exactly 360.000,
        # outside the required [0,360) range (same boundary artifact fixed
        # in fix_dir_360_wrap.py for the kit's own CSV writer). Wrap it back
        # to 0.000 at the source instead of shipping the artifact again.
        rounded = np.round(values, 3)
        return np.where(rounded >= 360.0, 0.0, rounded)

    for window in range(8):
        season = win_season[window]
        for hour in (0, 6, 12, 18):
            key = (window, 14, hour)
            assert key in block_starts, f"missing block {key}"
            start = block_starts[key]
            end = start + N_CELLS
            assert (df.loc[start:end - 1, "window"] == window).all()
            assert (df.loc[start:end - 1, "horizon"] == 14).all()
            assert (df.loc[start:end - 1, "hour"] == hour).all()

            bin_table = clim_by_bin[(season, hour)]
            assert len(bin_table) == N_CELLS
            assert (bin_table["point_id"].values == np.arange(N_CELLS)).all()

            for dst, src in zip(value_cols, clim_cols):
                vals = bin_table[src].values
                df.loc[start:end - 1, dst] = (
                    round3_wrap360(vals) if dst in dir_dst_cols else np.round(vals, 3)
                )

    non_d14_mask = ~d14_mask
    unchanged = df.loc[non_d14_mask].equals(df_before.loc[non_d14_mask])
    print(f"\nhorizon in (1,7) rows unchanged: {unchanged}")
    assert unchanged, "horizon 1/7 rows were modified - aborting write"

    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote updated {CSV_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(CSV_PATH, "submission.csv")
    print(f"Re-zipped {ZIP_PATH}")

    print("\n--- Before/after d14 summary (median across all d14 rows) ---")
    for col in ("q05", "q50", "q95"):
        print(f"  {col}: before={df_before.loc[d14_mask, col].median():.3f}  "
              f"after={df.loc[d14_mask, col].median():.3f}")
    for col in ("dir_05", "dir_50", "dir_95"):
        print(f"  {col}: before={df_before.loc[d14_mask, col].median():.3f}  "
              f"after={df.loc[d14_mask, col].median():.3f}")


if __name__ == "__main__":
    main()
