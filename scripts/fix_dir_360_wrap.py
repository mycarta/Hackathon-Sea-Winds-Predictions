#!/usr/bin/env python3
"""
Post-process phase_2/kit/phase_2/part1_forecast/submission.csv in place:
replace direction values == 360.000 with 0.000 in dir_05/dir_50/dir_95.

Root cause (see reports/task1_submission_validation_20260706.md): any
round(x, 3) step downstream of a %360 wrap can push an angle
infinitesimally below 360 (e.g. 359.9997) up to exactly 360.000, which is
out of the required [0, 360) range. 360 deg == 0 deg circularly, so this is
a safe value substitution, not a model change. This is a general-purpose,
idempotent fixer - re-run any time a new round-then-wrap boundary artifact
turns up (first use: 24 values from the kit's own CSV writer, 2026-07-06;
second use: 10 values from the d14-climatology-fix script's own rounding,
2026-07-06 - the exact count is expected to vary by run, so it is reported,
not asserted against a fixed constant).

Re-zips submission.zip after the fix (csv at archive root, matching the
kit's own convention).
"""

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
CSV_PATH = SUB_DIR / "submission.csv"
ZIP_PATH = SUB_DIR / "submission.zip"

DIR_COLS = ["dir_05", "dir_50", "dir_95"]


def main():
    df = pd.read_csv(CSV_PATH)

    n_fixed = 0
    for c in DIR_COLS:
        mask = df[c] >= 360.0
        n = int(mask.sum())
        if n:
            df.loc[mask, c] = 0.0
        n_fixed += n

    print(f"replaced {n_fixed} values == 360.000 with 0.000 across {DIR_COLS}")

    df.to_csv(CSV_PATH, index=False)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(CSV_PATH, "submission.csv")

    print(f"re-wrote {CSV_PATH}")
    print(f"re-zipped {ZIP_PATH}")


if __name__ == "__main__":
    main()
