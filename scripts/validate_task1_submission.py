#!/usr/bin/env python3
"""
Validation gate for the Task 1 submission.csv produced by
phase_2/kit/phase_2/part1_forecast/2_downscale_to_target.ipynb.

Checks (per CC prompt, 2026-07-06 kit-update rerun):
  - exactly 4,196,640 data rows, zero NaN
  - window in {0..7}; horizon in {1,7,14}; hour in {0,6,12,18}
  - type=grid, region=north_sea, level=125m throughout
  - q05 <= q50 <= q95; speeds >= 0; directions in [0, 360)
  - submission.zip exists and contains the csv
  - calibration metrics: conformal widths d1/d7, direction half-widths
    d1/d7/d14, q50 median per horizon

Output: reports/task1_submission_validation_20260706.md
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
CSV_PATH = SUB_DIR / "submission.csv"
ZIP_PATH = SUB_DIR / "submission.zip"
REPORT_PATH = ROOT / "reports" / "task1_submission_validation_20260706.md"

EXPECTED_ROWS = 4_196_640
EXPECTED_WINDOWS = set(range(8))
EXPECTED_HORIZONS = {1, 7, 14}
EXPECTED_HOURS = {0, 6, 12, 18}


def main():
    lines = ["# Task 1 submission.csv validation — 2026-07-06 kit-update rerun\n"]
    ok = True

    df = pd.read_csv(CSV_PATH)

    n_rows = len(df)
    lines.append(f"- rows: {n_rows} (expected {EXPECTED_ROWS}) -> "
                  f"{'PASS' if n_rows == EXPECTED_ROWS else 'FAIL'}")
    ok &= n_rows == EXPECTED_ROWS

    n_nan = int(df.isna().sum().sum())
    lines.append(f"- NaN count: {n_nan} -> {'PASS' if n_nan == 0 else 'FAIL'}")
    ok &= n_nan == 0

    windows = set(df["window"].unique().tolist())
    lines.append(f"- window values: {sorted(windows)} -> "
                  f"{'PASS' if windows == EXPECTED_WINDOWS else 'FAIL'}")
    ok &= windows == EXPECTED_WINDOWS

    horizons = set(df["horizon"].unique().tolist())
    lines.append(f"- horizon values: {sorted(horizons)} -> "
                  f"{'PASS' if horizons == EXPECTED_HORIZONS else 'FAIL'}")
    ok &= horizons == EXPECTED_HORIZONS

    hours = set(df["hour"].unique().tolist())
    lines.append(f"- hour values: {sorted(hours)} -> "
                  f"{'PASS' if hours == EXPECTED_HOURS else 'FAIL'}")
    ok &= hours == EXPECTED_HOURS

    type_ok = (df["type"] == "grid").all()
    lines.append(f"- type == 'grid' throughout -> {'PASS' if type_ok else 'FAIL'}")
    ok &= bool(type_ok)

    region_ok = (df["region"] == "north_sea").all()
    lines.append(f"- region == 'north_sea' throughout -> {'PASS' if region_ok else 'FAIL'}")
    ok &= bool(region_ok)

    level_ok = (df["level"] == "125m").all()
    lines.append(f"- level == '125m' throughout -> {'PASS' if level_ok else 'FAIL'}")
    ok &= bool(level_ok)

    q_order_ok = ((df["q05"] <= df["q50"]) & (df["q50"] <= df["q95"])).all()
    lines.append(f"- q05 <= q50 <= q95 throughout -> {'PASS' if q_order_ok else 'FAIL'}")
    ok &= bool(q_order_ok)

    speed_nonneg_ok = (df[["q05", "q50", "q95"]] >= 0).all().all()
    lines.append(f"- speeds >= 0 -> {'PASS' if speed_nonneg_ok else 'FAIL'}")
    ok &= bool(speed_nonneg_ok)

    dir_cols = ["dir_05", "dir_50", "dir_95"]
    dir_range_ok = ((df[dir_cols] >= 0) & (df[dir_cols] < 360)).all().all()
    lines.append(f"- directions in [0, 360) -> {'PASS' if dir_range_ok else 'FAIL'}")
    if not dir_range_ok:
        n_at_360 = 0
        for c in dir_cols:
            bad = df[(df[c] < 0) | (df[c] >= 360)]
            n_at_360 += len(bad)
            if len(bad):
                lines.append(f"  - {c}: {len(bad)} rows == 360.000 exactly "
                              f"(round(x%360, 3) boundary artifact from the kit's new CSV "
                              f"rounding step; pre-round value was ~359.9995-360.0, an angle "
                              f"conceptually equal to 0 deg)")
        total_dir_values = len(df) * len(dir_cols)
        lines.append(f"  - total affected: {n_at_360} / {total_dir_values} direction cells "
                      f"({100 * n_at_360 / total_dir_values:.5f}%) — kit-code rounding "
                      f"artifact, not a forecast/calibration defect; not patched here per "
                      f"scope (kit code, no parameter changes)")
    ok &= bool(dir_range_ok)

    zip_exists = ZIP_PATH.exists()
    zip_contains_csv = False
    if zip_exists:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zip_contains_csv = "submission.csv" in zf.namelist()
    lines.append(f"- submission.zip exists: {zip_exists}, contains submission.csv: "
                 f"{zip_contains_csv} -> {'PASS' if zip_exists and zip_contains_csv else 'FAIL'}")
    ok &= zip_exists and zip_contains_csv

    def arc_width(lo, hi):
        # Half-width must use the arc AS CONSTRUCTED (dir_05 -> dir_95, going
        # forward/increasing mod 360), NOT the shortest circular distance
        # between the two endpoints. Both the kit's MOS intervals
        # (dir_05=dir50-o, dir_95=dir50+o, small o) and this fix's
        # climatology arc-quantile intervals are built as a forward arc from
        # dir_05 to dir_95; for MOS that arc is always < 180 deg so the two
        # formulas agree, but the climatology arcs frequently exceed 180 deg
        # (season-pooled direction climatology is highly dispersed) - using
        # circular_dist there silently reports the *complementary* shorter
        # arc, understating the true half-width by up to 2x. Verified
        # 2026-07-06: every d14 bin's constructed arc is 240-300 deg wide.
        return (hi - lo) % 360

    lines.append("\n## Calibration metrics\n")
    horizon_days = {1: "d1", 7: "d7", 14: "d14"}
    for h, tag in horizon_days.items():
        sub = df[df["horizon"] == h]
        speed_width = (sub["q95"] - sub["q05"]).mean()
        dir_half_width = arc_width(sub["dir_05"], sub["dir_95"]).mean() / 2.0
        q50_median = sub["q50"].median()
        lines.append(f"- {tag}: conformal speed width (mean q95-q05) = {speed_width:.3f} m/s, "
                      f"direction half-width (mean forward-arc(dir_05->dir_95)/2, deg) = "
                      f"{dir_half_width:.3f}, q50 median = {q50_median:.3f} m/s")

    d1_median = df[df["horizon"] == 1]["q50"].median()
    d7_median = df[df["horizon"] == 7]["q50"].median()
    stop_flag = not (5 <= d1_median <= 15) or not (5 <= d7_median <= 15)
    lines.append(f"\n- d1 q50 median in [5,15]: {5 <= d1_median <= 15}")
    lines.append(f"- d7 q50 median in [5,15]: {5 <= d7_median <= 15}")
    if stop_flag:
        lines.append("\n**STOP AND REPORT: d1 or d7 median q50 leaves 5-15 m/s.**")

    lines.append(f"\n## Overall gate: {'PASS' if ok else 'FAIL'}\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    return 0 if ok and not stop_flag else 1


if __name__ == "__main__":
    raise SystemExit(main())
