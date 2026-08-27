"""Export the S3 winner-layout wind input as one hashable artifact.

Audit v2, Tier B anchor 5 (rung-2 protocol). Support script for
`notebooks/audit_s3_anchor_scaffold.ipynb`.

**Why this exists.** The S3 three-case run consumed its wind as 1,826 daily
AROME files read through `task2_scorer_replica.load_arome_series`, not as a
file. A manifest cell cannot SHA a loader. This script materialises the
identical series, at the identical pixel, as one CSV so the anchor
recomputation has a pinned input.

**What it deliberately does NOT do.** It stops at the native measurement
level the AROME files carry. No height adjustment of any kind is applied
here, and no simulation parameter appears in this file. Everything from the
loaded table onward is written by the principal in the notebook, from
Phase_2.pdf, with no repo code consulted. That separation is the point of
the anchor; do not "helpfully" extend this script past it.

**Gate (contract v2.1 §5b).** The extraction is asserted against the three
facts the S3 run recorded independently of any physics: the nearest-pixel
coordinates, the finite-step count, and the series endpoints. If the data
root ever changes under this script, those assertions fail loudly rather
than silently producing a different series.

Deterministic: file reads and a closed-form vector-to-polar conversion. No
stochastic step, so no seed applies.

Output: data/audit_s3_wind_input_cell63_20260824.csv
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
DATA_ROOT = Path(os.environ.get(
    "PHASE2_DATA_ROOT",
    REPO_ROOT / "phase_2" / "inference_2022" / "phase2_dataset_ship"))
OUT_CSV = REPO_ROOT / "data" / "audit_s3_wind_input_cell63_20260824.csv"

sys.path.insert(0, str(KIT_PHASE2))
sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
os.environ["PHASE2_DATA_ROOT"] = str(DATA_ROOT)

# Farm centre of the S3 winner-layout case, from data/task2_layout_winner.json
# ("farm_centre_lat", "farm_centre_lon"). Cell 63.
CENTRE_LAT = 52.50
CENTRE_LON = 3.00

# Facts the 2026-08-19 S3 run recorded (reports/three_case_scorer_20260818.md,
# reports/three_case_scorer_20260818.json case3_submitted_cell63). Used as
# extraction gates, not as inputs.
EXPECT_PIXEL_LAT = 52.4963
EXPECT_PIXEL_LON = 2.9933
EXPECT_N_STEPS = 14608
EXPECT_START = "2016-01-01 00:00:00"
EXPECT_END = "2020-12-31 21:00:00"
LEVEL = "125m"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t_start = time.time()
    import target_loader

    print(f"data root: {DATA_ROOT}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"dataset root not found: {DATA_ROOT}")

    static = target_loader.load_static()
    d2 = (static.lat - CENTRE_LAT) ** 2 + (static.lon - CENTRE_LON) ** 2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    px_lat, px_lon = float(static.lat[iy, ix]), float(static.lon[iy, ix])
    is_sea = bool(static.sea[iy, ix])
    print(f"nearest pixel to ({CENTRE_LAT}, {CENTRE_LON}): "
          f"({px_lat:.4f}, {px_lon:.4f})  sea={is_sea}  index=(y={iy}, x={ix})")
    assert is_sea, "nearest pixel is not sea"
    assert abs(px_lat - EXPECT_PIXEL_LAT) < 5e-5, (px_lat, EXPECT_PIXEL_LAT)
    assert abs(px_lon - EXPECT_PIXEL_LON) < 5e-5, (px_lon, EXPECT_PIXEL_LON)

    dates = sorted(target_loader.list_dates())
    print(f"{len(dates)} daily files to read")
    times, u_list, v_list = [], [], []
    t0 = time.time()
    for k, d in enumerate(dates):
        day = target_loader.load_day(d, levels=(LEVEL,))
        times.append(day.times)
        u_list.append(day.u[LEVEL][:, iy, ix])
        v_list.append(day.v[LEVEL][:, iy, ix])
        if (k + 1) % 400 == 0:
            print(f"  [{k + 1}/{len(dates)}] elapsed={time.time() - t0:.0f}s")

    times = np.concatenate(times)
    u = np.concatenate(u_list).astype(float)
    v = np.concatenate(v_list).astype(float)
    finite = np.isfinite(u) & np.isfinite(v)
    n_dropped = int((~finite).sum())
    times, u, v = times[finite], u[finite], v[finite]

    # Vector-to-polar, meteorological convention. This is how the data is
    # read, not how the farm is modelled.
    ws = np.sqrt(u ** 2 + v ** 2)
    wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0

    ts = pd.to_datetime(times)
    print(f"{ws.size} finite steps ({n_dropped} non-finite dropped), "
          f"{ts.min()} .. {ts.max()}, years {sorted(set(ts.year))}")
    print(f"mean ws at the native level: {ws.mean():.6f}")

    assert ws.size == EXPECT_N_STEPS, (ws.size, EXPECT_N_STEPS)
    assert str(ts.min()) == EXPECT_START, (str(ts.min()), EXPECT_START)
    assert str(ts.max()) == EXPECT_END, (str(ts.max()), EXPECT_END)

    df = pd.DataFrame({
        "time": ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "ws_native_ms": ws,
        "wd_deg": wd,
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.17g", lineterminator="\n")

    print(f"\nwrote {OUT_CSV}")
    print(f"  rows   {len(df)}")
    print(f"  bytes  {OUT_CSV.stat().st_size}")
    print(f"  SHA256 {sha256_of(OUT_CSV)}")
    print(f"total wall-clock {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
