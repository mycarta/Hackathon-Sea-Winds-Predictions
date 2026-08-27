#!/usr/bin/env python3
"""Width recovery (Option B, f=1.25) applied to the Pangu d7 submission (854984).

Matteo GO 2026-07-20, pre-registered single shot (no f iteration after the board
result; selection reverts to 854984 if worse). POST-PROCESSING width correction --
NOT a pipeline recalibration: the nominal alpha stays 0.90; f rescales the interval
half-widths of the Pangu d7 speed rows only, to bring ACTUAL board coverage closer
to nominal. q50 is NOT changed -- only q05 and q95 move, symmetrically about q50:

  q95_new = q50 + f*(q95 - q50)
  q05_new = max(0, q50 - f*(q50 - q05))

Base = 854984 = submission_pangu_d7_allhours_20260719.csv (SHA 488955...), verified
before any edit. Surgical line-level stream: only q05/q95 of the 32 horizon-7 speed
blocks (1,398,880 rows, all four init hours) change; q50, all directions, d1/d14,
and metadata are byte-identical. Output submission_pangu_d7_f125_20260720.csv (never
overwrites an existing CSV).

Pre-registered expectations (dev-year sweep + transfer bet; NOT measured on 2021):
  board coverage_d7 ~0.84 (dev coverage<->f slope, scale 1.27)
  board WS d7 ~24.3 (dev bowl ~4% fractional gain on 25.32)
  transfer bet: shape-invariance of the Winkler-optimal coverage; a scale
  misestimate of +/-20% still beats f=1.0 per the dev bowl.
Honest caveat (same as the D7 build): f* fit on dev-year 00 UTC data; transfer to
the withheld year and to 06/12/18 hours is expected, not measured. Phase-1
calibration study: per-horizon f transfers with a smaller gap than model changes.

Self-check (c) recomputes the dev-year Winkler/coverage at f=1.25 with THIS script's
scale function and asserts it matches the sweep row (0.9068 / 23.833) exactly.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
ARTIFACTS = _HERE / "artifacts"
BASE_CSV = ARTIFACTS / "submission_pangu_d7_allhours_20260719.csv"
BASE_SHA = "488955866949a6445fed37c817b0b192075dd0198dcc009787e2230a03fb70b8"
OUT_CSV = ARTIFACTS / "submission_pangu_d7_f125_20260720.csv"
F = 1.25
C_HOR, C_Q05, C_Q50, C_Q95 = 5, 8, 9, 10
N_TOTAL, N_D7 = 4_196_640, 1_398_880
SWEEP_COV, SWEEP_WINK = 0.9068, 23.8329       # exact sweep row f=1.25 (4 dp)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_q(v: float) -> str:
    s = f"{round(float(v), 3):.3f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def scale_halfwidths(q05, q50, q95, f):
    """Widen interval half-widths about q50 by factor f; clip q05>=0. q50 untouched."""
    hi = q50 + f * (q95 - q50)
    lo = np.maximum(0.0, q50 - f * (q50 - q05))
    return lo, hi


def main():
    # ---- [1] verify base (A2: 854984 immutably archived as the committed
    #          submission_pangu_d7_allhours_20260719.zip @ 11ac8d8) ----
    got = sha256(BASE_CSV)
    assert got == BASE_SHA, f"BASE 854984 SHA MISMATCH: {got}"
    print(f"[f125] base 854984 verified: {BASE_CSV.name} sha OK ({got})")

    # ---- [2] surgical stream: scale q05/q95 of d7 rows only ----
    n_d7 = 0; n_changed = 0
    with open(BASE_CSV, "r", newline="") as fin, open(OUT_CSV, "w", newline="") as fout:
        fout.write(fin.readline())                      # header
        for line in fin:
            f = line.rstrip("\n").split(",")
            if f[C_HOR] != "7":
                fout.write(line)                        # VERBATIM
                continue
            q05 = float(f[C_Q05]); q50 = float(f[C_Q50]); q95 = float(f[C_Q95])
            lo, hi = scale_halfwidths(np.array([q05]), np.array([q50]),
                                      np.array([q95]), F)
            f[C_Q05] = fmt_q(lo[0]); f[C_Q95] = fmt_q(hi[0])   # q50 (f[C_Q50]) untouched
            fout.write(",".join(f) + "\n")
            n_d7 += 1
    assert n_d7 == N_D7, f"d7 rows {n_d7} != {N_D7}"
    print(f"[f125] wrote {OUT_CSV.name}: d7 rows scaled {n_d7}")

    # ---- [4a/4b] verify: row count + changes confined to d7 q05/q95 only ----
    n_lines = 0
    with open(BASE_CSV) as fb, open(OUT_CSV) as fo:
        assert fb.readline() == fo.readline(), "header changed"
        for lb, lo_ in zip(fb, fo):
            n_lines += 1
            if lb == lo_:
                continue
            n_changed += 1
            a = lb.rstrip("\n").split(","); b = lo_.rstrip("\n").split(",")
            assert a[C_HOR] == "7", f"NON-d7 row changed at line {n_lines}"
            for j in range(len(a)):
                if j in (C_Q05, C_Q95):
                    continue
                assert a[j] == b[j], f"col {j} changed on d7 line {n_lines} (q50/dir must be identical)"
    assert n_lines == N_TOTAL, f"row count {n_lines} != {N_TOTAL}"
    print(f"[f125] verify OK: {n_lines} rows; changed {n_changed} (all d7, q05/q95 only; "
          f"q50 + all directions + d1/d14 byte-identical)")

    # ---- monotonicity + non-negativity on output ----
    import pandas as pd
    dn = pd.read_csv(OUT_CSV, usecols=["horizon", "q05", "q50", "q95"])
    d7 = dn[dn.horizon == 7]
    assert bool((d7.q05 <= d7.q50 + 1e-9).all()) and bool((d7.q50 <= d7.q95 + 1e-9).all())
    assert bool((dn.q05 >= 0).all()) and bool((dn.q05 <= dn.q95 + 1e-9).all())
    print(f"[f125] monotone q05<=q50<=q95 & q>=0 OK; d7 width mean "
          f"{(d7.q95-d7.q05).mean():.3f} (was ~13.47 at f=1.0)")

    # ---- [4c] dev-year self-check: scale fn on dev arm must match sweep f=1.25 ----
    import tier2_d7_widthcal_sweep as SW
    q05d, q50d, q95d, td, nd = SW.collect()
    lo, hi = scale_halfwidths(q05d, q50d, q95d, F)
    cov = round(float(np.mean((td >= lo) & (td <= hi))), 4)
    wink = round(float(np.mean(SW.winkler_score(td, lo, hi))), 4)
    print(f"[f125] dev self-check @f=1.25: coverage {cov} (sweep {SWEEP_COV}), "
          f"Winkler {wink} (sweep {SWEEP_WINK})")
    assert cov == SWEEP_COV and wink == SWEEP_WINK, "dev self-check != sweep f=1.25"

    # ---- hashes + zip ----
    out_zip = OUT_CSV.with_suffix(".zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_CSV, "submission.csv")
    out_sha = sha256(OUT_CSV); zip_sha = sha256(out_zip)
    summary = {"base_854984": BASE_CSV.name, "base_sha": BASE_SHA, "f": F,
               "out_csv": OUT_CSV.name, "out_csv_sha": out_sha,
               "out_zip": out_zip.name, "out_zip_sha": zip_sha,
               "rows_total": n_lines, "d7_rows_scaled": n_d7, "changed_lines": n_changed,
               "dev_selfcheck_coverage": cov, "dev_selfcheck_winkler": wink,
               "prereg_board_coverage": 0.84, "prereg_board_ws": 24.3}
    json.dump(summary, open(ARTIFACTS / "tier2_d7_f125_summary.json", "w"), indent=2)
    print(f"[f125] OUT CSV sha {out_sha}")
    print(f"[f125] OUT ZIP sha {zip_sha}")
    print(f"[f125] summary -> tier2_d7_f125_summary.json")


if __name__ == "__main__":
    main()
