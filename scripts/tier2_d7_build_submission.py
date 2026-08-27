#!/usr/bin/env python3
"""TIER-2 Option-C build: replace ALL d7 speed (hours 00/06/12/18) with Pangu.

Opus/Matteo Option-C GO 2026-07-19. Surgical, line-level edit of the designated
base = downloaded Codabench 835026 CSV (alpha=0.90, board 27.14). Only the
q05/q50/q95 of the 32 `horizon=7` blocks (8 windows x 4 hours = 1,398,880 rows)
are rewritten from the Pangu chain; every other row -- d1, d14, d7 direction, all
metadata -- is streamed VERBATIM (byte-identical). No pandas round-trip.

Chain (identical to the four-block validation; reuses its functions):
  arm_extracts_sub/extract_<date>_h<HH>.npz  (Pangu-24 coupled 125 m coarse d7)
  -> S.downscaled (pinned block-excluded downscaler, cache SHA b68eb5fe...)
  -> S.pg_interval(k=1.846 FROZEN)  [PG_LO 0.55, PG_HI 1.60]
  -> per-cell x season bias (S.build_bias, seed 42, tier2_d7_datemap.json)
  -> alpha 0.90 FROZEN tighten.
Frozen params: k=1.846, alpha=0.90, bias table = validated (no recalibration).

Provenance caveat (build note): the frozen bias table + k were MEASURED at 00 UTC
(four-block scorer, HOUR=0). Applying them to the 06/12/18 inits is the
expected-from-mechanism carryover (identical model, identical chain, only init
hour varies) -- NOT independently measured. All four seasonal blocks positive at
00 UTC is the supporting evidence.

Compliance (Clause 1): each init hour UTC lies within the window's context period
(context_end = issue day), so the analysis read is at/before the issue time.

Output: scripts/artifacts/submission_pangu_d7_allhours_20260719.csv (+ .zip).
NEVER writes phase_2/.../submission.csv. Asserts row count 4,196,640 and
byte-identity of every untouched row + all direction fields.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
sys.path.insert(0, str(_HERE.parent / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))
import tier2_eval_common as ec                        # noqa: E402
import tier2_d7_score_blocks as S                     # noqa: E402  (validated chain)
import downscaling as dn                              # noqa: E402
from tier2_f2_d14_precheck import DWN_CACHE           # noqa: E402

ARTIFACTS = _HERE / "artifacts"
BASE_ZIP = ARTIFACTS / "codabench_downloads_20260719" / "835026_submission__2_.zip"
BASE_CSV_SHA = "059e1c5643e49e62933d97c2125da28715efe03170536a8ffbf4e77512dbda17"
DWN_CACHE_SHA = "b3ae32c0bf4203351a03526a454030817f70adb588f55caefdb0b43b5a2d8703"
SUB_EXTRACTS = _HERE.parent / "data" / "arm_extracts_sub_2021_20260822"  # YEAR-VARYING:
# see data/PINNED_ARTIFACTS.md; the 2022 windows need this repointed (contract v2.1 5b).
K_PANGU = 1.846                                        # FROZEN (validated calib)
ALPHA = S.ALPHA                                        # 0.90 frozen
INIT_HOURS = (0, 6, 12, 18)
OUT_CSV = ARTIFACTS / "submission_pangu_d7_allhours_20260719.csv"

# 0-indexed CSV columns
C_WIN, C_LAT, C_LON, C_HOR, C_HOUR = 1, 3, 4, 5, 6
C_Q05, C_Q50, C_Q95 = 8, 9, 10
N_FP = 43715
N_D7_ROWS = 1_398_880
N_TOTAL = 4_196_640


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_q(v: float) -> str:
    """Kit-style float: round 3 dp, drop trailing zeros, keep one decimal."""
    s = f"{round(float(v), 3):.3f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def main():
    # fail-loud alpha guard (swap runbook amendment 2026-07-21): the Pangu d7
    # interval alpha is frozen at 0.90; abort if the score-blocks ALPHA ever drifts.
    assert abs(ALPHA - 0.90) < 1e-9, f"splice ALPHA={ALPHA} != 0.90 (frozen)"
    # ---- pin base CSV (extract from the archived download) ----
    scratch = Path(os.environ.get("TEMP", ".")) / "tier2_base835026"
    scratch.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        z.extract(name, scratch)
    base_csv = scratch / name
    got = sha256(base_csv)
    assert got == BASE_CSV_SHA, f"BASE SHA MISMATCH: {got} != {BASE_CSV_SHA}"
    print(f"[build] base 835026 pinned (sha OK): {base_csv}")

    # ---- pin downscaler + prep validated globals (S._DWN, S._YS, S._XS) ----
    got_d = sha256(DWN_CACHE)
    assert got_d == DWN_CACHE_SHA, f"DOWNSCALER SHA MISMATCH: {got_d}"
    S._prep()
    ys, xs = S._YS, S._XS
    assert ys.size == N_FP, ys.size
    print(f"[build] downscaler pinned (sha OK); footprint {ys.size}")

    # ---- frozen bias table (validated, seed 42) ----
    dm = json.load(open(ARTIFACTS / "tier2_d7_datemap.json"))
    bias = S.build_bias(S.pg_ctx, S.pg_coarse, dm["bias_issue_to_validV"])
    print(f"[build] frozen bias table rebuilt: seasons {sorted(bias)} "
          f"means {{" + ", ".join(f'{k}:{float(v.mean()):+.3f}' for k, v in bias.items()) + "}}")

    # ---- footprint lat/lon (target grid, canonical order) for ordering guard ----
    # Replicate the kit's EXACT rounding (build_forecast_submission.field_to_rows:
    # round(2)->float32, then write_submission round(2)); a plain float64 round
    # disagrees by 0.01 on ~11 .xx5-boundary cells and would false-trip the guard.
    st = dn._static()
    fp_lat = np.round(np.asarray(st.lat)[ys, xs].round(2).astype(np.float32), 2)
    fp_lon = np.round(np.asarray(st.lon)[ys, xs].round(2).astype(np.float32), 2)

    # ---- precompute replacement q-arrays per (window_col, hour), fp-order ----
    subdm = json.load(open(ARTIFACTS / "tier2_sub_datemap.json"))
    repl = {}                       # (wcol, hour) -> (lo, q50c, hi)  each len 43715
    nan_fb = {}
    for w in subdm["windows"]:
        wcol = w["window_id"] - 1                     # submission window col 0..7
        bb = bias[w["season"]]
        for H in INIT_HOURS:
            npz = SUB_EXTRACTS / f"extract_{pd.Timestamp(w['issue']):%Y%m%d}_h{H:02d}.npz"
            ctx = np.load(npz, allow_pickle=True)
            spd_fine = S.downscaled((ctx["d7_u"], ctx["d7_v"]))     # (479,433)
            q05f, q95f = S.pg_interval(None, spd_fine, K_PANGU)
            spd = spd_fine[ys, xs]; q05 = q05f[ys, xs]; q95 = q95f[ys, xs]
            q50c = spd - bb; q05c = q05 - bb; q95c = q95 - bb
            lo = np.maximum(0.0, q50c - ALPHA * (q50c - q05c))
            hi = q50c + ALPHA * (q95c - q50c)
            # kit field_to_rows post-step (build_forecast_submission.py:65-66):
            # enforce monotone quantiles + q05>=0 -- the SAME defensive step every
            # base row received; a no-op except a handful of near-zero-speed cells
            # where bias correction pushed q50 marginally negative.
            trip = np.sort(np.stack([lo, q50c, hi], axis=1), axis=1)
            lo = np.clip(trip[:, 0], 0.0, None); q50s = trip[:, 1]; hi = trip[:, 2]
            repl[(wcol, H)] = (lo, q50s, hi)
            nan_fb[(wcol, H)] = int(np.sum(~np.isfinite(lo) | ~np.isfinite(q50s) | ~np.isfinite(hi)))
    tot_nan = sum(nan_fb.values())
    print(f"[build] replacement arrays ready for {len(repl)} blocks; "
          f"NaN cells (fall back to base): {tot_nan}")

    # ---- stream base -> output, rewrite only d7 q-fields, verbatim otherwise ----
    counter = {}                    # (wcol,H) -> next cell index
    n_d7 = 0; n_fallback = 0; ord_bad = 0
    with open(base_csv, "r", newline="") as fin, open(OUT_CSV, "w", newline="") as fout:
        header = fin.readline()
        fout.write(header)
        for line in fin:
            # fast horizon test without full split for the 2/3 non-d7 lines
            # fields are unquoted; horizon is column index 5
            # (cheap split is fine and keeps code simple)
            f = line.rstrip("\n").split(",")
            if f[C_HOR] != "7":
                fout.write(line)                       # VERBATIM (byte-identical)
                continue
            wcol = int(f[C_WIN]); H = int(f[C_HOUR])
            k = counter.get((wcol, H), 0); counter[(wcol, H)] = k + 1
            # ordering guard: base row latlon must equal footprint cell k
            if abs(float(f[C_LAT]) - fp_lat[k]) > 0.005 or abs(float(f[C_LON]) - fp_lon[k]) > 0.005:
                ord_bad += 1
            lo, q50c, hi = repl[(wcol, H)]
            if np.isfinite(lo[k]) and np.isfinite(q50c[k]) and np.isfinite(hi[k]):
                f[C_Q05] = fmt_q(lo[k]); f[C_Q50] = fmt_q(q50c[k]); f[C_Q95] = fmt_q(hi[k])
            else:
                n_fallback += 1                         # keep base q-fields verbatim
            fout.write(",".join(f) + "\n")
            n_d7 += 1
    assert ord_bad == 0, f"ORDERING GUARD FAILED on {ord_bad} d7 rows (fp-order != block order)"
    assert n_d7 == N_D7_ROWS, f"d7 rows {n_d7} != {N_D7_ROWS}"
    for key, c in counter.items():
        assert c == N_FP, f"block {key} has {c} rows != {N_FP}"
    print(f"[build] wrote {OUT_CSV}: d7 rows {n_d7}, fallback-to-base {n_fallback}, ord_bad {ord_bad}")

    # ---- verify: row count + byte-identity of untouched rows + all directions ----
    n_lines = 0; n_diff = 0
    with open(base_csv) as fb, open(OUT_CSV) as fo:
        assert fb.readline() == fo.readline(), "header changed"
        for lb, lo_ in zip(fb, fo):
            n_lines += 1
            if lb == lo_:
                continue
            n_diff += 1
            a = lb.rstrip("\n").split(","); b = lo_.rstrip("\n").split(",")
            assert a[C_HOR] == "7", f"NON-d7 row changed at line {n_lines}"
            for j in range(len(a)):
                if j in (C_Q05, C_Q50, C_Q95):
                    continue
                assert a[j] == b[j], f"non-q field {j} changed on d7 line {n_lines}"
    assert n_lines == N_TOTAL, f"row count {n_lines} != {N_TOTAL}"
    print(f"[build] verify OK: {n_lines} rows; changed lines {n_diff} "
          f"(all d7, q-only); untouched rows + all directions byte-identical")

    # ---- zip + hashes ----
    out_zip = OUT_CSV.with_suffix(".zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_CSV, "submission.csv")
    out_sha = sha256(OUT_CSV); zip_sha = sha256(out_zip)
    summary = {"base": "835026", "base_csv_sha": BASE_CSV_SHA,
               "out_csv": str(OUT_CSV), "out_csv_sha": out_sha,
               "out_zip": str(out_zip), "out_zip_sha": zip_sha,
               "rows_total": n_lines, "d7_rows_replaced": n_d7,
               "changed_lines": n_diff, "nan_fallback_to_base": n_fallback,
               "k_pangu": K_PANGU, "alpha": ALPHA, "bias_per_season": 20,
               "nan_fallback_by_block": {f"{w}_{h}": v for (w, h), v in nan_fb.items()}}
    with open(ARTIFACTS / "tier2_d7_build_submission_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[build] OUT CSV sha  {out_sha}")
    print(f"[build] OUT ZIP sha  {zip_sha}")
    print(f"[build] summary -> {ARTIFACTS / 'tier2_d7_build_submission_summary.json'}")


if __name__ == "__main__":
    main()
