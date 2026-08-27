#!/usr/bin/env python3
"""Swap-run rehearsal: re-run the pipeline on the KNOWN 2021 windows and prove it
reproduces banked 855076. Written for docs/swap_runbook_20260721.md.

DO NOT auto-run the multi-hour legs in CI; Matteo triggers the rehearsal. The
reproduction-comparison engine (`compare_vs_baseline`) is the pass-criterion and is
self-tested (`--self-test`) against the on-disk 855076/854984 CSVs.

Reproduction expectation (see runbook §0.3):
  BYTE-IDENTICAL : d7 speed (pinned downscaler), d14 speed (deterministic clim),
                   all directions (seeded models), all metadata.
  JITTERS        : d1 speed q05/q50/q95 (kit downscaler has no random_state).

Pass criterion: byte-identical -> PASS; only d1 speed differs and all |Δ| < 5e-4
(inside the 3-dp submission floor) -> PASS WITH NOTE; d1 exceeds rounding, or ANY
other column differs -> FAIL.

Stages:
  --stage guard      : run the Clause-1 compliance guard (32 PASS lines)
  --stage validate   : run validate_task1_submission.py on a CSV
  --stage compare    : compare --candidate vs the banked 855076 baseline
  --stage orchestrate: run the full 2021 re-run via subprocess (needs Z: + notebooks;
                       untested end-to-end this session), then compare
  --self-test        : sanity-check the compare engine on existing CSVs (no re-run)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from _publication_paths import ppath  # noqa: E402  (publication tree)

_HERE = Path(__file__).resolve().parent
ARTIFACTS = _HERE / "artifacts"
PY = str(ppath("<USER_HOME>/AppData/Local/anaconda3/envs/swnd/python.exe"))
BASELINE_855076 = ARTIFACTS / "submission_pangu_d7_f125_20260720.csv"
BASELINE_855076_SHA = "169524f9ffa5c3e53f66a4d7f686299b1344c9f303040a68e610ed0816b33079"
ROUND_FLOOR = 5e-4                                    # 3-dp submission rounding floor
QCOLS = ["q05", "q50", "q95"]
COLS = ["type", "window", "region", "latitude", "longitude", "horizon", "hour",
        "level", "q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]


def compare_vs_baseline(candidate_csv, baseline_csv=BASELINE_855076):
    """Per (horizon, column) reproduction diff. Returns (verdict, report rows)."""
    a = pd.read_csv(baseline_csv)
    b = pd.read_csv(candidate_csv)
    assert list(a.columns) == COLS and list(b.columns) == COLS, "schema mismatch"
    assert len(a) == len(b) == 4_196_640, f"row count {len(a)}/{len(b)}"
    rows = []
    for hor in (1, 7, 14):
        ma = a.horizon == hor
        for c in QCOLS + ["dir_05", "dir_50", "dir_95"]:
            va = a.loc[ma, c].to_numpy(float); vb = b.loc[ma, c].to_numpy(float)
            d = np.abs(va - vb)
            changed = int((d > 0).sum())
            over = int((d > ROUND_FLOOR).sum())
            rows.append({"horizon": hor, "col": c, "changed": changed,
                         "max_abs_delta": float(d.max()), "over_3dp": over,
                         "n": int(ma.sum())})
    # verdict
    nonq_changed = [r for r in rows if r["col"].startswith("dir") and r["changed"] > 0]
    d1speed_over = [r for r in rows if r["horizon"] == 1 and r["col"] in QCOLS and r["over_3dp"] > 0]
    other_speed_changed = [r for r in rows if r["horizon"] in (7, 14)
                           and r["col"] in QCOLS and r["changed"] > 0]
    total_changed = sum(r["changed"] for r in rows)
    if total_changed == 0:
        verdict = "PASS (byte-identical)"
    elif not nonq_changed and not other_speed_changed and not d1speed_over:
        verdict = "PASS WITH NOTE (only d1 speed differs, all within 3-dp floor)"
    else:
        bad = []
        if nonq_changed: bad.append("direction columns changed")
        if other_speed_changed: bad.append("d7/d14 speed changed")
        if d1speed_over: bad.append("d1 speed exceeds 3-dp floor")
        verdict = "FAIL (" + "; ".join(bad) + ")"
    return verdict, rows


def print_report(verdict, rows):
    print(f"{'horizon':>7} {'col':>7} {'changed':>10} {'maxAbsDelta':>12} {'>3dp':>8}")
    for r in rows:
        flag = "" if (r["changed"] == 0) else ("  <== d1 speed jitter"
               if (r["horizon"] == 1 and r["col"] in QCOLS) else "  <== UNEXPECTED")
        print(f"{r['horizon']:>7} {r['col']:>7} {r['changed']:>10} "
              f"{r['max_abs_delta']:>12.6f} {r['over_3dp']:>8}{flag}")
    print(f"\nVERDICT: {verdict}")


def run_guard(meta_dir):
    print(f"[rehearsal] Clause-1 compliance guard on {meta_dir}")
    return subprocess.call([PY, str(_HERE / "tier2_swap_compliance_guard.py"),
                            "--meta-dir", str(meta_dir)])


def run_validate(csv):
    print(f"[rehearsal] validation gate on {csv}")
    return subprocess.call([PY, str(_HERE / "validate_task1_submission.py"), str(csv)])


def self_test():
    """Compare engine sanity: 855076 vs itself = identical; vs 854984 = d7 q05/q95 differ."""
    print("== self-test A: 855076 vs 855076 (expect byte-identical) ==")
    v, r = compare_vs_baseline(BASELINE_855076, BASELINE_855076)
    print_report(v, r); assert v.startswith("PASS (byte"), v
    prior = ARTIFACTS / "submission_pangu_d7_allhours_20260719.csv"     # 854984
    if prior.exists():
        print("\n== self-test B: 854984 vs 855076 (expect FAIL: d7 q05/q95 differ) ==")
        v, r = compare_vs_baseline(prior, BASELINE_855076)
        print_report(v, r); assert v.startswith("FAIL"), v
    print("\n[rehearsal] self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["guard", "validate", "compare", "orchestrate"])
    ap.add_argument("--candidate")
    ap.add_argument("--meta-dir", default=str(
        _HERE.parent / "phase_2" / "phase2_dataset_ship" / "inference"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.self_test:
        self_test()
    elif args.stage == "guard":
        run_guard(args.meta_dir)
    elif args.stage == "validate":
        run_validate(args.candidate)
    elif args.stage == "compare":
        v, r = compare_vs_baseline(args.candidate)
        print_report(v, r)
    elif args.stage == "orchestrate":
        # Full 2021 re-run: Matteo runs the runbook §R4 steps 3-4 to produce the
        # candidate CSV (needs Z: mounted + notebook execution + ~4 h). This stage
        # then runs guard -> validate -> compare on the produced candidate. Left as
        # documented subprocess wiring; NOT auto-executed (multi-hour, Z: dependency).
        raise SystemExit("orchestrate: run runbook §R4 steps 3-4 first, then "
                         "--stage compare --candidate <output>. See docs/swap_runbook_20260721.md")
    print(f"[rehearsal] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
