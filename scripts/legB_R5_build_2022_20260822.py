#!/usr/bin/env python3
"""Leg B R5: build the 2022 submission. One shot. Nothing is uploaded.

Matteo uploads; CC never submits. This script produces an artifact and reports
its SHA and row count. That is the whole of its authority.

CONTRACT v2.1 SECTION 5b. A production run on new inputs owes a changed-constant
list BEFORE Step 0, not after. The list is printed first, every entry resolved
against disk, and the run stops if any of them is missing or mis-hashed. The
July rehearsal passed while three silent-corruption paths were live precisely
because it re-ran the SAME year; this run varies the year, so the constants the
year touches are enumerated rather than assumed.

    CHANGED for 2022                          from -> to
      BASE_ZIP           the Leg A base       835026 (2021) -> legA_base_2022
      BASE_CSV_SHA       its inner CSV hash   059e1c56 -> 4f632fef
      SUB_EXTRACTS       window extracts      ..._2021_... -> ..._2022_...
      tier2_sub_datemap  window/season map    2021 file -> 2022 file
      OUT_CSV            output path          July's name -> a 2022 name

    UNCHANGED, and why each is correct to leave alone
      DWN_CACHE_SHA      the downscaler is fitted on 2020 and is year-independent
      tier2_d7_datemap   the BIAS dates are 2016-2020 by design; the bias table
                         is frozen and does not depend on the submission year
      K_PANGU 1.846      frozen calibration
      ALPHA 0.90         frozen
      INIT_HOURS         verified identical in the 2022 base
      N_FP 43715         verified: 43,715 rows per (window, hour) in the 2022 base
      N_TOTAL 4196640    verified in the 2022 base
      f 1.25             post-processing, proven byte-exact against 855076

THE DATEMAP CANNOT BE OVERRIDDEN BY ASSIGNMENT. `tier2_d7_build_submission.main`
reads it inline as `json.load(open(ARTIFACTS / "tier2_sub_datemap.json"))`, a
hardcoded path inside the function. So the 2021 file is backed up, the 2022
content is put in its place for the duration, and the original is restored in a
`finally` block and re-hashed. If that restore ever fails the script says so
loudly; the file is also committed, so git is the backstop.

GUARD A: Clause-1 compliance, `tier2_swap_compliance_guard.py`, one PASS line
per window x init hour, 32 for this layout. Verifies every ERA5 init timestamp
falls at or before its window's context_end.

GUARD B: `validate_task1_submission.py`. A BUG IN THAT SCRIPT IS WORKED AROUND
HERE AND REPORTED, NOT PATCHED: it has no `sys.argv` handling at all, so
`tier2_swap_rehearsal.run_validate(csv)` passes a path the validator silently
IGNORES, always reading the hardcoded
`phase_2/kit/phase_2/part1_forecast/submission.csv`. Anyone trusting that stage
would be validating a stale file, or none. This runner imports the module and
overrides CSV_PATH, ZIP_PATH and REPORT_PATH so the guard reads the artifact
actually built.

Run:  python scripts/legB_R5_build_2022_20260822.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACTS = HERE / "artifacts"

BASE_ZIP_2022 = ARTIFACTS / "submission_legA_base_2022_20260819.zip"
BASE_CSV_SHA_2022 = "4f632fef000d8526d64cce9ab5a01d14dd8aeefcf5e8e1fb1faee4ed5d3a0789"
EXTRACTS_2022 = REPO / "data" / "arm_extracts_sub_2022_20260822"
DATEMAP_2021 = ARTIFACTS / "tier2_sub_datemap.json"
DATEMAP_2022 = ARTIFACTS / "tier2_sub_datemap_2022.json"
DATEMAP_BAK = ARTIFACTS / "tier2_sub_datemap_2021_BACKUP_20260822.json"
META_DIR = REPO / "phase_2" / "phase2_dataset_ship" / "inference"

OUT = ARTIFACTS / "submission_legB_2022_20260822.csv"
OUT_F125 = ARTIFACTS / "submission_legB_2022_f125_20260822.csv"
SUMMARY = ARTIFACTS / "tier2_d7_build_submission_summary.json"
SUMMARY_BAK = ARTIFACTS / "tier2_d7_build_submission_summary_JULY_backup_R5.json"

DWN_SHA = "b3ae32c0bf4203351a03526a454030817f70adb588f55caefdb0b43b5a2d8703"
N_TOTAL, N_D7, N_FP = 4_196_640, 1_398_880, 43_715
F = 1.25
C_HOR, C_Q05, C_Q50, C_Q95 = 5, 8, 9, 10


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_q(v):
    s = "%.3f" % round(float(v), 3)
    s = s.rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def apply_f125(src, dst, f=F):
    n = 0
    with open(src, "r", newline="") as fin, open(dst, "w", newline="") as fout:
        fout.write(fin.readline())
        for line in fin:
            fl = line.rstrip("\n").split(",")
            if fl[C_HOR] != "7":
                fout.write(line)
                continue
            q05 = float(fl[C_Q05]); q50 = float(fl[C_Q50]); q95 = float(fl[C_Q95])
            fl[C_Q95] = fmt_q(q50 + f * (q95 - q50))
            fl[C_Q05] = fmt_q(max(0.0, q50 - f * (q50 - q05)))
            fout.write(",".join(fl) + "\n")
            n += 1
    return n


def step(n, t):
    print("\n" + "=" * 74)
    print("STEP %s: %s" % (n, t))
    print("=" * 74)


def main():
    t0 = time.time()

    # ---------------- STEP 0 PRECEDES EVERYTHING (contract 5b) -------------
    step("0", "changed-constant list, resolved before any compute")
    changed = [
        ("BASE_ZIP", "835026 (2021)", BASE_ZIP_2022.name, BASE_ZIP_2022),
        ("BASE_CSV_SHA", "059e1c56...", BASE_CSV_SHA_2022[:8] + "...", None),
        ("SUB_EXTRACTS", "arm_extracts_sub_2021_...", EXTRACTS_2022.name, EXTRACTS_2022),
        ("tier2_sub_datemap", "2021 windows", DATEMAP_2022.name, DATEMAP_2022),
        ("OUT_CSV", "July's filename", OUT.name, None),
    ]
    print("  %-20s %-26s %s" % ("constant", "from", "to"))
    missing = []
    for name, frm, to, path in changed:
        ok = "" if path is None else ("  [on disk]" if path.exists() else "  [MISSING]")
        if path is not None and not path.exists():
            missing.append(str(path))
        print("  %-20s %-26s %s%s" % (name, frm, to, ok))
    print("\n  UNCHANGED and correct to leave: DWN_CACHE_SHA (downscaler is")
    print("  year-independent), tier2_d7_datemap (bias dates are 2016-2020 by")
    print("  design), K_PANGU, ALPHA, INIT_HOURS, N_FP, N_TOTAL, f=1.25")
    if missing:
        print("\n  STOP. Inputs missing:")
        for m in missing:
            print("    " + m)
        sys.exit(2)

    n_ex = len(list(EXTRACTS_2022.glob("*.npz")))
    print("\n  2022 extracts present: %d of 32" % n_ex)
    assert n_ex == 32, "expected 32 extracts, found %d; run custody first" % n_ex

    got = sha256(BASE_ZIP_2022)
    print("  base zip  %s  %s" % (BASE_ZIP_2022.name, got[:16]))

    # ---------------- GUARD A ---------------------------------------------
    step("A", "Clause-1 compliance guard on the 2022 windows")
    rc = subprocess.call([sys.executable, str(HERE / "tier2_swap_compliance_guard.py"),
                          "--meta-dir", str(META_DIR)])
    assert rc == 0, "Clause-1 guard failed (rc=%d)" % rc

    # ---------------- BUILD ------------------------------------------------
    step("1", "build, with the 2022 constants overridden")
    if SUMMARY.exists():
        shutil.copy2(SUMMARY, SUMMARY_BAK)
        s_sha = sha256(SUMMARY)
    else:
        s_sha = None
    dm21_sha = sha256(DATEMAP_2021)
    shutil.copy2(DATEMAP_2021, DATEMAP_BAK)
    print("  2021 datemap backed up (%s)" % dm21_sha[:16])

    try:
        shutil.copy2(DATEMAP_2022, DATEMAP_2021)
        print("  2022 datemap in place for the duration (path is hardcoded in main())")

        sys.path.insert(0, str(HERE))
        sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
        sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))
        import tier2_d7_build_submission as B

        assert B.DWN_CACHE_SHA == DWN_SHA, "downscaler pin drifted: %s" % B.DWN_CACHE_SHA
        B.BASE_ZIP = BASE_ZIP_2022
        B.BASE_CSV_SHA = BASE_CSV_SHA_2022
        B.SUB_EXTRACTS = EXTRACTS_2022
        B.OUT_CSV = OUT
        print("  overrides applied; frozen file byte-unmodified\n")
        B.main()
    finally:
        shutil.copy2(DATEMAP_BAK, DATEMAP_2021)
        back = sha256(DATEMAP_2021)
        if back != dm21_sha:
            print("\n  *** 2021 DATEMAP DID NOT RESTORE: %s != %s ***" % (back, dm21_sha))
            print("  *** it is committed; restore with git checkout ***")
        else:
            DATEMAP_BAK.unlink()
            print("\n  2021 datemap restored and hash-verified (%s)" % back[:16])

    if SUMMARY.exists():
        (ARTIFACTS / "legB_2022_build_summary_20260822.json").write_text(
            SUMMARY.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    if s_sha is not None:
        shutil.copy2(SUMMARY_BAK, SUMMARY)
        assert sha256(SUMMARY) == s_sha, "July summary did not restore"
        SUMMARY_BAK.unlink()
        print("  July build summary restored (%s)" % s_sha[:16])

    # ---------------- f = 1.25 --------------------------------------------
    step("2", "apply f=1.25 (proven byte-exact against 855076 on 2026-08-22)")
    n = apply_f125(OUT, OUT_F125)
    assert n == N_D7, "d7 rows %d != %d" % (n, N_D7)
    print("  %d d7 rows scaled" % n)

    # ---------------- GUARD B ---------------------------------------------
    step("B", "submission validation gate")
    out_zip = OUT_F125.with_suffix(".zip")
    import zipfile
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_F125, "submission.csv")
    import validate_task1_submission as V
    # See the module docstring: the validator ignores argv entirely. Overriding
    # its module-level paths is the only way to point it at the real artifact.
    V.CSV_PATH = OUT_F125
    V.ZIP_PATH = out_zip
    V.REPORT_PATH = REPO / "reports" / "legB_2022_submission_validation_20260822.md"
    guard_b = V.main()
    # V.main() RETURNS 0/1; it does not raise. The first version of this runner
    # ignored the return value and printed "Built." under a FAILED guard, which
    # is the same silent-absorption failure this day has been cataloguing,
    # committed by the very script written to catch it. It is now load-bearing.
    if guard_b != 0:
        print("\n" + "!" * 74)
        print("GUARD B FAILED (rc=%d). The artifact exists but is NOT clean." % guard_b)
        print("Reporting and halting. Nothing uploaded, no decision taken.")
        print("!" * 74)

    # ---------------- report ----------------------------------------------
    step("3", "artifact")
    rows = sum(1 for _ in open(OUT_F125)) - 1
    res = {
        "csv": OUT_F125.name, "csv_sha256": sha256(OUT_F125),
        "csv_bytes": OUT_F125.stat().st_size,
        "zip": out_zip.name, "zip_sha256": sha256(out_zip),
        "zip_bytes": out_zip.stat().st_size,
        "rows": rows, "expected_rows": N_TOTAL,
        "pre_f125_csv": OUT.name, "pre_f125_sha256": sha256(OUT),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2))
    assert rows == N_TOTAL, "row count %d != %d" % (rows, N_TOTAL)
    if guard_b != 0:
        print("\nBuilt, GUARD B FAILED, HALTED. Not uploadable as it stands.")
        return 1
    print("\nBuilt and clean. NOT uploaded. Matteo uploads; CC never submits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
