#!/usr/bin/env python3
"""Leg B R4b: rebuild the 2021 submission, apply f=1.25, run gates G1/G2/G3.

Runs after the 32 2021 window extracts land. End to end, one command:

    python scripts/legB_R4b_run_20260822.py

WHY THIS RUNNER EXISTS RATHER THAN CALLING THE FROZEN SCRIPT DIRECTLY.

`tier2_d7_build_submission.py` opens its output with mode "w" at a HARDCODED
path: `scripts/artifacts/submission_pangu_d7_allhours_20260719.csv`. That file
is July's 854984 build, SHA `488955866949a644...`, and it is the pinned
`BASE_CSV`/`BASE_SHA` input of `tier2_d7_apply_f125.py:43-44`. Running the
frozen script as-is would DESTROY it in place, break that assertion, and erase
the July lineage. Its `.zip` and
`scripts/artifacts/tier2_d7_build_submission_summary.json` would go the same way.

So the runner overrides the module constants before calling `main()`, leaving
the frozen file byte-unmodified. This follows the repo's own precedent:
`ws_d7_feature_experiments.py:143` monkeypatches `forecast_hres` with the note
"kit files untouched". The summary JSON path is hardcoded inside `main()` and
cannot be overridden, so it is backed up and restored, hash-verified both ways.

Nothing at a canonical path is different when this script finishes. The rebuild
lands under explicit new names carrying their date.

THE f=1.25 TRANSFORM, and why it is re-implemented here rather than reused.

`tier2_d7_apply_f125.py` reads its base from a hardcoded path and asserts a
hardcoded SHA, so it cannot be pointed at the rebuild without a frozen edit. The
transform itself is three lines (`tier2_d7_apply_f125.py:14-16`):

    q95_new = q50 + f*(q95 - q50)
    q05_new = max(0, q50 - f*(q50 - q05))
    q50 unchanged, d7 speed rows only

Re-implementing it invites divergence, so it is PROVEN instead of trusted:
step 1 applies this implementation to July's 854984 and requires the result to
be BYTE-IDENTICAL to banked 855076, `169524f9ffa5c3e5...`. Both artifacts are in
custody and hash-verified today. If the reproduction is not exact, the run stops
before any rebuild happens. That turns "I think I copied the formula correctly"
into a proof against the only object that matters.

STAGES

    1  self-test: July 854984 + f=1.25 -> must reproduce 855076 exactly
    2  back up the summary JSON
    3  build the 2021 rebuild with OUT_CSV overridden
    4  restore the summary JSON, hash-verified
    5  apply f=1.25 to the rebuild
    6  gates G1/G2/G3 via legB_R4b_gates_20260822.py

Stop-and-report at stage 1 or 6 per the adopted gate. No decision is taken here
and nothing is submitted.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACTS = HERE / "artifacts"

JULY_ALLHOURS = ARTIFACTS / "submission_pangu_d7_allhours_20260719.csv"
JULY_ALLHOURS_SHA = "488955866949a6445fed37c817b0b192075dd0198dcc009787e2230a03fb70b8"
JULY_F125 = ARTIFACTS / "submission_pangu_d7_f125_20260720.csv"
JULY_F125_SHA = "169524f9ffa5c3e53f66a4d7f686299b1344c9f303040a68e610ed0816b33079"
SUMMARY = ARTIFACTS / "tier2_d7_build_submission_summary.json"
SUMMARY_BAK = ARTIFACTS / "tier2_d7_build_submission_summary_JULY_backup_20260822.json"

REBUILD = ARTIFACTS / "submission_legB_2021_rebuild_20260822.csv"
REBUILD_F125 = ARTIFACTS / "submission_legB_2021_rebuild_f125_20260822.csv"
SELFTEST = ARTIFACTS / "selftest_f125_reproduction_20260822.csv"

F = 1.25
C_HOR, C_Q05, C_Q50, C_Q95 = 5, 8, 9, 10
N_TOTAL, N_D7 = 4_196_640, 1_398_880


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_q(v):
    """Kit-style float, identical to tier2_d7_build_submission.fmt_q."""
    s = "%.3f" % round(float(v), 3)
    s = s.rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def apply_f125(src, dst, f=F):
    """q50 fixed; q05/q95 scaled about q50 on d7 speed rows only. Streamed."""
    n_d7 = 0
    with open(src, "r", newline="") as fin, open(dst, "w", newline="") as fout:
        fout.write(fin.readline())
        for line in fin:
            fields = line.rstrip("\n").split(",")
            if fields[C_HOR] != "7":
                fout.write(line)
                continue
            q05 = float(fields[C_Q05]); q50 = float(fields[C_Q50])
            q95 = float(fields[C_Q95])
            fields[C_Q95] = fmt_q(q50 + f * (q95 - q50))
            fields[C_Q05] = fmt_q(max(0.0, q50 - f * (q50 - q05)))
            fout.write(",".join(fields) + "\n")
            n_d7 += 1
    return n_d7


def step(n, title):
    print("\n" + "=" * 74)
    print("STAGE %d: %s" % (n, title))
    print("=" * 74)


def main():
    t0 = time.time()

    # ---------------- 1. prove the transform ---------------------------------
    step(1, "self-test: July 854984 + f=1.25 must reproduce banked 855076")
    for p, want in ((JULY_ALLHOURS, JULY_ALLHOURS_SHA), (JULY_F125, JULY_F125_SHA)):
        got = sha256(p)
        print("  %-46s %s" % (p.name, got[:16]))
        assert got == want, "%s SHA mismatch: %s != %s" % (p.name, got, want)
    print("  both July artifacts hash-verified")

    n = apply_f125(JULY_ALLHOURS, SELFTEST)
    got = sha256(SELFTEST)
    print("  transformed %d d7 rows -> %s" % (n, got[:16]))
    assert n == N_D7, "d7 row count %d != %d" % (n, N_D7)
    if got != JULY_F125_SHA:
        print("\n  STOP. The re-implemented f=1.25 does NOT reproduce 855076.")
        print("  got  %s" % got)
        print("  want %s" % JULY_F125_SHA)
        print("  Nothing is rebuilt. The transform must be proven before it is used.")
        sys.exit(3)
    print("  BYTE-IDENTICAL to 855076. The transform is proven, not assumed.")
    SELFTEST.unlink()

    # ---------------- 2. protect the July summary ----------------------------
    step(2, "back up the July build summary (its path is hardcoded in main())")
    if SUMMARY.exists():
        shutil.copy2(SUMMARY, SUMMARY_BAK)
        s_sha = sha256(SUMMARY)
        assert sha256(SUMMARY_BAK) == s_sha, "summary backup differs"
        print("  backed up %s (%s)" % (SUMMARY.name, s_sha[:16]))
    else:
        s_sha = None
        print("  no existing summary to protect")

    # ---------------- 3. build, with OUT_CSV overridden ----------------------
    step(3, "build the 2021 rebuild (frozen script, constants overridden)")
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
    sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))
    import tier2_d7_build_submission as B

    print("  SUB_EXTRACTS   %s" % B.SUB_EXTRACTS)
    print("  DWN_CACHE_SHA  %s" % B.DWN_CACHE_SHA)
    print("  OUT_CSV        %s" % B.OUT_CSV.name)
    assert B.OUT_CSV == JULY_ALLHOURS, (
        "OUT_CSV is %s, not the July path this runner is written to protect; "
        "re-read the frozen file before overriding it" % B.OUT_CSV)
    B.OUT_CSV = REBUILD
    print("  OUT_CSV overridden -> %s" % REBUILD.name)
    print("  (frozen file byte-unmodified; override is runner-side only)\n")

    t_build = time.time()
    B.main()
    print("\n  build wall time %.0f s" % (time.time() - t_build))

    # ---------------- 4. restore the July summary ----------------------------
    step(4, "restore the July build summary")
    new_summary = None
    if SUMMARY.exists():
        new_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        (ARTIFACTS / "legB_2021_rebuild_summary_20260822.json").write_text(
            json.dumps(new_summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("  rebuild summary saved as legB_2021_rebuild_summary_20260822.json")
    if s_sha is not None:
        shutil.copy2(SUMMARY_BAK, SUMMARY)
        assert sha256(SUMMARY) == s_sha, "July summary did not restore cleanly"
        SUMMARY_BAK.unlink()
        print("  July summary restored and hash-verified (%s)" % s_sha[:16])

    got = sha256(JULY_ALLHOURS)
    assert got == JULY_ALLHOURS_SHA, (
        "JULY 854984 WAS MODIFIED during the build: %s. This is the failure the "
        "runner exists to prevent; investigate before doing anything else." % got)
    print("  July 854984 still intact (%s)" % got[:16])

    # ---------------- 5. f=1.25 on the rebuild -------------------------------
    step(5, "apply the proven f=1.25 transform to the rebuild")
    n = apply_f125(REBUILD, REBUILD_F125)
    assert n == N_D7, "d7 row count %d" % n
    print("  %s" % REBUILD.name)
    print("    SHA-256 %s" % sha256(REBUILD))
    print("  %s" % REBUILD_F125.name)
    print("    SHA-256 %s" % sha256(REBUILD_F125))

    # ---------------- 6. gates ----------------------------------------------
    step(6, "gates G1 / G2 / G3")
    r = subprocess.run(
        [sys.executable, str(HERE / "legB_R4b_gates_20260822.py"),
         "--candidate", str(REBUILD_F125)],
        cwd=str(REPO))
    print("\ntotal wall time %.0f s" % (time.time() - t0))
    print("gate exit code %d (0 = G1 and G2 both pass)" % r.returncode)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
