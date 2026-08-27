#!/usr/bin/env python3
"""Repair the 2 non-monotone rows in the 2022 Leg B submission. Authorized fix.

The defect is in the kit's quantile post-processing, reproduced against the
organizers' own function in `scripts/kit_quantile_clamp_repro_20260822.py`: it
sorts the (q05, q50, q95) triple and THEN clips only q05 to zero, so a negative
q50 ends up below q05. The step that exists to guarantee monotonicity is the one
that breaks it.

THE CORRECTION, as authorized: clip all three at zero, THEN sort. Clipping
before sorting cannot break the ordering because the sort is last. Order of
operations is the whole bug.

APPLIED AS POST-PROCESSING, NOT AS A FROZEN EDIT. Same class as the f=1.25
rescaling: a documented, line-level pass over a finished CSV. The kit file and
`tier2_d7_build_submission.py` are untouched, so the organizers' code remains
byte-identical to what they shipped and our build stays reproducible from it.

SCOPE, asserted rather than hoped: this rewrites ONLY rows that actually violate
`q05 <= q50 <= q95` or `min >= 0`. Every other row is streamed VERBATIM, byte
for byte. The script reports the count and fails if it exceeds MAX_EXPECTED,
because a repair pass that silently touched thousands of rows would be a
different and much more alarming event than the one authorized.

Run:
    python scripts/legB_2022_repair_quantiles_20260822.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACTS = HERE / "artifacts"

SRC = ARTIFACTS / "submission_legB_2022_f125_20260822.csv"
SRC_SHA = "9755690ebc2b04701ebc2c36eddac4b9c0c9678fb74a0de8f1bb55706d3b7cfd"
OUT = ARTIFACTS / "submission_legB_2022_final_20260822.csv"
OUT_ZIP = ARTIFACTS / "submission_legB_2022_final_20260822.zip"

C_Q05, C_Q50, C_Q95 = 8, 9, 10
N_TOTAL = 4_196_640
MAX_EXPECTED = 10          # observed 2; a wider blast radius must stop the run


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


def main():
    got = sha256(SRC)
    print("source %s" % SRC.name)
    print("  SHA-256 %s" % got)
    assert got == SRC_SHA, "source SHA mismatch: %s != %s" % (got, SRC_SHA)
    print("  matches the R5 build")

    changed, n_rows = [], 0
    with open(SRC, "r", newline="") as fin, open(OUT, "w", newline="") as fout:
        fout.write(fin.readline())
        for line in fin:
            n_rows += 1
            f = line.rstrip("\n").split(",")
            a = float(f[C_Q05]); b = float(f[C_Q50]); c = float(f[C_Q95])
            if a <= b <= c and a >= 0.0:
                fout.write(line)                     # VERBATIM, byte-identical
                continue
            fixed = sorted((max(0.0, a), max(0.0, b), max(0.0, c)))
            f[C_Q05], f[C_Q50], f[C_Q95] = (fmt_q(fixed[0]), fmt_q(fixed[1]),
                                            fmt_q(fixed[2]))
            fout.write(",".join(f) + "\n")
            changed.append({
                "row": n_rows, "window": f[1], "lat": f[3], "lon": f[4],
                "horizon": f[5], "hour": f[6],
                "before": [a, b, c],
                "after": [float(f[C_Q05]), float(f[C_Q50]), float(f[C_Q95])],
            })

    assert n_rows == N_TOTAL, "row count %d != %d" % (n_rows, N_TOTAL)
    print("\nrows %d, repaired %d" % (n_rows, len(changed)))
    for ch in changed:
        print("  row %d  w%s h%s d%s  %s,%s"
              % (ch["row"], ch["window"], ch["hour"], ch["horizon"],
                 ch["lat"], ch["lon"]))
        print("    before (%+.3f, %+.3f, %+.3f)  ->  after (%+.3f, %+.3f, %+.3f)"
              % tuple(ch["before"] + ch["after"]))
    assert len(changed) <= MAX_EXPECTED, (
        "%d rows repaired, expected at most %d. A repair pass touching this many "
        "rows is a different event; stop and investigate."
        % (len(changed), MAX_EXPECTED))

    # ---- prove nothing else moved ---------------------------------------
    diff = 0
    with open(SRC) as fa, open(OUT) as fb:
        assert fa.readline() == fb.readline(), "header changed"
        for la, lb in zip(fa, fb):
            if la != lb:
                diff += 1
    print("\nlines differing from the source: %d" % diff)
    assert diff == len(changed), (
        "%d lines differ but %d were repaired" % (diff, len(changed)))
    print("  equals the repair count, so every untouched row is byte-identical")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT, "submission.csv")

    res = {
        "csv": OUT.name, "csv_sha256": sha256(OUT), "csv_bytes": OUT.stat().st_size,
        "zip": OUT_ZIP.name, "zip_sha256": sha256(OUT_ZIP),
        "zip_bytes": OUT_ZIP.stat().st_size,
        "rows": n_rows, "repaired": len(changed),
        "source": SRC.name, "source_sha256": SRC_SHA,
        "detail": changed,
    }
    print("\n" + json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
