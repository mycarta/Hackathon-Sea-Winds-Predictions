#!/usr/bin/env python3
"""Reproduction: the kit's quantile post-processing can emit a non-monotone row.

Drives the ORGANIZER'S OWN function,
`phase_2/kit/phase_2/part1_forecast/build_forecast_submission.field_to_rows`,
with crafted inputs. Nothing is re-implemented here; the two lines under test
are the kit's, executed in place:

    q = np.sort(out[["q05", "q50", "q95"]].values, axis=1)
    out["q05"], out["q50"], out["q95"] = np.clip(q[:, 0], 0, None), q[:, 1], q[:, 2]

THE DEFECT. The sort restores monotonicity, then the clip is applied to the
FIRST element only. If the sorted middle element (q50) is negative, clipping
q05 up to 0 lifts it ABOVE q50, breaking the invariant the sort had just
established. The step that exists to guarantee `q05 <= q50 <= q95` is the step
that violates it. q50 and q95 are also left negative, so `speeds >= 0` fails too.

WHICH INPUTS TRIGGER IT: any triple whose MIDDLE value, after sorting, is
negative. Equivalently, at least two of the three quantiles below zero.

A CORRECTION TO OUR OWN FIRST DIAGNOSIS, which is why this script exists before
the fix rather than after. We initially described the minimal case as
(0.009, -0.034, 0.009). That is the triple observed in our OUTPUT file, and it
does NOT reproduce the bug when fed to the kit: only one of its three values is
negative, so the sorted middle is +0.009 and the result is a clean
(0, 0.009, 0.009). Case B below demonstrates that explicitly. The observed
output arose because our pipeline applies a separate f=1.25 interval rescaling
AFTER the kit step, which mapped the kit's broken (0, -0.034, 0.0) onto
(0.009, -0.034, 0.009). Reporting the post-rescaling triple as the input would
have sent the organizers a case that does not reproduce.

Run:  python scripts/kit_quantile_clamp_repro_20260822.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
KIT = REPO / "phase_2" / "kit" / "phase_2" / "part1_forecast"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))

import build_forecast_submission as K   # noqa: E402

SHAPE = (479, 433)


def drive(triple):
    """Feed one (q05, q50, q95) triple through the kit's real field_to_rows."""
    m, _, _ = K._grid()
    ys, xs = np.where(m)
    y0, x0 = ys[0], xs[0]

    q05 = np.full(SHAPE, 5.0, dtype=np.float32)
    q50 = np.full(SHAPE, 8.0, dtype=np.float32)
    q95 = np.full(SHAPE, 12.0, dtype=np.float32)
    q05[y0, x0], q50[y0, x0], q95[y0, x0] = triple
    d = np.zeros(SHAPE, dtype=np.float32)

    rows = K.field_to_rows(0, 7, 6, q05, q50, q95, d, d, d)
    r = rows.iloc[0]
    return float(r["q05"]), float(r["q50"]), float(r["q95"])


def show(label, triple, expect_violation):
    got = drive(triple)
    mono = got[0] <= got[1] <= got[2]
    nonneg = min(got) >= 0.0
    print("\n%s" % label)
    print("  input           (%+.3f, %+.3f, %+.3f)" % triple)
    print("  kit output      (%+.3f, %+.3f, %+.3f)" % got)
    print("  q05 <= q50 <= q95 : %s" % ("OK" if mono else "**VIOLATED**"))
    print("  all >= 0          : %s" % ("OK" if nonneg else "**VIOLATED**"))
    violated = (not mono) or (not nonneg)
    assert violated == expect_violation, (
        "%s: expected violation=%s, got %s" % (label, expect_violation, violated))
    return got, mono, nonneg


def main():
    print("=" * 74)
    print("Kit quantile post-processing: reproduction")
    print("=" * 74)
    print("\nDriving build_forecast_submission.field_to_rows directly.")
    print("Kit file: %s" % (KIT / "build_forecast_submission.py"))

    src = (KIT / "build_forecast_submission.py").read_text(encoding="utf-8")
    assert 'np.clip(q[:, 0], 0, None), q[:, 1], q[:, 2]' in src, (
        "the kit's post-processing line has changed; re-read it before trusting "
        "this reproduction")
    print("The two lines under test are present in the kit source, verified.")

    # ---- Case A: the real defect -------------------------------------------
    show("CASE A  two values below zero -> the sorted middle is negative",
         (-0.050, -0.034, 0.000), expect_violation=True)
    print("  The sort gives (-0.050, -0.034, 0.000). The clip lifts q05 to 0,")
    print("  which is now ABOVE q50 = -0.034. The guarantee is broken BY the")
    print("  step that exists to enforce it, and q50 is left negative.")

    # ---- Case B: the triple we first reported, which does NOT reproduce -----
    show("CASE B  our first-reported triple, which does NOT reproduce",
         (0.009, -0.034, 0.009), expect_violation=False)
    print("  Only one value is negative, so the sorted middle is +0.009 and the")
    print("  output is clean. This is the triple we observed in our OUTPUT file,")
    print("  produced by a later f=1.25 rescaling, NOT an input that triggers")
    print("  the defect. Recorded so the diagnosis is not overstated.")

    # ---- Case C: all three negative ----------------------------------------
    show("CASE C  all three below zero", (-2.0, -1.0, -0.5), expect_violation=True)
    print("  q50 and q95 are returned negative as well: `speeds >= 0` fails on")
    print("  two of the three columns, not just the ordering.")

    # ---- Case D: the proposed correction ------------------------------------
    print("\n" + "-" * 74)
    print("PROPOSED CORRECTION: clip all three at zero, THEN sort")
    print("-" * 74)
    for triple in ((-0.050, -0.034, 0.000), (0.009, -0.034, 0.009), (-2.0, -1.0, -0.5)):
        fixed = np.sort(np.clip(np.array(triple, dtype=np.float64), 0.0, None))
        mono = fixed[0] <= fixed[1] <= fixed[2]
        nonneg = fixed.min() >= 0
        print("  (%+.3f, %+.3f, %+.3f) -> (%+.3f, %+.3f, %+.3f)  mono %s  nonneg %s"
              % (triple + tuple(fixed) + ("OK" if mono else "NO", "OK" if nonneg else "NO")))
    print("\n  Clipping before sorting cannot break the ordering, because the sort")
    print("  is the last operation. Clipping after sorting can, because the clip")
    print("  moves one element past another. Order of operations is the whole bug.")

    print("\n" + "=" * 74)
    print("REPRODUCED. Case A and Case C violate; Case B does not.")
    print("=" * 74)


if __name__ == "__main__":
    main()
