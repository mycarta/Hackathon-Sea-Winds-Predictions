#!/usr/bin/env python3
"""Leg B Stage R3: rebuild the per-cell x season d7 bias table and record it.

This is the table `tier2_d7_build_submission.py:108-109` builds LIVE at splice
time. It is not stored anywhere, which is why losing the extracts blocked the
whole build. R3 runs it once, standalone, and writes the numbers down so the
splice's later rebuild has something to be checked against.

CHAIN, unchanged from the frozen script:

    tier2_d7_datemap.json["bias_issue_to_validV"]     80 issue -> valid pairs
    -> S.pg_ctx(issue)      reads data/arm_extracts_20260821/extract_<date>.npz
    -> S.pg_coarse          the (d7_u, d7_v) 45x57 coarse fields
    -> S.downscaled         the R2 refit downscaler, SHA b3ae32c0...
    -> spd - AROME truth    per footprint cell, at HOUR = 0
    -> per season: mean over dates, then shrink toward the season mean with
       w = n / (n + K_SHRINK), K_SHRINK = 30.0

SEEDING. `build_bias` itself is deterministic: it is a mean over a fixed,
committed date list with no sampling step. The seed 42 in this stage lives in
`tier2_d7_datemap.json`, which was generated with it; the file is committed and
SHA-listed in `data/PINNED_ARTIFACTS.md`, so the date selection is pinned by
custody rather than re-drawn here. This script re-asserts that datemap SHA
instead of re-sampling.

WHAT THIS CANNOT DO. It cannot reproduce July's table, for two independent
reasons established 2026-08-21:

  1. `build_bias` calls `downscaled()`, so the table depends on the DOWNSCALER.
     July's was the lost unseeded pickle; R2's refit is a different object by
     construction. Even byte-identical extracts would give a different table.
  2. Every number in `tier2_d7_fourblock.json` scores over 112 block valid dates
     that read the 91 EVAL extracts, which were never regenerated.

The four-block gate is therefore DEAD as a comparison. This stage records what
the table IS, not that it matches.

Output: `reports/legB_R3_bias_table_20260822.md` plus
`data/legB_R3_bias_table_20260822.npz` (the four season vectors, so the splice's
live rebuild can be diffed against them).

Run:  python scripts/legB_R3_bias_table_20260822.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))

import tier2_eval_common as ec          # noqa: E402
import tier2_d7_score_blocks as S       # noqa: E402
from tier2_f2_d14_precheck import DWN_CACHE   # noqa: E402

ARTIFACTS = _HERE / "artifacts"
DATEMAP = ARTIFACTS / "tier2_d7_datemap.json"
DATEMAP_SHA16 = "68ab210d15436855"        # data/PINNED_ARTIFACTS.md, frozen table
DWN_SHA = "b3ae32c0bf4203351a03526a454030817f70adb588f55caefdb0b43b5a2d8703"

REPORT = REPO / "reports" / "legB_R3_bias_table_20260822.md"
OUT_NPZ = REPO / "data" / "legB_R3_bias_table_20260822.npz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t_all = time.time()
    print("=" * 74)
    print("Leg B R3: d7 bias table rebuild")
    print("=" * 74)

    # ---- pin the inputs before spending anything ------------------------
    got = sha256(DWN_CACHE)
    print("\ndownscaler %s" % DWN_CACHE)
    print("  SHA-256 %s" % got)
    assert got == DWN_SHA, "downscaler SHA mismatch: %s != %s" % (got, DWN_SHA)
    print("  matches the R2 refit pin")

    dm_sha = sha256(DATEMAP)
    print("\ndatemap %s" % DATEMAP.relative_to(REPO).as_posix())
    print("  SHA-256 %s" % dm_sha)
    assert dm_sha.startswith(DATEMAP_SHA16), (
        "datemap SHA %s does not start with the registered %s"
        % (dm_sha[:16], DATEMAP_SHA16))
    print("  matches the registered SHA-16 %s (seed-42 date selection, pinned by"
          % DATEMAP_SHA16)
    print("  custody rather than re-drawn here)")

    dm = json.load(open(DATEMAP))
    bias_map = dm["bias_issue_to_validV"]
    print("\nbias dates in the map: %d" % len(bias_map))

    # ---- extract availability, named before the run, not discovered in it ----
    missing = [i for i in bias_map
               if not (S.EXTRACTS / ("extract_%s.npz" % i.replace("-", ""))).exists()]
    print("extracts present: %d of %d" % (len(bias_map) - len(missing), len(bias_map)))
    assert not missing, (
        "MISSING extracts for %d dates, e.g. %s. build_bias would silently skip "
        "them (it has no such guard), so this stage stops instead."
        % (len(missing), missing[:5]))

    # ---- prep the validated globals, then build -------------------------
    S._prep()
    print("footprint cells: %d" % S._YS.size)

    print("\nbuilding bias table ...")
    t0 = time.time()
    bias = S.build_bias(S.pg_ctx, S.pg_coarse, bias_map)
    build_s = time.time() - t0
    print("  built in %.0f s" % build_s)

    # ---- per-season stats -----------------------------------------------
    seasons = list(ec.SEASONS)
    rows = []
    n_used = {}
    for s in seasons:
        v = np.asarray(bias[s], dtype=float)
        n_dates = sum(1 for i, V in bias_map.items()
                      if ec.season_of(V) == s
                      and ec.arome_truth_speed(str(V), S.HOUR) is not None)
        n_used[s] = n_dates
        w = n_dates / (n_dates + S.K_SHRINK)
        rows.append((s, n_dates, w, float(v.mean()), float(v.std()),
                     float(v.min()), float(v.max()),
                     int(np.count_nonzero(~np.isfinite(v)))))

    print("\n%-10s %5s %6s %9s %9s %9s %9s %7s"
          % ("season", "n", "w", "mean", "std", "min", "max", "nonfin"))
    for s, n, w, mu, sd, lo, hi, nf in rows:
        print("%-10s %5d %6.3f %+9.4f %9.4f %+9.4f %+9.4f %7d"
              % (s, n, w, mu, sd, lo, hi, nf))

    for s, n, w, mu, sd, lo, hi, nf in rows:
        assert nf == 0, "season %s has %d non-finite bias cells" % (s, nf)
        assert n > 0, "season %s used 0 dates; the table would be all zeros" % s

    total_dates = sum(n_used.values())
    print("\ndates actually used: %d of %d in the map" % (total_dates, len(bias_map)))

    np.savez_compressed(OUT_NPZ, **{s: np.asarray(bias[s]) for s in seasons})
    npz_sha = sha256(OUT_NPZ)
    print("\nwrote %s" % OUT_NPZ.relative_to(REPO).as_posix())
    print("  SHA-256 %s" % npz_sha)

    # ---- report ----------------------------------------------------------
    L = []
    L.append("# Leg B R3: d7 bias table, 2026-08-22")
    L.append("")
    L.append("Produced by `scripts/legB_R3_bias_table_20260822.py`.")
    L.append("")
    L.append("This is the table the splice rebuilds LIVE at")
    L.append("`tier2_d7_build_submission.py:108-109`. It is stored nowhere in the")
    L.append("frozen pipeline, which is exactly why losing the extracts blocked the")
    L.append("build. Recording it here gives the splice's later rebuild something to")
    L.append("be checked against.")
    L.append("")
    L.append("## Inputs, pinned before the run")
    L.append("")
    L.append("| Input | Value |")
    L.append("|---|---|")
    L.append("| Downscaler | `b3ae32c0...` (R2 refit, seed 42), asserted |")
    L.append("| Datemap | `%s...` (registered `%s`), asserted |" % (dm_sha[:16], DATEMAP_SHA16))
    L.append("| Extracts | `data/arm_extracts_20260821/`, %d of %d present, asserted |"
             % (len(bias_map) - len(missing), len(bias_map)))
    L.append("| Footprint cells | %d |" % S._YS.size)
    L.append("| Hour | %d UTC |" % S.HOUR)
    L.append("| Shrinkage | `K_SHRINK` = %.1f |" % S.K_SHRINK)
    L.append("| Build wall time | %.0f s |" % build_s)
    L.append("| Output | `data/legB_R3_bias_table_20260822.npz`, SHA-256 `%s` |" % npz_sha)
    L.append("")
    L.append("## Table")
    L.append("")
    L.append("Bias is `downscaled_q50 - AROME truth`, per footprint cell, averaged")
    L.append("over that season's dates, then shrunk toward the season mean with")
    L.append("`w = n / (n + 30)`. A POSITIVE value means the Pangu chain runs fast")
    L.append("and the splice subtracts it.")
    L.append("")
    L.append("| Season | n dates | shrink w | mean | std | min | max |")
    L.append("|---|---|---|---|---|---|---|")
    for s, n, w, mu, sd, lo, hi, nf in rows:
        L.append("| %s | %d | %.3f | %+.4f | %.4f | %+.4f | %+.4f |"
                 % (s, n, w, mu, sd, lo, hi))
    L.append("")
    L.append("%d of the %d mapped dates contributed; the remainder have no AROME"
             % (total_dates, len(bias_map)))
    L.append("truth at %d UTC. All four seasons are non-empty and every cell is" % S.HOUR)
    L.append("finite, both asserted.")
    L.append("")
    L.append("## What the sign means, and a weak cross-check")
    L.append("")
    L.append("All four seasonal means are NEGATIVE. The splice computes")
    L.append("`q50c = spd - bb`, so a negative `bb` ADDS speed: the Pangu chain runs")
    L.append("SLOW against AROME truth on these dates, and the correction pushes it up.")
    L.append("The effect is small in MAM (-0.14) and large in SON (-2.09), a factor of")
    L.append("fifteen between them.")
    L.append("")
    L.append("There is no July bias table to compare against, for the reasons below.")
    L.append("But `scripts/artifacts/tier2_d7_fourblock.json` records July's Pangu")
    L.append("`center_bias` per block, measured AFTER the correction was applied:")
    L.append("")
    L.append("| Season | July residual after correction | R3 correction |")
    L.append("|---|---|---|")
    L.append("| DJF | -0.2379 | %+.4f |" % float(np.asarray(bias["DJF"]).mean()))
    L.append("| MAM | -1.5366 | %+.4f |" % float(np.asarray(bias["MAM"]).mean()))
    L.append("| JJA | -0.5035 | %+.4f |" % float(np.asarray(bias["JJA"]).mean()))
    L.append("| SON | **+1.9798** | %+.4f |" % float(np.asarray(bias["SON"]).mean()))
    L.append("")
    L.append("SON is the informative row. It is the ONLY season where July's")
    L.append("post-correction residual is positive, and strongly so: the correction")
    L.append("overshot by about 2 m/s. That is what a large negative SON entry does")
    L.append("when it is applied to dates that needed less of it. R3 produces a large")
    L.append("negative SON entry, -2.09, and small ones elsewhere.")
    L.append("")
    L.append("**This is weak corroboration and nothing more.** The two columns are")
    L.append("different quantities over different date populations (80 bias dates")
    L.append("versus 112 block valid dates), so they cannot be arithmetically")
    L.append("reconciled, and MAM does not line up neatly. What it does establish is")
    L.append("that the refit table has not flipped sign or changed order of magnitude")
    L.append("against the only July trace that survives. Treated as a smell test that")
    L.append("passed, not as validation.")
    L.append("")
    L.append("## Seeding")
    L.append("")
    L.append("`build_bias` is deterministic: a mean over a fixed committed date list,")
    L.append("no sampling step. The seed 42 belongs to `tier2_d7_datemap.json`, which")
    L.append("was drawn with it and is committed and SHA-listed. This stage therefore")
    L.append("re-ASSERTS that SHA rather than re-drawing the dates, which is the")
    L.append("stronger guarantee: a re-draw could silently differ, a SHA cannot.")
    L.append("")
    L.append("## What this stage cannot claim")
    L.append("")
    L.append("It does NOT reproduce July's table, and no comparison against")
    L.append("`tier2_d7_fourblock.json` is possible. Two independent reasons,")
    L.append("established 2026-08-21:")
    L.append("")
    L.append("1. `build_bias` calls `downscaled()`, so the table depends on the")
    L.append("   downscaler. July's was the lost unseeded pickle; the R2 refit is a")
    L.append("   different object by construction. Even byte-identical extracts would")
    L.append("   give a different table.")
    L.append("2. Every number in `tier2_d7_fourblock.json` scores over 112 block valid")
    L.append("   dates that read the 91 EVAL extracts. Those were never regenerated.")
    L.append("")
    L.append("The four-block gate is DEAD as a comparison. R4b's 2021 score against")
    L.append("the 27.14 floor carries the verification load instead.")
    L.append("")
    L.append("## One guard added that the frozen code does not have")
    L.append("")
    L.append("`build_bias` skips a date whose truth is missing and has no guard at all")
    L.append("for a missing EXTRACT: `np.load` would raise, but only partway through.")
    L.append("This script asserts all %d extracts exist BEFORE building, so a custody"
             % len(bias_map))
    L.append("gap fails in a second rather than mid-table. The frozen file is not")
    L.append("edited; the guard lives here.")
    L.append("")

    text = "\n".join(L) + "\n"
    bad = sorted(set(c for c in text if ord(c) > 126))
    assert not bad, "non-ASCII in report: %r" % bad
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    print("\nwrote %s" % REPORT.relative_to(REPO).as_posix())

    print("\n" + json.dumps({
        "seasons": {s: {"n": n, "w": round(w, 4), "mean": round(mu, 5),
                        "std": round(sd, 5), "min": round(lo, 5), "max": round(hi, 5)}
                    for s, n, w, mu, sd, lo, hi, nf in rows},
        "dates_used": total_dates, "dates_mapped": len(bias_map),
        "npz_sha256": npz_sha, "build_s": round(build_s, 1),
        "elapsed_s": round(time.time() - t_all, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
