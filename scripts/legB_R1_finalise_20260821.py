#!/usr/bin/env python3
"""Leg B R1 finalise: verify the 80 rebuilt extracts, witness-check, manifest, commit-ready.

Runs after tier2_pangu_rollout.py completes. Does four things:

  1. Verifies all 80 expected extracts exist, one per bias date, with the right
     arrays and no d14 payload (the rebuild harvests step 7 only).
  2. Checks each against the July witnesses where one exists. These are the
     `d7_fallback_frac` values logged by the July batch: a pure function of the
     7-step rollout and the coupler, touching neither the downscaler nor the
     scorer, so they are the only recorded quantity that witnesses the EXTRACTS
     rather than the pipeline downstream of them.
  3. Copies the extracts INTO THE REPO. Matteo's instruction, 2026-08-21: the
     custody class that lost the originals was "a file the pipeline depends on,
     living outside version control". Version-controlled extracts cost a
     checkout to restore, not eight hours.
  4. Writes MANIFEST.md with 80 SHA-256s, the ERA5 source and fetch date, the
     thread pin, and the wall time.

The working copy under <PROTECTED_ARTIFACTS>/ is kept as well; the two
are byte-identical and the copy is verified by hash after writing.

Run:  python scripts/legB_R1_finalise_20260821.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
from _publication_paths import ppath  # noqa: E402  (publication tree)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SRC = str(ppath("<PROTECTED_ARTIFACTS>/arm_extracts_20260821"))
DST = os.path.join(REPO, "data", "arm_extracts_20260821")
DATEMAP = os.path.join(REPO, "scripts", "artifacts", "tier2_d7_datemap.json")
WITNESSES = os.path.join(REPO, "scripts", "artifacts", "legB_R3_witnesses_20260821.json")
MANIFEST = os.path.join(DST, "MANIFEST.md")

# Recorded from the run, not inferred.
WALL_S = 26883
THREADS = 16
RUN_DATE = "2026-08-21"
ERA5_SOURCE = ("gs://weatherbench2/datasets/era5/"
               "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr")
ONNX_SHA = "613a5c140a1399abcaffb4dbce32af373a1f5f56c515704f5be61925bb9fdcfd"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    dm = json.load(open(DATEMAP))
    bias = dm["bias_issue_to_validV"]
    dates = sorted(bias.keys())
    assert len(dates) == 80, "expected 80 bias dates, got %d" % len(dates)

    print("=" * 74)
    print("Leg B R1 finalise: 80 rebuilt arm extracts")
    print("=" * 74)

    # ------------------------------------------------------- 1. verify
    rows = []
    problems = []
    for d in dates:
        tag = d.replace("-", "")
        p = os.path.join(SRC, "extract_%s.npz" % tag)
        if not os.path.isfile(p):
            problems.append("MISSING %s" % p)
            continue
        z = np.load(p, allow_pickle=False)
        keys = set(z.files)
        if not {"d7_u", "d7_v", "issue_date", "valid_d7", "meta"} <= keys:
            problems.append("KEYS %s: %s" % (tag, sorted(keys)))
            continue
        if any(k.startswith("d14_") for k in keys):
            problems.append("D14 PAYLOAD in %s (harvest should be step 7 only)" % tag)
        u, v = z["d7_u"], z["d7_v"]
        if u.shape != (45, 57) or v.shape != (45, 57):
            problems.append("SHAPE %s: %s %s" % (tag, u.shape, v.shape))
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            problems.append("NONFINITE %s" % tag)
        if str(z["issue_date"]) != d:
            problems.append("ISSUE MISMATCH %s: file says %s" % (d, z["issue_date"]))
        if str(z["valid_d7"]) != bias[d]:
            problems.append("VALID MISMATCH %s: file says %s, datemap says %s"
                            % (d, z["valid_d7"], bias[d]))
        meta = json.loads(str(z["meta"]))
        rows.append({
            "issue": d,
            "valid_d7": str(z["valid_d7"]),
            "bytes": os.path.getsize(p),
            "sha256": sha256(p),
            "fallback_frac": meta["d7_fallback_frac"],
        })

    print("\nverified %d of 80 extracts" % len(rows))
    if problems:
        print("\nPROBLEMS (%d):" % len(problems))
        for x in problems[:20]:
            print("  " + x)
        sys.exit(3)
    print("  all present, d7_u/d7_v (45,57) finite, no d14 payload,")
    print("  issue_date and valid_d7 agree with the committed datemap")

    sizes = [r["bytes"] for r in rows]
    print("  size %d to %d bytes, total %d" % (min(sizes), max(sizes), sum(sizes)))
    shas = set(r["sha256"] for r in rows)
    assert len(shas) == 80, "duplicate content across extracts: %d unique" % len(shas)
    print("  80 distinct SHA-256s, no duplicates")

    # ------------------------------------------------------- 2. witnesses
    W = json.load(open(WITNESSES))["witnesses"]
    print("\nwitness check against the July batch")
    wrows = []
    for d in sorted(W):
        jul = W[d]["fallback_frac"]
        rec = next((r for r in rows if r["issue"] == d), None)
        if rec is None:
            print("  %s not among the 80" % d)
            continue
        new = rec["fallback_frac"]
        exact = (new == jul)
        degenerate = (jul in (0.0, 1.0))
        wrows.append((d, jul, new, exact, degenerate))
        print("  %s  July %.16f  rebuild %.16f  %s%s"
              % (d, jul, new, "EXACT" if exact else "DIFFERS",
                 "  (degenerate, saturated)" if degenerate else ""))
    n_exact = sum(1 for w in wrows if w[3])
    n_inform = sum(1 for w in wrows if w[3] and not w[4])
    print("  %d of %d match exactly; %d of those are informative (not saturated)"
          % (n_exact, len(wrows), n_inform))
    if n_exact != len(wrows):
        print("\n  A WITNESS DIFFERS. Stop and report before the refit.")
        sys.exit(3)

    # ------------------------------------------------------- 3. copy into repo
    if not os.path.isdir(DST):
        os.makedirs(DST)
    print("\ncopying into the repo at %s" % os.path.relpath(DST, REPO))
    for r in rows:
        tag = r["issue"].replace("-", "")
        name = "extract_%s.npz" % tag
        shutil.copy2(os.path.join(SRC, name), os.path.join(DST, name))
        got = sha256(os.path.join(DST, name))
        assert got == r["sha256"], "copy differs for %s" % name
    print("  80 files copied, every one hash-verified after the copy")

    # ------------------------------------------------------- 4. manifest
    L = []
    L.append("# MANIFEST: rebuilt Leg B arm extracts, 2026-08-21")
    L.append("")
    L.append("**80 Pangu d7 coarse extracts**, regenerated after the July originals were")
    L.append("lost. Produced by `scripts/tier2_pangu_rollout.py` on %s." % RUN_DATE)
    L.append("")
    L.append("## Why these are in the repo")
    L.append("")
    L.append("The custody failure that lost the originals was not a backup failure. It was")
    L.append("a file the pipeline asserts by SHA living outside version control, in a")
    L.append("folder named like a scratch area, gitignored by pattern, never listed. These")
    L.append("are committed so that restoring them costs a checkout rather than 7.5 hours")
    L.append("of CPU. At 1.6 MB total that is a cheap insurance premium.")
    L.append("")
    L.append("A working copy is also kept at `%s`," % SRC)
    L.append("byte-identical and hash-verified after copying. The pipeline may read either.")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append("| Generator | `scripts/tier2_pangu_rollout.py` |")
    L.append("| Date list | `scripts/artifacts/legB_bias_issue_dates_80.txt` (committed) |")
    L.append("| Source of dates | `bias_issue_to_validV` in `scripts/artifacts/tier2_d7_datemap.json`, seed 42, 20 per season |")
    L.append("| Model | Pangu `pangu_weather_24.onnx`, SHA-256 `%s` |" % ONNX_SHA)
    L.append("| Initial states | ERA5 via WeatherBench2, `%s` |" % ERA5_SOURCE)
    L.append("| ERA5 fetch date | %s (fetched live at run time; no local cache exists) |" % RUN_DATE)
    L.append("| Init hour | 00 UTC on each issue date (compliance clause 1) |")
    L.append("| Rollout | 7 x 24 h steps, harvest step 7 only |")
    L.append("| `intra_op_num_threads` | **%d**, pinned via `--threads` |" % THREADS)
    L.append("| Wall time | **%d s (%.2f h)** for 80 extracts, 0 skipped |" % (WALL_S, WALL_S / 3600.0))
    L.append("| Total size | %d bytes across 80 files |" % sum(sizes))
    L.append("")
    L.append("## Thread pin, and what it does and does not claim")
    L.append("")
    L.append("Measured in R0 (`reports/legB_R0_preflight_20260821.md`): one 24 h step ran")
    L.append("47.4 s unpinned, 47.8 s at 8 threads and 45.2 s at 16, and all three produced")
    L.append("**bitwise identical** output from the same initial state. Pinning therefore")
    L.append("buys provenance, not correctness, on this machine. The claim recorded here is")
    L.append("run-to-run determinism at a known thread count; cross-machine reproducibility")
    L.append("is NOT claimed.")
    L.append("")
    L.append("## Relationship to the lost originals")
    L.append("")
    L.append("These are **method-identical, not bit-identical by assumption**. The one")
    L.append("place where that assumption can be tested is the `d7_fallback_frac` recorded")
    L.append("per date in the July session log, which is a pure function of the rollout and")
    L.append("the coupler and touches neither the downscaler nor the scorer. Three of the")
    L.append("80 bias dates have such a witness:")
    L.append("")
    L.append("| Issue date | July value | Rebuild | Verdict |")
    L.append("|---|---|---|---|")
    for d, jul, new, exact, degen in wrows:
        note = "exact match" + (", but degenerate (saturated at 1.0)" if degen else "")
        L.append("| %s | %.16f | %.16f | %s |" % (d, jul, new, note))
    L.append("")
    L.append("Only **%d** of the three is informative; the others are saturated and would" % n_inform)
    L.append("match trivially. That one is a 16-significant-digit reproduction of the")
    L.append("extract path a month later, on a fresh ERA5 fetch, at a different thread pin.")
    L.append("It is evidence, not proof: 77 of the 80 have no recorded witness because none")
    L.append("was ever taken. Stage R4, the 2021 re-run, carries the rest of the")
    L.append("verification load.")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append("| Issue date | Valid d7 | Bytes | `d7_fallback_frac` | SHA-256 |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        L.append("| %s | %s | %d | %.6f | `%s` |"
                 % (r["issue"], r["valid_d7"], r["bytes"], r["fallback_frac"], r["sha256"]))
    L.append("")
    L.append("Each file holds `d7_u`, `d7_v` (45x57 float32), plus `issue_date`,")
    L.append("`init_hour`, `valid_d7`, `valid_d14` and `meta`. No `d14_*` arrays: the")
    L.append("rebuild harvests step 7 only. Verified on all 80.")
    L.append("")

    text = "\n".join(L) + "\n"
    bad = sorted(set(c for c in text if ord(c) > 126))
    assert not bad, "non-ASCII in manifest: %r" % bad
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("\nwrote %s" % os.path.relpath(MANIFEST, REPO))

    summary = {
        "extracts": 80, "wall_s": WALL_S, "threads": THREADS,
        "total_bytes": sum(sizes),
        "witnesses_checked": len(wrows), "witnesses_exact": n_exact,
        "witnesses_informative": n_inform,
        "repo_path": os.path.relpath(DST, REPO).replace("\\", "/"),
        "working_path": SRC,
    }
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
