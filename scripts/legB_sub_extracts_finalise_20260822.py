#!/usr/bin/env python3
"""Custody for the rebuilt submission-window extracts (amendment 3, 2026-08-22).

Same treatment R1's 80 bias extracts got: verified, copied INTO THE REPO,
manifested with SHA-256s, and registered. Parameterised by year because there
are two sets, 2021 (the floor run) and 2022 (the submission run), built from
different date lists into different directories.

    python scripts/legB_sub_extracts_finalise_20260822.py --year 2021
    python scripts/legB_sub_extracts_finalise_20260822.py --year 2022

WHAT THESE ARE. 32 files per year, `extract_YYYYMMDD_hHH.npz`, the 8 submission
window issue dates crossed with the 4 init hours (00, 06, 12, 18). Read by
`tier2_d7_build_submission.py:129` via the `SUB_EXTRACTS` constant.

WHY THIS SCRIPT EXISTS AT ALL. The 2021 originals were lost with the
`tier2_smoke/` folder and had NEVER been listed in `data/PINNED_ARTIFACTS.md`,
unlike the downscaler and the bias extracts. Nobody knew they were load-bearing
until a run was attempted three days later. That single omission cost 6.0 h of
unbudgeted Pangu compute. Register entry 6 is the corrective record; this script
is the remedy.

CHECKS, all of which must pass before anything is copied:

  1. All 32 expected files exist: every issue date in the year's date list,
     crossed with every init hour. Named explicitly, so a partial run cannot
     look like a complete one.
  2. Each holds `d7_u` and `d7_v`, 45x57, float32, finite.
  3. No `d14_*` payload: the rebuild harvests step 7 only.
  4. `issue_date` in the file agrees with the filename and with the datemap.
  5. All 32 SHA-256s distinct. Identical hashes would mean the rollout
     silently reused an initial state across dates or hours.

Then the files are copied into the repo and every copy is re-hashed and checked
against its source, because a copy that is not verified is not custody.

Run only after the extracts for that year are complete. It refuses to write a
partial manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from _publication_paths import ppath  # noqa: E402  (publication tree)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACTS = HERE / "artifacts"

INIT_HOURS = (0, 6, 12, 18)
N_EXPECTED = 32
SHAPE = (45, 57)

MODEL_SHA = "613a5c140a1399abcaffb4dbce32af373a1f5f56c515704f5be61925bb9fdcfd"
ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"

YEARS = {
    "2021": {
        "src": ppath("<PROTECTED_ARTIFACTS>/arm_extracts_sub_2021_20260822"),
        "dst": REPO / "data" / "arm_extracts_sub_2021_20260822",
        "dates": ARTIFACTS / "legB_sub_issue_dates_2021_8.txt",
        "datemap": ARTIFACTS / "tier2_sub_datemap.json",
        "role": "the FLOOR run: rebuild of the lost originals, used for R4b and "
                "the G1/G2 comparison against banked 855076",
    },
    "2022": {
        "src": ppath("<PROTECTED_ARTIFACTS>/arm_extracts_sub_2022_20260822"),
        "dst": REPO / "data" / "arm_extracts_sub_2022_20260822",
        "dates": ARTIFACTS / "legB_sub_issue_dates_2022_8.txt",
        "datemap": ARTIFACTS / "tier2_sub_datemap_2022.json",
        "role": "the SUBMISSION run: built unconditionally under amendment 2, so "
                "R5's long pole is done whether or not the floor passes",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True, choices=sorted(YEARS))
    a = ap.parse_args()
    cfg = YEARS[a.year]
    src, dst = cfg["src"], cfg["dst"]

    print("=" * 74)
    print("Custody: %s submission-window extracts" % a.year)
    print("=" * 74)
    print("source %s" % src)
    print("repo   %s" % dst.relative_to(REPO).as_posix())

    issues = [ln.strip() for ln in cfg["dates"].read_text(encoding="utf-8").split("\n")
              if ln.strip()]
    assert len(issues) == 8, "expected 8 issue dates, got %d" % len(issues)
    dm = json.loads(cfg["datemap"].read_text(encoding="utf-8"))
    dm_issue = {w["issue"]: w for w in dm["windows"]}
    assert sorted(dm_issue) == sorted(issues), "date list and datemap disagree"

    # ---- 1. every expected file, named explicitly --------------------------
    expected = [(d, H, "extract_%s_h%02d.npz" % (d.replace("-", ""), H))
                for d in issues for H in INIT_HOURS]
    assert len(expected) == N_EXPECTED

    problems, rows = [], []
    for d, H, name in expected:
        p = src / name
        if not p.exists():
            problems.append("MISSING %s" % name)
            continue
        z = np.load(p, allow_pickle=True)
        keys = set(z.files)
        tag = "%s h%02d" % (d, H)
        if not {"d7_u", "d7_v"} <= keys:
            problems.append("KEYS %s: %s" % (tag, sorted(keys)))
            continue
        if any(k.startswith("d14") for k in keys):
            problems.append("D14 PAYLOAD in %s (harvest should be step 7 only)" % tag)
        u, v = z["d7_u"], z["d7_v"]
        if u.shape != SHAPE or v.shape != SHAPE:
            problems.append("SHAPE %s: %s %s" % (tag, u.shape, v.shape))
            continue
        if u.dtype != np.float32 or v.dtype != np.float32:
            problems.append("DTYPE %s: %s %s" % (tag, u.dtype, v.dtype))
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            problems.append("NONFINITE %s" % tag)
        if "issue_date" in keys:
            got = str(z["issue_date"])
            if d not in got:
                problems.append("ISSUE MISMATCH %s: file says %s" % (tag, got))
        rows.append({"issue": d, "hour": H, "name": name,
                     "bytes": os.path.getsize(p), "sha256": sha256(p),
                     "season": dm_issue[d]["season"],
                     "valid_d7": dm_issue[d]["valid_d7"]})

    print("\nverified %d of %d" % (len(rows), N_EXPECTED))
    if problems:
        print("\nPROBLEMS (%d). Nothing copied, no manifest written." % len(problems))
        for q in problems[:20]:
            print("  " + q)
        sys.exit(2)

    shas = set(r["sha256"] for r in rows)
    assert len(shas) == N_EXPECTED, (
        "only %d distinct SHA-256s among %d files: the rollout reused an initial "
        "state across dates or hours" % (len(shas), N_EXPECTED))
    print("all %d SHA-256s distinct" % N_EXPECTED)

    total = sum(r["bytes"] for r in rows)
    print("sizes %d to %d bytes, %d total"
          % (min(r["bytes"] for r in rows), max(r["bytes"] for r in rows), total))

    # ---- 2. copy into the repo, verify every copy --------------------------
    dst.mkdir(parents=True, exist_ok=True)
    for r in rows:
        shutil.copy2(src / r["name"], dst / r["name"])
    for r in rows:
        got = sha256(dst / r["name"])
        assert got == r["sha256"], "copy differs for %s" % r["name"]
    print("copied %d files into the repo, every copy hash-verified" % len(rows))

    # ---- 3. manifest -------------------------------------------------------
    L = []
    L.append("# MANIFEST: Leg B submission-window extracts, %s" % a.year)
    L.append("")
    L.append("**32 Pangu d7 coarse extracts**, 8 window issue dates x 4 init hours.")
    L.append("Produced by `scripts/tier2_pangu_rollout.py`, custody by")
    L.append("`scripts/legB_sub_extracts_finalise_20260822.py` on %s."
             % datetime.now().strftime("%Y-%m-%d"))
    L.append("")
    L.append("Role: %s." % cfg["role"])
    L.append("")
    L.append("## Why these are in the repo")
    L.append("")
    L.append("The 2021 originals were lost with the `tier2_smoke/` folder AND had")
    L.append("never been listed in `data/PINNED_ARTIFACTS.md`, unlike the downscaler")
    L.append("and the bias extracts. Nobody knew they were load-bearing until a run")
    L.append("was attempted three days later. That single omission cost 6.0 h of")
    L.append("unbudgeted Pangu compute across two years of windows.")
    L.append("")
    L.append("Committed so that restoring them costs a checkout, not three hours of")
    L.append("CPU. See register entry 6 for the corrective record.")
    L.append("")
    L.append("A working copy is kept at `%s`," % cfg["src"].as_posix())
    L.append("byte-identical and hash-verified after copying.")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append("| Generator | `scripts/tier2_pangu_rollout.py` |")
    L.append("| Date list | `%s` (committed) |" % cfg["dates"].relative_to(REPO).as_posix())
    L.append("| Datemap | `%s` (committed) |" % cfg["datemap"].relative_to(REPO).as_posix())
    L.append("| Init hours | %s |" % (INIT_HOURS,))
    L.append("| Harvest | step 7 only (d7); no d14 payload, asserted |")
    L.append("| Model | `pangu_weather_24.onnx`, SHA-256 `%s` |" % MODEL_SHA)
    L.append("| Initial states | ERA5 via WeatherBench2, `%s` |" % ERA5)
    L.append("| Thread pin | `intra_op_num_threads` = 16 |")
    L.append("| Files | %d |" % N_EXPECTED)
    L.append("| Total size | %d B |" % total)
    L.append("| Read at | `scripts/tier2_d7_build_submission.py:129` via `SUB_EXTRACTS` |")
    L.append("")
    L.append("## Checks that passed before this manifest was written")
    L.append("")
    L.append("1. All 32 expected files present, enumerated explicitly from the date")
    L.append("   list crossed with the init hours, so a partial run cannot pass.")
    L.append("2. `d7_u` and `d7_v` present, 45x57, float32, all finite.")
    L.append("3. No `d14_*` payload.")
    L.append("4. `issue_date` agrees with the filename and the datemap.")
    L.append("5. All 32 SHA-256s distinct. Identical hashes would mean the rollout")
    L.append("   silently reused an initial state across dates or hours.")
    L.append("")
    L.append("The script exits without copying or writing if any check fails.")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append("| Issue | Hour | Season | valid_d7 | Bytes | SHA-256 |")
    L.append("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["issue"], r["hour"])):
        L.append("| %s | %02d | %s | %s | %d | `%s` |"
                 % (r["issue"], r["hour"], r["season"], r["valid_d7"],
                    r["bytes"], r["sha256"]))
    L.append("")

    text = "\n".join(L) + "\n"
    bad = sorted(set(c for c in text if ord(c) > 126))
    assert not bad, "non-ASCII in manifest: %r" % bad
    (dst / "MANIFEST.md").write_text(text, encoding="utf-8", newline="\n")
    print("wrote %s" % (dst / "MANIFEST.md").relative_to(REPO).as_posix())

    print("\n" + json.dumps({
        "year": a.year, "files": len(rows), "distinct_shas": len(shas),
        "total_bytes": total,
        "repo_dir": dst.relative_to(REPO).as_posix(),
    }, indent=2))
    print("\nNEXT: git add the directory, then flip register entry 6b/6c to PRESENT.")


if __name__ == "__main__":
    main()
