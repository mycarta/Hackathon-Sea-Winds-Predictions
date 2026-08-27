"""Row-level verification of the uploaded Task 1 artifact (submission 897665).

CC dispatch 2026-08-27, item 5. Strictly read-only: the submission is opened
from inside its zip, nothing is rewritten, and no violation is repaired. Per
the dispatch, "any non-zero is an escalation, not a fix".

**Which artifact.** The board id 897665 appears nowhere in the repo record
(`grep -rn 897665` returns only the deliverables-manifest line in
`OBLIGATIONS.md` and the 2026-08-25 dispatch that seeded it), so the id cannot
be resolved to a file from inside the repo. This script therefore verifies the
artifact that the record DOES identify as the final Task 1 upload -- the 2022
Leg B build of 2026-08-22 -- and asserts its SHA-256 against the two hashes
logged in `LLM_AGENT_LOG.md:4299-4302` before checking a single row. If the
hashes match, the file under test is the one the log says was handed to Matteo
for upload; confirming that upload is 897665 remains Matteo's to state.

    csv  scripts/artifacts/submission_legB_2022_final_20260822.csv
         5c6c9f6ad96ed4a56fcde6204d0d521002de024e59e2edb68568a14904b4b9a2
    zip  scripts/artifacts/submission_legB_2022_final_20260822.zip
         d44cfee92f0eb4ac2ada41dd7c4bacc779d85041b086293e07e4cf2d16a5e945

The four checks are run against the CSV **inside the zip**, because the zip is
what is uploaded; the loose CSV is hashed separately and compared, so a
divergence between the two would itself be a finding.

**Checks (dispatch wording).**
  1. row count against the expected 4,196,640
     = 43,715 points x 8 windows x 3 horizons x 4 hours (Phase_2.pdf)
  2. quantile ordering q05 <= q50 <= q95 on every row
  3. non-negativity of the wind-speed quantiles
  4. direction values inside range

Note on check 2 and 4: the ordering check applies to the SPEED quantiles only.
Direction quantiles are circular -- dir_05 > dir_95 is the normal encoding of an
arc that crosses north and is not a violation -- so directions are checked for
range membership [0, 360) only, which is what the dispatch asks for. Counting
circular directions as an ordering violation would manufacture ~1e6 false
findings.

Counts of violations are reported, not samples, as the dispatch requires. A
small witness set is additionally written to the JSON for any non-zero count so
an escalation has something to look at, but the headline is the count.

Deterministic: a pure scan of a SHA-pinned file. No stochastic step, so no seed
applies.

Reads : scripts/artifacts/submission_legB_2022_final_20260822.{zip,csv}
Writes: reports/audit_submission_rowcheck_20260827.json
"""
from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SUB_CSV = REPO_ROOT / "scripts" / "artifacts" / "submission_legB_2022_final_20260822.csv"
SUB_ZIP = REPO_ROOT / "scripts" / "artifacts" / "submission_legB_2022_final_20260822.zip"
OUT_JSON = REPO_ROOT / "reports" / "audit_submission_rowcheck_20260827.json"

# LLM_AGENT_LOG.md:4299-4302
CSV_SHA256 = "5c6c9f6ad96ed4a56fcde6204d0d521002de024e59e2edb68568a14904b4b9a2"
ZIP_SHA256 = "d44cfee92f0eb4ac2ada41dd7c4bacc779d85041b086293e07e4cf2d16a5e945"

# Phase_2.pdf submission-size constant, mirrored in CLAUDE.md.
EXPECTED_ROWS = 4_196_640
EXPECTED_POINTS = 43_715
EXPECTED_WINDOWS = 8
EXPECTED_HORIZONS = 3
EXPECTED_HOURS = 4

INNER_NAME = "submission.csv"
CHUNK = 500_000
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]
MAX_WITNESSES = 5


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_member(zpath: Path, member: str) -> str:
    h = hashlib.sha256()
    with zipfile.ZipFile(zpath) as z, z.open(member) as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    result: dict = {
        "generated_by": "scripts/audit_submission_897665_rowcheck_20260827.py",
        "dispatch": "CC dispatch 2026-08-27, item 5",
        "board_id_claimed": 897665,
        "board_id_resolvable_from_repo": False,
        "artifact_zip": str(SUB_ZIP.relative_to(REPO_ROOT)).replace("\\", "/"),
        "artifact_csv": str(SUB_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
    }

    for p in (SUB_CSV, SUB_ZIP):
        if not p.exists():
            raise SystemExit(f"missing artifact: {p}")

    print("hashing artifacts ...")
    csv_sha, zip_sha = sha256_of(SUB_CSV), sha256_of(SUB_ZIP)
    inner_sha = sha256_of_member(SUB_ZIP, INNER_NAME)
    result["sha256"] = {
        "csv_actual": csv_sha, "csv_expected": CSV_SHA256,
        "csv_match": csv_sha == CSV_SHA256,
        "zip_actual": zip_sha, "zip_expected": ZIP_SHA256,
        "zip_match": zip_sha == ZIP_SHA256,
        "zip_inner_csv_actual": inner_sha,
        "zip_inner_matches_loose_csv": inner_sha == csv_sha,
    }
    for k in ("csv_match", "zip_match", "zip_inner_matches_loose_csv"):
        print(f"  {k}: {result['sha256'][k]}")
    if not (result["sha256"]["csv_match"] and result["sha256"]["zip_match"]):
        raise SystemExit(
            "SHA mismatch against LLM_AGENT_LOG.md:4299-4302 -- the file under "
            "test is not the logged artifact. Halting rather than reporting "
            "checks on an unidentified file."
        )

    with zipfile.ZipFile(SUB_ZIP) as z:
        names = z.namelist()
    result["zip_members"] = names
    if names != [INNER_NAME]:
        raise SystemExit(f"zip must contain exactly [{INNER_NAME!r}], has {names}")

    # ── Scan ────────────────────────────────────────────────────────────
    n_rows = 0
    v_order = 0            # q05 <= q50 <= q95 violated
    v_negative = 0         # any speed quantile < 0
    v_dir_range = 0        # any direction outside [0, 360)
    v_nan = 0
    witnesses: dict[str, list] = {"order": [], "negative": [], "dir_range": []}
    points: set[tuple] = set()
    windows: set = set()
    horizons: set = set()
    hours: set = set()
    combo_counter: dict = {}

    print(f"scanning {INNER_NAME} in {CHUNK:,}-row chunks ...")
    with zipfile.ZipFile(SUB_ZIP) as z, z.open(INNER_NAME) as fh:
        reader = pd.read_csv(fh, chunksize=CHUNK)
        for ci, ch in enumerate(reader):
            n_rows += len(ch)

            q05 = ch["q05"].to_numpy(dtype=float)
            q50 = ch["q50"].to_numpy(dtype=float)
            q95 = ch["q95"].to_numpy(dtype=float)
            d = ch[DIR_COLS].to_numpy(dtype=float)

            bad_order = ~((q05 <= q50) & (q50 <= q95))
            bad_neg = (q05 < 0) | (q50 < 0) | (q95 < 0)
            bad_dir = ~((d >= 0.0) & (d < 360.0)).all(axis=1)
            bad_nan = (~np.isfinite(np.column_stack([q05, q50, q95]))).any(axis=1) | \
                      (~np.isfinite(d)).any(axis=1)

            v_order += int(bad_order.sum())
            v_negative += int(bad_neg.sum())
            v_dir_range += int(bad_dir.sum())
            v_nan += int(bad_nan.sum())

            for key, mask in (("order", bad_order), ("negative", bad_neg),
                              ("dir_range", bad_dir)):
                if mask.any() and len(witnesses[key]) < MAX_WITNESSES:
                    take = ch.loc[mask].head(MAX_WITNESSES - len(witnesses[key]))
                    witnesses[key].extend(
                        take.to_dict(orient="records"))

            points.update(map(tuple, ch[["latitude", "longitude"]].to_numpy()))
            windows.update(ch["window"].unique().tolist())
            horizons.update(ch["horizon"].unique().tolist())
            hours.update(ch["hour"].unique().tolist())

            key_counts = ch.groupby(["window", "horizon", "hour"]).size()
            for k, v in key_counts.items():
                combo_counter[k] = combo_counter.get(k, 0) + int(v)

            if (ci + 1) % 2 == 0:
                print(f"  {n_rows:,} rows ... ({time.time()-t0:.0f}s)")

    # ── Structure ───────────────────────────────────────────────────────
    n_points = len(points)
    expected_per_combo = n_points
    combos_wrong = {str(k): v for k, v in combo_counter.items()
                    if v != expected_per_combo}

    result["checks"] = {
        "row_count": {
            "actual": n_rows, "expected": EXPECTED_ROWS,
            "violations": 0 if n_rows == EXPECTED_ROWS else abs(n_rows - EXPECTED_ROWS),
            "pass": n_rows == EXPECTED_ROWS,
        },
        "quantile_ordering_q05_le_q50_le_q95": {
            "violations": v_order, "pass": v_order == 0,
        },
        "speed_non_negativity": {
            "violations": v_negative, "pass": v_negative == 0,
        },
        "direction_in_range_0_360": {
            "violations": v_dir_range, "pass": v_dir_range == 0,
        },
    }
    result["supporting"] = {
        "non_finite_values": {"violations": v_nan, "pass": v_nan == 0},
        "distinct_points": {"actual": n_points, "expected": EXPECTED_POINTS,
                            "pass": n_points == EXPECTED_POINTS},
        "windows": {"actual": sorted(windows), "expected_count": EXPECTED_WINDOWS,
                    "pass": len(windows) == EXPECTED_WINDOWS},
        "horizons": {"actual": sorted(horizons), "expected_count": EXPECTED_HORIZONS,
                     "pass": len(horizons) == EXPECTED_HORIZONS},
        "hours": {"actual": sorted(hours), "expected_count": EXPECTED_HOURS,
                  "pass": len(hours) == EXPECTED_HOURS},
        "cross_product_complete": {
            "n_combinations": len(combo_counter),
            "expected_combinations": EXPECTED_WINDOWS * EXPECTED_HORIZONS * EXPECTED_HOURS,
            "combinations_with_wrong_row_count": combos_wrong,
            "pass": (len(combo_counter) == EXPECTED_WINDOWS * EXPECTED_HORIZONS * EXPECTED_HOURS
                     and not combos_wrong),
        },
    }
    result["witnesses"] = {k: v for k, v in witnesses.items() if v}
    result["wall_clock_s"] = time.time() - t0

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    print("\n" + "=" * 68)
    print("ITEM 5 -- violation counts (zero is the expected answer)")
    for name, c in result["checks"].items():
        print(f"  {name:38s} violations={c['violations']:<10} "
              f"{'PASS' if c['pass'] else 'FAIL'}")
    print("-" * 68)
    for name, c in result["supporting"].items():
        print(f"  {name:38s} {'PASS' if c['pass'] else 'FAIL'}")
    print("=" * 68)
    all_pass = all(c["pass"] for c in result["checks"].values()) and \
               all(c["pass"] for c in result["supporting"].values())
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL -- ESCALATE, DO NOT FIX'}")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
