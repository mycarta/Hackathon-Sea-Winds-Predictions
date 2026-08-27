"""Pre-A6 assertion: the build is the 2022 windows, and it is not the 2021 build.

CC dispatch 2026-08-19: "Before A6 (validation gate), add one check the runbook
did not have: assert that the output's issue dates are all in 2022 and none in
2021."

The submission CSV carries `window`, `horizon` and `hour` but no issue date, so
the issue date is only meaningful through the mapping the pipeline itself used,
`splits.eval_windows()`. This script therefore asserts on both sides of that
mapping, and then adds a content check so the answer does not rest on metadata
alone:

  1. `splits._installed_windows_year()` is 2022.
  2. Every one of the 8 windows has `context_end` (the issue date), `predict_*`
     and all three `score_days` in 2022, and NONE in 2021.
  3. The submission has the contracted shape: 4,196,640 rows, windows 0..7,
     horizons {1,7,14}, hours {0,6,12,18}, 96 blocks of 43,715 rows.
  4. Content check: the submission differs from the archived 2021 build. A build
     that silently reused 2021 inputs would satisfy 1-3 and fail here. Compared
     by SHA-256 of the value columns, and by the fraction of q50 values that
     differ.

Exit code 1 on any failure, so it can gate A6 in a shell chain.

Read-only with respect to the submission. Deterministic.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
SHIP = REPO_ROOT / "phase_2" / "phase2_dataset_ship"
SUB = KIT_PHASE2 / "part1_forecast" / "submission.csv"
SUB_2021 = KIT_PHASE2 / "part1_forecast" / "submission_2021.csv"
OUT_JSON = REPO_ROOT / "reports" / "assert_2022_issue_dates_20260819.json"

EXPECTED_ROWS = 4_196_640
N_CELLS = 43_715
VALUE_COLS = ["q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]


def main() -> None:
    os.environ.setdefault("PHASE2_DATA_ROOT", str(SHIP))
    sys.path.insert(0, str(KIT_PHASE2))
    sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
    import splits

    failures: list[str] = []
    out: dict = {"generated_by": "scripts/assert_2022_issue_dates_20260819.py"}

    # ── 1 + 2: window dates ─────────────────────────────────────────────
    installed = splits._installed_windows_year()
    print(f"[1] splits._installed_windows_year() = {installed}")
    if installed != 2022:
        failures.append(f"installed windows year is {installed}, expected 2022")

    windows = splits.eval_windows()
    print(f"[2] issue dates (context_end) per window:")
    issue_years = []
    for w in windows:
        dates = [w["context_start"], w["context_end"], w["predict_start"], w["predict_end"]]
        dates += list(w["score_days"].values())
        years = sorted({int(str(d)[:4]) for d in dates})
        issue_years.append(int(str(w["context_end"])[:4]))
        ok = years == [2022]
        print(f"    window {w['id']}: issue {w['context_end']}  all dates years={years}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"window {w['id']} has non-2022 dates: {years}")
        if 2021 in years:
            failures.append(f"window {w['id']} contains a 2021 date")
    out["issue_years"] = issue_years
    if set(issue_years) != {2022}:
        failures.append(f"issue years are {sorted(set(issue_years))}, expected {{2022}}")

    # ── 3: submission shape ─────────────────────────────────────────────
    print(f"\n[3] submission shape: {SUB}")
    df = pd.read_csv(SUB)
    print(f"    rows={len(df):,}")
    if len(df) != EXPECTED_ROWS:
        failures.append(f"rows={len(df)}, expected {EXPECTED_ROWS}")
    wins = sorted(df["window"].unique().tolist())
    hors = sorted(df["horizon"].unique().tolist())
    hrs = sorted(df["hour"].unique().tolist())
    print(f"    windows={wins}  horizons={hors}  hours={hrs}")
    if wins != list(range(8)):
        failures.append(f"windows={wins}, expected 0..7")
    if hors != [1, 7, 14]:
        failures.append(f"horizons={hors}, expected [1,7,14]")
    if hrs != [0, 6, 12, 18]:
        failures.append(f"hours={hrs}, expected [0,6,12,18]")
    grp = df.groupby(["window", "horizon", "hour"], sort=False).size()
    print(f"    blocks={len(grp)}  all {N_CELLS} rows: {bool((grp == N_CELLS).all())}")
    if len(grp) != 96 or not (grp == N_CELLS).all():
        failures.append(f"expected 96 blocks of {N_CELLS}, got {len(grp)}")
    out["shape"] = {"rows": int(len(df)), "windows": wins, "horizons": hors,
                    "hours": hrs, "blocks": int(len(grp))}

    # ── 4: content differs from the archived 2021 build ─────────────────
    print(f"\n[4] content check against the archived 2021 build")
    if not SUB_2021.exists():
        print(f"    archived 2021 submission not found at {SUB_2021}; check SKIPPED")
        out["content_check"] = {"status": "skipped", "reason": "no 2021 archive"}
    else:
        old = pd.read_csv(SUB_2021)
        def vsha(d):
            return hashlib.sha256(
                d[VALUE_COLS].to_csv(index=False, float_format="%.10g").encode()
            ).hexdigest()
        sha_new, sha_old = vsha(df), vsha(old)
        same = sha_new == sha_old
        print(f"    2021 archive value-SHA : {sha_old[:32]}...")
        print(f"    2022 build   value-SHA : {sha_new[:32]}...")
        frac = float("nan")
        if len(old) == len(df):
            frac = float((df["q50"].to_numpy() != old["q50"].to_numpy()).mean())
            print(f"    fraction of q50 values differing: {frac:.4f}")
        print(f"    identical to 2021 build: {same}  -> {'FAIL' if same else 'OK'}")
        if same:
            failures.append("submission is byte-identical to the archived 2021 build")
        out["content_check"] = {"sha_2022": sha_new, "sha_2021": sha_old,
                                "identical": same, "frac_q50_differing": frac}

    out["failures"] = failures
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print(f"2022-ISSUE-DATE ASSERTION: {'FAIL (' + str(len(failures)) + ')' if failures else 'PASS'}")
    for x in failures:
        print(f"  - {x}")
    print("=" * 70)
    print(f"wrote {OUT_JSON}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
