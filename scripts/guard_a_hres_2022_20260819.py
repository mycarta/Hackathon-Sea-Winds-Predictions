"""Guard A: the 2022 HRES driver is visible, and it does not perturb 2021.

CC dispatch 2026-08-19, condition 4 on the blocker-4 approval. Runs BEFORE Leg A.

Two assertions:

  A1. `forecast_hres.build_hres_table([issue])` returns exactly 20,520 rows at
      EVERY one of the eight 2022 window issue dates, with finite driver columns.
      20,520 = 2,565 grid points x 8 lead/hour feature blocks, the same count the
      pre-update probe returned for a working 2021 issue date.

  A2. The same call at a 2021 issue date is byte-identical to what the pre-update
      configuration produced. "Pre-update" is reconstructed exactly: kit 9edd92b
      added `inference/window_*/context_hres_north_sea.parquet` to
      `config.hres_parquets()`, so the pre-update source set is that list minus
      the inference entries, i.e. the two train parquets alone. The table is
      built both ways and compared by SHA-256 of a canonical serialisation, so
      the check is on content, not on row count.

A2 is the one that matters: it proves that making the 2022 driver visible cannot
have changed a single 2021 feature value, which is what makes the 2021 rehearsal
still a valid reference for Guard B.

Read-only. Deterministic. No stochastic step, so no seed applies.
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

EXPECTED_ROWS = 20_520
REF_2021_ISSUE = "2021-01-14"          # window 1 issue date of the archived 2021 set
OUT_JSON = REPO_ROOT / "reports" / "guard_a_hres_2022_20260819.json"


def table_sha(df: pd.DataFrame) -> str:
    """SHA-256 of a canonical serialisation: sorted columns, stable row order."""
    d = df.copy()
    sort_cols = [c for c in ("lat", "lon", "lead", "hour") if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    d = d.reindex(sorted(d.columns), axis=1)
    return hashlib.sha256(
        d.to_csv(index=False, float_format="%.10g").encode("utf-8")
    ).hexdigest()


def main() -> None:
    os.environ.setdefault("PHASE2_DATA_ROOT", str(SHIP))
    os.chdir(KIT_PHASE2)
    sys.path.insert(0, str(KIT_PHASE2))
    sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
    sys.path.insert(0, str(KIT_PHASE2 / "part1_forecast"))

    import config
    import splits
    import forecast_hres as fh

    failures: list[str] = []
    out: dict = {"generated_by": "scripts/guard_a_hres_2022_20260819.py"}

    # ── Source inventory ────────────────────────────────────────────────
    paths = [Path(p) for p in config.hres_parquets()]
    infer_paths = [p for p in paths if "context_hres_north_sea" in p.name]
    train_paths = [p for p in paths if "context_hres_north_sea" not in p.name]
    print(f"config.hres_parquets() -> {len(paths)} files "
          f"({len(train_paths)} train, {len(infer_paths)} window context)")
    for p in paths:
        print(f"    {p}")
    out["hres_parquets"] = [str(p) for p in paths]
    if len(infer_paths) != 8:
        failures.append(f"expected 8 window-context parquets, found {len(infer_paths)}")

    installed_year = splits._installed_windows_year()
    windows = splits.eval_windows()
    print(f"\nsplits._installed_windows_year() = {installed_year}")
    print(f"splits.eval_windows() -> {len(windows)} windows, "
          f"{windows[0]['context_start']} .. {windows[-1]['predict_end']}")
    out["installed_windows_year"] = installed_year
    if installed_year != 2022:
        failures.append(f"installed windows year is {installed_year}, expected 2022")

    # ── A1: 20,520 rows at every 2022 issue date ────────────────────────
    print("\n[A1] build_hres_table at each 2022 issue date")
    a1 = {}
    for w in windows:
        issue = pd.Timestamp(w["context_end"])
        te = fh.build_hres_table([issue], with_truth=False)
        drivers = [c for c in ("fcst_u", "fcst_v", "fcst_speed") if c in te.columns]
        finite = {c: float(te[c].notna().mean()) for c in drivers} if len(te) else {}
        ok = len(te) == EXPECTED_ROWS and all(v == 1.0 for v in finite.values())
        status = "PASS" if ok else "FAIL"
        print(f"  window {w['id']}  issue {issue.date()}  rows={len(te):,}  "
              + ", ".join(f"{c}={v:.3f}" for c, v in finite.items()) + f"  {status}")
        a1[str(issue.date())] = {"rows": int(len(te)), "finite": finite, "pass": ok}
        if not ok:
            failures.append(f"A1 window {w['id']} ({issue.date()}): rows={len(te)}, finite={finite}")
        if issue.year != 2022:
            failures.append(f"A1 window {w['id']} issue year is {issue.year}, expected 2022")
    out["a1"] = a1

    # ── A2: 2021 unchanged by the new sources ───────────────────────────
    print(f"\n[A2] build_hres_table at {REF_2021_ISSUE}, with vs without the 2022 sources")
    issue21 = pd.Timestamp(REF_2021_ISSUE)

    te_after = fh.build_hres_table([issue21], with_truth=False)
    sha_after = table_sha(te_after)

    # Reconstruct the pre-9edd92b source set: train parquets only.
    orig = config.hres_parquets
    fh._load_hres.cache_clear() if hasattr(fh._load_hres, "cache_clear") else None
    config.hres_parquets = lambda: train_paths
    try:
        if hasattr(fh._load_hres, "cache_clear"):
            fh._load_hres.cache_clear()
        te_before = fh.build_hres_table([issue21], with_truth=False)
        sha_before = table_sha(te_before)
    finally:
        config.hres_parquets = orig
        if hasattr(fh._load_hres, "cache_clear"):
            fh._load_hres.cache_clear()

    same = sha_before == sha_after and len(te_before) == len(te_after)
    print(f"  train-parquets-only : rows={len(te_before):,}  sha={sha_before[:32]}...")
    print(f"  with 2022 context   : rows={len(te_after):,}  sha={sha_after[:32]}...")
    print(f"  identical: {same}  -> {'PASS' if same else 'FAIL'}")
    out["a2"] = {
        "issue": REF_2021_ISSUE,
        "rows_before": int(len(te_before)), "rows_after": int(len(te_after)),
        "sha_before": sha_before, "sha_after": sha_after, "identical": same,
    }
    if not same:
        failures.append("A2: the 2022 sources changed the 2021 HRES feature table")

    # ── Verdict ─────────────────────────────────────────────────────────
    out["failures"] = failures
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    if failures:
        print(f"GUARD A: FAIL ({len(failures)})")
        for x in failures:
            print(f"  - {x}")
    else:
        print("GUARD A: PASS")
    print("=" * 70)
    print(f"wrote {OUT_JSON}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
