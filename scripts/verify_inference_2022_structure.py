"""S1.1 structure verification gate: 2022 inference set versus the 2021 set.

CC dispatch 2026-08-18, block S1.1. Runs BEFORE any compute. The dispatch rule
is explicit: if anything here fails, stop and report; do not adapt the pipeline
to a changed structure without a decision from Matteo.

Compares, field by field, the newly pinned final-evaluation set at
`phase_2/inference_2022/inference/` (Zenodo 20874645, `inference_2022.zip`,
SHA-256 bb329f04..., pinned by scripts/fetch_pin_inference_2022.py) against the
2021 set the swap runbook was written against, at
`phase_2/phase2_dataset_ship/inference/`.

What is compared:
  1. Window count and directory naming.
  2. Per-window file set.
  3. metadata.json key set, context length in days, predict length, and the
     horizon set implied by score_days.
  4. Parquet schemas: column names and dtypes, for both context files.
  5. Row counts and distinct point counts per context file.
  6. Target grid size, from the static file the submission is written against.
  7. Implied submission shape against the 4,196,640 row contract.

Read-only and deterministic. No stochastic step, so no seed applies.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = REPO_ROOT / "phase_2" / "phase2_dataset_ship" / "inference"
NEW_DIR = REPO_ROOT / "phase_2" / "inference_2022" / "inference"
FOOTPRINT_PARQUET = (REPO_ROOT / "phase_2" / "phase2_dataset_ship" / "static"
                     / "footprint_points.parquet")

CONTEXT_FILES = ("context_hres_north_sea.parquet", "context_reanalysis_north_sea.parquet")
EXPECTED_WINDOWS = 8
EXPECTED_POINTS = 43_715
EXPECTED_ROWS = 4_196_640      # 43,715 x 8 windows x 3 horizons x 4 hours

failures: list[str] = []
notes: list[str] = []


def check(label: str, old, new) -> bool:
    same = old == new
    status = "MATCH" if same else "DIFFER"
    print(f"  {label:52s} {status}")
    if not same:
        print(f"      2021: {old}")
        print(f"      2022: {new}")
        failures.append(f"{label}: 2021={old} 2022={new}")
    return same


def window_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("window_")],
                  key=lambda p: int(p.name.split("_")[1]))


def main() -> None:
    print("=" * 78)
    print("S1.1 STRUCTURE VERIFICATION GATE  -  2022 final-evaluation set vs 2021 set")
    print("=" * 78)

    old_w, new_w = window_dirs(OLD_DIR), window_dirs(NEW_DIR)
    print("\n[1] Window inventory")
    check("window count", len(old_w), len(new_w))
    check("window directory names", [p.name for p in old_w], [p.name for p in new_w])
    if len(new_w) != EXPECTED_WINDOWS:
        failures.append(f"expected {EXPECTED_WINDOWS} windows, found {len(new_w)}")

    print("\n[2] Per-window file set")
    for o, n in zip(old_w, new_w):
        of = sorted(p.name for p in o.iterdir() if p.is_file())
        nf = sorted(p.name for p in n.iterdir() if p.is_file())
        check(f"{n.name} file set", of, nf)

    print("\n[3] metadata.json fields and window geometry")
    meta_new = []
    for o, n in zip(old_w, new_w):
        om = json.loads((o / "metadata.json").read_text())
        nm = json.loads((n / "metadata.json").read_text())
        meta_new.append(nm)
        check(f"{n.name} metadata key set", sorted(om), sorted(nm))
        check(f"{n.name} score_days key set", sorted(om["score_days"]), sorted(nm["score_days"]))
        oc = (pd.Timestamp(om["context_end"]) - pd.Timestamp(om["context_start"])).days
        nc = (pd.Timestamp(nm["context_end"]) - pd.Timestamp(nm["context_start"])).days
        check(f"{n.name} context span (days)", oc, nc)
        op = (pd.Timestamp(om["predict_end"]) - pd.Timestamp(om["predict_start"])).days
        np_ = (pd.Timestamp(nm["predict_end"]) - pd.Timestamp(nm["predict_start"])).days
        check(f"{n.name} predict span (days)", op, np_)
        # horizon set implied by score_days offsets from predict_start
        oh = sorted((pd.Timestamp(v) - pd.Timestamp(om["predict_start"])).days + 1
                    for v in om["score_days"].values())
        nh = sorted((pd.Timestamp(v) - pd.Timestamp(nm["predict_start"])).days + 1
                    for v in nm["score_days"].values())
        check(f"{n.name} implied horizon set", oh, nh)
        # gap between context_end and predict_start
        og = (pd.Timestamp(om["predict_start"]) - pd.Timestamp(om["context_end"])).days
        ng = (pd.Timestamp(nm["predict_start"]) - pd.Timestamp(nm["context_end"])).days
        check(f"{n.name} context_end -> predict_start gap (days)", og, ng)

    print("\n[4] Years covered by the new set")
    yrs = sorted({pd.Timestamp(m["predict_start"]).year for m in meta_new})
    print(f"  predict_start years: {yrs}")
    if yrs != [2022]:
        notes.append(f"new set predict years = {yrs}, expected [2022]")
    print(f"  window date ranges:")
    for m in meta_new:
        print(f"    window {m['id']}: context {m['context_start']}..{m['context_end']}  "
              f"predict {m['predict_start']}..{m['predict_end']}  "
              f"score d1={m['score_days']['d1']} d7={m['score_days']['d7']} d14={m['score_days']['d14']}")

    print("\n[5] Parquet schema, row counts, point counts")
    for fname in CONTEXT_FILES:
        o = pd.read_parquet(old_w[0] / fname)
        n = pd.read_parquet(new_w[0] / fname)
        check(f"{fname} columns", list(o.columns), list(n.columns))
        check(f"{fname} dtypes", [str(t) for t in o.dtypes], [str(t) for t in n.dtypes])
        latlon = [c for c in ("latitude", "longitude") if c in o.columns]
        if latlon:
            check(f"{fname} distinct points",
                  int(o[latlon].drop_duplicates().shape[0]),
                  int(n[latlon].drop_duplicates().shape[0]))
        print(f"    {fname}: 2021 rows={len(o):,}  2022 rows={len(n):,}")
        if len(o) != len(n):
            notes.append(f"{fname} row count differs (2021 {len(o):,} vs 2022 {len(n):,})")

    print("\n[6] Row counts across ALL windows (new set)")
    for fname in CONTEXT_FILES:
        counts = [len(pd.read_parquet(w / fname)) for w in new_w]
        print(f"    {fname}: {counts}")

    print("\n[7] Target grid and implied submission shape")
    # The target grid is defined by footprint_points.parquet, NOT by the raw
    # arome_static seamask (which counts every AROME sea pixel, 75,653, a
    # different quantity). footprint_points.parquet ships inside phase2_dataset.zip,
    # whose MD5 is byte-identical between the pinned v3 and the v4 record, so the
    # target grid cannot have changed with the new inference set.
    fp = pd.read_parquet(FOOTPRINT_PARQUET)
    n_pts = int(fp[["lat", "lon"]].drop_duplicates().shape[0])
    print(f"    footprint_points.parquet distinct points: {n_pts:,}")
    check("target grid point count", EXPECTED_POINTS, n_pts)
    implied = EXPECTED_POINTS * len(new_w) * 3 * 4
    print(f"    implied submission rows = {EXPECTED_POINTS:,} x {len(new_w)} x 3 x 4 = {implied:,}")
    check("implied submission rows", EXPECTED_ROWS, implied)

    print("\n" + "=" * 78)
    if failures:
        print(f"GATE RESULT: FAIL  ({len(failures)} difference(s))")
        for f in failures:
            print(f"  - {f}")
    else:
        print("GATE RESULT: PASS  (no structural difference found)")
    if notes:
        print(f"\nNotes ({len(notes)}), not gate failures:")
        for n in notes:
            print(f"  - {n}")
    print("=" * 78)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    os.environ.setdefault("PHASE2_DATA_ROOT", str(REPO_ROOT / "phase_2" / "phase2_dataset_ship"))
    main()
