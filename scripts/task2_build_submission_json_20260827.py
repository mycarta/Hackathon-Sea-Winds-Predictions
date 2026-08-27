"""Build the clean Task 2 submission.json from the frozen layout winner.

CC dispatch 2026-08-27, item 6. `data/task2_layout_winner.json` is opened
READ-ONLY and is never rewritten; this script only projects it onto the
organizers' six-key schema and validates the result.

**Schema, verified against Phase_2.pdf rather than assumed.** The dispatch
supplies six keys from `task2_siting_constraints_and_decisions.md` and asks CC
to check them against its own copy of the PDF before building. `phase_2/Phase
2.pdf` page 6, section "What you submit (submission.json)", prints:

    {
      "team": "...",
      "farm_centre_lat": 53.5, "farm_centre_lon": 1.5,
      "turbine_key": "IEA_22MW",
      "layout_x_m": [ ...55 values... ], "layout_y_m": [ ...55 values... ]
    }

which is the same six keys, in the same order, with no seventh. The diff
against the project record is EMPTY. The page also states "Positions are in
metres relative to the farm centre", and that the optional
predicted_q05/q50/q95 power-forecast fields "enable the bidding bonus" -- those
are omitted here per the parked decision, which forfeits the bidding bonus by
choice, not by oversight.

**Byte-identity of the payload.** The dispatch requires the coordinates and the
turbine list to be byte-identical to the winner file. This script does not
re-derive, re-round or re-order anything: it copies the parsed values through
unchanged, then proves identity two ways -- exact float equality element by
element, and a verbatim substring match of the serialised `layout_x_m`,
`layout_y_m` and `turbine_key` blocks against the winner file's own text. Both
must hold or the script refuses to write.

**Validation.** The kit's own `wind_farm_simulator.validate_layout` is run on
the written file, with the Phase_2.pdf p.5 constraints: 15 x 15 km box, exactly
55 turbines, minimum spacing 5 D at D = 284 m (>= 1420 m).

The 2026-08-25 blocker was that `validate_layout` could not run because py_wake
is absent from the `swnd` conda environment. py_wake 2.6.17 IS present in the
`pywake` environment on this machine, so the resolution is to run this script
there:

    conda run -n pywake python scripts/task2_build_submission_json_20260827.py

Deterministic: a pure projection of a SHA-pinned JSON plus closed-form geometry
checks. No stochastic step, so no seed applies.

Reads : data/task2_layout_winner.json   (read-only, SHA asserted)
        phase_2/kit/phase_2/wind_farm_simulator.py
Writes: data/submission/submission.json
        reports/task2_submission_build_20260827.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
DATA_ROOT = REPO_ROOT / "phase_2" / "inference_2022" / "phase2_dataset_ship"

sys.path.insert(0, str(KIT_PHASE2))
sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
if DATA_ROOT.is_dir():
    os.environ.setdefault("PHASE2_DATA_ROOT", str(DATA_ROOT))

WINNER_JSON = REPO_ROOT / "data" / "task2_layout_winner.json"
OUT_JSON = REPO_ROOT / "data" / "submission" / "submission.json"
OUT_REPORT = REPO_ROOT / "reports" / "task2_submission_build_20260827.json"

# Winner-file identity, from the c9f61803 lineage named in the dispatch.
WINNER_SHA256 = "c9f618034ebeeab9cbbf163d8c4006e9d1eb1f9e7f9bc6fd4e43752ab60f6f1d"

# Entrant name, confirmed by Matteo in the 2026-08-27 dispatch.
TEAM = "Matteo Niccoli"

# The six keys, in the order Phase_2.pdf p.6 prints them.
SCHEMA_KEYS = ["team", "farm_centre_lat", "farm_centre_lon", "turbine_key",
               "layout_x_m", "layout_y_m"]

# Phase_2.pdf p.5 constraints.
BOX_SIZE_M = 15_000.0
MAX_TURBINES = 55
MIN_SPACING_D = 5.0
DIAMETER_M = 284.0
EXPECTED_TURBINE_KEY = "IEA_22MW"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not WINNER_JSON.exists():
        raise SystemExit(f"missing winner file: {WINNER_JSON}")
    winner_sha = sha256_of(WINNER_JSON)
    if winner_sha != WINNER_SHA256:
        raise SystemExit(f"winner-file SHA mismatch:\n  expected {WINNER_SHA256}"
                         f"\n  actual   {winner_sha}")
    print(f"winner file verified: sha256={winner_sha[:12]}...")

    winner_text = WINNER_JSON.read_text(encoding="utf-8")
    winner = json.loads(winner_text)

    # ── Project onto the six-key schema ─────────────────────────────────
    sub = OrderedDict()
    sub["team"] = TEAM
    sub["farm_centre_lat"] = winner["farm_centre_lat"]
    sub["farm_centre_lon"] = winner["farm_centre_lon"]
    sub["turbine_key"] = winner["turbine_key"]
    sub["layout_x_m"] = winner["layout_x_m"]
    sub["layout_y_m"] = winner["layout_y_m"]

    assert list(sub.keys()) == SCHEMA_KEYS, list(sub.keys())
    if sub["turbine_key"] != EXPECTED_TURBINE_KEY:
        raise SystemExit(f"turbine_key is {sub['turbine_key']!r}, "
                         f"expected {EXPECTED_TURBINE_KEY!r}")
    if len(sub["layout_x_m"]) != MAX_TURBINES or len(sub["layout_y_m"]) != MAX_TURBINES:
        raise SystemExit(f"expected {MAX_TURBINES} turbines, got "
                         f"{len(sub['layout_x_m'])}/{len(sub['layout_y_m'])}")

    # ── Byte-identity proof 1: exact float equality ─────────────────────
    for key in ("layout_x_m", "layout_y_m"):
        for i, (a, b) in enumerate(zip(sub[key], winner[key])):
            if a != b or repr(a) != repr(b):
                raise SystemExit(f"{key}[{i}] diverges: {a!r} vs {b!r}")
    print(f"payload float-equality check: PASS "
          f"({2 * MAX_TURBINES} coordinates, exact)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(sub, indent=2) + "\n"
    OUT_JSON.write_text(text, encoding="utf-8")

    # ── Byte-identity proof 2: serialised blocks appear verbatim ────────
    for key in ("turbine_key", "layout_x_m", "layout_y_m"):
        block = json.dumps({key: sub[key]}, indent=2)
        inner = block[block.index("\n") + 1:block.rindex("\n")]
        if inner not in winner_text:
            raise SystemExit(f"serialised {key} block is not verbatim in the "
                             f"winner file")
    print("payload verbatim-substring check: PASS "
          "(turbine_key, layout_x_m, layout_y_m)")

    out_sha = sha256_of(OUT_JSON)

    # ── Diff against the winner file ────────────────────────────────────
    dropped = [k for k in winner if k not in sub]
    added = [k for k in sub if k not in winner]
    changed = {k: {"winner": winner[k], "submission": sub[k]}
               for k in sub if k in winner and sub[k] != winner[k]}
    print(f"\ndiff vs winner file:")
    print(f"  dropped keys ({len(dropped)}): {dropped}")
    print(f"  added keys   ({len(added)}): {added}")
    print(f"  changed values: {sorted(changed)}")
    for k, v in changed.items():
        print(f"    {k}: {v['winner']!r} -> {v['submission']!r}")

    # ── Kit validate_layout ─────────────────────────────────────────────
    print("\nrunning kit wind_farm_simulator.validate_layout ...")
    from wind_farm_simulator import validate_layout
    on_disk = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    ok, errors = validate_layout(
        on_disk["layout_x_m"], on_disk["layout_y_m"],
        box_size_m=BOX_SIZE_M, max_turbines=MAX_TURBINES,
        min_spacing_d=MIN_SPACING_D, diameter_m=DIAMETER_M,
    )
    print(f"  validate_layout(box_size_m={BOX_SIZE_M}, "
          f"max_turbines={MAX_TURBINES}, min_spacing_d={MIN_SPACING_D}, "
          f"diameter_m={DIAMETER_M})")
    print(f"  -> ({ok}, {errors})")

    # Independent restatement of the two numeric constraints, so a silent
    # tolerance inside validate_layout cannot pass unnoticed.
    import numpy as np
    x = np.asarray(on_disk["layout_x_m"], dtype=float)
    y = np.asarray(on_disk["layout_y_m"], dtype=float)
    dmat = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    iu = np.triu_indices(x.size, 1)
    min_pair = float(dmat[iu].min())
    max_abs = float(max(abs(x).max(), abs(y).max()))
    print(f"  independent check: min pairwise spacing "
          f"{min_pair:.6f} m (required >= {MIN_SPACING_D * DIAMETER_M:.1f} m, "
          f"margin {min_pair - MIN_SPACING_D * DIAMETER_M:+.6f} m)")
    print(f"  independent check: max |coordinate| {max_abs:.6f} m "
          f"(box half-width {BOX_SIZE_M / 2:.1f} m)")

    report = {
        "generated_by": "scripts/task2_build_submission_json_20260827.py",
        "dispatch": "CC dispatch 2026-08-27, item 6",
        "team": TEAM,
        "schema_keys": SCHEMA_KEYS,
        "schema_source": "phase_2/Phase 2.pdf p.6, 'What you submit (submission.json)'",
        "schema_diff_vs_project_record": "empty -- same six keys, same order",
        "winner_file": str(WINNER_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "winner_sha256": winner_sha,
        "output_file": str(OUT_JSON.relative_to(REPO_ROOT)).replace("\\", "/"),
        "output_sha256": out_sha,
        "output_bytes": OUT_JSON.stat().st_size,
        "diff_vs_winner": {
            "dropped_keys": dropped,
            "added_keys": added,
            "changed_values": {k: {"winner": v["winner"],
                                   "submission": v["submission"]}
                               for k, v in changed.items()},
        },
        "payload_identity": {
            "float_equality": True,
            "verbatim_substring": True,
            "n_coordinates": 2 * MAX_TURBINES,
        },
        "validate_layout": {
            "constraints": {
                "box_size_m": BOX_SIZE_M, "max_turbines": MAX_TURBINES,
                "min_spacing_d": MIN_SPACING_D, "diameter_m": DIAMETER_M,
                "min_spacing_m": MIN_SPACING_D * DIAMETER_M,
            },
            "is_valid": bool(ok),
            "errors": list(errors),
            "verbatim": f"({ok}, {errors})",
        },
        "independent_constraint_check": {
            "min_pairwise_spacing_m": min_pair,
            "spacing_margin_m": min_pair - MIN_SPACING_D * DIAMETER_M,
            "max_abs_coordinate_m": max_abs,
            "box_half_width_m": BOX_SIZE_M / 2,
        },
        "bonus_fields_omitted": ["predicted_q05", "predicted_q50", "predicted_q95"],
        "bonus_fields_note": ("optional per Phase_2.pdf p.6; omitted per the "
                              "parked decision, which forfeits the bidding bonus"),
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 68)
    print(f"submission written: {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"  sha256 : {out_sha}")
    print(f"  bytes  : {OUT_JSON.stat().st_size}")
    print(f"  keys   : {list(on_disk.keys())}")
    print(f"validate_layout: {'PASS' if ok else 'FAIL'}")
    print("=" * 68)
    if not ok:
        raise SystemExit("validate_layout FAILED -- escalate, do not fix here")


if __name__ == "__main__":
    main()
