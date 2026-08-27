"""Task 2 QA Check 1: geometric validity of the Stage 3 winner layout.

CC prompt (2026-07-13), "Task 2 QA: verify Stage 3 layout (3 checks)",
Check 1. Verification only - reads data/task2_layout_winner.json and
checks turbine count, pairwise spacing (>=5D=1420m), box bounds
(+/-7500m), and duplicate positions. Writes no new layout data.

Deterministic - pure geometry, no stochastic step.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WINNER_JSON = REPO_ROOT / "data" / "task2_layout_winner.json"

DIAMETER_M = 284.0
MIN_SPACING_D = 5.0
MIN_SPACING_M = MIN_SPACING_D * DIAMETER_M  # 1420 m
HALF_BOX_M = 7500.0
DUP_TOL_M = 1e-6


def main() -> None:
    with open(WINNER_JSON) as f:
        winner = json.load(f)
    x = np.array(winner["layout_x_m"], dtype=float)
    y = np.array(winner["layout_y_m"], dtype=float)

    print(f"=== Check 1: geometric validity ({WINNER_JSON.name}) ===\n")

    # 1. count
    n = x.size
    assert x.size == y.size, "x_m/y_m length mismatch"
    count_pass = (n == 55)
    print(f"1. Turbine count: {n} (expect 55) -> {'PASS' if count_pass else 'FAIL'}")

    # 2. pairwise spacing
    pts = np.column_stack([x, y])
    dists = []
    for i, j in combinations(range(n), 2):
        d = float(np.hypot(x[i] - x[j], y[i] - y[j]))
        dists.append((d, i, j))
    dists.sort(key=lambda t: t[0])
    min_d, i_min, j_min = dists[0]
    spacing_pass = min_d >= MIN_SPACING_M * (1 - 1e-9)
    print(f"2. Min pairwise distance: {min_d:.4f} m (require >= {MIN_SPACING_M:.0f} m = "
          f"{MIN_SPACING_D}D) at turbines {i_min}/{j_min} -> "
          f"{'PASS' if spacing_pass else 'FAIL'}")
    violations = [(d, i, j) for d, i, j in dists if d < MIN_SPACING_M * (1 - 1e-9)]
    print(f"   violating pairs: {len(violations)}")
    for d, i, j in violations:
        print(f"     turbines {i:2d}/{j:2d}: ({x[i]:.2f},{y[i]:.2f}) - ({x[j]:.2f},{y[j]:.2f})  "
              f"d={d:.4f} m  shortfall={MIN_SPACING_M - d:.4f} m")

    # 3. box bounds
    max_abs_x = float(np.max(np.abs(x)))
    max_abs_y = float(np.max(np.abs(y)))
    box_pass = (max_abs_x <= HALF_BOX_M * (1 + 1e-9)) and (max_abs_y <= HALF_BOX_M * (1 + 1e-9))
    print(f"3. Max |x|: {max_abs_x:.2f} m, max |y|: {max_abs_y:.2f} m "
          f"(require <= {HALF_BOX_M:.0f} m) -> {'PASS' if box_pass else 'FAIL'}")

    # 4. duplicates
    n_dupe_pairs = sum(1 for d, _, _ in dists if d < DUP_TOL_M)
    dupe_pass = (n_dupe_pairs == 0)
    print(f"4. Duplicate positions (distance < {DUP_TOL_M} m): {n_dupe_pairs} -> "
          f"{'PASS' if dupe_pass else 'FAIL'}")

    overall = count_pass and spacing_pass and box_pass and dupe_pass
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
