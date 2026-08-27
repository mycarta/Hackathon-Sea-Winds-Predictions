"""Task 2 QA Check 4: spread-grid full-series validation at cell 63.

CC prompt (2026-07-13), "Task 2 QA continued: spread-grid + spacing fix",
Check 4. The Stage 1-3 grid candidates in
reports/task2_layout_optimization_20260713.md were all compact (5-5.4D
spacing) - the max-spread uniform grid (~7-8D, the true competitor to
Stage 3's boundary-loaded winner per the prompt's Sickler et al. 2023
citation, 8.88D regular layout) was screened by the fast 16-sector rose
(known to misjudge grids, see the predecessor report's "fast-screen
reversal" section) but never run through the full 14,608-step real AROME
replica. This script closes that gap.

Reuses `task2_scorer_replica` (wind load, turbine, `run_case` - the SAME
full-series code path Stage 1-3 used in `task2_layout_validate.py`, not a
new construction) and `task2_layout_search`'s geometry primitives
(`make_rect_grid`, `rotate`, `check_valid`, `MIN_SPACING_M` - pure
geometry, not physics, imported not reimplemented).

Three configs:
    1. uniform_7D_baseline - the kit's own stock 8x7 grid at 7D=1988m
       spacing, already in `task2_layout_search.py`'s `aspect_configs()`
       Stage 1 candidate pool - geometry reproduced bit-for-bit here via
       the same `make_rect_grid`/`rotate` calls (not re-derived). At
       orientation=10 deg (the Stage 1 winning orientation for the
       WINNING config, a different, more elongated shape) this
       particular 8x7/near-square grid is geometrically INVALID
       (out_of_box - confirmed against the regenerated
       `data/task2_layout_candidates.csv`: valid ONLY at 0/90 deg out of
       the full 0-170 deg Stage 1 sweep, since an asymmetric 8x7
       rectangle only clears the axis-aligned square box unrotated or at
       a right angle). Run instead at its own valid orientation, 0 deg -
       also the kit's own `grid_layout`/`turbines_catalog` default
       (`rotation_deg=0.0`) - so this is still a faithful "kit stock
       config" run, just not force-rotated to a 10 deg that doesn't fit
       it.
    2. 8x7_max_spacing - n_A=8,n_B=7, isotropic spacing bisected to the
       largest value that still fits 55 turbines in the 15x15km box at
       10 deg (per prompt instruction - this config's box-fit was
       verified valid at 10 deg by construction, since the bisection is
       run AT that orientation).
    3. 7x8_max_spacing - n_A=7,n_B=8 (swapped aspect), same bisection,
       also at 10 deg.

Deterministic - real AROME data, closed-form geometry + bisection
search, no stochastic step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import task2_scorer_replica as rep
import task2_layout_search as search

CENTRE_LAT, CENTRE_LON = 52.50, 3.00
ORIENTATION_DEG = 10.0        # Stage 1 winning orientation, per prompt instruction
BASELINE_ORIENTATION_DEG = 0.0  # uniform_7D_baseline is out_of_box at 10 deg; 0 deg is its
                                 # own valid orientation (also the kit's own default)
DIAMETER_M = rep.DIAMETER_M
OUT_CSV = REPO_ROOT / "data" / "task2_qa_spread_grid.csv"

STAGE3_CF = 0.4721
STAGE3_WAKE = 0.0827


def largest_valid_uniform_spacing(n_A: int, n_B: int, orientation_deg: float,
                                  lo: float = search.MIN_SPACING_M, hi_guess: float = 3000.0,
                                  iters: int = 50) -> float:
    """Largest ISOTROPIC spacing (same value both axes) that keeps an
    n_A x n_B grid, rotated by orientation_deg, inside the box and above
    min spacing. Bisection, same style as
    task2_layout_search._largest_valid_spacing_B."""
    def valid_at(s: float) -> bool:
        x, y = search.make_rect_grid(n_A, n_B, s, s)
        x, y = search.rotate(x, y, orientation_deg)
        return search.check_valid(x, y)[0]

    if not valid_at(lo):
        raise RuntimeError(f"n_A={n_A},n_B={n_B}: lo spacing {lo}m is already invalid")
    hi = hi_guess
    if valid_at(hi):
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if valid_at(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    print(f"Loading real AROME at cell 63 ({CENTRE_LAT},{CENTRE_LON}), all training years...")
    times, ws_hub, wd = rep.load_arome_series(CENTRE_LAT, CENTRE_LON)
    turbine = rep.build_turbine()

    configs = []

    x0, y0 = search.make_rect_grid(8, 7, 7 * DIAMETER_M, 7 * DIAMETER_M)
    x0, y0 = search.rotate(x0, y0, ORIENTATION_DEG)
    valid0, reason0 = search.check_valid(x0, y0)
    print(f"uniform_7D_baseline @ {ORIENTATION_DEG:.0f} deg: valid={valid0} "
          f"{'' if valid0 else reason0} - " +
          ("using this orientation" if valid0 else
           f"falling back to {BASELINE_ORIENTATION_DEG:.0f} deg (its own valid orientation)"))
    if not valid0:
        x0, y0 = search.make_rect_grid(8, 7, 7 * DIAMETER_M, 7 * DIAMETER_M)
        x0, y0 = search.rotate(x0, y0, BASELINE_ORIENTATION_DEG)
        baseline_orient = BASELINE_ORIENTATION_DEG
    else:
        baseline_orient = ORIENTATION_DEG
    configs.append({"label": "uniform_7D_baseline", "n_A": 8, "n_B": 7,
                    "spacing_m": 7 * DIAMETER_M, "orientation_deg": baseline_orient,
                    "x": x0, "y": y0})

    for n_A, n_B, label in [(8, 7, "8x7_max_spacing"), (7, 8, "7x8_max_spacing")]:
        s = largest_valid_uniform_spacing(n_A, n_B, ORIENTATION_DEG)
        x, y = search.make_rect_grid(n_A, n_B, s, s)
        x, y = search.rotate(x, y, ORIENTATION_DEG)
        configs.append({"label": label, "n_A": n_A, "n_B": n_B, "spacing_m": s,
                        "orientation_deg": ORIENTATION_DEG, "x": x, "y": y})

    rows = []
    for cfg in configs:
        x, y = cfg["x"], cfg["y"]
        valid, reason = search.check_valid(x, y)
        print(f"\n=== {cfg['label']} (n_A={cfg['n_A']},n_B={cfg['n_B']}, "
              f"spacing={cfg['spacing_m']:.1f} m = {cfg['spacing_m']/DIAMETER_M:.2f}D, "
              f"orientation={cfg['orientation_deg']:.0f} deg) ===")
        print(f"  geometry valid: {valid} {'' if valid else reason}")
        if not valid:
            rows.append({"label": cfg["label"], "n_A": cfg["n_A"], "n_B": cfg["n_B"],
                        "spacing_m": cfg["spacing_m"], "spacing_D": cfg["spacing_m"] / DIAMETER_M,
                        "orientation_deg": cfg["orientation_deg"], "valid": False, "invalid_reason": reason})
            continue
        res = rep.run_case(cfg["label"], times, ws_hub, wd, turbine, x, y)
        rows.append({"label": cfg["label"], "n_A": cfg["n_A"], "n_B": cfg["n_B"],
                    "spacing_m": cfg["spacing_m"], "spacing_D": cfg["spacing_m"] / DIAMETER_M,
                    "orientation_deg": cfg["orientation_deg"], "valid": True, "invalid_reason": None, **res})

    df = pd.DataFrame(rows)
    df["stage3_cf"] = STAGE3_CF
    df["stage3_wake"] = STAGE3_WAKE
    df["cf_diff_vs_stage3_pp"] = (df["capacity_factor"] - STAGE3_CF) * 100
    df["wake_diff_vs_stage3_pp"] = (df["wake_loss_fraction"] - STAGE3_WAKE) * 100
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    print("\n=== Check 4 summary vs Stage 3 winner (CF=47.21%, wake=8.27%) ===")
    for _, r in df.iterrows():
        if not r["valid"]:
            print(f"  {r['label']:20s}: INVALID ({r['invalid_reason']})")
            continue
        print(f"  {r['label']:20s}: CF={r['capacity_factor']*100:6.2f}%  "
              f"wake={r['wake_loss_fraction']*100:6.2f}%  AEP={r['aep_gwh']:8.2f} GWh  "
              f"spacing={r['spacing_D']:.2f}D  vs Stage3 CF diff={r['cf_diff_vs_stage3_pp']:+.2f}pp")


if __name__ == "__main__":
    main()
