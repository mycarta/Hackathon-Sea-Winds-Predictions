"""Task 2 QA Check 2: run the kit's OWN wind_farm_simulator.simulate_year()
(not task2_scorer_replica) on the Stage 3 boundary-loaded winner layout.

CC prompt (2026-07-13), "Task 2 QA: verify Stage 3 layout (3 checks)",
Check 2. Loads data/task2_layout_winner.json (the 55-turbine boundary-
loaded layout) and real AROME wind at cell 63 (52.50N, 3.00E), all
training years (2016-2020, 14,608 native 3-hourly steps) - the identical
series task2_layout_search.py / task2_layout_validate.py used (real
AROME, 125m->170m shear, alpha=0.11) - via
task2_scorer_replica.load_arome_series (imported, not reimplemented, so
the wind input can't drift from the Stage 1-3 pipeline).

Feeds that wind + layout through the kit's own FarmLayout / WindSeries /
simulate_year / turbines_catalog.load_turbine classes
(phase_2/kit/phase_2/wind_farm_simulator.py, turbines_catalog.py) -
deliberately NOT importing task2_scorer_replica's independent PyWake
construction - to check whether the kit's own code reproduces the
reported result (CF=47.21%, wake=8.27%,
data/task2_layout_winner.json's "reported" block).

Note: the kit's real IEA-22MW CSV curve
(phase_2/data/wind_data/turbines/iea_22mw_power_ct.csv) is absent from
this checkout, so turbines_catalog.load_turbine("IEA_22MW") falls back to
its built-in generic cubic-ramp curve - confirmed bit-identical to
data/iea22mw_power_ct.csv (the curve task2_scorer_replica.py and the
Stage 1-3 pipeline use) by comparing data/task2_welltie_results.csv case 2
against data/task2_welltie_kit_crosscheck.csv
(reports/task2_retest_welltie_20260713.md, Task F.4) - same fallback
formula, materialised to CSV once. Not a confound here.

Deterministic - real AROME data, closed-form layout load from JSON, no
stochastic step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
WINNER_JSON = REPO_ROOT / "data" / "task2_layout_winner.json"
OUT_CSV = REPO_ROOT / "data" / "task2_qa_kit_crosscheck.csv"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(KIT_PHASE2))

CENTRE_LAT, CENTRE_LON = 52.50, 3.00
DIAMETER_M = 284.0
BOX_M = 15000.0
N_TURB = 55
MIN_SPACING_D = 5.0

REPORTED_CF = 0.4721
REPORTED_WAKE = 0.0827
CF_TOL = 0.005     # +/- 0.5 pp
WAKE_TOL = 0.005   # +/- 0.5 pp


def main() -> None:
    import task2_scorer_replica as rep  # real AROME loader (identical series as Stage 1-3)
    from turbines_catalog import load_turbine, _TURBINES_DIR
    from wind_farm_simulator import FarmLayout, WindSeries, simulate_year, validate_layout

    print(f"kit's real IEA-22MW CSV curve present ({_TURBINES_DIR}): {_TURBINES_DIR.exists()}")

    with open(WINNER_JSON) as f:
        winner = json.load(f)
    x = np.array(winner["layout_x_m"], dtype=float)
    y = np.array(winner["layout_y_m"], dtype=float)
    print(f"loaded {x.size} turbines from {WINNER_JSON.name} "
          f"(source_stage={winner['source_stage']})")

    ok, errs = validate_layout(x, y, box_size_m=BOX_M, max_turbines=N_TURB,
                               min_spacing_d=MIN_SPACING_D, diameter_m=DIAMETER_M)
    print(f"kit validate_layout: {ok} {errs}")

    print(f"\nLoading real AROME at cell 63 ({CENTRE_LAT},{CENTRE_LON}), all training years...")
    times, ws_hub, wd = rep.load_arome_series(CENTRE_LAT, CENTRE_LON)

    wind = WindSeries(pd.DataFrame({"time": times, "ws": ws_hub, "wd": wd}))
    turbine = load_turbine("IEA_22MW")
    layout = FarmLayout(x_m=x, y_m=y, turbine=turbine)

    print(f"\nRunning kit's own simulate_year() ({wind.n_steps} steps)...")
    result = simulate_year(layout, wind)

    cf_diff_pp = (result.capacity_factor - REPORTED_CF) * 100
    wake_diff_pp = (result.wake_loss_fraction - REPORTED_WAKE) * 100
    cf_ok = abs(cf_diff_pp) <= CF_TOL * 100
    wake_ok = abs(wake_diff_pp) <= WAKE_TOL * 100

    print(f"\n=== Check 2: kit's own simulate_year() vs reported ===")
    print(f"kit CF:        {result.capacity_factor*100:.4f}%   reported: {REPORTED_CF*100:.2f}%   "
          f"diff: {cf_diff_pp:+.4f} pp   tol +/-{CF_TOL*100:.1f} pp -> {'PASS' if cf_ok else 'FAIL'}")
    print(f"kit wake loss: {result.wake_loss_fraction*100:.4f}%   reported: {REPORTED_WAKE*100:.2f}%   "
          f"diff: {wake_diff_pp:+.4f} pp   tol +/-{WAKE_TOL*100:.1f} pp -> {'PASS' if wake_ok else 'FAIL'}")
    print(f"kit AEP: {result.aep_gwh:.2f} GWh   rated: {result.rated_capacity_mw:.1f} MW   "
          f"n_steps: {wind.n_steps}")

    row = {
        "label": "kit_own_simulate_year_stage3_winner_cell63",
        "n_turbines": layout.n_turbines,
        "n_steps": wind.n_steps,
        "aep_gwh": result.aep_gwh,
        "capacity_factor": result.capacity_factor,
        "wake_loss_fraction": result.wake_loss_fraction,
        "rated_capacity_mw": result.rated_capacity_mw,
        "reported_capacity_factor": REPORTED_CF,
        "reported_wake_loss_fraction": REPORTED_WAKE,
        "cf_diff_pp": cf_diff_pp,
        "wake_loss_diff_pp": wake_diff_pp,
        "cf_within_tolerance_0.5pp": cf_ok,
        "wake_within_tolerance_0.5pp": wake_ok,
    }
    pd.DataFrame([row]).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
