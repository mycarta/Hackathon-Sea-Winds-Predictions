"""Cross-check: run the KIT'S OWN `wind_farm_simulator.simulate_year` code
directly, on the same wind data as `task2_scorer_replica.py` Case 2, to
determine whether a mismatch against the SUBMISSION.md CF~54%/wake~6%
target is a bug in our independent replica or a property of the kit
itself.

Task2 CC prompt (2026-07-13), Task F diagnosis. Unlike
`task2_scorer_replica.py` (which deliberately does NOT import
`wind_farm_simulator.py`, to make the well-tie a genuine independent
check), this script deliberately DOES import and call the kit's own
`simulate_year` - the entire point here is to get the kit's actual
ground-truth output on a specific input, not to re-verify our physics.

Same wind data as replica Case 2: kit's own Tier-0 bootstrap synthetic
wind (`synthetic_generator.bootstrap_year(hist, block_days=14, seed=0)`),
same Dogger-Bank-like centre (54.5N, 2.0E) that `2_simulator_intro.ipynb`
cell 7 uses, same plain 55-turbine 7D-spacing grid.

Deterministic (seed=0, matches the kit's own notebook exactly).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
DATA_ROOT = REPO_ROOT / "phase_2" / "phase2_dataset_ship"
OUT_CSV = REPO_ROOT / "data" / "task2_welltie_kit_crosscheck.csv"

sys.path.insert(0, str(KIT_PHASE2))
sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
sys.path.insert(0, str(KIT_PHASE2 / "part2_siting"))
os.environ.setdefault("PHASE2_DATA_ROOT", str(DATA_ROOT))

DOGGER_LAT, DOGGER_LON = 54.5, 2.0
HUB_M = 170.0
N_TURB = 55
BOX_M = 15000.0
TARGET_CF = 0.538
TARGET_WAKE_LOSS = 0.059


def main() -> None:
    cwd = os.getcwd()
    os.chdir(KIT_PHASE2)
    try:
        import synthetic_generator as sg
        import synth_wind
        from turbines_catalog import get_spec, load_turbine, _TURBINES_DIR
        from wind_farm_simulator import FarmLayout, grid_layout, simulate_year, validate_layout

        hist = sg.load_coarse_history()
        synth = sg.bootstrap_year(hist, block_days=14, seed=0)
        wind_cache = synth_wind.SynthWindCache(synth, hub_height_m=HUB_M)
        ws = wind_cache.get(DOGGER_LAT, DOGGER_LON)
        print(f"wind series: {ws.n_steps} steps, mean ws={ws.df['ws'].mean():.2f} m/s, "
              f"duration_hours={ws.duration_hours:.1f}")
        print(f"turbines_catalog._TURBINES_DIR exists (real IEA22MW curve present?): "
              f"{_TURBINES_DIR.exists()}")

        spec = get_spec("IEA_22MW")
        x, y = grid_layout(n_turbines=N_TURB, spacing_d=7, diameter_m=spec.diameter_m, rotation_deg=0)
        ok, errs = validate_layout(x, y, box_size_m=BOX_M, max_turbines=N_TURB,
                                   min_spacing_d=5.0, diameter_m=spec.diameter_m)
        print("layout valid:", ok, errs)

        turbine = load_turbine("IEA_22MW")
        layout = FarmLayout(x_m=x, y_m=y, turbine=turbine)
        year_res = simulate_year(layout, ws)
    finally:
        os.chdir(cwd)

    row = {
        "label": "kit_own_code_synthetic_dogger_54.5_2.0",
        "rated_capacity_mw": year_res.rated_capacity_mw,
        "aep_gwh": year_res.aep_gwh,
        "capacity_factor": year_res.capacity_factor,
        "wake_loss_fraction": year_res.wake_loss_fraction,
        "target_cf": TARGET_CF,
        "target_wake_loss": TARGET_WAKE_LOSS,
        "cf_diff_pp": (year_res.capacity_factor - TARGET_CF) * 100,
        "wake_loss_diff_pp": (year_res.wake_loss_fraction - TARGET_WAKE_LOSS) * 100,
    }
    print(f"\nKIT'S OWN CODE (wind_farm_simulator.simulate_year), called directly:")
    print(f"  rated capacity: {row['rated_capacity_mw']:.0f} MW")
    print(f"  AEP:            {row['aep_gwh']:.0f} GWh/year")
    print(f"  capacity factor:{row['capacity_factor']*100:.2f}%  (target {TARGET_CF*100:.1f}%, "
          f"diff {row['cf_diff_pp']:+.1f} pp)")
    print(f"  wake loss:      {row['wake_loss_fraction']*100:.2f}%  (target {TARGET_WAKE_LOSS*100:.1f}%, "
          f"diff {row['wake_loss_diff_pp']:+.1f} pp)")

    pd.DataFrame([row]).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
