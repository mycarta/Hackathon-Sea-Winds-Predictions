"""S3 like-for-like scorer run: three cases, one wind input, one code path.

CC dispatch 2026-08-18, block S3, authorized 2026-08-19. Read-only. No
submissions, no refits, no layout changes, no parameter changes.

**Why.** S2 established that cell 63 has BETTER wind resource than the organizer
baseline centre (+4.90% pooled mean v^3), and that about 90% of the 5.99 pp CF
gap sits in gross capacity factor rather than wake. That was inference from
stored metrics. S3 is the direct test: run the same scorer, on the same wind
input, through the same code path, at both centres, and separate the site effect
from the layout effect by measurement instead of by argument.

**The three cases** (dispatch wording, verbatim):
  1. plain 7D grid, 55 x IEA_22MW, centre 53.5N 1.5E
  2. plain 7D grid, 55 x IEA_22MW, centre cell 63 (52.50N, 3.00E)
  3. our submitted Stage 3 layout at cell 63

Cases 1 and 2 differ ONLY in wind input (the layout geometry is byte-identical,
produced by the same `grid_layout()` call), so their difference is the pure site
effect. Cases 2 and 3 share a wind input and differ ONLY in layout, so their
difference is the pure layout effect.

**Code path.** `scripts/task2_scorer_replica.py`, the independent PyWake
construction, as the dispatch specifies. The kit's OWN
`wind_farm_simulator.simulate_year()` is ALSO run on all three cases as a
same-path confirmation, because the two agree to full float precision on
identical input: `data/task2_welltie_results.csv` row 2 and
`data/task2_welltie_kit_crosscheck.csv` both report capacity_factor
0.45924871878723555 for the same case. Running both costs seconds and removes
any question about which scorer produced the table.

**Power curve: NOT substituted, per the dispatch.** Both paths use the kit's
synthetic generic cubic-ramp fallback, because no real IEA 22 MW curve exists in
this checkout. The point of S3 is to see what OUR chain returns for THEIR
configuration, so the curve is left exactly as the pipeline has it.

**Deliberately NOT done:** no comparison against `2_simulator_intro.ipynb`. The
organizers confirmed 2026-08-10 that it runs synthetic wind at Dogger Bank
(54.5N, 2.0E) and reproduces no baseline figure.

Deterministic: real AROME series, closed-form layout geometry, no stochastic
step, so no seed applies.

Outputs: reports/three_case_scorer_20260818.json and
data/task2_three_case_scorer_20260818.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
DATA_ROOT = REPO_ROOT / "phase_2" / "phase2_dataset_ship"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(KIT_PHASE2))
sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
os.environ.setdefault("PHASE2_DATA_ROOT", str(DATA_ROOT))

BASELINE_CENTRE = (53.5, 1.5)
CELL63_CENTRE = (52.50, 3.00)
WINNER_JSON = REPO_ROOT / "data" / "task2_layout_winner.json"
OUT_JSON = REPO_ROOT / "reports" / "three_case_scorer_20260818.json"
OUT_CSV = REPO_ROOT / "data" / "task2_three_case_scorer_20260818.csv"

# Farm constants, Phase_2.pdf p.5, mirrored in build_recon_20260718.py:46-47
CAPACITY_MW = 1210.0
N_TURBINES = 55


def site_cost_inputs(centre_lat: float, centre_lon: float,
                     x_m: np.ndarray, y_m: np.ndarray) -> tuple[float, float]:
    """Mean water depth and mean shore distance over the turbine positions.

    Identical method to task2_lcoe/build_recon_20260718.py:122-139, which is how
    the banked LCOE 95.168 was produced: sample the EMODnet raster at every
    turbine position, then take the arithmetic mean.
    """
    import task2_depth_sampler as ds
    from task2_projection import local_xy_to_latlon
    lat, lon = local_xy_to_latlon(centre_lat, centre_lon, x_m, y_m)
    depths = ds.depth_at_array(lat, lon)
    shore = np.array([ds.dist_to_coast_at(float(a), float(o)) for a, o in zip(lat, lon)])
    if not np.isfinite(depths).all():
        raise RuntimeError("non-finite depth at a turbine position")
    if not np.isfinite(shore).all():
        raise RuntimeError("non-finite shore distance at a turbine position")
    return float(np.mean(depths)), float(np.mean(shore))


def main() -> None:
    t_start = time.time()
    import task2_scorer_replica as rep
    from cost_model import CostParameters, evaluate_farm
    from turbines_catalog import load_turbine
    from wind_farm_simulator import FarmLayout, WindSeries, simulate_year

    params = CostParameters()
    turbine_rep = rep.build_turbine()
    turbine_kit = load_turbine("IEA_22MW")

    # ── Layouts ─────────────────────────────────────────────────────────
    gx, gy = rep.grid_layout()          # plain 7D grid, 55 turbines, 0 deg
    with open(WINNER_JSON) as f:
        winner = json.load(f)
    wx = np.array(winner["layout_x_m"], dtype=float)
    wy = np.array(winner["layout_y_m"], dtype=float)
    assert gx.size == N_TURBINES and wx.size == N_TURBINES

    print(f"plain 7D grid: {gx.size} turbines, spacing {rep.SPACING_D}D x "
          f"{rep.DIAMETER_M} m = {rep.SPACING_D * rep.DIAMETER_M:.0f} m, "
          f"extent {gx.max()-gx.min():.0f} x {gy.max()-gy.min():.0f} m")
    print(f"submitted layout: {wx.size} turbines, {winner['source_stage']}, "
          f"orientation {winner['orientation_deg']} deg, "
          f"extent {wx.max()-wx.min():.0f} x {wy.max()-wy.min():.0f} m")

    # ── Wind: load once per distinct centre, reuse across cases ─────────
    series = {}
    for name, (lat, lon) in {"baseline": BASELINE_CENTRE, "cell63": CELL63_CENTRE}.items():
        print(f"\nloading real AROME at {name} ({lat}, {lon}) ...")
        t0 = time.time()
        times, ws_hub, wd = rep.load_arome_series(lat, lon)
        series[name] = (times, ws_hub, wd)
        ts = pd.to_datetime(times)
        print(f"  {ws_hub.size} steps, {ts.min()} .. {ts.max()}, "
              f"years {sorted(set(ts.year))}, {time.time()-t0:.0f}s")

    cases = [
        ("case1_grid_baseline_centre", "plain 7D grid", BASELINE_CENTRE, gx, gy, "baseline"),
        ("case2_grid_cell63", "plain 7D grid", CELL63_CENTRE, gx, gy, "cell63"),
        ("case3_submitted_cell63", "submitted Stage 3 boundary-loaded",
         CELL63_CENTRE, wx, wy, "cell63"),
    ]

    rows = []
    for key, layout_name, (clat, clon), x_m, y_m, wind_key in cases:
        times, ws_hub, wd = series[wind_key]
        ts = pd.to_datetime(times)
        print(f"\n=== {key} ===")

        # Path A: the independent replica (dispatch-specified primary).
        r = rep.run_case(key, times, ws_hub, wd, turbine_rep, x_m, y_m)

        # Path B: the kit's own simulate_year, same wind, same layout.
        wind = WindSeries(pd.DataFrame({"time": times, "ws": ws_hub, "wd": wd}))
        kit_res = simulate_year(FarmLayout(x_m=x_m, y_m=y_m, turbine=turbine_kit), wind)
        print(f"  kit simulate_year: CF={kit_res.capacity_factor*100:.4f}%  "
              f"wake={kit_res.wake_loss_fraction*100:.4f}%  AEP={kit_res.aep_gwh:.2f} GWh")

        cf_net = r["capacity_factor"]
        wake = r["wake_loss_fraction"]
        cf_gross = cf_net / (1.0 - wake)

        depth_m, dist_km = site_cost_inputs(clat, clon, x_m, y_m)
        cb = evaluate_farm(capacity_mw=CAPACITY_MW, n_turbines=N_TURBINES,
                           aep_gwh=r["aep_gwh"], water_depth_m=depth_m,
                           distance_to_shore_km=dist_km, params=params)

        rows.append({
            "case": key, "layout": layout_name,
            "centre_lat": clat, "centre_lon": clon,
            "n_steps": r["n_steps"],
            "wind_years": ",".join(str(y) for y in sorted(set(ts.year))),
            "wind_start": str(ts.min()), "wind_end": str(ts.max()),
            "mean_ws_hub": r["mean_ws_hub"],
            "cf_gross": cf_gross,
            "wake_loss_fraction": wake,
            "cf_net": cf_net,
            "aep_gwh": r["aep_gwh"],
            "mean_depth_m": depth_m,
            "mean_shore_distance_km": dist_km,
            "lcoe_eur_per_mwh": cb.lcoe_eur_per_mwh,
            "capex_eur": cb.capex_eur,
            "opex_eur_per_year": cb.opex_eur_per_year,
            "rated_capacity_mw": r["rated_capacity_mw"],
            "kit_cf_net": kit_res.capacity_factor,
            "kit_wake_loss_fraction": kit_res.wake_loss_fraction,
            "kit_aep_gwh": kit_res.aep_gwh,
            "kit_vs_replica_cf_diff_pp": (kit_res.capacity_factor - cf_net) * 100,
            "kit_vs_replica_wake_diff_pp": (kit_res.wake_loss_fraction - wake) * 100,
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # ── Effect decomposition ────────────────────────────────────────────
    c1, c2, c3 = rows[0], rows[1], rows[2]
    effects = {
        "site_effect_same_layout": {
            "description": "case2 minus case1: identical plain 7D grid, different centre",
            "cf_net_pp": (c2["cf_net"] - c1["cf_net"]) * 100,
            "cf_gross_pp": (c2["cf_gross"] - c1["cf_gross"]) * 100,
            "wake_pp": (c2["wake_loss_fraction"] - c1["wake_loss_fraction"]) * 100,
            "aep_gwh": c2["aep_gwh"] - c1["aep_gwh"],
            "lcoe_eur_per_mwh": c2["lcoe_eur_per_mwh"] - c1["lcoe_eur_per_mwh"],
            "mean_ws_hub_ms": c2["mean_ws_hub"] - c1["mean_ws_hub"],
        },
        "layout_effect_same_site": {
            "description": "case3 minus case2: identical wind at cell 63, different layout",
            "cf_net_pp": (c3["cf_net"] - c2["cf_net"]) * 100,
            "cf_gross_pp": (c3["cf_gross"] - c2["cf_gross"]) * 100,
            "wake_pp": (c3["wake_loss_fraction"] - c2["wake_loss_fraction"]) * 100,
            "aep_gwh": c3["aep_gwh"] - c2["aep_gwh"],
            "lcoe_eur_per_mwh": c3["lcoe_eur_per_mwh"] - c2["lcoe_eur_per_mwh"],
        },
        "organizer_reported": {"cf": 0.532, "wake": 0.071, "aep_gwh": 5635.0, "lcoe": 83.1},
        "case1_vs_organizer": {
            "cf_net_pp": (0.532 - c1["cf_net"]) * 100,
            "wake_pp": (0.071 - c1["wake_loss_fraction"]) * 100,
            "aep_gwh": 5635.0 - c1["aep_gwh"],
            "lcoe_eur_per_mwh": 83.1 - c1["lcoe_eur_per_mwh"],
        },
        "banked_case3_reference": {"cf": 0.4721, "wake": 0.0827, "aep_gwh": 5003.92},
    }

    out = {
        "generated_by": "scripts/task2_three_case_scorer_20260818.py",
        "cases": rows, "effects": effects,
        "wall_clock_s": time.time() - t_start,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 92)
    print(f"{'case':30s} {'CFgross':>9s} {'wake':>8s} {'CFnet':>8s} {'AEP GWh':>10s} {'LCOE':>9s}")
    for r in rows:
        print(f"{r['case']:30s} {r['cf_gross']*100:8.2f}% {r['wake_loss_fraction']*100:7.2f}% "
              f"{r['cf_net']*100:7.2f}% {r['aep_gwh']:10.1f} {r['lcoe_eur_per_mwh']:8.2f}")
    print("=" * 92)
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_JSON}")
    print(f"total wall-clock {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
