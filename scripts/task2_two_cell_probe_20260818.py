"""Residual probe for the two-cell check (CC dispatch 2026-08-18, block S2.3).

The S2.3 decomposition in scripts/task2_two_cell_check_20260818.py lands at
79.02 EUR/MWh when it adopts BOTH the organizer's site (53.5N, 1.5E) and the
organizer's energy (5,635 GWh), against their reported 83.1. The kit cost model
therefore does not reproduce 83.1 from the organizer's own stated inputs, and
the sign of the residual is the opposite of the site/energy story. This script
characterises that residual rather than leaving it as an unexplained number.

It sweeps the free inputs of the shipped kit cost model
(phase_2/kit/phase_2/cost_model.py) at the organizer's stated AEP and asks:
which combinations land on 83.1?

Read-only, deterministic, no stochastic step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "phase_2" / "kit" / "phase_2"))

from cost_model import CostParameters, evaluate_farm  # noqa: E402

CAPACITY_MW = 1210.0
N_TURBINES = 55
ORG_AEP_GWH = 5635.0
ORG_LCOE = 83.1
BASE_DEPTH_M = 25.6299991607666      # sampler value at 53.5N, 1.5E
BASE_DIST_KM = 64.00936126708984     # sampler value at 53.5N, 1.5E


def lcoe(depth_m: float, dist_km: float, aep_gwh: float = ORG_AEP_GWH) -> float:
    return evaluate_farm(
        capacity_mw=CAPACITY_MW, n_turbines=N_TURBINES, aep_gwh=aep_gwh,
        water_depth_m=depth_m, distance_to_shore_km=dist_km,
        params=CostParameters(),
    ).lcoe_eur_per_mwh


def main() -> None:
    p = CostParameters()
    print(f"target: organizer-reported LCOE {ORG_LCOE} EUR/MWh at AEP {ORG_AEP_GWH} GWh\n")

    print("-- kit-default probes at the organizer AEP --")
    probes = {
        "evaluate_farm defaults (depth=30, dist=60)": (30.0, 60.0),
        "sampler site (depth=25.63, dist=64.01)": (BASE_DEPTH_M, BASE_DIST_KM),
        "sampler depth, kit-default dist": (BASE_DEPTH_M, 60.0),
        "kit-doc depth 26 m, kit-default dist": (26.0, 60.0),
        "kit-doc depth 26 m, sampler dist": (26.0, BASE_DIST_KM),
    }
    for label, (d, k) in probes.items():
        v = lcoe(d, k)
        print(f"  {label:46s} -> {v:7.3f}  (delta vs {ORG_LCOE}: {v - ORG_LCOE:+.3f})")

    print("\n-- what single input reproduces 83.1 at the organizer AEP? --")
    # Distance needed, holding sampler depth.
    dists = np.arange(0.0, 400.0, 0.01)
    vals = np.array([lcoe(BASE_DEPTH_M, float(x)) for x in dists[::100]])
    coarse = dists[::100]
    idx = int(np.argmin(np.abs(vals - ORG_LCOE)))
    lo = max(0.0, coarse[idx] - 1.0)
    fine = np.arange(lo, lo + 2.0, 0.001)
    fvals = np.array([lcoe(BASE_DEPTH_M, float(x)) for x in fine])
    dist_star = float(fine[int(np.argmin(np.abs(fvals - ORG_LCOE)))])
    print(f"  distance_to_shore_km needed (depth held at {BASE_DEPTH_M:.2f} m): {dist_star:.2f} km")

    # Depth needed, holding sampler distance.
    depths = np.arange(0.0, 200.0, 0.001)
    dvals = np.array([lcoe(float(x), BASE_DIST_KM) for x in depths[::1000]])
    dcoarse = depths[::1000]
    didx = int(np.argmin(np.abs(dvals - ORG_LCOE)))
    dlo = max(0.0, dcoarse[didx] - 1.0)
    dfine = np.arange(dlo, dlo + 2.0, 0.001)
    dfvals = np.array([lcoe(float(x), BASE_DIST_KM) for x in dfine])
    depth_star = float(dfine[int(np.argmin(np.abs(dfvals - ORG_LCOE)))])
    print(f"  water_depth_m needed (distance held at {BASE_DIST_KM:.2f} km): {depth_star:.2f} m")

    # AEP needed at the sampler site to give 83.1.
    aep_star = None
    aeps = np.arange(3000.0, 8000.0, 0.5)
    avals = np.array([lcoe(BASE_DEPTH_M, BASE_DIST_KM, float(a)) for a in aeps[::100]])
    acoarse = aeps[::100]
    aidx = int(np.argmin(np.abs(avals - ORG_LCOE)))
    alo = acoarse[aidx] - 60.0
    afine = np.arange(alo, alo + 120.0, 0.1)
    afvals = np.array([lcoe(BASE_DEPTH_M, BASE_DIST_KM, float(a)) for a in afine])
    aep_star = float(afine[int(np.argmin(np.abs(afvals - ORG_LCOE)))])
    print(f"  aep_gwh needed (site held at sampler values): {aep_star:.1f} GWh "
          f"(vs organizer-stated {ORG_AEP_GWH:.0f})")

    print("\n-- CRF / lifetime sanity (unchanged kit defaults) --")
    print(f"  discount_rate {p.discount_rate}, lifetime {p.project_lifetime_years} yr, CRF {p.crf:.6f}")


if __name__ == "__main__":
    main()
