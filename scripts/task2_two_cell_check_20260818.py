"""Two-cell check: kit baseline centre (53.5N, 1.5E) versus cell 63 (52.50N, 3.00E).

CC dispatch 2026-08-18, block S2. Read-only diagnostic. No layout is changed,
no submission is touched, nothing is refitted, no AEP is recomputed.

**Question.** The corrected organizer baseline is CF ~53.2%, AEP ~5,635 GWh,
wake ~7.1%, LCOE ~83.1 EUR/MWh at 53.5N 1.5E on real wind
(organizer board reply, 2026-08-10). Our submitted layout is CF 47.21%,
wake 8.27%, AEP 5,003.9 GWh, LCOE 95.168 EUR/MWh at cell 63. The standing
candidate explanation is that the two sites differ under the depth constraint
and so the figures are not like-for-like. That explanation is plausible and
NOT established. This script produces the numbers that decide it.

**What it computes, and from where.**

  1. Centre depth and distance-to-coast at both centres, from the EMODnet
     raster via the existing sampler (scripts/task2_depth_sampler.py, which
     reads phase_2/phase2_dataset_ship/static/bathymetry/emodnet_northsea_1km.nc).
  2. 15x15 km box depth statistics at both centres, on the SAME 250 m grid
     that scripts/task2_shallow_box_retest.py:48-54 used for the cell 63
     screen (BOX_M=15_000, GRID_SPACING_M=250). Note the dispatch describes
     that screen as "1 km sampling"; the committed screen is finer, at 250 m.
     Both are reported so the discrepancy is visible rather than reconciled.
  3. The kit foundation cost term, evaluated at each centre depth using the
     formula as it actually appears in the shipped code
     (phase_2/kit/phase_2/cost_model.py:125-127).
  4. Wind resource: mean ws at 125 m and mean ws^3, pooled 2016-2020 and per
     year, plus each cell's rank, read from data/cell_resource_ranking.csv
     (built by scripts/task2_rank_cells_by_resource.py over the AROME
     training corpus). Read, not recomputed: that CSV IS the extraction, and
     re-running the 1826-file aggregation would produce the same numbers at
     ~40 min cost.
  5. 16-sector direction distributions for both cells from
     data/cell_direction_rose_16sector.csv (same script), plus the angular
     offset between the two dominant sectors.
  6. An LCOE decomposition that separates the depth/distance-driven component
     of the 83.1 vs 95.168 gap from the energy-driven component, using the
     kit's own compute_capex / compute_opex / compute_lcoe unchanged.

**Deterministic.** Static raster reads, CSV reads and closed-form arithmetic.
No stochastic step anywhere, so no seed is required.

Output: reports/two_cell_check_20260818.json (machine-readable record of every
number quoted in reports/two_cell_check_20260818.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(KIT_PHASE2))

import task2_depth_sampler as depth_sampler  # noqa: E402
from cost_model import (  # noqa: E402
    CostParameters, compute_capex, compute_lcoe, compute_opex, evaluate_farm,
)

# ── Sites under comparison ──────────────────────────────────────────────
BASELINE_LAT, BASELINE_LON = 53.5, 1.5     # organizer baseline centre
CELL63_LAT, CELL63_LON = 52.50, 3.00       # our submitted siting

# ── Box screen constants, copied from task2_shallow_box_retest.py:38-39 ──
BOX_M = 15_000.0
GRID_SPACING_M = 250.0
DEPTH_THRESHOLD_M = 50.0

# ── Farm constants, from build_recon_20260718.py:46-47 (Phase_2.pdf p.5) ─
CAPACITY_MW = 1210.0
N_TURBINES = 55

# ── Recorded figures under test (NOT recomputed here) ───────────────────
OURS_AEP_GWH = 5003.923          # data/task2_layout_final_validation.csv
OURS_CF = 0.47209
OURS_WAKE = 0.0827
OURS_LCOE = 95.168               # task2_lcoe/RECON_20260718.md
OURS_MEAN_DEPTH_M = 38.179       # mean over 55 turbines, recon baseline
OURS_MEAN_SHORE_KM = 79.197      # mean over 55 turbines, recon baseline
ORG_AEP_GWH = 5635.0             # organizer board reply 2026-08-10
ORG_CF = 0.532
ORG_WAKE = 0.071
ORG_LCOE = 83.1

RANKING_CSV = REPO_ROOT / "data" / "cell_resource_ranking.csv"
ROSE_CSV = REPO_ROOT / "data" / "cell_direction_rose_16sector.csv"
WINNER_JSON = REPO_ROOT / "data" / "task2_layout_winner.json"
OUT_JSON = REPO_ROOT / "reports" / "two_cell_check_20260818.json"


def box_grid(spacing_m: float = GRID_SPACING_M) -> tuple[np.ndarray, np.ndarray]:
    """Identical to task2_shallow_box_retest.box_grid (same constants)."""
    half = BOX_M / 2
    n = int(round(BOX_M / spacing_m)) + 1
    coords = np.linspace(-half, half, n)
    return np.meshgrid(coords, coords)


def box_stats(centre_lat: float, centre_lon: float) -> dict:
    x_m, y_m = box_grid()
    depth = depth_sampler.depth_at_xy(
        centre_lat, centre_lon, x_m.ravel(), y_m.ravel()
    ).reshape(x_m.shape)
    finite = np.isfinite(depth)
    sea = finite & (depth > 0)
    shallow = sea & (depth <= DEPTH_THRESHOLD_M)
    return {
        "n_samples": int(depth.size),
        "grid_spacing_m": GRID_SPACING_M,
        "max_depth_in_box_m": float(np.nanmax(depth)) if finite.any() else float("nan"),
        "min_depth_in_box_m": float(np.nanmin(depth[sea])) if sea.any() else float("nan"),
        "mean_depth_in_box_m": float(np.nanmean(depth[sea])) if sea.any() else float("nan"),
        "frac_box_le_50m": float(shallow.sum()) / depth.size,
        "frac_land_or_nodata": float((~finite).sum() + (finite & (depth <= 0)).sum()) / depth.size,
    }


def foundation_per_kw(depth_m: float, params: CostParameters) -> float:
    """kit cost_model.py:125-127, quoted verbatim in structure."""
    return params.foundation_eur_per_kw + max(
        0.0, depth_m - params.foundation_depth_ref_m
    ) * params.foundation_depth_slope_eur_per_kw_per_m


def cell_row(df: pd.DataFrame, lat: float, lon: float) -> dict | None:
    hit = df[(np.isclose(df["lat"], lat)) & (np.isclose(df["lon"], lon))]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def rose_row(df: pd.DataFrame, cell_id: int) -> dict:
    hit = df[df["cell_id"] == cell_id]
    if hit.empty:
        raise RuntimeError(f"cell_id {cell_id} not present in {ROSE_CSV}")
    return hit.iloc[0].to_dict()


def circular_offset_deg(a_deg: float, b_deg: float) -> float:
    """Smallest signed angular separation a -> b, in (-180, 180]."""
    return float((b_deg - a_deg + 180.0) % 360.0 - 180.0)


def lcoe_at(depth_m: float, distance_km: float, aep_gwh: float,
            params: CostParameters) -> dict:
    capex, comp = compute_capex(CAPACITY_MW, depth_m, distance_km, params)
    opex = compute_opex(CAPACITY_MW, distance_km, params)
    lcoe, parts = compute_lcoe(aep_gwh * 1000.0, capex, opex, params)
    return {
        "water_depth_m": depth_m,
        "distance_to_shore_km": distance_km,
        "aep_gwh": aep_gwh,
        "capex_eur": capex,
        "capex_foundation_eur": comp["foundation"],
        "capex_export_cable_eur": comp["export_cable"],
        "opex_eur_per_year": opex,
        "lcoe_eur_per_mwh": lcoe,
        "lcoe_capex_eur_per_mwh": parts["capex"],
        "lcoe_opex_eur_per_mwh": parts["opex"],
    }


def main() -> None:
    params = CostParameters()
    out: dict = {"generated_by": "scripts/task2_two_cell_check_20260818.py"}

    # ── 1. Depth and distance to coast at the two centres ───────────────
    sites = {}
    for key, (lat, lon) in {
        "baseline_centre": (BASELINE_LAT, BASELINE_LON),
        "cell63": (CELL63_LAT, CELL63_LON),
    }.items():
        sites[key] = {
            "lat": lat, "lon": lon,
            "centre_depth_m": depth_sampler.depth_at(lat, lon),
            "centre_dist_coast_km": depth_sampler.dist_to_coast_at(lat, lon),
            "box": box_stats(lat, lon),
        }
        sites[key]["foundation_eur_per_kw"] = foundation_per_kw(
            sites[key]["centre_depth_m"], params
        )
    out["sites"] = sites

    # ── 2. Resource, from the committed ranking ─────────────────────────
    rank_df = pd.read_csv(RANKING_CSV)
    out["ranking_csv_n_cells"] = int(len(rank_df))
    best = rank_df.loc[rank_df["mean_ws3_pooled"].idxmax()]
    out["domain_best"] = {
        "cell_id": int(best["cell_id"]), "lat": float(best["lat"]),
        "lon": float(best["lon"]), "mean_ws3_pooled": float(best["mean_ws3_pooled"]),
    }
    for key, (lat, lon) in {
        "baseline_centre": (BASELINE_LAT, BASELINE_LON),
        "cell63": (CELL63_LAT, CELL63_LON),
    }.items():
        row = cell_row(rank_df, lat, lon)
        if row is None:
            sites[key]["in_159_eligible_set"] = False
            sites[key]["resource"] = None
            continue
        sites[key]["in_159_eligible_set"] = True
        sites[key]["resource"] = {
            "cell_id": int(row["cell_id"]),
            "rank_of_159": int(row["rank"]),
            "depth_m_in_ranking": float(row["depth_m"]),
            "n_arome_pixels": int(row["n_arome_pixels"]),
            "mean_ws_pooled": float(row["mean_ws_pooled"]),
            "mean_ws3_pooled": float(row["mean_ws3_pooled"]),
            "mean_ws_per_year": {str(y): float(row[f"mean_ws_{y}"])
                                 for y in (2016, 2017, 2018, 2019, 2020)},
            "mean_ws3_per_year": {str(y): float(row[f"mean_ws3_{y}"])
                                  for y in (2016, 2017, 2018, 2019, 2020)},
            "frac_of_domain_best_ws3": float(row["mean_ws3_pooled"]) / float(best["mean_ws3_pooled"]),
        }

    b_ws3 = sites["baseline_centre"]["resource"]["mean_ws3_pooled"]
    c_ws3 = sites["cell63"]["resource"]["mean_ws3_pooled"]
    b_ws = sites["baseline_centre"]["resource"]["mean_ws_pooled"]
    c_ws = sites["cell63"]["resource"]["mean_ws_pooled"]
    out["resource_comparison"] = {
        "cell63_over_baseline_ws3_ratio": c_ws3 / b_ws3,
        "cell63_over_baseline_ws3_pct": (c_ws3 / b_ws3 - 1.0) * 100.0,
        "cell63_over_baseline_ws_ratio": c_ws / b_ws,
        "cell63_over_baseline_ws_pct": (c_ws / b_ws - 1.0) * 100.0,
        "per_year_ws3_pct": {
            str(y): (sites["cell63"]["resource"]["mean_ws3_per_year"][str(y)]
                     / sites["baseline_centre"]["resource"]["mean_ws3_per_year"][str(y)] - 1.0) * 100.0
            for y in (2016, 2017, 2018, 2019, 2020)
        },
    }

    # ── 3. 16-sector roses ──────────────────────────────────────────────
    rose_df = pd.read_csv(ROSE_CSV)
    sector_cols = [c for c in rose_df.columns if c.startswith("sector_")]
    sector_centres = [float(c.split("_")[2].replace("deg", "")) + 11.25 for c in sector_cols]
    roses = {}
    for key in ("baseline_centre", "cell63"):
        cid = sites[key]["resource"]["cell_id"]
        r = rose_row(rose_df, cid)
        probs = [float(r[c]) for c in sector_cols]
        dom = int(np.argmax(probs))
        roses[key] = {
            "cell_id": cid,
            "sector_lower_edges_deg": [float(c.split("_")[2].replace("deg", "")) for c in sector_cols],
            "probabilities": probs,
            "dominant_sector_index": dom,
            "dominant_sector_lower_edge_deg": float(sector_cols[dom].split("_")[2].replace("deg", "")),
            "dominant_sector_centre_deg": sector_centres[dom],
            "dominant_sector_prob": probs[dom],
        }
    roses["dominant_sector_offset_deg"] = circular_offset_deg(
        roses["baseline_centre"]["dominant_sector_centre_deg"],
        roses["cell63"]["dominant_sector_centre_deg"],
    )
    roses["l1_distance_between_roses"] = float(np.abs(
        np.array(roses["baseline_centre"]["probabilities"])
        - np.array(roses["cell63"]["probabilities"])
    ).sum())
    out["roses"] = roses

    # ── 4. Reproduce our banked LCOE, then decompose the gap ────────────
    repro = evaluate_farm(
        capacity_mw=CAPACITY_MW, n_turbines=N_TURBINES, aep_gwh=OURS_AEP_GWH,
        water_depth_m=OURS_MEAN_DEPTH_M, distance_to_shore_km=OURS_MEAN_SHORE_KM,
        params=params,
    )
    out["banked_lcoe_reproduction"] = {
        "recorded_lcoe_eur_per_mwh": OURS_LCOE,
        "recomputed_lcoe_eur_per_mwh": repro.lcoe_eur_per_mwh,
        "abs_diff": abs(repro.lcoe_eur_per_mwh - OURS_LCOE),
        "inputs": {
            "capacity_mw": CAPACITY_MW, "aep_gwh": OURS_AEP_GWH,
            "water_depth_m": OURS_MEAN_DEPTH_M,
            "distance_to_shore_km": OURS_MEAN_SHORE_KM,
        },
    }

    b_depth = sites["baseline_centre"]["centre_depth_m"]
    b_dist = sites["baseline_centre"]["centre_dist_coast_km"]
    c_depth = sites["cell63"]["centre_depth_m"]
    c_dist = sites["cell63"]["centre_dist_coast_km"]

    # Step A: our actual banked configuration.
    step_a = lcoe_at(OURS_MEAN_DEPTH_M, OURS_MEAN_SHORE_KM, OURS_AEP_GWH, params)
    # Step B: move the SITE to the baseline centre, hold our energy fixed.
    step_b = lcoe_at(b_depth, b_dist, OURS_AEP_GWH, params)
    # Step C: additionally adopt the organizer's energy at that site.
    step_c = lcoe_at(b_depth, b_dist, ORG_AEP_GWH, params)
    # Depth-only variant: hold distance at our value, change depth only.
    depth_only = lcoe_at(b_depth, OURS_MEAN_SHORE_KM, OURS_AEP_GWH, params)
    # Kit-default-distance variants, for the "60 km never overridden" question.
    kit_default_ours = lcoe_at(OURS_MEAN_DEPTH_M, params.distance_to_shore_km, OURS_AEP_GWH, params)
    kit_default_base = lcoe_at(b_depth, params.distance_to_shore_km, ORG_AEP_GWH, params)
    # Centre-depth (not turbine-mean) variant at cell 63, for the table.
    centre_depth_cell63 = lcoe_at(c_depth, c_dist, OURS_AEP_GWH, params)

    out["lcoe_decomposition"] = {
        "step_a_ours_as_banked": step_a,
        "step_b_site_moved_energy_held": step_b,
        "step_c_site_and_energy_moved": step_c,
        "depth_only_variant": depth_only,
        "kit_default_60km_ours": kit_default_ours,
        "kit_default_60km_baseline_site_org_energy": kit_default_base,
        "cell63_centre_depth_variant": centre_depth_cell63,
        "organizer_reported_lcoe": ORG_LCOE,
        "total_gap_eur_per_mwh": step_a["lcoe_eur_per_mwh"] - ORG_LCOE,
        "site_component_eur_per_mwh": step_a["lcoe_eur_per_mwh"] - step_b["lcoe_eur_per_mwh"],
        "energy_component_eur_per_mwh": step_b["lcoe_eur_per_mwh"] - step_c["lcoe_eur_per_mwh"],
        "residual_vs_organizer_eur_per_mwh": step_c["lcoe_eur_per_mwh"] - ORG_LCOE,
        "depth_only_component_eur_per_mwh": step_a["lcoe_eur_per_mwh"] - depth_only["lcoe_eur_per_mwh"],
        "distance_only_component_eur_per_mwh": depth_only["lcoe_eur_per_mwh"] - step_b["lcoe_eur_per_mwh"],
    }

    out["recorded_figures"] = {
        "ours": {"cf": OURS_CF, "wake": OURS_WAKE, "aep_gwh": OURS_AEP_GWH, "lcoe": OURS_LCOE},
        "organizer": {"cf": ORG_CF, "wake": ORG_WAKE, "aep_gwh": ORG_AEP_GWH, "lcoe": ORG_LCOE},
        "cf_gap_pp": (ORG_CF - OURS_CF) * 100.0,
        "aep_gap_pct": (ORG_AEP_GWH / OURS_AEP_GWH - 1.0) * 100.0,
    }

    # ── 5. Split the CF gap into a wake part and a gross-conversion part ──
    # CF_net = CF_gross * (1 - wake_loss), so CF_gross = CF_net / (1 - wake).
    # Both wake fractions are recorded values, not recomputed here.
    ours_gross = OURS_CF / (1.0 - OURS_WAKE)
    org_gross = ORG_CF / (1.0 - ORG_WAKE)
    ours_at_org_wake = ours_gross * (1.0 - ORG_WAKE)
    out["cf_gap_decomposition"] = {
        "ours_cf_net": OURS_CF,
        "organizer_cf_net": ORG_CF,
        "total_cf_gap_pp": (ORG_CF - OURS_CF) * 100.0,
        "ours_cf_gross": ours_gross,
        "organizer_cf_gross": org_gross,
        "gross_cf_gap_pp": (org_gross - ours_gross) * 100.0,
        "ours_cf_net_if_given_organizer_wake": ours_at_org_wake,
        "wake_attributable_pp": (ours_at_org_wake - OURS_CF) * 100.0,
        "gross_attributable_pp": (ORG_CF - ours_at_org_wake) * 100.0,
        "gross_share_of_gap": (ORG_CF - ours_at_org_wake) / (ORG_CF - OURS_CF),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
