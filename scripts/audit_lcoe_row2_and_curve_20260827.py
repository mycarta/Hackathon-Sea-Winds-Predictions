"""Audit recompute: Table 2 row 2 LCOE, and the published-curve substitution.

CC dispatch 2026-08-27, items 0 and 0b. Read-only with respect to every frozen
deliverable: no submission is touched, no layout is changed, no model is
refitted, `data/task2_layout_winner.json` is opened read-only.

Both items are run from ONE script because they share the expensive step -- the
AROME series load at the two centres -- and because running them apart would
risk the two answers coming from different wind inputs.

──────────────────────────────────────────────────────────────────────────────
ITEM 0 -- Table 2 row 2 LCOE recompute
──────────────────────────────────────────────────────────────────────────────
External audit finding, verbatim arithmetic: the report states 95.17 EUR/MWh
for row 3 against 96.31 for row 2. With no yield-dependent cost term, scaling
row 3 by the AEP ratio gives 95.17 * (5,003.9 / 4,931.3) = 96.57, not 96.31;
about 0.26 EUR/MWh is unexplained.

This script recomputes row 2 from the scorer replica for that case, through the
same `cost_model.evaluate_farm` call path that produced row 3, and decomposes
the residual exactly:

    LCOE = (CAPEX * CRF + OPEX) / AEP_MWh          [cost_model.compute_lcoe]

    naive  = LCOE_row3 * (AEP_row3 / AEP_row2) = N_row3 / AEP_row2
    actual = N_row2 / AEP_row2
    actual - naive = (N_row2 - N_row3) / AEP_row2

so the residual is entirely a NUMERATOR difference. The audit's premise -- that
row 2 and row 3 share a numerator and differ only in yield -- is what this
script tests. Rows 2 and 3 sit at the SAME centre (cell 63) but carry DIFFERENT
layouts, and `site_cost_inputs` samples mean water depth and mean shore
distance over the TURBINE POSITIONS, so a layout change moves the cost inputs
as well as the yield.

──────────────────────────────────────────────────────────────────────────────
ITEM 0b -- published IEA 22 MW curve vs the kit synthetic ramp
──────────────────────────────────────────────────────────────────────────────
Hypothesis only, not a report claim. The replica drives the wake model with the
kit's synthetic generic cubic ramp, rated at 12.0 m/s -- a hardcoded
`rated_ws` fallback in `turbines_catalog._generic_power_ct`, not a property of
the IEA 22 MW machine. The published IEA-22-280-RWT reference turbine is rated
at 11.1349 m/s (extracted and pinned by
`scripts/audit_iea22_published_curve_extract_20260827.py`).

This script reruns the organizer's own baseline centre (53.5N, 1.5E) with the
plain 7D grid, everything else unchanged, once per curve, and reports gross CF,
net CF and wake side by side.

**Comparison target, corrected.** The organizers corrected the Task 2 baseline
on the Codabench board on 2026-08-10 to: CF 53.2 % NET, wake 7.1 %, AEP 5,635
GWh, LCOE 83.1 EUR/MWh. Those are the figures used here. The pre-correction
figures (CF ~54 %, wake ~6 %, LCOE ~82) are NOT used anywhere in this script.

Deterministic: real AROME series, closed-form layout geometry, tabular power
curves, no stochastic step, so no seed applies. The only randomness-adjacent
input, the kit's synthetic bootstrap wind, is not used here at all.

Reads : phase_2/phase2_dataset_ship (AROME, via kit target_loader)
        data/iea22mw_power_ct.csv                       (kit synthetic ramp)
        data/iea22_280_rwt/iea22_280_rwt_wisdem_power_ct.csv  (published)
        data/task2_layout_winner.json                   (read-only)
Writes: reports/audit_lcoe_row2_and_curve_20260827.json
        data/audit_lcoe_row2_and_curve_20260827.csv
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"

# Amendment v2.1 §5b, changed-constant discipline. `task2_scorer_replica` and
# `task2_three_case_scorer_20260818` both declare
# DATA_ROOT = phase_2/phase2_dataset_ship and then `os.environ.setdefault`,
# so on a machine where that path does not exist they SILENTLY fall through to
# whatever PHASE2_DATA_ROOT the shell happens to carry -- and if the shell
# carries nothing, target_loader.list_dates() returns an empty list rather
# than raising. That directory does not exist in this checkout; the ship layout
# moved under phase_2/inference_2022/ (see
# docs/audit/anchor6_poststatement_20260825.md:93,140). This script therefore
# resolves the root explicitly, asserts it, and asserts the resulting date
# count, instead of inheriting ambient shell state.
DATA_ROOT = REPO_ROOT / "phase_2" / "inference_2022" / "phase2_dataset_ship"
EXPECTED_TRAIN_DATES = 1826          # 2016-01-01 .. 2020-12-31 inclusive
EXPECTED_STEPS = 14608               # 1826 days x 8 three-hourly steps

# `task2_depth_sampler.EMODNET_PATH` hardcodes the same stale
# phase_2/phase2_dataset_ship root and does NOT consult PHASE2_DATA_ROOT
# (task2_depth_sampler.py:50-53), so it raises FileNotFoundError in this
# checkout. The file itself is present under the moved root. This script
# repoints the sampler's loader at runtime -- the sampling logic in
# depth_at_array is untouched, only path resolution changes -- and pins the
# file by SHA-256 so the substitution cannot silently pick up a different
# raster. The recomputed depths are then gated against the banked three-case
# values, which is the real proof that this is the same raster the 2026-08-18
# run used. The sampler script itself is left unmodified: per the dispatch,
# "a guard or a check that does not fire is a finding, not a fix".
BATHY_NC = DATA_ROOT / "static" / "bathymetry" / "emodnet_northsea_1km.nc"
BATHY_SHA256 = "8d61153684f1e5420442b9e0d491e5c04359e2ce3af9f810d14f0633810217c6"

# Banked values from the 2026-08-18 S3 run, reports/three_case_scorer_20260818.json.
# Used as an equivalence gate, not as an input.
BANKED = {
    "row2": {"aep_gwh": 4931.292000610045, "mean_depth_m": 36.891818063909355,
             "mean_shore_distance_km": 79.82806091308593,
             "lcoe_eur_per_mwh": 96.31472107673332},
    "row3": {"aep_gwh": 5003.92289397341, "mean_depth_m": 38.178909024325286,
             "mean_shore_distance_km": 79.19674044522372,
             "lcoe_eur_per_mwh": 95.16756451257942},
}

sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(KIT_PHASE2))
sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
if not DATA_ROOT.is_dir():
    raise SystemExit(f"dataset root not found: {DATA_ROOT}")
os.environ["PHASE2_DATA_ROOT"] = str(DATA_ROOT)   # set, not setdefault

BASELINE_CENTRE = (53.5, 1.5)
CELL63_CENTRE = (52.50, 3.00)
WINNER_JSON = REPO_ROOT / "data" / "task2_layout_winner.json"
PUBLISHED_CSV = REPO_ROOT / "data" / "iea22_280_rwt" / "iea22_280_rwt_wisdem_power_ct.csv"
OUT_JSON = REPO_ROOT / "reports" / "audit_lcoe_row2_and_curve_20260827.json"
OUT_CSV = REPO_ROOT / "data" / "audit_lcoe_row2_and_curve_20260827.csv"

# Phase_2.pdf p.5. Nameplate is a hard constant, independent of which power
# curve is loaded, so CF is reported against it as well as against the
# curve-derived rating.
CAPACITY_MW = 1210.0
N_TURBINES = 55

# Organizer correction, Codabench board, the organizer, 2026-08-10.
ORGANIZER = {"cf_net": 0.532, "wake": 0.071, "aep_gwh": 5635.0, "lcoe": 83.1}

# The two numbers the external audit quotes from report Table 2.
REPORT_ROW2_LCOE = 96.31
REPORT_ROW3_LCOE = 95.17


def build_turbine_from_csv(csv_path: Path, name: str):
    """Same construction as task2_scorer_replica.build_turbine, with the curve
    CSV as a parameter instead of a module constant. Kept byte-for-byte
    equivalent in every argument so the curve is the ONLY thing that varies."""
    from py_wake.wind_turbines import WindTurbine
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    import task2_scorer_replica as rep

    df = pd.read_csv(csv_path)
    ws = df["wind_speed_ms"].to_numpy()
    power = df["power_w"].to_numpy()
    ct = df["ct"].to_numpy()
    order = np.argsort(ws)
    ws, power, ct = ws[order], power[order], ct[order]
    mask = (ws > 0) & (power >= 0)
    power_ct = PowerCtTabular(
        ws=ws[mask], power=power[mask], power_unit="W", ct=ct[mask],
        ws_cutin=3.0, ws_cutout=25.0, power_idle=0, ct_idle=0, method="linear",
    )
    return WindTurbine(name=name, diameter=rep.DIAMETER_M,
                       hub_height=rep.HUB_HEIGHT_M, powerCtFunction=power_ct)


def rated_wind_speed(csv_path: Path) -> tuple[float, float]:
    """(rated power MW, first wind speed reaching it) for a curve CSV."""
    df = pd.read_csv(csv_path).sort_values("wind_speed_ms")
    peak_w = float(df["power_w"].max())
    hit = df.loc[df["power_w"] >= peak_w - 1e3, "wind_speed_ms"]
    return peak_w / 1e6, float(hit.iloc[0])


def repoint_depth_sampler() -> str:
    """Point task2_depth_sampler at the bathymetry raster's real location.

    Verifies the raster by SHA-256 first, then replaces the module's cached
    loader with one bound to the verified path. Returns the SHA for the record.
    """
    import xarray as xr
    import task2_depth_sampler as ds

    if not BATHY_NC.exists():
        raise SystemExit(f"bathymetry raster not found: {BATHY_NC}")
    h = hashlib.sha256()
    with open(BATHY_NC, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != BATHY_SHA256:
        raise SystemExit(f"bathymetry SHA mismatch: expected {BATHY_SHA256}, "
                         f"got {actual}")

    dataset = xr.load_dataset(BATHY_NC)
    ds._load = lambda *a, **k: dataset          # noqa: SLF001 - deliberate
    print(f"depth sampler repointed to {BATHY_NC}")
    print(f"  raster sha256 verified: {actual[:12]}...")
    return actual


def main() -> None:
    t_start = time.time()
    import task2_scorer_replica as rep
    from task2_three_case_scorer_20260818 import site_cost_inputs
    from cost_model import CostParameters, evaluate_farm

    bathy_sha = repoint_depth_sampler()
    params = CostParameters()

    kit_curve_csv = rep.TURBINE_CSV
    turbine_kit_ramp = build_turbine_from_csv(
        kit_curve_csv, "IEA 22 MW (kit synthetic ramp)")
    turbine_published = build_turbine_from_csv(
        PUBLISHED_CSV, "IEA 22 MW (IEA-22-280-RWT, WISDEM)")

    kit_rated_mw, kit_rated_ws = rated_wind_speed(kit_curve_csv)
    pub_rated_mw, pub_rated_ws = rated_wind_speed(PUBLISHED_CSV)
    print(f"kit synthetic ramp : rated {kit_rated_mw:.4f} MW at {kit_rated_ws:.4f} m/s")
    print(f"published WISDEM   : rated {pub_rated_mw:.4f} MW at {pub_rated_ws:.4f} m/s")

    # ── Layouts ─────────────────────────────────────────────────────────
    gx, gy = rep.grid_layout()
    with open(WINNER_JSON) as f:
        winner = json.load(f)
    wx = np.array(winner["layout_x_m"], dtype=float)
    wy = np.array(winner["layout_y_m"], dtype=float)
    assert gx.size == N_TURBINES and wx.size == N_TURBINES

    # ── Wind, loaded once per centre ────────────────────────────────────
    import target_loader
    n_dates = len(target_loader.list_dates())
    if n_dates != EXPECTED_TRAIN_DATES:
        raise SystemExit(
            f"AROME date count {n_dates} != expected {EXPECTED_TRAIN_DATES}; "
            f"data root is {DATA_ROOT}. Refusing to run: an empty or partial "
            f"root would produce plausible-looking but wrong CF/LCOE numbers."
        )
    print(f"AROME date count verified: {n_dates} days under {DATA_ROOT}")

    series = {}
    for name, (lat, lon) in {"baseline": BASELINE_CENTRE,
                             "cell63": CELL63_CENTRE}.items():
        print(f"\nloading real AROME at {name} ({lat}, {lon}) ...")
        t0 = time.time()
        times, ws_hub, wd = rep.load_arome_series(lat, lon)
        series[name] = (times, ws_hub, wd)
        ts = pd.to_datetime(times)
        if ws_hub.size != EXPECTED_STEPS:
            raise SystemExit(f"{name}: {ws_hub.size} steps != {EXPECTED_STEPS}")
        if sorted(set(ts.year)) != [2016, 2017, 2018, 2019, 2020]:
            raise SystemExit(f"{name}: unexpected years {sorted(set(ts.year))}")
        print(f"  {ws_hub.size} steps, {ts.min()} .. {ts.max()}, "
              f"years {sorted(set(ts.year))}, {time.time()-t0:.0f}s")

    cases = [
        # key, centre, layout, wind key, turbine, curve label
        ("row2_grid_cell63_kitramp", CELL63_CENTRE, (gx, gy), "cell63",
         turbine_kit_ramp, "kit synthetic ramp"),
        ("row3_submitted_cell63_kitramp", CELL63_CENTRE, (wx, wy), "cell63",
         turbine_kit_ramp, "kit synthetic ramp"),
        ("baseline_grid_kitramp", BASELINE_CENTRE, (gx, gy), "baseline",
         turbine_kit_ramp, "kit synthetic ramp"),
        ("baseline_grid_published", BASELINE_CENTRE, (gx, gy), "baseline",
         turbine_published, "published IEA-22-280-RWT (WISDEM)"),
    ]

    rows = []
    for key, (clat, clon), (x_m, y_m), wind_key, turbine, curve_label in cases:
        times, ws_hub, wd = series[wind_key]
        print(f"\n=== {key} ===")
        r = rep.run_case(key, times, ws_hub, wd, turbine, x_m, y_m)

        cf_net_internal = r["capacity_factor"]
        wake = r["wake_loss_fraction"]
        aep = r["aep_gwh"]
        # CF against the Phase_2.pdf nameplate, so the two curves compare on a
        # fixed denominator rather than on each curve's own rated power.
        cf_net = aep * 1000.0 / (CAPACITY_MW * 8760.0)
        cf_gross = cf_net / (1.0 - wake)

        depth_m, dist_km = site_cost_inputs(clat, clon, x_m, y_m)
        cb = evaluate_farm(capacity_mw=CAPACITY_MW, n_turbines=N_TURBINES,
                           aep_gwh=aep, water_depth_m=depth_m,
                           distance_to_shore_km=dist_km, params=params)

        rows.append({
            "case": key,
            "curve": curve_label,
            "centre_lat": clat, "centre_lon": clon,
            "n_steps": r["n_steps"],
            "mean_ws_hub": r["mean_ws_hub"],
            "cf_gross": cf_gross,
            "wake_loss_fraction": wake,
            "cf_net": cf_net,
            "cf_net_replica_internal": cf_net_internal,
            "rated_capacity_mw_from_curve": r["rated_capacity_mw"],
            "aep_gwh": aep,
            "mean_depth_m": depth_m,
            "mean_shore_distance_km": dist_km,
            "capex_eur": cb.capex_eur,
            "opex_eur_per_year": cb.opex_eur_per_year,
            "lcoe_eur_per_mwh": cb.lcoe_eur_per_mwh,
            "lcoe_capex_component": cb.lcoe_components["capex"],
            "lcoe_opex_component": cb.lcoe_components["opex"],
        })
        print(f"  cf_gross={cf_gross*100:.4f}%  wake={wake*100:.4f}%  "
              f"cf_net={cf_net*100:.4f}%  AEP={aep:.2f} GWh  "
              f"LCOE={cb.lcoe_eur_per_mwh:.4f}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    by = {r["case"]: r for r in rows}

    # ── ITEM 0: exact decomposition of the 0.26 residual ────────────────
    r2, r3 = by["row2_grid_cell63_kitramp"], by["row3_submitted_cell63_kitramp"]
    crf = params.crf
    n2 = r2["capex_eur"] * crf + r2["opex_eur_per_year"]      # annualised cost, EUR/yr
    n3 = r3["capex_eur"] * crf + r3["opex_eur_per_year"]
    a2_mwh, a3_mwh = r2["aep_gwh"] * 1000.0, r3["aep_gwh"] * 1000.0

    naive_row2 = r3["lcoe_eur_per_mwh"] * (a3_mwh / a2_mwh)
    residual = r2["lcoe_eur_per_mwh"] - naive_row2
    # Attribute the residual to its two cost terms.
    d_capex_term = (r2["capex_eur"] - r3["capex_eur"]) * crf / a2_mwh
    d_opex_term = (r2["opex_eur_per_year"] - r3["opex_eur_per_year"]) / a2_mwh

    item0 = {
        "question": "Is report Table 2 row 2 LCOE = 96.31 EUR/MWh correct?",
        "recomputed_row2_lcoe_eur_per_mwh": round(r2["lcoe_eur_per_mwh"], 2),
        "recomputed_row2_lcoe_full_precision": r2["lcoe_eur_per_mwh"],
        "report_row2_lcoe": REPORT_ROW2_LCOE,
        "recomputed_row3_lcoe_eur_per_mwh": round(r3["lcoe_eur_per_mwh"], 2),
        "report_row3_lcoe": REPORT_ROW3_LCOE,
        "aep_gwh_row2": r2["aep_gwh"],
        "aep_gwh_row3": r3["aep_gwh"],
        "cost_terms_row2": {
            "capex_eur": r2["capex_eur"], "opex_eur_per_year": r2["opex_eur_per_year"],
            "crf": crf, "annualised_cost_eur_per_year": n2,
            "mean_depth_m": r2["mean_depth_m"],
            "mean_shore_distance_km": r2["mean_shore_distance_km"],
        },
        "cost_terms_row3": {
            "capex_eur": r3["capex_eur"], "opex_eur_per_year": r3["opex_eur_per_year"],
            "crf": crf, "annualised_cost_eur_per_year": n3,
            "mean_depth_m": r3["mean_depth_m"],
            "mean_shore_distance_km": r3["mean_shore_distance_km"],
        },
        "any_yield_dependent_cost_term": False,
        "yield_dependent_term_name": None,
        "naive_aep_scaled_row2_lcoe": naive_row2,
        "residual_eur_per_mwh": residual,
        "residual_attribution": {
            "capex_difference_term": d_capex_term,
            "opex_difference_term": d_opex_term,
            "sum": d_capex_term + d_opex_term,
        },
        "explanation": (
            "No cost term depends on yield: compute_capex takes "
            "(capacity_mw, water_depth_m, distance_to_shore_km) and compute_opex "
            "takes (capacity_mw, distance_to_shore_km); AEP enters only as the "
            "denominator in compute_lcoe. The audit's residual is therefore NOT a "
            "yield-dependent cost term -- it is a numerator difference. Rows 2 and "
            "3 share a centre but not a layout, and site_cost_inputs samples mean "
            "water depth and mean shore distance over the turbine POSITIONS, so "
            "the boundary-loaded row-3 layout sits in deeper water than the plain "
            "7D grid of row 2 and pays more foundation CAPEX. Scaling row 3 by the "
            "AEP ratio silently carries row 3's cost numerator onto row 2's yield."
        ),
    }

    # ── Equivalence gate against the banked 2026-08-18 S3 run ──────────
    gate = {}
    for tag, row in (("row2", r2), ("row3", r3)):
        b = BANKED[tag]
        gate[tag] = {k: {"recomputed": row[k], "banked": b[k],
                         "abs_diff": abs(row[k] - b[k])} for k in b}
    worst = max(v["abs_diff"] for t in gate.values() for v in t.values())
    gate["max_abs_diff"] = worst
    gate["pass"] = worst < 1e-6
    if not gate["pass"]:
        print(f"WARNING: recompute diverges from the banked S3 run "
              f"(max abs diff {worst:.3e}); the numbers below are NOT a "
              f"like-for-like reproduction.")
    else:
        print(f"equivalence gate vs banked S3 run: PASS "
              f"(max abs diff {worst:.3e})")
    item0["equivalence_gate_vs_banked_s3_20260818"] = gate

    # ── ITEM 0b: curve substitution at the organizer baseline centre ────
    bk, bp = by["baseline_grid_kitramp"], by["baseline_grid_published"]
    # The organizers publish a NET CF and a wake loss, not a gross CF. The
    # like-for-like comparison for a gross figure is therefore against their
    # IMPLIED gross, cf_net / (1 - wake), not against their net number. Both
    # framings are reported so the comparison cannot be read the wrong way.
    org_cf_gross_implied = ORGANIZER["cf_net"] / (1.0 - ORGANIZER["wake"])
    gap_kit = bk["cf_gross"] - ORGANIZER["cf_net"]
    gap_pub = bp["cf_gross"] - ORGANIZER["cf_net"]
    gross_gap_kit = bk["cf_gross"] - org_cf_gross_implied
    gross_gap_pub = bp["cf_gross"] - org_cf_gross_implied
    item0b = {
        "question": ("Does substituting the published IEA-22-280-RWT WISDEM curve "
                     "for the kit synthetic ramp close the gross-CF gap against "
                     "the organizers' corrected 53.2 % net baseline?"),
        "centre": {"lat": BASELINE_CENTRE[0], "lon": BASELINE_CENTRE[1]},
        "layout": "plain 7D grid, 55 x IEA 22 MW",
        "organizer_corrected_baseline": ORGANIZER,
        "organizer_correction_source":
            "Codabench board, the organizer, 2026-08-10 (supersedes the CF ~54 % / "
            "wake ~6 % / LCOE ~82 figures)",
        "kit_synthetic_ramp": {
            "rated_power_mw": kit_rated_mw, "rated_wind_speed_ms": kit_rated_ws,
            "cf_gross": bk["cf_gross"], "wake_loss_fraction": bk["wake_loss_fraction"],
            "cf_net": bk["cf_net"], "aep_gwh": bk["aep_gwh"],
            "lcoe_eur_per_mwh": bk["lcoe_eur_per_mwh"],
        },
        "published_wisdem": {
            "rated_power_mw": pub_rated_mw, "rated_wind_speed_ms": pub_rated_ws,
            "cf_gross": bp["cf_gross"], "wake_loss_fraction": bp["wake_loss_fraction"],
            "cf_net": bp["cf_net"], "aep_gwh": bp["aep_gwh"],
            "lcoe_eur_per_mwh": bp["lcoe_eur_per_mwh"],
        },
        "delta_published_minus_kit": {
            "cf_gross_pp": (bp["cf_gross"] - bk["cf_gross"]) * 100,
            "cf_net_pp": (bp["cf_net"] - bk["cf_net"]) * 100,
            "wake_pp": (bp["wake_loss_fraction"] - bk["wake_loss_fraction"]) * 100,
            "aep_gwh": bp["aep_gwh"] - bk["aep_gwh"],
            "lcoe_eur_per_mwh": bp["lcoe_eur_per_mwh"] - bk["lcoe_eur_per_mwh"],
        },
        "organizer_implied_gross_cf": org_cf_gross_implied,
        "gross_cf_gap_vs_organizer_IMPLIED_GROSS_pp": {
            "kit_ramp": gross_gap_kit * 100,
            "published": gross_gap_pub * 100,
            "closed_by_pp": (abs(gross_gap_kit) - abs(gross_gap_pub)) * 100,
            "note": ("like-for-like gross vs gross. Positive = our replica "
                     "produces more gross energy than the organizers' baseline "
                     "implies."),
        },
        "wake_vs_organizer_pp": {
            "kit_ramp": (bk["wake_loss_fraction"] - ORGANIZER["wake"]) * 100,
            "published": (bp["wake_loss_fraction"] - ORGANIZER["wake"]) * 100,
        },
        "gross_cf_gap_vs_organizer_net_532_pp": {
            "kit_ramp": gap_kit * 100,
            "published": gap_pub * 100,
            "closed_by_pp": (gap_kit - gap_pub) * 100,
        },
        "net_cf_gap_vs_organizer_net_532_pp": {
            "kit_ramp": (bk["cf_net"] - ORGANIZER["cf_net"]) * 100,
            "published": (bp["cf_net"] - ORGANIZER["cf_net"]) * 100,
        },
    }

    out = {
        "generated_by": "scripts/audit_lcoe_row2_and_curve_20260827.py",
        "dispatch": "CC dispatch 2026-08-27, items 0 and 0b",
        "wind_source": "real AROME 125 m, nearest pixel, TRAIN_YEARS, sheared to 170 m",
        "data_root": str(DATA_ROOT),
        "bathymetry_raster": str(BATHY_NC),
        "bathymetry_sha256": bathy_sha,
        "cases": rows,
        "item0_lcoe_row2": item0,
        "item0b_published_curve": item0b,
        "wall_clock_s": time.time() - t_start,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\n" + "=" * 72)
    print("ITEM 0")
    print(f"  row 2 recomputed LCOE : {r2['lcoe_eur_per_mwh']:.2f} EUR/MWh "
          f"(report says {REPORT_ROW2_LCOE})")
    print(f"  row 3 recomputed LCOE : {r3['lcoe_eur_per_mwh']:.2f} EUR/MWh "
          f"(report says {REPORT_ROW3_LCOE})")
    print(f"  naive AEP-scaled row 2: {naive_row2:.2f}")
    print(f"  residual              : {residual:+.4f} EUR/MWh")
    print(f"    capex-difference term {d_capex_term:+.4f}")
    print(f"    opex-difference term  {d_opex_term:+.4f}")
    print(f"    sum                   {d_capex_term + d_opex_term:+.4f}")
    print("=" * 72)
    print("ITEM 0b  (organizer baseline centre, plain 7D grid)")
    print(f"  {'':22s} {'kit ramp':>12s} {'published':>12s}")
    print(f"  {'rated ws (m/s)':22s} {kit_rated_ws:12.4f} {pub_rated_ws:12.4f}")
    print(f"  {'gross CF (%)':22s} {bk['cf_gross']*100:12.4f} {bp['cf_gross']*100:12.4f}")
    print(f"  {'wake (%)':22s} {bk['wake_loss_fraction']*100:12.4f} "
          f"{bp['wake_loss_fraction']*100:12.4f}")
    print(f"  {'net CF (%)':22s} {bk['cf_net']*100:12.4f} {bp['cf_net']*100:12.4f}")
    print(f"  {'AEP (GWh)':22s} {bk['aep_gwh']:12.2f} {bp['aep_gwh']:12.2f}")
    print(f"  organizer corrected: net CF 53.2 %, wake 7.1 %, AEP 5,635 GWh")
    print(f"  organizer IMPLIED gross CF = 53.2/(1-0.071) = "
          f"{org_cf_gross_implied*100:.2f} %")
    print(f"  gross vs IMPLIED GROSS : kit {gross_gap_kit*100:+.2f} pp -> "
          f"published {gross_gap_pub*100:+.2f} pp")
    print(f"  gross vs the 53.2 NET  : kit {gap_kit*100:+.2f} pp -> "
          f"published {gap_pub*100:+.2f} pp   (not like-for-like)")
    print(f"  wake vs organizer 7.1 %: kit "
          f"{(bk['wake_loss_fraction']-ORGANIZER['wake'])*100:+.3f} pp -> "
          f"published {(bp['wake_loss_fraction']-ORGANIZER['wake'])*100:+.3f} pp")
    print("=" * 72)
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
