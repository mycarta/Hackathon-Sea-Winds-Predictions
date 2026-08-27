"""Task 2 Step 3: robustness + full-series validation of the Stage 1-3 winners.

Task2 CC prompt (2026-07-13 afternoon), "Robustness check" + "Validation"
sections. Loads the same cell-63 real AROME series as
`task2_layout_search.py` (deterministic - identical series both times) and
runs the WELL-TIE-VERIFIED `task2_scorer_replica.run_case` (full per-
timestep PyWake simulation, not the fast binned rose) on each of the three
Stage 1/2/3 winning layouts, both pooled (2016-2020) and per year, so the
selection is made on real numbers, not the fast screen.

**Added rigor step, not in the original prompt but directly motivated by
what this script found:** the full-series validation REVERSED the fast-
screen ranking (Stage 3 - fast-screen's WORST candidate at 46.8% - came
out BEST for real, 47.2% vs Stage 1's real 43.4%; Stage 1's fast-screen
49.1% shrank to a real 43.4%, i.e. the fast screen's error was not just
noise, it inverted the stage ordering). That single result already proves
the fast 16x5-bin rose is not a reliable fine-grained ranker for a
REGULAR grid's wake loss (coarse direction bins interact very differently
with a grid's sharp row/column alignment angles than with an irregular
perimeter layout - see the report for the full mechanism). Given that,
trusting the fast screen's #1 pick INSIDE Stage 1's own 132 candidates is
equally suspect - so before finalising, we also re-validate the top 10
Stage-1 fast-screen candidates on the full series (cheap: ~2s x 10 = 20s)
to check whether a different grid config/orientation is the true best
grid-type candidate. This does not re-run Stage 2/3 variations or Stage 4
- scope stays bounded to "is Stage 1's own pick actually its best".

Selects the final layout by pooled CF, tie-broken toward the higher
worst-year CF if within 0.5 CF point of each other (stability
preference, per the prompt: "We select on stability, not just pooled
mean").

Deterministic - no stochastic step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import task2_scorer_replica as rep

CANDIDATES_CSV = REPO_ROOT / "data" / "task2_layout_candidates.csv"
OUT_ROBUSTNESS_CSV = REPO_ROOT / "data" / "task2_layout_robustness.csv"
OUT_WINNER_JSON = REPO_ROOT / "data" / "task2_layout_winner.json"
OUT_FINAL_VALIDATION_CSV = REPO_ROOT / "data" / "task2_layout_final_validation.csv"

CENTRE_LAT, CENTRE_LON = 52.50, 3.00
YEARS = (2016, 2017, 2018, 2019, 2020)
STABILITY_MARGIN_PP = 0.5   # prefer higher worst-year CF within this pooled-CF margin


TOP_N_STAGE1_RECHECK = 10


def load_candidates_df() -> pd.DataFrame:
    df = pd.read_csv(CANDIDATES_CSV)
    df["x_m"] = df["x_m"].apply(json.loads)
    df["y_m"] = df["y_m"].apply(json.loads)
    return df


def load_top_stage1_candidates(df: pd.DataFrame, n: int = TOP_N_STAGE1_RECHECK) -> dict:
    sub = df[(df["stage"] == "1_grid_orientation") & (df["valid"])]
    top = sub.sort_values("capacity_factor", ascending=False).head(n)
    out = {}
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        label = f"stage1_rank{rank}_{row['config']}_o{row['orientation_deg']:.0f}"
        out[label] = {
            "config": row["config"], "orientation_deg": float(row["orientation_deg"]),
            "x_m": np.array(row["x_m"]), "y_m": np.array(row["y_m"]),
            "fast_cf": float(row["capacity_factor"]), "fast_wake_loss": float(row["wake_loss_fraction"]),
            "fast_rank": rank,
        }
    return out


def load_stage_winner_layouts(df: pd.DataFrame) -> dict:
    layouts = {}
    for stage, label in [("1_grid_orientation", "stage1_oriented_grid"),
                         ("2_staggered", "stage2_staggered"),
                         ("3_boundary_loaded", "stage3_boundary_loaded")]:
        sub = df[(df["stage"] == stage) & (df["valid"])]
        if sub.empty:
            print(f"WARNING: no valid candidate for {stage}, skipping")
            continue
        if stage == "1_grid_orientation":
            row = sub.loc[sub["capacity_factor"].idxmax()]
        else:
            row = sub.iloc[0]
        layouts[label] = {
            "config": row["config"], "orientation_deg": float(row["orientation_deg"]),
            "x_m": np.array(row["x_m"]), "y_m": np.array(row["y_m"]),
            "fast_cf": float(row["capacity_factor"]), "fast_wake_loss": float(row["wake_loss_fraction"]),
        }
    return layouts


def main() -> None:
    print("Loading cell-63 AROME wind (2016-2020, nearest pixel) - identical series as task2_layout_search.py...")
    times, ws_hub, wd = rep.load_arome_series(CENTRE_LAT, CENTRE_LON)
    years_arr = pd.to_datetime(times).year.to_numpy()

    candidates_df = load_candidates_df()
    layouts = load_stage_winner_layouts(candidates_df)
    turbine = rep.build_turbine()

    # ── Added rigor: re-check whether Stage 1's fast-screen #1 pick is
    # actually the best REAL candidate among Stage 1's own 132 valid grids
    # (motivated by the Stage 1-vs-3 reversal found below on first run -
    # see module docstring). Pooled-only (no per-year) for these extra
    # candidates - cheap triage, not full robustness treatment.
    print(f"\n=== Re-checking top {TOP_N_STAGE1_RECHECK} Stage-1 fast-screen candidates on the full series ===")
    top_stage1 = load_top_stage1_candidates(candidates_df)
    recheck_rows = []
    for label, lay in top_stage1.items():
        res = rep.run_case(label, times, ws_hub, wd, turbine, lay["x_m"], lay["y_m"])
        recheck_rows.append({"label": label, "fast_rank": lay["fast_rank"], "config": lay["config"],
                            "orientation_deg": lay["orientation_deg"], "fast_cf": lay["fast_cf"],
                            **res})
    recheck_df = pd.DataFrame(recheck_rows).sort_values("capacity_factor", ascending=False)
    print(recheck_df[["fast_rank", "config", "orientation_deg", "fast_cf", "capacity_factor",
                      "wake_loss_fraction"]].to_string(index=False))
    true_best_stage1 = recheck_df.iloc[0]
    fast_pick_real_cf = recheck_df[recheck_df["fast_rank"] == 1]["capacity_factor"].iloc[0]
    if true_best_stage1["fast_rank"] != 1:
        print(f"\nFast screen's #1 pick is NOT the real-CF best among the top {TOP_N_STAGE1_RECHECK}: "
              f"fast #1 gets real CF={fast_pick_real_cf*100:.2f}%, but fast-rank "
              f"{int(true_best_stage1['fast_rank'])} gets real CF={true_best_stage1['capacity_factor']*100:.2f}%. "
              f"Using the real-CF best as 'stage1_oriented_grid' going forward.")
        layouts["stage1_oriented_grid"] = {
            "config": true_best_stage1["config"], "orientation_deg": float(true_best_stage1["orientation_deg"]),
            "x_m": top_stage1[true_best_stage1["label"]]["x_m"],
            "y_m": top_stage1[true_best_stage1["label"]]["y_m"],
            "fast_cf": float(true_best_stage1["fast_cf"]),
            "fast_wake_loss": float(recheck_df.iloc[0]["wake_loss_fraction"]),
        }
    else:
        print(f"\nFast screen's #1 pick is confirmed the real-CF best among the top {TOP_N_STAGE1_RECHECK} "
              f"(real CF={fast_pick_real_cf*100:.2f}%). No change.")
    recheck_df.to_csv(REPO_ROOT / "data" / "task2_layout_stage1_recheck.csv", index=False)
    print(f"wrote {REPO_ROOT / 'data' / 'task2_layout_stage1_recheck.csv'}")

    rows = []
    pooled_results = {}
    for label, lay in layouts.items():
        x, y = lay["x_m"], lay["y_m"]
        print(f"\n=== {label} (config={lay['config']}, orientation={lay['orientation_deg']:.0f} deg) ===")
        pooled = rep.run_case(f"{label}_pooled", times, ws_hub, wd, turbine, x, y)
        pooled_results[label] = pooled
        rows.append({"label": label, "period": "pooled_2016_2020", **pooled,
                    "fast_cf": lay["fast_cf"], "fast_wake_loss": lay["fast_wake_loss"]})
        year_cfs = []
        for yr in YEARS:
            m = years_arr == yr
            yr_res = rep.run_case(f"{label}_{yr}", times[m], ws_hub[m], wd[m], turbine, x, y)
            rows.append({"label": label, "period": str(yr), **yr_res,
                        "fast_cf": np.nan, "fast_wake_loss": np.nan})
            year_cfs.append(yr_res["capacity_factor"])
        year_cfs = np.array(year_cfs)
        print(f"  per-year CF: {[f'{c*100:.1f}%' for c in year_cfs]}  "
              f"worst={year_cfs.min()*100:.1f}%  spread={100*(year_cfs.max()-year_cfs.min()):.1f}pp")

    robustness_df = pd.DataFrame(rows)
    robustness_df.to_csv(OUT_ROBUSTNESS_CSV, index=False)
    print(f"\nwrote {OUT_ROBUSTNESS_CSV}")

    # ── Select final winner: pooled CF, tie-broken toward stability ────
    pooled_cf = {k: v["capacity_factor"] for k, v in pooled_results.items()}
    best_label = max(pooled_cf, key=pooled_cf.get)
    close = [k for k in pooled_cf if pooled_cf[best_label] - pooled_cf[k] <= STABILITY_MARGIN_PP / 100]
    if len(close) > 1:
        worst_year_cf = {}
        for k in close:
            yr_rows = robustness_df[(robustness_df["label"] == k) & (robustness_df["period"] != "pooled_2016_2020")]
            worst_year_cf[k] = yr_rows["capacity_factor"].min()
        best_label = max(worst_year_cf, key=worst_year_cf.get)
        print(f"\nTie within {STABILITY_MARGIN_PP}pp on pooled CF among {close}; "
              f"selected by worst-year CF: {best_label}")

    winner = layouts[best_label]
    winner_result = pooled_results[best_label]
    print(f"\n=== FINAL WINNER: {best_label} ===")
    print(f"  config={winner['config']} orientation={winner['orientation_deg']:.0f} deg")
    print(f"  full-series CF={winner_result['capacity_factor']*100:.2f}%  "
          f"AEP={winner_result['aep_gwh']:.0f} GWh  wake_loss={winner_result['wake_loss_fraction']*100:.2f}%")

    stage1_pooled = pooled_results.get("stage1_oriented_grid")
    sanity_cf_ok = 0.45 <= winner_result["capacity_factor"] <= 0.55
    sanity_wake_ok = (stage1_pooled is None or best_label == "stage1_oriented_grid"
                      or winner_result["wake_loss_fraction"] <= stage1_pooled["wake_loss_fraction"] + 1e-9)
    print(f"  sanity: CF in [45,55]% -> {sanity_cf_ok}   "
          f"wake_loss <= Stage 1 baseline -> {sanity_wake_ok}")

    submission = {
        "team": "cc_task2_layout_search",
        "farm_centre_lat": CENTRE_LAT,
        "farm_centre_lon": CENTRE_LON,
        "turbine_key": "IEA_22MW",
        "layout_x_m": winner["x_m"].round(2).tolist(),
        "layout_y_m": winner["y_m"].round(2).tolist(),
        "source_stage": best_label,
        "source_config": winner["config"],
        "orientation_deg": winner["orientation_deg"],
        "reported": {
            "aep_gwh": round(float(winner_result["aep_gwh"]), 2),
            "capacity_factor": round(float(winner_result["capacity_factor"]), 4),
            "wake_loss_fraction": round(float(winner_result["wake_loss_fraction"]), 4),
            "n_steps": int(winner_result["n_steps"]),
            "wind_source": "real AROME 2016-2020, nearest pixel to (52.50N,3.00E), 125m->170m shear",
        },
    }
    with open(OUT_WINNER_JSON, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"wrote {OUT_WINNER_JSON}")

    final_df = pd.DataFrame([{"label": best_label, **winner_result,
                             "sanity_cf_in_45_55": sanity_cf_ok, "sanity_wake_le_stage1": sanity_wake_ok}])
    final_df.to_csv(OUT_FINAL_VALIDATION_CSV, index=False)
    print(f"wrote {OUT_FINAL_VALIDATION_CSV}")


if __name__ == "__main__":
    main()
