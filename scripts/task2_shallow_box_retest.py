"""Shallow-box feasibility retest, two thresholds, top-20 shallow-centre cells.

Task2 CC prompt (2026-07-13), Task D. The 2026-07-10 shallow-box test
(scripts/task2_shallow_box_feasibility.py) ran on the top-10 cells by pure
wind resource, all of which turned out to sit in 57-78 m water - none
came close to all-shallow under 50 m. This retest instead restricts the
candidate pool up front to cells whose CENTRE is already <=50 m deep
(`data/cell_resource_ranking.csv`, `depth_m<=50`), takes the top 20 of
those by pooled mean ws^3, and tests each one's 15x15 km box at BOTH
50 m (the brief's constraint) and 60 m (kit's own `DEFAULT_MAX_DEPTH_M`,
also the internationally-recognised fixed-bottom / monopile-jacket limit)
so the sensitivity of the "all shallow" test to the still-unresolved
50-vs-60 m discrepancy (Task A.5 of the 2026-07-10 report) is visible.

Same 250 m box grid as the 2026-07-10 script (finer than the requested
<=1 km). Uses scripts/task2_depth_sampler.py (which now sources its
projection from scripts/task2_projection.py, Task E).

Deterministic - static bathymetry read + grid sampling, no stochastic step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import task2_depth_sampler as depth_sampler

RANKING_CSV = REPO_ROOT / "data" / "cell_resource_ranking.csv"
OUT_DIR = REPO_ROOT / "data" / "task2_shallow_box"
OUT_RETEST_CSV = REPO_ROOT / "data" / "task2_shallow_box_retest.csv"

BOX_M = 15_000.0
GRID_SPACING_M = 250.0
CENTRE_DEPTH_LIMIT_M = 50.0   # candidate-pool filter: only shallow-centre cells
THRESHOLDS_M = (50.0, 60.0)
TOP_N = 20
MATERIAL_DIFF_FRAC = 0.05     # >5 percentage points of box area -> "materially different"

PRIORITY_CELLS = (63, 119, 72)


def box_grid(spacing_m: float = GRID_SPACING_M) -> tuple[np.ndarray, np.ndarray]:
    half = BOX_M / 2
    n = int(round(BOX_M / spacing_m)) + 1
    coords = np.linspace(-half, half, n)
    x, y = np.meshgrid(coords, coords)
    return x, y


def box_stats(depth: np.ndarray, threshold_m: float) -> dict:
    finite = np.isfinite(depth)
    shallow = finite & (depth > 0) & (depth <= threshold_m)
    frac_shallow = float(shallow.sum()) / depth.size
    max_depth = float(np.nanmax(depth)) if finite.any() else float("nan")
    frac_land_or_nodata = float((~finite).sum() + (depth <= 0).sum()) / depth.size
    return {
        "mask": shallow,
        "frac_shallow": frac_shallow,
        "max_depth_m": max_depth,
        "frac_land_or_nodata": frac_land_or_nodata,
        "all_shallow": bool(frac_shallow >= 1.0 - 1e-9 and frac_land_or_nodata == 0.0),
    }


def main() -> None:
    df = pd.read_csv(RANKING_CSV)
    shallow_pool = (
        df[df["depth_m"] <= CENTRE_DEPTH_LIMIT_M]
        .sort_values("mean_ws3_pooled", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    shallow_pool["shallow_rank"] = np.arange(1, len(shallow_pool) + 1)
    print(f"{len(df[df['depth_m'] <= CENTRE_DEPTH_LIMIT_M])} cells with centre depth "
          f"<= {CENTRE_DEPTH_LIMIT_M:.0f} m; top {len(shallow_pool)} by mean_ws3_pooled selected")

    missing_priority = set(PRIORITY_CELLS) - set(shallow_pool["cell_id"])
    if missing_priority:
        print(f"WARNING: priority cells not in top-{TOP_N} shallow pool: {missing_priority}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_m, y_m = box_grid()

    # Priority cells first, then the rest in shallow-resource-rank order.
    ordered = pd.concat([
        shallow_pool[shallow_pool["cell_id"].isin(PRIORITY_CELLS)],
        shallow_pool[~shallow_pool["cell_id"].isin(PRIORITY_CELLS)],
    ]).reset_index(drop=True)

    rows = []
    for _, r in ordered.iterrows():
        cell_id = int(r["cell_id"])
        clat, clon = float(r["lat"]), float(r["lon"])
        depth = depth_sampler.depth_at_xy(clat, clon, x_m.ravel(), y_m.ravel()).reshape(x_m.shape)

        stats = {t: box_stats(depth, t) for t in THRESHOLDS_M}

        row = {
            "cell_id": cell_id, "shallow_rank": int(r["shallow_rank"]),
            "global_rank": int(r["rank"]), "lat": clat, "lon": clon,
            "centre_depth_m": depth_sampler.depth_at(clat, clon),
            "mean_ws3_pooled": float(r["mean_ws3_pooled"]),
            "is_priority_cell": cell_id in PRIORITY_CELLS,
            "max_depth_in_box_m": stats[THRESHOLDS_M[0]]["max_depth_m"],  # threshold-independent
        }
        for t in THRESHOLDS_M:
            s = stats[t]
            suffix = f"{int(t)}m"
            row[f"frac_box_le_{suffix}"] = s["frac_shallow"]
            row[f"all_shallow_{suffix}"] = s["all_shallow"]

        diff = abs(stats[60.0]["frac_shallow"] - stats[50.0]["frac_shallow"])
        row["threshold_sensitivity_delta"] = diff
        row["materially_different_60_vs_50"] = bool(diff > MATERIAL_DIFF_FRAC)

        # Save masks for both thresholds (none of these 20 cells were part of
        # the 2026-07-10 top-10 run, so no pre-existing 50 m mask to diff against).
        mask_paths = {}
        for t in THRESHOLDS_M:
            suffix = f"{int(t)}m"
            mask_path = OUT_DIR / f"cell{cell_id:03d}_depth_le{suffix}_mask.npy"
            np.save(mask_path, stats[t]["mask"])
            mask_paths[t] = mask_path
        row["mask_npy_50m"] = str(mask_paths[50.0].relative_to(REPO_ROOT))
        row["mask_npy_60m"] = str(mask_paths[60.0].relative_to(REPO_ROOT))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, t in zip(axes, THRESHOLDS_M):
            im = ax.pcolormesh(x_m / 1000, y_m / 1000, depth, cmap="Blues", vmin=0, vmax=100, shading="auto")
            ax.contour(x_m / 1000, y_m / 1000, depth, levels=[t], colors="crimson", linewidths=2)
            ax.set_aspect("equal")
            ax.set_xlabel("x (km)")
            ax.set_title(f"<= {t:.0f} m: {stats[t]['frac_shallow']*100:.0f}% of box")
        axes[0].set_ylabel("y (km)")
        plt.colorbar(im, ax=axes, label="depth (m)", shrink=0.8)
        fig.suptitle(f"cell {cell_id} ({clat:.2f}N,{clon:.2f}E), centre depth "
                     f"{row['centre_depth_m']:.1f} m")
        png_path = OUT_DIR / f"cell{cell_id:03d}_depth_box_retest.png"
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        row["plot_png"] = str(png_path.relative_to(REPO_ROOT))

        rows.append(row)
        print(f"cell {cell_id} (shallow_rank {row['shallow_rank']}, centre {row['centre_depth_m']:.1f}m): "
              f"50m={stats[50.0]['frac_shallow']*100:.1f}%  60m={stats[60.0]['frac_shallow']*100:.1f}%  "
              f"max_depth_in_box={stats[50.0]['max_depth_m']:.1f}m"
              + ("  <-- PRIORITY" if cell_id in PRIORITY_CELLS else ""))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_RETEST_CSV, index=False)
    print(f"\nwrote {OUT_RETEST_CSV} ({len(out_df)} rows)")
    n_all_shallow_50 = int(out_df["all_shallow_50m"].sum())
    n_all_shallow_60 = int(out_df["all_shallow_60m"].sum())
    print(f"all_shallow at 50m: {n_all_shallow_50}/{len(out_df)}   "
          f"all_shallow at 60m: {n_all_shallow_60}/{len(out_df)}")


if __name__ == "__main__":
    main()
