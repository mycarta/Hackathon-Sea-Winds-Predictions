"""Shallow-box feasibility + depth-mask export for the top-10 resource cells.

Task2 CC prompt (2026-07-10), Task C.5-C.6. For each of the top-10 cells in
data/cell_resource_ranking.csv (by mean_ws3_pooled), test whether a 15x15 km
box centred on the cell fits entirely within depth <= 50 m, using
scripts/task2_depth_sampler.py on a 250 m grid (finer than the requested
<=1 km, still cheap: 61x61 = 3721 points/box). Reports the fraction of the
box under 50 m and the max depth in the box (unscored diagnostic - an
implied foundation-cost win if the whole box is shallow). Also exports the
boolean <=50 m mask in the box's local (x_m, y_m) layout coordinates as
.npy + a PNG, for later use as a per-turbine placement constraint.

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
OUT_SUMMARY_CSV = REPO_ROOT / "data" / "task2_shallow_box_summary.csv"

BOX_M = 15_000.0
GRID_SPACING_M = 250.0
DEPTH_THRESHOLD_M = 50.0
TOP_N = 10


def box_grid(spacing_m: float = GRID_SPACING_M) -> tuple[np.ndarray, np.ndarray]:
    half = BOX_M / 2
    n = int(round(BOX_M / spacing_m)) + 1
    coords = np.linspace(-half, half, n)
    x, y = np.meshgrid(coords, coords)
    return x, y


def main() -> None:
    df = pd.read_csv(RANKING_CSV).sort_values("mean_ws3_pooled", ascending=False).head(TOP_N)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x_m, y_m = box_grid()
    rows = []
    for _, r in df.iterrows():
        cell_id = int(r["cell_id"])
        clat, clon = float(r["lat"]), float(r["lon"])
        depth = depth_sampler.depth_at_xy(clat, clon, x_m.ravel(), y_m.ravel()).reshape(x_m.shape)

        finite = np.isfinite(depth)
        shallow = finite & (depth > 0) & (depth <= DEPTH_THRESHOLD_M)
        frac_shallow = float(shallow.sum()) / depth.size
        max_depth = float(np.nanmax(depth)) if finite.any() else float("nan")
        min_depth = float(np.nanmin(depth[depth > 0])) if (finite & (depth > 0)).any() else float("nan")
        frac_land_or_nodata = float((~finite).sum() + (depth <= 0).sum()) / depth.size

        mask_path = OUT_DIR / f"cell{cell_id:03d}_depth_le50_mask.npy"
        np.save(mask_path, shallow)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.pcolormesh(x_m / 1000, y_m / 1000, depth, cmap="Blues", vmin=0, vmax=80, shading="auto")
        ax.contour(x_m / 1000, y_m / 1000, depth, levels=[DEPTH_THRESHOLD_M], colors="crimson", linewidths=2)
        plt.colorbar(im, ax=ax, label="depth (m)")
        ax.set_aspect("equal")
        ax.set_xlabel("x (km, east of centre)")
        ax.set_ylabel("y (km, north of centre)")
        ax.set_title(f"cell {cell_id} ({clat:.2f}N,{clon:.2f}E)\n"
                     f"{frac_shallow*100:.0f}% of box <= {DEPTH_THRESHOLD_M:.0f} m")
        fig.tight_layout()
        png_path = OUT_DIR / f"cell{cell_id:03d}_depth_box.png"
        fig.savefig(png_path, dpi=140)
        plt.close(fig)

        rows.append({
            "cell_id": cell_id, "rank": int(r["rank"]), "lat": clat, "lon": clon,
            "centre_depth_m": depth_sampler.depth_at(clat, clon),
            "frac_box_le_50m": frac_shallow,
            "max_depth_in_box_m": max_depth,
            "min_depth_in_box_m": min_depth,
            "frac_box_land_or_nodata": frac_land_or_nodata,
            "all_shallow": bool(frac_shallow >= 1.0 - 1e-9 and frac_land_or_nodata == 0.0),
            "mask_npy": str(mask_path.relative_to(REPO_ROOT)),
            "plot_png": str(png_path.relative_to(REPO_ROOT)),
        })
        print(f"cell {cell_id}: {frac_shallow*100:.1f}% box <=50m, max depth {max_depth:.1f} m, "
              f"land/nodata {frac_land_or_nodata*100:.1f}%")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"wrote {OUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
