"""Bathymetry contour map for Task 2 siting groundwork.

Task2 CC prompt (2026-07-10), Task B.1. Contours at 30/40/50/60/70 m
(50 m emphasized) over the EMODnet grid, restricted to the requested zone
intersected with the file's actual coverage (lat 50.00-56.50 N, lon
-5.00-9.50 E - the prompt's 51-60N window exceeds the file; we plot what
exists and say so on the figure). Overlays: the 159 eligible siting cells
(zone.py, target n reanalysis n sea domain, 0.25 deg reanalysis grid) and the
kit baseline centre (53.5N, 1.5E). No bank names are printed - EMODnet ships
no name field and we did not cross-reference an external gazetteer, per the
prompt's "no guessed names" instruction.

Deterministic (static NetCDF read, matplotlib render) - no stochastic step.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
EMODNET_PATH = (
    REPO_ROOT / "phase_2" / "phase2_dataset_ship" / "static" / "bathymetry"
    / "emodnet_northsea_1km.nc"
)
OUT_PNG = REPO_ROOT / "data" / "task2_bathymetry_contour_map.png"
OUT_CELLS_CSV = REPO_ROOT / "data" / "task2_allowed_159_cells.csv"

BASELINE_LAT, BASELINE_LON = 53.5, 1.5
REQUESTED_LAT = (51.0, 60.0)
REQUESTED_LON = (-3.5, 6.0)
NORTH_SEA_LON_EDGE = -2.0  # footprint.py LON_EAST_OF


def get_allowed_cells() -> pd.DataFrame:
    """The 159 target n reanalysis n sea cells, via the kit's own zone.py."""
    sys.path.insert(0, str(KIT_PHASE2))
    sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
    os.environ.setdefault("PHASE2_DATA_ROOT", str(REPO_ROOT / "phase_2" / "phase2_dataset_ship"))
    cwd = os.getcwd()
    os.chdir(KIT_PHASE2)
    try:
        import zone
        lats, lons, allowed = zone._grid(max_depth_m=None)
    finally:
        os.chdir(cwd)
    assert allowed.sum() == 159, f"expected 159 allowed cells, got {allowed.sum()}"
    ii, jj = np.where(allowed)
    return pd.DataFrame({"cell_id": np.arange(len(ii)), "lat": lats[ii], "lon": lons[jj]})


def main() -> None:
    ds = xr.load_dataset(EMODNET_PATH)
    file_lat = (float(ds.lat.min()), float(ds.lat.max()))
    file_lon = (float(ds.lon.min()), float(ds.lon.max()))
    plot_lat = (max(REQUESTED_LAT[0], file_lat[0]), min(REQUESTED_LAT[1], file_lat[1]))
    plot_lon = (max(REQUESTED_LON[0], file_lon[0]), min(REQUESTED_LON[1], file_lon[1]))

    sub = ds.sel(lat=slice(*plot_lat), lon=slice(*plot_lon))
    depth = sub["water_depth_m"].where(sub["water_depth_m"] > 0)  # sea only

    cells = get_allowed_cells()
    OUT_CELLS_CSV.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT_CELLS_CSV, index=False)

    fig, ax = plt.subplots(figsize=(11, 9))
    levels = [30, 40, 50, 60, 70]
    cs = ax.contour(
        sub.lon, sub.lat, depth, levels=levels, colors="0.4", linewidths=0.9,
    )
    ax.clabel(cs, levels, fmt="%d m", fontsize=8)
    cs50 = ax.contour(sub.lon, sub.lat, depth, levels=[50], colors="crimson", linewidths=2.2)
    ax.clabel(cs50, [50], fmt="%d m", fontsize=9, colors="crimson")
    im = ax.pcolormesh(sub.lon, sub.lat, depth, cmap="Blues", vmin=0, vmax=100, shading="auto", alpha=0.55)
    plt.colorbar(im, ax=ax, label="water depth (m)", shrink=0.8)

    ax.scatter(
        cells["lon"], cells["lat"], marker="s", s=28, facecolor="none",
        edgecolor="black", linewidth=0.8, label=f"159 eligible siting cells (0.25° grid)",
    )
    ax.scatter(
        [BASELINE_LON], [BASELINE_LAT], marker="*", s=260, color="gold",
        edgecolor="black", linewidth=1.0, zorder=5,
        label=f"kit baseline centre ({BASELINE_LAT}N, {BASELINE_LON}E)",
    )
    ax.axvline(NORTH_SEA_LON_EDGE, color="tab:orange", ls="--", lw=1.2,
               label=f"North-Sea-side edge (lon ≥ {NORTH_SEA_LON_EDGE}°, footprint.py)")

    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.set_title(
        "EMODnet water depth, North Sea — 30/40/50(bold)/60/70 m contours\n"
        f"plotted extent lat {plot_lat[0]:.1f}–{plot_lat[1]:.1f}°N "
        f"(EMODnet file covers only to {file_lat[1]:.1f}°N; requested {REQUESTED_LAT[1]:.0f}°N not available)"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_aspect(1.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CELLS_CSV} ({len(cells)} rows)")


if __name__ == "__main__":
    main()
