"""Rank the 159 Task-2 eligible siting cells by AROME wind resource, 2016-2020.

Task2 CC prompt (2026-07-10), Task C. Per eligible cell (zone.py's target n
reanalysis n sea domain, 0.25 deg reanalysis grid, confirmed 159 cells in
scripts/task2_bathymetry_map.py / reports/task2_scorer_bathymetry_ranking_*.md
Task A5):

    - mean(ws^3) pooled over all AROME 125 m pixel/timestep samples that fall
      inside the cell (2016-2020, all 8 daily target hours) - the primary,
      energy-proportional ranking metric.
    - mean(ws^3) and mean(ws) per year (5 values each), for cross-year
      stability screening.
    - a pooled 16-sector (22.5 deg) direction-probability histogram, for
      later layout-orientation work. NOTE this is a different sector width
      than the simulator's own 12-sector (30 deg) TI convention in
      wind_farm_simulator.py::N_DIRECTION_SECTORS - both are legitimate, for
      different purposes; do not conflate them.

Spatial aggregation choice (undocumented in the CC prompt, made explicit
here): a reanalysis cell typically contains several AROME pixels. We treat
every (AROME sea pixel, timestep) pair inside the cell as one independent
resource sample and average ws / ws^3 over ALL of them - i.e. mean-of-cube,
not cube-of-the-cell-mean-wind. This preserves the sub-cell wind-speed
variance in the energy metric (standard wind-resource-assessment practice)
and is deliberately NOT the same operation as coarsen.coarsen_uv (vector
mean of u,v used elsewhere in the kit for coarsening a *field*, e.g. in
zone.py's sea-fraction calc) - that would average away variance before
cubing and understate the energy metric.

Pixel-to-cell membership reuses coarsen.py's own nearest-reanalysis-centre
rule (part0_dataset_setup/coarsen.py::_cell_ids) for consistency with how
zone.py defines the 159 cells. Land AROME pixels (arome_static.seamask==0)
are excluded before aggregation.

Deterministic - full-corpus aggregation, no stochastic step, no seed needed.
Long-running (~1826 daily files); logs progress every 200 days.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
DATA_ROOT = REPO_ROOT / "phase_2" / "phase2_dataset_ship"

sys.path.insert(0, str(KIT_PHASE2))
sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
os.environ.setdefault("PHASE2_DATA_ROOT", str(DATA_ROOT))

OUT_RANKING_CSV = REPO_ROOT / "data" / "cell_resource_ranking.csv"
OUT_ROSE_CSV = REPO_ROOT / "data" / "cell_direction_rose_16sector.csv"

N_SECTORS = 16
SECTOR_WIDTH = 360.0 / N_SECTORS
YEARS = (2016, 2017, 2018, 2019, 2020)


def main() -> None:
    t0 = time.time()
    os.chdir(KIT_PHASE2)
    import zone
    import coarsen
    import target_loader
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import task2_depth_sampler as depth_sampler

    # ── 159 eligible cells (reanalysis 0.25 deg grid) ──────────────────────
    lats, lons, allowed = zone._grid(max_depth_m=None)
    nlat, nlon = allowed.shape
    ii, jj = np.where(allowed)
    n_cells = len(ii)
    assert n_cells == 159, f"expected 159 allowed cells, got {n_cells}"
    cell_lat = lats[ii]
    cell_lon = lons[jj]
    allowed_flat_ids = ii * nlon + jj
    flat_to_local = {int(fid): k for k, fid in enumerate(allowed_flat_ids)}

    # ── AROME static grid: pixel -> reanalysis-cell membership, sea mask ───
    static = target_loader.load_static()
    target_lat, target_lon = static.lat, static.lon
    sea = static.sea

    cell_id_flat, valid, _nlat_ra, _nlon_ra = coarsen._cell_ids(
        target_lat, target_lon, lats, lons
    )

    # Vectorized membership: local cell index per AROME pixel, -1 if not in
    # one of the 159 allowed cells or not sea.
    lut = np.full(nlat * nlon, -1, dtype=np.int64)
    for fid, local in flat_to_local.items():
        lut[fid] = local
    pixel_local_cell = np.where(valid, lut[np.clip(cell_id_flat, 0, nlat * nlon - 1)], -1)
    pixel_local_cell = pixel_local_cell.reshape(target_lat.shape)
    pixel_local_cell = np.where(sea, pixel_local_cell, -1)

    sel_mask = pixel_local_cell >= 0
    n_sel = int(sel_mask.sum())
    print(f"AROME grid {target_lat.shape}, {n_sel} sea pixels fall inside the 159 cells")
    sel_local_idx = pixel_local_cell[sel_mask]  # (n_sel,)
    pixel_counts = np.bincount(sel_local_idx, minlength=n_cells)  # per-cell pixel count

    # ── Accumulators ────────────────────────────────────────────────────
    pooled_sum_ws = np.zeros(n_cells)
    pooled_sum_ws3 = np.zeros(n_cells)
    year_sum_ws = {y: np.zeros(n_cells) for y in YEARS}
    year_sum_ws3 = {y: np.zeros(n_cells) for y in YEARS}
    year_n_timesteps = {y: 0 for y in YEARS}
    sector_count = np.zeros((n_cells, N_SECTORS), dtype=np.int64)

    dates = [d for d in target_loader.list_dates() if d.year in YEARS]
    dates.sort()
    print(f"{len(dates)} AROME daily files to process, years {YEARS}")

    for k, d in enumerate(dates):
        day = target_loader.load_day(d, levels=("125m",))
        u = day.u["125m"][:, sel_mask]   # (8, n_sel)
        v = day.v["125m"][:, sel_mask]   # (8, n_sel)
        n_t = u.shape[0]
        ws = np.sqrt(u * u + v * v)
        ws3 = ws ** 3
        wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        sector = np.minimum((wd / SECTOR_WIDTH).astype(np.int64), N_SECTORS - 1)

        for t in range(n_t):
            finite = np.isfinite(ws[t])
            idx_t = sel_local_idx[finite]
            pooled_sum_ws += np.bincount(idx_t, weights=ws[t][finite], minlength=n_cells)
            pooled_sum_ws3 += np.bincount(idx_t, weights=ws3[t][finite], minlength=n_cells)
            year_sum_ws[d.year] += np.bincount(idx_t, weights=ws[t][finite], minlength=n_cells)
            year_sum_ws3[d.year] += np.bincount(idx_t, weights=ws3[t][finite], minlength=n_cells)
            combined = idx_t * N_SECTORS + sector[t][finite]
            sc = np.bincount(combined, minlength=n_cells * N_SECTORS)
            sector_count += sc.reshape(n_cells, N_SECTORS)
            year_n_timesteps[d.year] += 1

        if (k + 1) % 200 == 0 or k == len(dates) - 1:
            print(f"  [{k+1}/{len(dates)}] {d}  elapsed={time.time()-t0:.0f}s")

    pooled_n = pixel_counts * sum(year_n_timesteps.values())
    mean_ws_pooled = pooled_sum_ws / np.maximum(pooled_n, 1)
    mean_ws3_pooled = pooled_sum_ws3 / np.maximum(pooled_n, 1)

    rows = []
    for c in range(n_cells):
        row = {
            "cell_id": c,
            "lat": float(cell_lat[c]),
            "lon": float(cell_lon[c]),
            "depth_m": depth_sampler.depth_at(float(cell_lat[c]), float(cell_lon[c])),
            "n_arome_pixels": int(pixel_counts[c]),
            "mean_ws3_pooled": float(mean_ws3_pooled[c]),
            "mean_ws_pooled": float(mean_ws_pooled[c]),
        }
        for y in YEARS:
            n_y = pixel_counts[c] * year_n_timesteps[y]
            row[f"mean_ws3_{y}"] = float(year_sum_ws3[y][c] / max(n_y, 1))
            row[f"mean_ws_{y}"] = float(year_sum_ws[y][c] / max(n_y, 1))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("mean_ws3_pooled", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    OUT_RANKING_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_RANKING_CSV, index=False)
    print(f"wrote {OUT_RANKING_CSV} ({len(df)} rows)")

    rose_rows = []
    for c in range(n_cells):
        total = sector_count[c].sum()
        probs = sector_count[c] / max(total, 1)
        rose_rows.append({"cell_id": c, "lat": float(cell_lat[c]), "lon": float(cell_lon[c]),
                          **{f"sector_{i}_{int(i*SECTOR_WIDTH)}deg": float(probs[i]) for i in range(N_SECTORS)}})
    rose_df = pd.DataFrame(rose_rows)
    rose_df.to_csv(OUT_ROSE_CSV, index=False)
    print(f"wrote {OUT_ROSE_CSV} ({len(rose_df)} rows)")
    print(f"total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
