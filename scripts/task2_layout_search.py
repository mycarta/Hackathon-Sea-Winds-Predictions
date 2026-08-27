"""Task 2 Step 3: layout optimization at cell 63, Stages 1-3.

Task2 CC prompt (2026-07-13 afternoon), "Layout optimization at cell 63".
Siting locked: 52.50N, 3.00E (cell 63, rank 12 globally, rank 1 among
depth<=50m centre cells, entire 15x15km box <=50m - no depth mask needed,
see reports/task2_retest_welltie_20260713.md Task D).

Reuses the WELL-TIE-VERIFIED physics from scripts/task2_scorer_replica.py
(build_turbine, build_wake_model, derive_ti_per_sector - unchanged, that
script is not modified here) but adds a probability-weighted "16-sector
binned rose" fast evaluator for the Stage 1-3 sweep (hundreds of
candidates - too many for a full 14,608-step replica run each). Full
per-timestep validation happens in scripts/task2_layout_validate.py.

**Fast-evaluator design, documented (undocumented in the prompt beyond
"16-sector binned rose"):** initially tried ONE representative sample per
22.5-degree sector using the energy-equivalent speed cbrt(mean(ws^3))
(matching the Task C mean-ws^3 philosophy) - this was wrong and caught by
a sanity check: it gave CF=80.8%, physically implausible (offshore CF
rarely exceeds ~60%, and the well-tie's real-data ceiling anywhere was
~50%). Root cause: the IEA 22 MW curve SATURATES at rated power above
~12 m/s (`data/iea22mw_power_ct.csv`); the cube-root-of-cube-mean speed
sits above 12 m/s for the dominant SW sectors (pooled cbrt(mean ws^3) =
12.4 m/s vs pooled arithmetic mean 10.2 m/s, Task C data) purely from
distribution variance (power-mean inequality), which then pushes a
single representative sample per sector into rated/saturated output far
more often than the real distribution would - a single point cannot
represent a saturating nonlinearity. Fixed by binning each sector's
REAL wind-speed sample population into 5 equal-probability quantile bins
(so both below-rated and at-rated conditions are represented, each with
correct probability mass) instead of one point - 16 sectors x 5 bins = 80
representative (ws, wd, P) samples, still ~180x fewer than the full
14,608-step series. AEP is the probability-weighted average farm power x
8760 h, not a trapezoidal time integral (there is no real time axis for
a rose/quantile table).

**TI is a non-issue for this evaluator, verified empirically**: with
`BastankhahGaussianDeficit`'s k fixed at 0.0324555 (Task A.1, 2026-07-10
report) and no turbulenceModel active, simulated power is bit-identical
across TI in [0.03, 0.18] (checked directly: same total power to 12
significant figures). We still compute a real per-sector TI (from the
pinned 12-sector Charnock formula, applied to the full multi-year series)
for construction completeness, but its value cannot affect any result
here.

Box is FIXED axis-aligned (|x_m|,|y_m| <= 7500 m) per
`wind_farm_simulator.validate_layout`'s own convention - "orientation"
rotates the TURBINE PATTERN placed inside it, not the box itself.

Deterministic - no random step anywhere in Stages 1-3 (pure grid
enumeration + rotation + geometric constraint checks).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from py_wake.site.xrsite import XRSite

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import task2_scorer_replica as rep

CENTRE_LAT, CENTRE_LON = 52.50, 3.00
N_TURBINES = 55
DIAMETER_M = rep.DIAMETER_M
MIN_SPACING_D = 5.0
MIN_SPACING_M = MIN_SPACING_D * DIAMETER_M   # 1420 m
BOX_HALF_M = 7500.0

# QA fix (2026-07-13, "Task 2 QA continued: spread-grid + spacing fix"):
# the Stage 3 interior sub-grid was built at exactly MIN_SPACING_M=1420m,
# which (after task2_layout_validate.py's .round(2) JSON export) rounded
# 6 of its pairwise distances down to 1419.993m - 7mm under the 1420m
# floor (reports/task2_qa_verification_20260713.md, Check 1). Pure
# rounding artifact (rotation preserves distance exactly; only the 2-
# decimal serialization ate the margin), but a 1m construction margin
# makes the export robust to that rounding regardless of orientation -
# cheaper than requiring every downstream consumer to round more
# precisely. Perimeter spacing (1500m, unaffected) and the box/spacing
# validity floor (MIN_SPACING_M=1420m, the hard 5D requirement) are
# unchanged.
INTERIOR_SPACING_M = MIN_SPACING_M + 1.0     # 1421 m
N_ROSE_SECTORS = 16
ORIENTATIONS_DEG = np.arange(0, 180, 10)     # 0..170 step 10 (18 values)

OUT_CANDIDATES_CSV = REPO_ROOT / "data" / "task2_layout_candidates.csv"
OUT_FIGURE = REPO_ROOT / "data" / "task2_layout_cf_vs_orientation.png"
OUT_STAGE_WINNERS_JSON = REPO_ROOT / "data" / "task2_layout_stage_winners.json"


# ── Fast 16-sector x 5-speed-bin rose (built once from the full cell-63 series) ─

N_SPEED_BINS = 5


def build_rose(ws_hub: np.ndarray, wd: np.ndarray, n_sectors: int = N_ROSE_SECTORS,
               n_speed_bins: int = N_SPEED_BINS):
    """16 direction sectors x n_speed_bins equal-probability speed quantile
    bins = up to 80 representative (direction, speed, probability) samples.
    Each bin's representative speed is the ARITHMETIC mean of the samples in
    that bin (not cube-mean - see module docstring for why a cubic
    representative speed is wrong through a saturating power curve); using
    several quantile bins per sector instead of one point is what correctly
    captures the below-rated/at-rated split."""
    wd = np.asarray(wd, dtype=float) % 360
    ws_hub = np.asarray(ws_hub, dtype=float)
    sector_width = 360.0 / n_sectors
    sector_centres = np.arange(n_sectors) * sector_width
    directions, speeds, probs = [], [], []
    n_total = wd.size
    for c in sector_centres:
        lo, hi = (c - sector_width / 2) % 360, (c + sector_width / 2) % 360
        mask = (wd >= lo) & (wd < hi) if lo < hi else (wd >= lo) | (wd < hi)
        sector_ws = np.sort(ws_hub[mask])
        n = sector_ws.size
        if n == 0:
            continue
        bins = np.array_split(sector_ws, min(n_speed_bins, n))
        for b in bins:
            directions.append(c)
            speeds.append(float(b.mean()))
            probs.append(b.size / n_total)
    directions = np.array(directions)
    speeds = np.array(speeds)
    probs = np.array(probs)
    assert abs(probs.sum() - 1.0) < 1e-9
    return directions, probs, speeds


def ti_for_directions(ws_hub: np.ndarray, wd: np.ndarray, directions: np.ndarray) -> np.ndarray:
    ti_centres, ti_vals = rep.derive_ti_per_sector(ws_hub, wd)   # pinned 12-sector formula
    width = 360.0 / len(ti_centres)
    idx = np.round(directions / width).astype(int) % len(ti_centres)
    return ti_vals[idx]


def build_site_weighted(ws: np.ndarray, wd: np.ndarray, ti: np.ndarray, prob: np.ndarray) -> XRSite:
    n = ws.size
    ds = xr.Dataset(
        data_vars={"WS": (("time",), ws), "WD": (("time",), wd % 360),
                  "TI": (("time",), ti), "P": (("time",), prob)},
        coords={"time": np.arange(n)},
    )
    return XRSite(ds=ds)


def evaluate_binned(x_m: np.ndarray, y_m: np.ndarray, ws: np.ndarray, wd: np.ndarray,
                    ti: np.ndarray, prob: np.ndarray, turbine) -> dict:
    site = build_site_weighted(ws, wd, ti, prob)
    wake = rep.build_wake_model(turbine, site)
    sim = wake(x_m, y_m, ws=ws, wd=wd, time=True)
    power_w = np.asarray(sim.Power.values)
    while power_w.ndim > 2 and power_w.shape[-1] == 1:
        power_w = power_w[..., 0]
    per_turbine_w = power_w.T if power_w.shape[0] == x_m.size else power_w
    per_turbine_mw = per_turbine_w / 1e6
    farm_mw_per_sector = per_turbine_mw.sum(axis=1)

    mean_farm_mw = float(np.sum(prob * farm_mw_per_sector))
    wake_free_mw_per_sector = (np.asarray(turbine.power(ws), dtype=float) / 1e6) * x_m.size
    mean_wf_mw = float(np.sum(prob * wake_free_mw_per_sector))

    aep_gwh = mean_farm_mw * 8760 / 1000
    aep_wf_gwh = mean_wf_mw * 8760 / 1000
    rated_mw = x_m.size * (turbine.power(np.array([15.0])) / 1e6).item()
    cf = aep_gwh * 1000 / (rated_mw * 8760) if rated_mw > 0 else 0.0
    wake_loss = 1 - aep_gwh / aep_wf_gwh if aep_wf_gwh > 0 else 0.0
    return {"aep_gwh": aep_gwh, "capacity_factor": cf, "wake_loss_fraction": wake_loss}


# ── Layout geometry ──────────────────────────────────────────────────

def make_rect_grid(n_A: int, n_B: int, spacing_A: float, spacing_B: float,
                   n_total: int = N_TURBINES) -> tuple[np.ndarray, np.ndarray]:
    """n_A = crosswind count (tight axis), n_B = downwind count (wide axis)."""
    pts = []
    for j in range(n_B):
        for i in range(n_A):
            if len(pts) >= n_total:
                break
            pts.append((i * spacing_A, j * spacing_B))
        if len(pts) >= n_total:
            break
    x = np.array([p[0] for p in pts]) - np.mean([p[0] for p in pts])
    y = np.array([p[1] for p in pts]) - np.mean([p[1] for p in pts])
    return x, y


def rotate(x: np.ndarray, y: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    if deg == 0:
        return x, y
    a = np.deg2rad(deg)
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)


def check_valid(x: np.ndarray, y: np.ndarray, box_half: float = BOX_HALF_M,
                min_spacing: float = MIN_SPACING_M) -> tuple[bool, str]:
    if x.size != N_TURBINES:
        return False, f"wrong_turbine_count_{x.size}"
    if np.any(np.abs(x) > box_half * (1 + 1e-9)) or np.any(np.abs(y) > box_half * (1 + 1e-9)):
        return False, "out_of_box"
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.hypot(dx, dy)
    np.fill_diagonal(dist, np.inf)
    if dist.min() < min_spacing * (1 - 1e-9):
        return False, "spacing_violation"
    return True, "ok"


def _largest_valid_spacing_B(n_A: int, n_B: int, spacing_A: float, hi_guess: float,
                             lo: float = MIN_SPACING_M, iters: int = 30) -> float:
    """Largest spacing_B (unrotated) that keeps the (possibly TRUNCATED, for
    n_A*n_B > N_TURBINES) grid inside the box. Truncation drops points from
    the last downwind column, which shifts the centroid off-symmetric and
    makes the true max extent bigger than a naive `box_size/(n_B-1)` formula
    predicts (caught by the smoke test: several "analytic max" configs came
    out geometrically invalid) - bisect empirically instead of guessing."""
    x, y = make_rect_grid(n_A, n_B, spacing_A, hi_guess)
    if check_valid(x, y)[0]:
        return hi_guess
    hi = hi_guess
    for _ in range(iters):
        mid = (lo + hi) / 2
        x, y = make_rect_grid(n_A, n_B, spacing_A, mid)
        if check_valid(x, y)[0]:
            lo = mid
        else:
            hi = mid
    return lo


def aspect_configs() -> list[dict]:
    configs = [{
        "name": "uniform_7D_baseline", "n_A": 8, "n_B": 7,
        "spacing_A": 7 * DIAMETER_M, "spacing_B": 7 * DIAMETER_M,
    }]
    for n_A in range(5, 12):
        n_B = int(np.ceil(N_TURBINES / n_A))
        analytic_guess = (2 * BOX_HALF_M) / (n_B - 1) if n_B > 1 else 2 * BOX_HALF_M
        max_fit_B = _largest_valid_spacing_B(n_A, n_B, MIN_SPACING_M, analytic_guess)
        spacing_B_options = sorted(set(
            v for v in [MIN_SPACING_M, min(2 * MIN_SPACING_M, max_fit_B), max_fit_B]
            if v >= MIN_SPACING_M - 1e-6
        ))
        for spacing_B in spacing_B_options:
            configs.append({
                "name": f"nA{n_A}_nB{n_B}_spB{spacing_B:.0f}",
                "n_A": n_A, "n_B": n_B, "spacing_A": MIN_SPACING_M, "spacing_B": spacing_B,
            })
    return configs


def perimeter_points(n_per_side: int, half: float = BOX_HALF_M) -> tuple[np.ndarray, np.ndarray]:
    """n_per_side points per edge, EACH EDGE STARTING AT A CORNER (corner shared
    with the previous edge, not duplicated) - this guarantees every adjacent
    pair (including across a corner) is exactly `2*half/n_per_side` apart in a
    straight line, unlike a naive constant-arclength walk (which shortens the
    chord at corners via the triangle inequality and can violate min spacing -
    caught by the smoke test: naive 42-point walk gave a 1010 m corner spacing
    against a 1420 m requirement)."""
    side = 2 * half
    step = side / n_per_side
    corners = np.array([(-half, -half), (half, -half), (half, half), (-half, half)])
    xs, ys = [], []
    for c in range(4):
        cx, cy = corners[c]
        nx, ny = corners[(c + 1) % 4]
        for k in range(n_per_side):
            t = k / n_per_side
            xs.append(cx + (nx - cx) * t)
            ys.append(cy + (ny - cy) * t)
    return np.array(xs), np.array(ys)


def make_boundary_loaded(orientation_deg: float, n_per_side: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """n_per_side=10 -> step=1500m (>=1420 min spacing) -> 40 perimeter points,
    15 interior (3x5 grid at INTERIOR_SPACING_M=1421m, 1m above the 1420m
    floor so post-rounding export can't dip below it - see
    INTERIOR_SPACING_M docstring - well clear of the perimeter, see
    module docstring / report for the clearance calc)."""
    px, py = perimeter_points(n_per_side)
    n_interior = N_TURBINES - px.size
    ix, iy = make_rect_grid(3, 5, INTERIOR_SPACING_M, INTERIOR_SPACING_M, n_total=n_interior)
    ix, iy = rotate(ix, iy, orientation_deg)
    x = np.concatenate([px, ix])
    y = np.concatenate([py, iy])
    return x, y


def _staggered_raw(n_A: int, n_B: int, spacing_A: float, spacing_B: float,
                   offset_frac: float) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for j in range(n_B):
        row_offset = spacing_A * offset_frac if (j % 2 == 1) else 0.0
        for i in range(n_A):
            if len(pts) >= N_TURBINES:
                break
            pts.append((i * spacing_A + row_offset, j * spacing_B))
        if len(pts) >= N_TURBINES:
            break
    x = np.array([p[0] for p in pts]) - np.mean([p[0] for p in pts])
    y = np.array([p[1] for p in pts]) - np.mean([p[1] for p in pts])
    return x, y


def make_staggered(n_A: int, n_B: int, spacing_A: float, spacing_B: float,
                   orientation_deg: float) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Classic brick pattern: alternating rows offset by half the crosswind
    spacing. This grows the crosswind bounding extent by spacing_A/2, which
    can push a config that was already tight against the box (built at
    exactly the min spacing) out of bounds after rotation. If the full 0.5
    offset doesn't fit, we shrink the offset fraction (0.5 -> 0.4 -> ... ->
    0.1) rather than silently falling back to zero stagger - returns
    (x, y, offset_frac_used, is_full_stagger)."""
    for offset_frac in (0.5, 0.4, 0.3, 0.2, 0.1):
        x0, y0 = _staggered_raw(n_A, n_B, spacing_A, spacing_B, offset_frac)
        x, y = rotate(x0, y0, orientation_deg)
        valid, _ = check_valid(x, y)
        if valid:
            return x, y, offset_frac, offset_frac == 0.5
    x0, y0 = _staggered_raw(n_A, n_B, spacing_A, spacing_B, 0.0)
    x, y = rotate(x0, y0, orientation_deg)
    return x, y, 0.0, False


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading cell-63 AROME wind (2016-2020, nearest pixel)...")
    times, ws_hub, wd = rep.load_arome_series(CENTRE_LAT, CENTRE_LON)
    rose_centres, rose_prob, rose_ws = build_rose(ws_hub, wd)   # up to 80 (dir, P, ws) samples
    rose_ti = ti_for_directions(ws_hub, wd, rose_centres)
    print(f"built {rose_centres.size} representative (direction, speed) samples "
          f"({N_ROSE_SECTORS} sectors x up to {N_SPEED_BINS} speed bins)")
    print("per-sector summary (probability-weighted mean speed within sector):")
    sector_probs = {}
    for c in np.unique(rose_centres):
        m = rose_centres == c
        sector_p = float(rose_prob[m].sum())
        sector_probs[c] = sector_p
        sector_ws_mean = float(np.sum(rose_prob[m] * rose_ws[m]) / sector_p)
        print(f"  {c:6.1f} deg: {sector_p*100:5.1f}%  mean ws={sector_ws_mean:5.2f} m/s  "
              f"speed range [{rose_ws[m].min():.1f},{rose_ws[m].max():.1f}]")
    dominant = max(sector_probs, key=sector_probs.get)
    print(f"most-probable single sector: {dominant:.0f} deg "
          f"(prevailing band per Friday's rose: ~200-230 deg, SW)")

    turbine = rep.build_turbine()

    # ── Stage 1: oriented rectangular grids ──────────────────────────
    print("\n=== Stage 1: oriented rectangular grid sweep ===")
    rows = []
    for cfg in aspect_configs():
        for orient in ORIENTATIONS_DEG:
            x0, y0 = make_rect_grid(cfg["n_A"], cfg["n_B"], cfg["spacing_A"], cfg["spacing_B"])
            x, y = rotate(x0, y0, float(orient))
            valid, reason = check_valid(x, y)
            row = {
                "stage": "1_grid_orientation", "config": cfg["name"], "orientation_deg": float(orient),
                "n_A": cfg["n_A"], "n_B": cfg["n_B"], "spacing_A_m": cfg["spacing_A"],
                "spacing_B_m": cfg["spacing_B"], "valid": valid, "invalid_reason": None if valid else reason,
                "x_m": x.round(2).tolist(), "y_m": y.round(2).tolist(),
            }
            if valid:
                res = evaluate_binned(x, y, rose_ws, rose_centres, rose_ti, rose_prob, turbine)
                row.update(res)
            else:
                row.update({"aep_gwh": np.nan, "capacity_factor": np.nan, "wake_loss_fraction": np.nan})
            rows.append(row)

    stage1_df = pd.DataFrame(rows)
    n_valid = int(stage1_df["valid"].sum())
    print(f"Stage 1: {len(stage1_df)} candidates, {n_valid} geometrically valid "
          f"(rest excluded: out-of-box or spacing violation at that rotation)")
    stage1_valid = stage1_df[stage1_df["valid"]].copy()
    stage1_winner = stage1_valid.loc[stage1_valid["capacity_factor"].idxmax()]
    print(f"Stage 1 winner: config={stage1_winner['config']} orientation={stage1_winner['orientation_deg']:.0f}deg "
          f"CF={stage1_winner['capacity_factor']*100:.2f}% wake={stage1_winner['wake_loss_fraction']*100:.2f}%")

    # Figure: CF vs orientation, one line per config. Markers only where a
    # config is geometrically valid at that orientation - connecting lines
    # across a gap (e.g. an elongated config invalid near 45 deg because it
    # no longer fits the box, see check_valid/out_of_box) would visually
    # imply an interpolated value that was never evaluated.
    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg_name, grp in stage1_valid.groupby("config"):
        grp = grp.sort_values("orientation_deg")
        ax.plot(grp["orientation_deg"], grp["capacity_factor"] * 100, marker="o", ms=4,
               lw=0, alpha=0.8, label=cfg_name)
    ax.axhline(stage1_winner["capacity_factor"] * 100, ls="--", color="black", lw=1,
              label=f"Stage 1 winner ({stage1_winner['capacity_factor']*100:.1f}%)")
    ax.set(xlabel="grid orientation (deg from north)", ylabel="capacity factor (%, fast 16-sector rose)",
          title="Stage 1 — CF vs orientation, cell 63 (52.5N, 3.0E)\n"
                "markers only: a missing point means that config didn't fit the box at that orientation")
    ax.legend(fontsize=6, ncol=3, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIGURE, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_FIGURE}")

    # ── Stage 2: staggered / brick pattern on Stage 1 winner's config ─
    print("\n=== Stage 2: staggered / brick pattern ===")
    w = stage1_winner
    x2, y2, offset_frac, is_full = make_staggered(
        int(w["n_A"]), int(w["n_B"]), w["spacing_A_m"], w["spacing_B_m"], float(w["orientation_deg"]))
    valid2, reason2 = check_valid(x2, y2)
    stage2_row = {
        "stage": "2_staggered", "config": w["config"] + "_staggered",
        "orientation_deg": float(w["orientation_deg"]), "n_A": w["n_A"], "n_B": w["n_B"],
        "spacing_A_m": w["spacing_A_m"], "spacing_B_m": w["spacing_B_m"],
        "stagger_offset_frac": offset_frac, "full_half_row_stagger": is_full,
        "valid": valid2, "invalid_reason": None if valid2 else reason2,
        "x_m": x2.round(2).tolist(), "y_m": y2.round(2).tolist(),
    }
    if valid2:
        res2 = evaluate_binned(x2, y2, rose_ws, rose_centres, rose_ti, rose_prob, turbine)
        stage2_row.update(res2)
        tag = "full 0.5x row" if is_full else f"reduced to {offset_frac}x row (full stagger didn't fit the box)"
        print(f"Stage 2 ({tag}): CF={res2['capacity_factor']*100:.2f}% wake={res2['wake_loss_fraction']*100:.2f}% "
              f"(vs Stage 1 winner {stage1_winner['capacity_factor']*100:.2f}%)")
    else:
        stage2_row.update({"aep_gwh": np.nan, "capacity_factor": np.nan, "wake_loss_fraction": np.nan})
        print(f"Stage 2: INVALID even at minimal stagger ({reason2})")
    rows.append(stage2_row)

    # ── Stage 3: boundary-loaded ──────────────────────────────────────
    print("\n=== Stage 3: boundary-loaded (perimeter + interior) ===")
    x3, y3 = make_boundary_loaded(float(w["orientation_deg"]))
    valid3, reason3 = check_valid(x3, y3)
    stage3_row = {
        "stage": "3_boundary_loaded", "config": "boundary_loaded_40perim_15interior",
        "orientation_deg": float(w["orientation_deg"]), "n_A": np.nan, "n_B": np.nan,
        "spacing_A_m": INTERIOR_SPACING_M, "spacing_B_m": np.nan,
        "valid": valid3, "invalid_reason": None if valid3 else reason3,
        "x_m": x3.round(2).tolist(), "y_m": y3.round(2).tolist(),
    }
    if valid3:
        res3 = evaluate_binned(x3, y3, rose_ws, rose_centres, rose_ti, rose_prob, turbine)
        stage3_row.update(res3)
        print(f"Stage 3: CF={res3['capacity_factor']*100:.2f}% wake={res3['wake_loss_fraction']*100:.2f}% "
              f"(vs Stage 1 winner {stage1_winner['capacity_factor']*100:.2f}%)")
    else:
        stage3_row.update({"aep_gwh": np.nan, "capacity_factor": np.nan, "wake_loss_fraction": np.nan})
        print(f"Stage 3: INVALID ({reason3})")
    rows.append(stage3_row)

    full_df = pd.DataFrame(rows)
    full_df.to_csv(OUT_CANDIDATES_CSV, index=False)
    print(f"\nwrote {OUT_CANDIDATES_CSV} ({len(full_df)} candidates)")

    winners = {
        "stage1": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stage1_winner.to_dict().items()},
        "stage2": stage2_row,
        "stage3": stage3_row,
        "rose": {"direction_deg": rose_centres.tolist(), "prob": rose_prob.tolist(),
                "ws_bin_mean": rose_ws.tolist(), "ti": rose_ti.tolist()},
    }
    with open(OUT_STAGE_WINNERS_JSON, "w") as f:
        json.dump(winners, f, indent=2, default=str)
    print(f"wrote {OUT_STAGE_WINNERS_JSON}")

    print("\n=== Stage 1-3 summary (fast 16-sector rose CF) ===")
    for label, row in [("Stage 1 (oriented grid)", stage1_winner.to_dict()),
                       ("Stage 2 (staggered)", stage2_row),
                       ("Stage 3 (boundary-loaded)", stage3_row)]:
        cf = row.get("capacity_factor", float("nan"))
        wl = row.get("wake_loss_fraction", float("nan"))
        print(f"  {label:28s}: CF={cf*100:5.2f}%  wake={wl*100:5.2f}%" if row.get("valid") else
              f"  {label:28s}: INVALID")


if __name__ == "__main__":
    main()
