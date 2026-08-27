"""Independent PyWake scorer replica + well-tie against the kit's baseline.

Task2 CC prompt (2026-07-13), Task F. Builds the wake simulation from bare
py_wake calls using ONLY the pinned physics from
`reports/task2_scorer_bathymetry_ranking_20260710.md` (Task A) - it does
NOT import `wind_farm_simulator.simulate()` - so a match against the kit's
own reported numbers is a genuine independent check, not a tautology.

Pinned construction (Task A.1, A.2, A.3):
    - PropagateDownwind + BastankhahGaussianDeficit() (all defaults:
      k=0.0324555, ceps=0.2, ctlim=0.899), superpositionModel=LinearSum()
      passed EXPLICITLY (not relying on the PyWake default).
    - No turbulenceModel, no blockageModel, no rotorAvgModel.
    - Per-sector TI (Charnock-style, 12 sectors) fed to the XRSite, ported
      from wind_farm_simulator.derive_ti_per_sector (same formula, not
      imported - see _derive_ti_per_sector below).
    - IEA 22 MW turbine built from data/iea22mw_power_ct.csv (the kit's own
      synthetic fallback curve, Task A.3) via PowerCtTabular directly.
    - 55-turbine, 7D-spacing, 0-deg grid layout, ported from
      wind_farm_simulator.grid_layout (pure geometry, no physics content).

**Shear reference-height deviation from the literal prompt text, flagged:**
The 2026-07-13 prompt says to match "10 m -> 170 m for reanalysis ws10
series," but also says to use AROME (125 m native) as the site data. Those
two instructions are inconsistent for AROME input: AROME already reports
wind at 125 m, so a 10-m-reference shear factor is the wrong call site
(Task A.2 found the kit itself uses a DIFFERENT call site,
`FineSynthCache` in `3b_farm_optimization_refined.ipynb`, at 125 m -> hub
for exactly this situation). Applying the 10 m factor to 125 m data would
multiply speed by (170/10)^0.11 = 1.366x instead of the physically correct
(170/125)^0.11 = 1.034x - a 32-percentage-point error, not a rounding
difference. This script uses 125 m -> 170 m (REF_HEIGHT_M=125.0) and
documents the deviation here rather than silently complying with a
physically incoherent instruction; see the report for the numeric check.

**Wind-source caveat found while preparing this well-tie (see report Task
F.0):** the kit's OWN reference pipeline that SUBMISSION.md cites for the
baseline CF~54%/wake~6% numbers runs on a stochastically-generated
synthetic wind year (`synthetic_generator.py`, fit to real AROME-coarsened
climatology but a single seeded synthetic realisation), not real AROME -
and the one shipped notebook that actually executes `simulate_year` on a
plain 7D grid (`2_simulator_intro.ipynb`) does so at a DIFFERENT centre,
Dogger-Bank-like (54.5N, 2.0E), not (53.5N, 1.5E). This script therefore
runs three cases, not one, to separate "is our physics correct" from "does
the wind input match":

    1. PRIMARY / GATE - real AROME 2016-2020 @ nearest pixel to
       (53.5N, 1.5E), as literally instructed.
    2. PHYSICS CROSS-CHECK - the kit's OWN synthetic Tier-0 bootstrap wind
       (`synthetic_generator.bootstrap_year`, seed=0) at Dogger Bank
       (54.5N, 2.0E), reusing the kit's data-generation code but feeding
       it through OUR independently-built PyWake objects. If this
       reproduces `2_simulator_intro.ipynb`'s own printed CF/wake numbers,
       our physics construction is confirmed correct independent of any
       wind-source question.
    3. LOCATION CHECK - real AROME at Dogger Bank (54.5N, 2.0E), to see
       whether real wind at the *notebook's actual* baseline location is
       closer to the SUBMISSION.md target than real wind at 53.5/1.5.

Deterministic (case 2's `bootstrap_year(..., seed=0)` is seeded, matching
the kit's own seed exactly - no other stochastic step).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.site.xrsite import XRSite
from py_wake.superposition_models import LinearSum

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
DATA_ROOT = REPO_ROOT / "phase_2" / "phase2_dataset_ship"
TURBINE_CSV = REPO_ROOT / "data" / "iea22mw_power_ct.csv"
OUT_CSV = REPO_ROOT / "data" / "task2_welltie_results.csv"

sys.path.insert(0, str(KIT_PHASE2))
sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))
sys.path.insert(0, str(KIT_PHASE2 / "part2_siting"))
os.environ.setdefault("PHASE2_DATA_ROOT", str(DATA_ROOT))

# ── Pinned constants (Task A) ───────────────────────────────────────────
DIAMETER_M = 284.0
HUB_HEIGHT_M = 170.0
N_TURBINES = 55
SPACING_D = 7.0
SHEAR_ALPHA = 0.11
REF_HEIGHT_M = 125.0   # see module docstring for why not 10 m
N_TI_SECTORS = 12
BASELINE_CENTRE = (53.5, 1.5)     # SUBMISSION.md's stated baseline
DOGGER_CENTRE = (54.5, 2.0)       # 2_simulator_intro.ipynb's actual demo centre
TARGET_CF = 0.538
TARGET_WAKE_LOSS = 0.059
CF_TOLERANCE = 0.02
WAKE_TOLERANCE_PP = 0.02


# ── Turbine (ported CSV load, Task A.3) ─────────────────────────────────

def build_turbine() -> WindTurbine:
    df = pd.read_csv(TURBINE_CSV)
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
    return WindTurbine(name="IEA 22 MW (kit fallback curve)", diameter=DIAMETER_M,
                       hub_height=HUB_HEIGHT_M, powerCtFunction=power_ct)


# ── Layout (ported geometry, wind_farm_simulator.grid_layout) ──────────

def grid_layout(n_turbines: int = N_TURBINES, spacing_d: float = SPACING_D,
                diameter_m: float = DIAMETER_M) -> tuple[np.ndarray, np.ndarray]:
    spacing_m = spacing_d * diameter_m
    n_rows = int(np.ceil(np.sqrt(n_turbines)))
    n_cols = int(np.ceil(n_turbines / n_rows))
    pts = []
    for r in range(n_rows):
        for c in range(n_cols):
            if len(pts) >= n_turbines:
                break
            pts.append((c * spacing_m, r * spacing_m))
    x = np.array([p[0] for p in pts]) - np.mean([p[0] for p in pts])
    y = np.array([p[1] for p in pts]) - np.mean([p[1] for p in pts])
    return x, y


# ── TI (ported formula, wind_farm_simulator.derive_ti_per_sector) ──────

def _charnock_ti(mean_ws_ms: float, a: float = 0.05, b: float = 0.4) -> float:
    if mean_ws_ms < 0.5:
        return a + b / 0.5
    return a + b / mean_ws_ms


def derive_ti_per_sector(ws: np.ndarray, wd: np.ndarray, n_sectors: int = N_TI_SECTORS,
                         min_ti: float = 0.03, max_ti: float = 0.18,
                         default_ti: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    ws = np.asarray(ws, dtype=float)
    wd = np.asarray(wd, dtype=float) % 360
    sector_width = 360 / n_sectors
    centres = np.arange(n_sectors) * sector_width + sector_width / 2
    ti = np.full(n_sectors, default_ti)
    for i, c in enumerate(centres):
        lo = (c - sector_width / 2) % 360
        hi = (c + sector_width / 2) % 360
        mask = (wd >= lo) & (wd < hi) if lo < hi else (wd >= lo) | (wd < hi)
        n = int(mask.sum())
        if n >= 30:
            ti[i] = _charnock_ti(float(np.mean(ws[mask])))
    return centres, np.clip(ti, min_ti, max_ti)


def build_site(ws: np.ndarray, wd: np.ndarray) -> XRSite:
    wd = np.asarray(wd, dtype=float) % 360
    sector_centres, ti_per_sector = derive_ti_per_sector(ws, wd)
    bin_idx = np.minimum((wd / (360 / N_TI_SECTORS)).astype(int), N_TI_SECTORS - 1)
    ti_per_step = ti_per_sector[bin_idx]
    n = ws.size
    ds = xr.Dataset(
        data_vars={
            "WS": (("time",), ws), "WD": (("time",), wd),
            "TI": (("time",), ti_per_step), "P": (("time",), np.full(n, 1.0 / n)),
        },
        coords={"time": np.arange(n)},
    )
    return XRSite(ds=ds)


def build_wake_model(turbine: WindTurbine, site: XRSite) -> PropagateDownwind:
    return PropagateDownwind(
        site=site, windTurbines=turbine,
        wake_deficitModel=BastankhahGaussianDeficit(),
        superpositionModel=LinearSum(),   # explicit, per Task F instruction
    )


# ── Simulation (ported integration logic, wind_farm_simulator.simulate) ─

def run_case(label: str, times: np.ndarray, ws_hub: np.ndarray, wd: np.ndarray,
            turbine: WindTurbine, x_m: np.ndarray, y_m: np.ndarray) -> dict:
    t0 = time.time()
    site = build_site(ws_hub, wd)
    wake = build_wake_model(turbine, site)

    sim = wake(x_m, y_m, ws=ws_hub, wd=wd, time=True)
    power_w = np.asarray(sim.Power.values)
    while power_w.ndim > 2 and power_w.shape[-1] == 1:
        power_w = power_w[..., 0]
    per_turbine_w = power_w.T if power_w.shape[0] == x_m.size else power_w
    per_turbine_mw = per_turbine_w / 1e6
    farm_mw = per_turbine_mw.sum(axis=1)

    wake_free_mw = (np.asarray(turbine.power(ws_hub), dtype=float) / 1e6) * x_m.size

    times_s = pd.to_datetime(times)
    n_steps = ws_hub.size
    dt_span_h = (times_s[-1] - times_s[0]).total_seconds() / 3600
    avg_step_h = dt_span_h / max(n_steps - 1, 1)
    duration_hours = dt_span_h + avg_step_h
    dx = duration_hours / max(n_steps, 1)

    energy_mwh = float(np.trapezoid(farm_mw, dx=dx))
    energy_wf_mwh = float(np.trapezoid(wake_free_mw, dx=dx))
    scale = 8760.0 / duration_hours if duration_hours > 0 else 1.0
    aep_gwh = energy_mwh * scale / 1000
    aep_wf_gwh = energy_wf_mwh * scale / 1000

    rated_mw = x_m.size * (turbine.power(np.array([15.0])) / 1e6).item()
    cf = aep_gwh * 1000 / (rated_mw * 8760) if rated_mw > 0 else 0.0
    wake_loss = 1 - aep_gwh / aep_wf_gwh if aep_wf_gwh > 0 else 0.0

    elapsed = time.time() - t0
    result = {
        "label": label, "n_steps": n_steps, "duration_hours": duration_hours,
        "aep_gwh": aep_gwh, "capacity_factor": cf, "wake_loss_fraction": wake_loss,
        "rated_capacity_mw": rated_mw, "mean_ws_hub": float(np.mean(ws_hub)),
        "elapsed_s": elapsed,
    }
    print(f"[{label}] n_steps={n_steps} AEP={aep_gwh:.0f} GWh  CF={cf*100:.1f}%  "
          f"wake_loss={wake_loss*100:.2f}%  mean_ws={result['mean_ws_hub']:.2f} m/s  "
          f"({elapsed:.1f}s)")
    return result


# ── Wind sources ─────────────────────────────────────────────────────

def load_arome_series(target_lat: float, target_lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real AROME 125 m at the single nearest pixel to (target_lat, target_lon),
    all TRAIN_YEARS, native 3-hourly cadence, sheared 125 m -> hub."""
    import target_loader
    static = target_loader.load_static()
    d2 = (static.lat - target_lat) ** 2 + (static.lon - target_lon) ** 2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    px_lat, px_lon = float(static.lat[iy, ix]), float(static.lon[iy, ix])
    print(f"  nearest AROME pixel to ({target_lat},{target_lon}): "
          f"({px_lat:.4f},{px_lon:.4f}), sea={bool(static.sea[iy, ix])}")

    dates = sorted(target_loader.list_dates())
    times, u_list, v_list = [], [], []
    t0 = time.time()
    for k, d in enumerate(dates):
        day = target_loader.load_day(d, levels=("125m",))
        times.append(day.times)
        u_list.append(day.u["125m"][:, iy, ix])
        v_list.append(day.v["125m"][:, iy, ix])
        if (k + 1) % 400 == 0:
            print(f"    [{k+1}/{len(dates)}] elapsed={time.time()-t0:.0f}s")
    times = np.concatenate(times)
    u = np.concatenate(u_list).astype(float)
    v = np.concatenate(v_list).astype(float)
    finite = np.isfinite(u) & np.isfinite(v)
    times, u, v = times[finite], u[finite], v[finite]

    ws_ref = np.sqrt(u ** 2 + v ** 2)
    wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    shear_factor = (HUB_HEIGHT_M / REF_HEIGHT_M) ** SHEAR_ALPHA
    ws_hub = ws_ref * shear_factor
    print(f"  loaded {ws_hub.size} finite AROME steps, shear factor "
          f"({HUB_HEIGHT_M:.0f}/{REF_HEIGHT_M:.0f})^{SHEAR_ALPHA} = {shear_factor:.4f}")
    return times, ws_hub, wd


def load_kit_synthetic_series(target_lat: float, target_lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The kit's OWN synthetic wind (Tier-0 bootstrap, seed=0), as
    2_simulator_intro.ipynb builds it, reused verbatim for data generation
    only (not for the physics under test)."""
    import synthetic_generator as sg
    import synth_wind
    hist = sg.load_coarse_history()
    synth = sg.bootstrap_year(hist, block_days=14, seed=0)
    cache = synth_wind.SynthWindCache(synth, hub_height_m=HUB_HEIGHT_M)
    series = cache.get(target_lat, target_lon)
    times = series.df["time"].to_numpy()
    ws_hub = series.df["ws"].to_numpy()
    wd = series.df["wd"].to_numpy()
    print(f"  kit synthetic (Tier-0 bootstrap, seed=0) @ ({target_lat},{target_lon}): "
          f"{ws_hub.size} steps, mean ws={ws_hub.mean():.2f} m/s")
    return times, ws_hub, wd


def main() -> None:
    turbine = build_turbine()
    x_m, y_m = grid_layout()
    rows = []

    print("\n=== Case 1 (PRIMARY / GATE): real AROME @ baseline centre (53.5N,1.5E) ===")
    times1, ws1, wd1 = load_arome_series(*BASELINE_CENTRE)
    r1 = run_case("arome_baseline_53.5_1.5", times1, ws1, wd1, turbine, x_m, y_m)
    rows.append(r1)

    print("\n=== Case 2 (physics cross-check): kit's own synthetic wind @ Dogger (54.5N,2.0E) ===")
    times2, ws2, wd2 = load_kit_synthetic_series(*DOGGER_CENTRE)
    r2 = run_case("kit_synthetic_dogger_54.5_2.0", times2, ws2, wd2, turbine, x_m, y_m)
    rows.append(r2)

    print("\n=== Case 3 (location check): real AROME @ Dogger (54.5N,2.0E) ===")
    times3, ws3, wd3 = load_arome_series(*DOGGER_CENTRE)
    r3 = run_case("arome_dogger_54.5_2.0", times3, ws3, wd3, turbine, x_m, y_m)
    rows.append(r3)

    df = pd.DataFrame(rows)
    df["target_cf"] = TARGET_CF
    df["target_wake_loss"] = TARGET_WAKE_LOSS
    df["cf_diff_pp"] = (df["capacity_factor"] - TARGET_CF) * 100
    df["wake_loss_diff_pp"] = (df["wake_loss_fraction"] - TARGET_WAKE_LOSS) * 100
    df["cf_within_tolerance"] = df["cf_diff_pp"].abs() <= CF_TOLERANCE * 100
    df["wake_within_tolerance"] = df["wake_loss_diff_pp"].abs() <= WAKE_TOLERANCE_PP * 100
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")
    print(df[["label", "aep_gwh", "capacity_factor", "wake_loss_fraction",
              "cf_diff_pp", "wake_loss_diff_pp", "cf_within_tolerance", "wake_within_tolerance"]]
          .to_string(index=False))

    gate_row = df[df["label"] == "arome_baseline_53.5_1.5"].iloc[0]
    gate_pass = bool(gate_row["cf_within_tolerance"] and gate_row["wake_within_tolerance"])
    print(f"\nGATE (case 1, real AROME @ 53.5,1.5): {'PASS' if gate_pass else 'FAIL'} "
          f"(tolerance +/-{CF_TOLERANCE*100:.0f} CF pts, +/-{WAKE_TOLERANCE_PP*100:.0f} pp wake loss)")


if __name__ == "__main__":
    main()
