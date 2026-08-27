"""Export the IEA_22MW power/Ct curve exactly as the Task-2 kit builds it.

Task2 CC prompt (2026-07-10), Task A.3. No real IEA 22 MW CSV exists anywhere
in the repo or the Zenodo Phase-2 dataset: `turbines_catalog._TURBINES_DIR`
(`phase_2/kit/data/wind_data/turbines/`) does not exist, so
`turbines_catalog.load_turbine('IEA_22MW')` falls through to
`_generic_power_ct()` - a synthetic cubic-ramp curve, NOT a published IEA 22MW
curve. This script reproduces that fallback numerically (pure numpy, mirrors
`phase_2/kit/phase_2/turbines_catalog.py::_generic_power_ct` line-for-line) so
we have an auditable CSV of what the kit (and, if it mirrors the kit, the
scorer) actually simulates. Deterministic - no stochastic step, no seed
needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "iea22mw_power_ct.csv"

# TurbineSpec("IEA_22MW", ...) fields, from turbines_catalog.py CATALOG.
CUT_IN_MS = 3.0
CUT_OUT_MS = 25.0
RATED_POWER_MW = 22.0


def generic_power_ct(cut_in_ms: float, cut_out_ms: float, rated_power_mw: float) -> pd.DataFrame:
    """Verbatim port of turbines_catalog._generic_power_ct (kit source, read-only)."""
    rated_ws = 12.0
    ws = np.concatenate([
        np.linspace(0, cut_in_ms, 5, endpoint=False),
        np.linspace(cut_in_ms, rated_ws, 25),
        np.linspace(rated_ws, cut_out_ms, 15),
        np.linspace(cut_out_ms, cut_out_ms + 1, 3),
    ])
    power = np.zeros_like(ws)
    ct = np.zeros_like(ws)
    rated_w = rated_power_mw * 1e6
    for i, w in enumerate(ws):
        if w < cut_in_ms or w > cut_out_ms:
            power[i], ct[i] = 0.0, 0.0
        elif w < rated_ws:
            r = (w - cut_in_ms) / (rated_ws - cut_in_ms)
            power[i] = rated_w * r ** 3
            ct[i] = min(0.85, 4.0 * r * (1 - r) + 0.55)
        else:
            power[i] = rated_w
            r = (w - rated_ws) / (cut_out_ms - rated_ws)
            ct[i] = max(0.2, 0.55 * (1 - r))
    return pd.DataFrame({"wind_speed_ms": ws, "power_w": power, "ct": ct})


def main() -> None:
    df = generic_power_ct(CUT_IN_MS, CUT_OUT_MS, RATED_POWER_MW)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
