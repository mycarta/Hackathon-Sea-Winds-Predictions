#!/usr/bin/env python3
"""TIER-2 Night-1 gate F1: coupling / representativeness floor.

Phase 0 F1 (confirmed 2026-07-17). On a few NON-EXCLUDED dates, compare the
Pangu-coupled 125 m coarse field against the shipped AROME coarse truth (u125c),
at analysis lead 0 (Pangu is initialised FROM ERA5, so at lead 0 Pangu-coarse ==
ERA5-coupled-coarse) -- this isolates the ERA5->AROME product/representativeness
gap the reused downscaler + mean-only bias table must absorb (risk #1). Forecast
drift is tested separately in F2.

Reports, per date and pooled over the 45x57 coarse cells:
  * speed q50 RMSE and mean bias (coupled - u125c)
  * u/v component bias
  * FRACTION of cells taking the 925 hPa fallback branch (z1000 <= 125 m)
Deterministic; no stochastic step. NO pipeline file touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import tier2_eval_common as ec               # noqa: E402
from tier2_pangu_couple import couple_box    # noqa: E402
from tier2_era5_fetch import fetch_box        # noqa: E402

ARTIFACTS = _HERE / "artifacts"
# 4 non-excluded dates, one per season, spread across training years (00 UTC).
DATES = ["2016-01-20", "2016-05-15", "2018-07-20", "2019-10-15"]
HOUR = 0


def main():
    excl = ec.exclusion_set()
    import pandas as pd
    per = []
    su = {"se2": 0.0, "n": 0, "sbias": 0.0, "ubias": 0.0, "vbias": 0.0,
          "fb": 0, "cells": 0}
    t0 = time.time()
    for ds in DATES:
        d = pd.Timestamp(ds).date()
        assert d not in excl, f"{ds} is inside block+/-buffer!"
        box = fetch_box(ds, HOUR)
        u125, v125, fallback = couple_box(box)
        truth = ec.load_u125c(ds, HOUR)
        assert truth is not None, f"no coarse truth for {ds}"
        tu, tv = truth
        m = np.isfinite(tu) & np.isfinite(tv) & np.isfinite(u125) & np.isfinite(v125)
        spd = np.sqrt(u125 ** 2 + v125 ** 2)
        tspd = np.sqrt(tu ** 2 + tv ** 2)
        rmse = float(np.sqrt(np.mean((spd[m] - tspd[m]) ** 2)))
        sbias = float(np.mean(spd[m] - tspd[m]))
        ubias = float(np.mean(u125[m] - tu[m]))
        vbias = float(np.mean(v125[m] - tv[m]))
        fbfrac = float(np.mean(fallback[m]))
        per.append({"date": ds, "season": ec.season_of(ds), "n_cells": int(m.sum()),
                    "speed_rmse": round(rmse, 4), "speed_bias": round(sbias, 4),
                    "u_bias": round(ubias, 4), "v_bias": round(vbias, 4),
                    "fallback_925_frac": round(fbfrac, 4),
                    "mean_truth_speed": round(float(tspd[m].mean()), 3)})
        su["se2"] += float(np.sum((spd[m] - tspd[m]) ** 2))
        su["n"] += int(m.sum())
        su["sbias"] += float(np.sum(spd[m] - tspd[m]))
        su["ubias"] += float(np.sum(u125[m] - tu[m]))
        su["vbias"] += float(np.sum(v125[m] - tv[m]))
        su["fb"] += int(fallback[m].sum())
        su["cells"] += int(m.sum())
        print(f"[F1] {ds} {ec.season_of(ds)}: speed RMSE={rmse:.3f} bias={sbias:+.3f} "
              f"u_bias={ubias:+.3f} v_bias={vbias:+.3f} 925fallback={fbfrac:.1%} "
              f"(truth mean {tspd[m].mean():.2f} m/s)")

    pooled = {
        "speed_rmse": round(float(np.sqrt(su["se2"] / su["n"])), 4),
        "speed_bias": round(su["sbias"] / su["n"], 4),
        "u_bias": round(su["ubias"] / su["n"], 4),
        "v_bias": round(su["vbias"] / su["n"], 4),
        "fallback_925_frac": round(su["fb"] / su["cells"], 4),
        "n_cells": su["n"],
    }
    print(f"[F1] POOLED: speed RMSE={pooled['speed_rmse']:.3f} "
          f"bias={pooled['speed_bias']:+.3f} 925fallback={pooled['fallback_925_frac']:.1%}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = {"gate": "F1_coupling_floor", "lead": "0 (analysis)", "hour": HOUR,
           "source": "WeatherBench2 ERA5 (anon GCS)", "alpha": 0.11,
           "dates": DATES, "per_date": per, "pooled": pooled,
           "elapsed_sec": round(time.time() - t0, 1)}
    with open(ARTIFACTS / "tier2_f1_coupling.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[F1] wrote {ARTIFACTS / 'tier2_f1_coupling.json'} in {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
