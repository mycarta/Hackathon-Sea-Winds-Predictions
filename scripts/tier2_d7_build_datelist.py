#!/usr/bin/env python3
"""TIER-2 ARM-D7 issue-date list (eval remaining + bias N=20/season + spd_infl calib).

Opus-authorized, Matteo GO 2026-07-18. d14 killed -> 7-step rollouts only. Each
d7 forecast valid at date W comes from a rollout issued at W-7. Seeded (A3).

Sets (issue dates, all rolled out 7 steps, harvest step 7):
  * EVAL: I7 = {W-7 : W in the 112 block valid dates}, MINUS the 21 DJF issue dates
    already harvested by the completed I14_DJF 14-step batch (extract_2017-01-02..
    2017-01-22). -> 91 new issue dates. (These lie inside block+/-buffer -- eval only.)
  * BIAS: for 20 seeded, season-balanced NON-EXCLUDED valid dates V per season
    (same pool logic as ws_d7_block_refit.py:186-205, N 40->20 per Matteo's logged
    deviation, K=30 unchanged), issue = V-7. -> 80 issue dates.
  * CALIB: the 15 monthly spd_infl calibration issue dates (2,6,10 x 2016-2020),
    non-excluded (calibrate_intervals treats these as issue dates D, valid D+7).

Writes:
  scripts/artifacts/tier2_d7_issue_dates.txt   (deduped union, one per line)
  scripts/artifacts/tier2_d7_datemap.json      (eval W-map, bias V-map, calib list)
so the scorer reuses the exact seeded sets.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))
sys.path.insert(0, str(_HERE.parent / "phase_2" / "kit" / "phase_2"))
import tier2_eval_common as ec                       # noqa: E402
import target_loader                                 # noqa: E402
import config                                        # noqa: E402

SEED = 42
BIAS_PER_SEASON = 20                                  # logged deviation from 25
ARTIFACTS = _HERE / "artifacts"
# DJF issue dates already harvested (d7 served by the completed 14-step batch)
DJF_DONE_ISSUE = {(pd.Timestamp(w) - timedelta(days=7)).date()
                  for w in ec.block_days()["DJF"]
                  if (pd.Timestamp(w) - timedelta(days=7)).date()
                  <= pd.Timestamp("2017-01-22").date()}


def main():
    excl = ec.exclusion_set()

    # ---- EVAL: I7 minus DJF-done ----
    eval_map = {}          # issue_date -> valid W
    for W in ec.block_valid_dates():
        D = (pd.Timestamp(W) - timedelta(days=7)).date()
        if D in DJF_DONE_ISSUE:
            continue
        eval_map[str(D)] = str(W)
    eval_issue = sorted(eval_map)

    # ---- BIAS: seeded season-balanced non-excluded V (issue = V-7) ----
    all_targets = sorted(target_loader.list_dates(config.target_root()))
    pool = {s: [] for s in ec.SEASONS}
    for V in all_targets:
        if V.year not in (2016, 2017, 2018, 2019, 2020):
            continue
        D = V - timedelta(days=7)
        if V in excl or D in excl:
            continue
        pool[ec.SEASON_OF_MONTH[V.month]].append(V)
    rng = np.random.default_rng(SEED)
    bias_map = {}          # issue_date (V-7) -> valid V
    for s in ec.SEASONS:
        cand = pool[s]
        pick = rng.choice(len(cand), size=min(BIAS_PER_SEASON, len(cand)), replace=False)
        for i in sorted(pick):
            V = cand[i]
            bias_map[str((V - timedelta(days=7)))] = str(V)
    bias_issue = sorted(bias_map)

    # ---- CALIB: 15 monthly issue dates, non-excluded ----
    calib_all = [pd.Timestamp(f"{y}-{m:02d}-15").date()
                 for y in (2016, 2017, 2018, 2019, 2020) for m in (2, 6, 10)]
    calib_issue = sorted(str(d) for d in calib_all if d not in excl)

    union = sorted(set(eval_issue) | set(bias_issue) | set(calib_issue))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "tier2_d7_issue_dates.txt").write_text("\n".join(union) + "\n")
    with open(ARTIFACTS / "tier2_d7_datemap.json", "w") as f:
        json.dump({"seed": SEED, "bias_per_season": BIAS_PER_SEASON,
                   "eval_issue_to_validW": eval_map, "bias_issue_to_validV": bias_map,
                   "calib_issue": calib_issue,
                   "djf_done_issue": sorted(str(d) for d in DJF_DONE_ISSUE)}, f, indent=2)
    print(f"EVAL remaining: {len(eval_issue)}  BIAS: {len(bias_issue)}  "
          f"CALIB: {len(calib_issue)}  UNION distinct: {len(union)}")
    print(f"est wall @43s/step, 7 steps: {len(union)*7*43/3600:.1f} h "
          f"(+21 DJF eval already in hand)")
    print(f"wrote {ARTIFACTS/'tier2_d7_issue_dates.txt'} and datemap.json")


if __name__ == "__main__":
    main()
