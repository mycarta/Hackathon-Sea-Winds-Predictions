#!/usr/bin/env python3
"""TIER-2 Option-C submission rollout date list: the 8 window issue dates.

Opus/Matteo Option-C GO 2026-07-19. Each of the 8 inference windows is issued at
its `context_end` date (00 UTC); the d7 forecast is valid `context_end + 7 days`
(matches metadata.json `score_days.d7`). Option C rolls each issue date from four
ERA5 inits (00/06/12/18 UTC) to reach the four d7 valid hours with the validated
Pangu-24 chain. This script emits the issue dates + a datemap so the rollout and
surgical build reuse the exact same window->date->season mapping. Deterministic
(reads metadata.json only; no sampling).

Writes:
  scripts/artifacts/tier2_sub_window_issue_dates.txt  (8 issue dates, one per line)
  scripts/artifacts/tier2_sub_datemap.json            (window/issue/valid/season)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))
import tier2_eval_common as ec                       # noqa: E402

INF = _HERE.parent / "phase_2" / "phase2_dataset_ship" / "inference"
ARTIFACTS = _HERE / "artifacts"
INIT_HOURS = (0, 6, 12, 18)


def main():
    rows = []
    for md in sorted(glob.glob(str(INF / "window_*" / "metadata.json"))):
        d = json.load(open(md))
        issue = pd.Timestamp(d["context_end"]).date()        # 00 UTC issue day
        valid_d7 = pd.Timestamp(d["score_days"]["d7"]).date()
        assert valid_d7 == issue + timedelta(days=7), (md, issue, valid_d7)
        rows.append({"window_id": d["id"], "issue": str(issue),
                     "valid_d7": str(valid_d7),
                     "season": ec.SEASON_OF_MONTH[valid_d7.month]})
    rows.sort(key=lambda r: r["window_id"])
    issues = sorted({r["issue"] for r in rows})
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "tier2_sub_window_issue_dates.txt").write_text("\n".join(issues) + "\n")
    with open(ARTIFACTS / "tier2_sub_datemap.json", "w") as f:
        json.dump({"init_hours": list(INIT_HOURS), "windows": rows}, f, indent=2)
    print(f"windows: {len(rows)}  distinct issue dates: {len(issues)}")
    for r in rows:
        print(f"  w{r['window_id']}: issue {r['issue']} 00UTC -> d7 valid {r['valid_d7']} "
              f"({r['season']})")
    n_roll = len(issues) * len(INIT_HOURS)
    print(f"rollouts: {len(issues)} dates x {len(INIT_HOURS)} inits = {n_roll} "
          f"(~{n_roll*360.6/3600:.1f} h @ 360.6 s/7-step)")
    print(f"wrote {ARTIFACTS/'tier2_sub_window_issue_dates.txt'} and tier2_sub_datemap.json")


if __name__ == "__main__":
    main()
