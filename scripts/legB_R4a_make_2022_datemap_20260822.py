#!/usr/bin/env python3
"""Build the 2022 submission-window datemap and issue-date list for Leg B.

Amendment 2 of the 2026-08-22 dispatch: the 2022 window extracts launch
immediately and unconditionally once the 2021 set finishes. They need a date
list, and the splice needs a datemap, and BOTH are year-varying.

`scripts/artifacts/tier2_sub_datemap.json` is a **2021** file. It is listed in
`data/PINNED_ARTIFACTS.md` under "Frozen artifacts NOT SHA-asserted in code"
with SHA-16 `404a065b2776ded4`. It is frozen for 2021 and WRONG for 2022, which
makes it a second entry on the contract v2.1 §5b changed-constant list, next to
`SUB_EXTRACTS`. This script writes a SEPARATE 2022 file and does not touch the
2021 one.

Derivation, straight from the shipped organizer metadata, nothing inferred:

    phase_2/phase2_dataset_ship/inference/window_<N>/metadata.json
      issue    = context_end        (the last day of context; the analysis the
                                     forecast is issued from)
      valid_d7 = score_days["d7"]
      season   = ec.season_of(valid_d7)

Cross-checked against the 2021 datemap's own construction: window 1 there is
issue 2021-01-14 / valid_d7 2021-01-21, and 2022 window 1 is context_end
2022-01-14 / d7 2022-01-21. Same offsets, same shape.

The script ASSERTS valid_d7 - issue == 7 days for all eight windows, and that
the eight window ids are exactly 1..8.

Outputs:
    scripts/artifacts/tier2_sub_datemap_2022.json
    scripts/artifacts/legB_sub_issue_dates_2022_8.txt

Both must be `git add -f`ed: `scripts/artifacts/` is gitignored by pattern, and
that pattern silently dropped the R1 date list once already.

Run:  python scripts/legB_R4a_make_2022_datemap_20260822.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))

import tier2_eval_common as ec        # noqa: E402

WINDOWS = REPO / "phase_2" / "phase2_dataset_ship" / "inference"
ARTIFACTS = _HERE / "artifacts"
OUT_MAP = ARTIFACTS / "tier2_sub_datemap_2022.json"
OUT_DATES = ARTIFACTS / "legB_sub_issue_dates_2022_8.txt"
MAP_2021 = ARTIFACTS / "tier2_sub_datemap.json"
INIT_HOURS = [0, 6, 12, 18]
LEAD_DAYS = 7


def main():
    assert WINDOWS.is_dir(), "2022 inference windows not found at %s" % WINDOWS

    windows = []
    for n in range(1, 9):
        md = WINDOWS / ("window_%d" % n) / "metadata.json"
        assert md.exists(), "missing %s" % md
        m = json.loads(md.read_text(encoding="utf-8"))
        assert m["id"] == n, "window_%d/metadata.json declares id %r" % (n, m["id"])
        issue = date.fromisoformat(m["context_end"])
        valid = date.fromisoformat(m["score_days"]["d7"])
        assert valid - issue == timedelta(days=LEAD_DAYS), (
            "window %d: valid_d7 - issue = %s, expected %d days"
            % (n, valid - issue, LEAD_DAYS))
        assert issue.year == 2022, "window %d issue is %s, not 2022" % (n, issue)
        windows.append({"window_id": n, "issue": issue.isoformat(),
                        "valid_d7": valid.isoformat(),
                        "season": ec.season_of(valid.isoformat())})

    ids = [w["window_id"] for w in windows]
    assert ids == list(range(1, 9)), ids
    issues = [w["issue"] for w in windows]
    assert len(set(issues)) == 8, "duplicate issue dates: %s" % issues

    payload = {"init_hours": INIT_HOURS, "windows": windows}
    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAP.write_text(json.dumps(payload, indent=2) + "\n",
                       encoding="utf-8", newline="\n")
    OUT_DATES.write_text("\n".join(issues) + "\n", encoding="utf-8", newline="\n")

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    print("2022 submission-window datemap")
    print("%-4s %-12s %-12s %s" % ("id", "issue", "valid_d7", "season"))
    for w in windows:
        print("%-4d %-12s %-12s %s"
              % (w["window_id"], w["issue"], w["valid_d7"], w["season"]))

    seasons = {}
    for w in windows:
        seasons[w["season"]] = seasons.get(w["season"], 0) + 1
    print("\nseason counts: %s" % json.dumps(seasons, sort_keys=True))

    # The 2021 map for comparison. Same shape is expected; same DATES would be a bug.
    old = json.loads(MAP_2021.read_text(encoding="utf-8"))
    old_issues = [w["issue"] for w in old["windows"]]
    assert not (set(old_issues) & set(issues)), "2021 and 2022 issue dates overlap"
    old_seasons = {}
    for w in old["windows"]:
        old_seasons[w["season"]] = old_seasons.get(w["season"], 0) + 1
    print("2021 season counts: %s" % json.dumps(old_seasons, sort_keys=True))
    if seasons != old_seasons:
        print("\nNOTE: the season MIX differs between 2021 and 2022. The bias table is")
        print("per season, so the two years do not weight it identically. Stated")
        print("because it is a real difference between the floor run and the")
        print("submission run, not because it is a defect.")
    else:
        print("\nSeason mix is identical to 2021, so the bias table is weighted the")
        print("same way in both years.")

    print("\nwrote %s" % OUT_MAP.relative_to(REPO).as_posix())
    print("  SHA-256 %s" % sha(OUT_MAP))
    print("wrote %s" % OUT_DATES.relative_to(REPO).as_posix())
    print("  SHA-256 %s" % sha(OUT_DATES))
    print("\nBoth need `git add -f` (scripts/artifacts/ is gitignored by pattern).")


if __name__ == "__main__":
    main()
