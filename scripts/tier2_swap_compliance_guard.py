#!/usr/bin/env python3
"""Compliance guard (Clause 1): every Pangu ERA5 init timestamp <= context_end.

The d14 recon (commit 5195f0d) confirmed init <= context_end holds by construction
(issue = context_end date; init hours 00/06/12/18 all fall on context_end <
predict_start) but there was NO runtime assert. This is that assert, run as a
pre-flight before any swap-year rollout: it reads each window's metadata.json and
verifies every (context_end + init_hour) < predict_start, failing loudly with the
offending timestamp if violated. Prints one PASS line per (window x init hour) --
32 for the 8-window x 4-init layout.

Usage:
  python tier2_swap_compliance_guard.py --meta-dir <inference_dir> [--init-hours 0,6,12,18]
  # inference_dir contains window_*/metadata.json (context_end, predict_start)

Importable: check_init_compliance(meta_dir, init_hours) -> list of dict rows;
assert_init_compliance(...) raises AssertionError on any violation. The rollout
script calls the latter when --compliance-meta-dir is passed (opt-in; the default
inference path is unchanged, so byte-for-byte reproduction is preserved).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

DEFAULT_INIT_HOURS = (0, 6, 12, 18)


def check_init_compliance(meta_dir, init_hours=DEFAULT_INIT_HOURS):
    """Return one row per (window, init_hour): the init datetime vs the window's
    context_end / predict_start, and whether init < predict_start (compliant)."""
    rows = []
    mds = sorted(glob.glob(os.path.join(str(meta_dir), "window_*", "metadata.json")))
    if not mds:
        raise FileNotFoundError(f"no window_*/metadata.json under {meta_dir}")
    for md in mds:
        d = json.load(open(md))
        ce = pd.Timestamp(d["context_end"])                 # 00:00 of context_end date
        ps = pd.Timestamp(d["predict_start"])               # 00:00 of predict_start date
        for h in init_hours:
            init_dt = ce + pd.Timedelta(hours=int(h))        # analysis time actually loaded
            # compliant iff the init analysis precedes the prediction period AND
            # falls on the context_end date (i.e. within the context window)
            ok = (init_dt < ps) and (init_dt.normalize() == ce.normalize())
            rows.append({"window_id": d["id"], "init_hour": int(h),
                         "init_dt": str(init_dt), "context_end": str(ce.date()),
                         "predict_start": str(ps.date()), "ok": bool(ok)})
    return rows


def assert_init_compliance(meta_dir, init_hours=DEFAULT_INIT_HOURS, verbose=True):
    rows = check_init_compliance(meta_dir, init_hours)
    bad = [r for r in rows if not r["ok"]]
    if verbose:
        for r in rows:
            tag = "PASS" if r["ok"] else "**FAIL**"
            print(f"  [guard] window {r['window_id']} init {r['init_hour']:02d}UTC: "
                  f"init={r['init_dt']}  context_end={r['context_end']}  "
                  f"predict_start={r['predict_start']}  {tag}")
    if bad:
        first = bad[0]
        raise AssertionError(
            f"Clause-1 VIOLATION: {len(bad)} init timestamp(s) not <= context_end. "
            f"First: window {first['window_id']} init {first['init_hour']:02d}UTC "
            f"({first['init_dt']}) vs predict_start {first['predict_start']}.")
    if verbose:
        print(f"  [guard] ALL {len(rows)} init timestamps <= context_end (Clause 1 OK)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", required=True,
                    help="dir containing window_*/metadata.json")
    ap.add_argument("--init-hours", default="0,6,12,18")
    args = ap.parse_args()
    ih = tuple(int(x) for x in args.init_hours.split(","))
    rows = assert_init_compliance(Path(args.meta_dir), ih)
    print(f"[guard] {len(rows)} checks, all PASS")


if __name__ == "__main__":
    main()
