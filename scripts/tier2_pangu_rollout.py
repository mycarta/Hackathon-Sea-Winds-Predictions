#!/usr/bin/env python3
"""TIER-2 Pangu rollout batch generator -> per-date coarse 125 m extracts.

Opus-authorized Tier-2 build (Phase 0 confirmed 2026-07-17). Driver = Pangu
pangu_weather_24.onnx (CC BY-NC-SA 4.0, non-commercial), CPU EP. ERA5 initial
states from WeatherBench2 (anon GCS), single analysis time per issue date at
00 UTC (Clause 1). The global state stays in RAM (never written to disk); only
the tiny 45x57 coarse extract is persisted.

Per issue date D (00 UTC): 14-step (or 7-step) 24 h rollout, harvest the states
valid at D+7 (step 7) and D+14 (step 14), couple each to 125 m on the 45x57 grid
(tier2_pangu_couple), and write an incremental npz:
    <outdir>/extract_<YYYYMMDD>.npz : d7_u,d7_v,d14_u,d14_v (45,57) + fallback fracs
Resumable (decision 6): existing extracts are skipped; each file is written to a
.tmp then renamed, so an interrupted batch resumes rather than restarts.

Measured (tier2_smoke): ~46 s / 24 h step on CPU, ~32 GB peak RAM. Session loaded
ONCE and reused across all dates. Sequential only (2x32 GB would page).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from tier2_pangu_couple import couple_global   # noqa: E402
from tier2_era5_fetch import fetch_global        # noqa: E402
from _publication_paths import ppath  # noqa: E402  (publication tree)

# Local C: copy: cold reads of the 1.18 GB model off the Z: network drive take
# ~9 min; a local copy loads in ~5 s. (SHA-256 613a5c14..., verified in the smoke.)
MODEL = ppath("<USER_HOME>/tier2_model/pangu_weather_24.onnx",
              must_exist=False)          # the fallback below covers absence
MODEL_ZFALLBACK = ppath("<NETWORK_SHARE>/Matteo/large downloads/tier2_smoke/"
                        "models/pangu_weather_24.onnx", must_exist=False)
DEFAULT_OUT = ppath("<NETWORK_SHARE>/Matteo/large downloads/tier2_smoke/"
                    "arm_extracts", must_exist=False)   # output directory

# Pinned-artifact SHA (contract v2.1 §5a remediation, 2026-08-20). Previously this
# hash was a comment only, so a corrupted or substituted ONNX would have loaded
# silently; the downscaler pin in tier2_d7_build_submission.py:55,100-101 fails
# loud, and this now mirrors it. Assert only: no behaviour change on a good file.
MODEL_SHA = "613a5c140a1399abcaffb4dbce32af373a1f5f56c515704f5be61925bb9fdcfd"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class PanguRunner:
    """Persistent 24 h Pangu session; roll out one initial state, harvest leads."""

    def __init__(self, model=MODEL, threads: int = 0):
        import onnxruntime as ort
        if not Path(model).exists() and MODEL_ZFALLBACK.exists():
            model = MODEL_ZFALLBACK
        # Pinned-artifact gate (v2.1 §5a): verify whichever copy actually resolved,
        # C: primary or Z: fallback. Measured 1.0 s on the local copy (1.18 GB,
        # 2026-08-20); a cold Z: read is far slower, which is itself a reason to
        # keep the C: copy present.
        got_m = sha256(model)
        assert got_m == MODEL_SHA, f"MODEL SHA MISMATCH: {got_m} != {MODEL_SHA} ({model})"
        so = ort.SessionOptions()
        # CRITICAL: the default CPU mem-arena + mem-pattern pre-commit ~32 GB of
        # virtual memory, which grows pagefile.sys on the near-full C: drive to
        # exhaustion (measured). Disabling both cuts peak RSS 32 GB -> ~3 GB and
        # commit to ~3.4 GB with NO step-time penalty (43 s/step). Verified 2026-07-17.
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Thread pin (Leg B rebuild, 2026-08-21). Default 0 leaves ORT to choose,
        # which is the July behaviour byte-for-byte. Setting it records the count
        # that produced the artifacts, so a later re-run can reproduce the same
        # arithmetic on a differently loaded machine. Measured on this box: 24 h
        # step 47.4 s unpinned vs 45.2 s at 16 threads, and the outputs of
        # unpinned, 8 and 16 threads are BITWISE IDENTICAL, so the pin buys
        # provenance rather than correctness here. See
        # reports/legB_R0_preflight_20260821.md.
        if threads:
            so.intra_op_num_threads = int(threads)
        t0 = time.time()
        self.sess = ort.InferenceSession(str(model), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.load_s = time.time() - t0
        self.upper_in = next(i.name for i in self.sess.get_inputs() if len(i.shape) == 4)
        self.surf_in = next(i.name for i in self.sess.get_inputs() if len(i.shape) == 3)
        self.upper_out = next(o.name for o in self.sess.get_outputs() if len(o.shape) == 4)
        self.surf_out = next(o.name for o in self.sess.get_outputs() if len(o.shape) == 3)
        print(f"[runner] session loaded in {self.load_s:.1f}s "
              f"(intra_op_num_threads={threads or 'ORT default'})")

    def rollout(self, upper, surface, harvest=(7, 14)):
        """Return {lead_step: (upper, surface)} at the requested 24 h step counts."""
        cu, cs = upper.astype("float32"), surface.astype("float32")
        want = set(harvest)
        out = {}
        for k in range(1, max(harvest) + 1):
            cu, cs = self.sess.run([self.upper_out, self.surf_out],
                                   {self.upper_in: cu, self.surf_in: cs})
            if k in want:
                out[k] = (cu.copy(), cs.copy())
        return out


def generate(issue_dates, harvest, outdir: Path, runner: PanguRunner,
             init_hour: int = 0, name_hour: bool = False):
    """Roll out each issue date from its ERA5 init at `init_hour` UTC.

    init_hour defaults to 0 (the original 00-UTC batch behaviour, byte-unchanged).
    Option C (2026-07-19) drives init_hour in {0,6,12,18} to reach the four d7
    valid hours; those runs pass name_hour=True so the extract filename carries the
    hour (`extract_<date>_h<HH>.npz`) and the four inits never collide. Clause 1:
    init_hour UTC on the issue date lies within the window's context period, so the
    analysis read is at/before the issue time.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    log = []
    for i, D in enumerate(issue_dates, 1):
        D = pd.Timestamp(D)
        tag = f"{D:%Y%m%d}"
        fname = f"extract_{tag}_h{init_hour:02d}.npz" if name_hour else f"extract_{tag}.npz"
        dst = outdir / fname
        if dst.exists():
            print(f"[{i}/{len(issue_dates)}] {fname} exists -> skip")
            continue
        t0 = time.time()
        upper, surface = fetch_global(D, hour=init_hour)
        states = runner.rollout(upper, surface, harvest=harvest)
        data, meta = {}, {}
        for step, (u, s) in states.items():
            u125, v125, fb = couple_global(u, s)
            lead = "d7" if step == 7 else "d14" if step == 14 else f"d{step}"
            data[f"{lead}_u"] = u125
            data[f"{lead}_v"] = v125
            meta[f"{lead}_fallback_frac"] = float(np.mean(fb))
        tmp = outdir / f".{tag}_h{init_hour:02d}.tmp.npz"
        np.savez_compressed(tmp, issue_date=str(D.date()), init_hour=int(init_hour),
                            valid_d7=str((D + pd.Timedelta(days=7)).date()),
                            valid_d14=str((D + pd.Timedelta(days=14)).date()),
                            meta=json.dumps(meta), **data)
        os.replace(tmp, dst)                                   # atomic
        dt = time.time() - t0
        log.append({"issue": str(D.date()), "init_hour": int(init_hour),
                    "sec": round(dt, 1), **meta})
        print(f"[{i}/{len(issue_dates)}] {fname} leads={sorted(states)} "
              f"init={init_hour:02d}UTC {dt:.0f}s fb={meta}")
        del upper, surface, states
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", help="issue dates YYYY-MM-DD (00 UTC)")
    ap.add_argument("--dates-file", help="file with one issue date per line")
    ap.add_argument("--harvest", default="7,14", help="comma steps to harvest")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--init-hour", type=int, default=0,
                    help="ERA5 init hour UTC (Option C: 0,6,12,18)")
    ap.add_argument("--name-hour", action="store_true",
                    help="carry init hour in the extract filename (extract_<date>_h<HH>.npz)")
    ap.add_argument("--threads", type=int, default=0,
                    help="intra_op_num_threads for the CPU EP. 0 leaves ORT to "
                         "choose, which is the July behaviour. Set it to record "
                         "the count that produced the artifacts.")
    args = ap.parse_args()
    dates = list(args.dates or [])
    if args.dates_file:
        dates += [ln.strip() for ln in open(args.dates_file) if ln.strip()]
    dates = sorted(set(dates))
    harvest = tuple(int(x) for x in args.harvest.split(","))
    print(f"[batch] {len(dates)} issue dates, harvest steps {harvest}, "
          f"init {args.init_hour:02d}UTC, out {args.outdir}")
    runner = PanguRunner(Path(args.model), threads=args.threads)
    t0 = time.time()
    log = generate(dates, harvest, Path(args.outdir), runner,
                   init_hour=args.init_hour, name_hour=args.name_hour)
    print(f"[batch] generated {len(log)} new extracts in {time.time()-t0:.0f}s "
          f"({len(dates)-len(log)} skipped)")


if __name__ == "__main__":
    main()
