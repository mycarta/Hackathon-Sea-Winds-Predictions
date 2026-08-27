#!/usr/bin/env python3
"""Leg B rebuild, R0: preflight before committing 8 hours of CPU.

Checks, in the order they can fail cheaply:

  1. Pinned-artifact gate (contract v2.1 5a(iii)): SHA-verify the Pangu ONNX in
     both its resolved location and its protected copy, since a frozen run is on
     the calendar.
  2. Disk headroom on C:, against the measured ~3.4 GB commit of the rollout and
     the trivial output size.
  3. THE ONE THAT CAN KILL THE DAY: the WeatherBench2 ERA5 mirror. Initial states
     are fetched live from anonymous GCS at run time and nothing is cached
     locally, so a rollout that fails on date 60 of 80 wastes most of a day.
     This opens the store and pulls one real initial state for the first bias
     date, end to end.
  4. Thread pinning. tier2_pangu_rollout.py does not set intra_op_num_threads, so
     ONNX Runtime's CPU provider picks a count from the machine and may reorder
     float reductions if that count differs between runs. One 24 h step is timed
     at the default and at a pinned count, to measure both the cost of pinning
     and whether the two agree bitwise on identical input.

READ-ONLY. Writes one report. Rolls no date and produces no extract.

Run:  python scripts/legB_R0_preflight_20260821.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from _publication_paths import ppath  # noqa: E402  (publication tree)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

MODEL_C = ppath("<USER_HOME>/tier2_model/pangu_weather_24.onnx",
                must_exist=False)
MODEL_PROTECTED = ppath("<PROTECTED_ARTIFACTS>/pangu_weather_24.onnx",
                        must_exist=False)
MODEL_SHA = "613a5c140a1399abcaffb4dbce32af373a1f5f56c515704f5be61925bb9fdcfd"
DATEMAP = REPO / "scripts" / "artifacts" / "tier2_d7_datemap.json"
OUT = REPO / "reports" / "legB_R0_preflight_20260821.md"

# Overridable so the pinned count can be tuned against the default without
# guessing: pinning is only free if it matches whatever ORT picks on its own.
PIN_THREADS = int(os.environ.get("LEGB_PIN_THREADS", "8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    t_all = time.time()
    L = []
    ok = True

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    say("# Leg B rebuild, R0 preflight")
    say("")
    say("**2026-08-21**, `scripts/legB_R0_preflight_20260821.py`. Read-only.")
    say("")

    # ---------------------------------------------------------------- 1. SHA
    say("## 1. Pinned-artifact gate (v2.1 5a(iii))")
    say("")
    say("| Copy | Present | SHA-256 matches |")
    say("|---|---|---|")
    for label, p in (("resolved load path (C:)", MODEL_C),
                     ("protected copy", MODEL_PROTECTED)):
        if not p.exists():
            say("| %s | NO | n/a |" % label)
            ok = False
            continue
        t0 = time.time()
        got = sha256(p)
        good = got == MODEL_SHA
        ok = ok and good
        say("| %s | yes | **%s** (%.1f s) |" % (label, "yes" if good else "NO", time.time() - t0))
    say("")

    # --------------------------------------------------------------- 2. disk
    say("## 2. Disk")
    say("")
    # Was a hardcoded drive letter. The repo's own anchor is the drive we
    # actually care about, and it needs no configuration.
    _drive = str(Path(REPO).anchor)
    total, used, free = shutil.disk_usage(_drive)
    free_gb = free / 1e9
    say("%s free **%.1f GB**. The rollout commits about 3.4 GB with the memory "
        "options the production script sets, and 80 extracts at about 20 KB each "
        "is under 2 MB of output." % (_drive, free_gb))
    disk_ok = free_gb > 8.0
    ok = ok and disk_ok
    say("")
    say("Headroom verdict: **%s**" % ("OK" if disk_ok else "TOO TIGHT"))
    say("")

    # ------------------------------------------------------------- 3. dates
    dm = json.load(open(DATEMAP))
    bias_dates = sorted(dm["bias_issue_to_validV"].keys())
    say("## 3. Work list")
    say("")
    say("`bias_issue_to_validV` holds **%d** issue dates, %s to %s, seed %s, "
        "%s per season. Only these are needed: the July batch also covered 91 eval "
        "and 13 calib dates for scoring, which the rebuild does not repeat."
        % (len(bias_dates), bias_dates[0], bias_dates[-1],
           dm.get("seed"), dm.get("bias_per_season")))
    say("")
    if len(bias_dates) != 80:
        say("**UNEXPECTED: %d dates, not 80.**" % len(bias_dates))
        ok = False

    # -------------------------------------------------------------- 4. ERA5
    say("## 4. ERA5 reachability (the one that can kill the day)")
    say("")
    era_ok = False
    upper = surface = None
    try:
        import tier2_era5_fetch as ef
        say("Store: `%s`" % ef.WB2_ZARR)
        say("")
        t0 = time.time()
        ds = ef._ds()
        t_open = time.time() - t0
        say("- store opened in **%.1f s**" % t_open)
        t0 = time.time()
        upper, surface = ef.fetch_global(bias_dates[0], hour=0)
        t_fetch = time.time() - t0
        say("- one initial state for %s 00 UTC fetched in **%.1f s**" % (bias_dates[0], t_fetch))
        say("- upper %s %s, surface %s %s"
            % (upper.shape, upper.dtype, surface.shape, surface.dtype))
        shape_ok = (upper.shape == (5, 13, 721, 1440) and surface.shape == (4, 721, 1440))
        fin_ok = bool(np.isfinite(upper).all() and np.isfinite(surface).all())
        say("- shapes as Pangu expects: **%s**" % ("yes" if shape_ok else "NO"))
        say("- all finite: **%s**" % ("yes" if fin_ok else "NO"))
        era_ok = shape_ok and fin_ok
        say("")
        say("- projected fetch cost across 80 dates: about **%.0f min** of the run, "
            "on top of inference" % (80 * t_fetch / 60.0))
    except Exception as exc:
        say("**FAILED: %s: %s**" % (type(exc).__name__, exc))
    ok = ok and era_ok
    say("")
    say("ERA5 verdict: **%s**" % ("REACHABLE" if era_ok else "BLOCKED"))
    say("")

    # ------------------------------------------------------------ 5. threads
    say("## 5. Thread pinning")
    say("")
    say("Logical CPUs visible: **%s**. `tier2_pangu_rollout.py` sets no "
        "`intra_op_num_threads`, so ONNX Runtime chooses." % os.cpu_count())
    say("")
    if era_ok:
        import onnxruntime as ort
        say("onnxruntime **%s**, provider CPUExecutionProvider." % ort.__version__)
        say("")

        def build(threads):
            so = ort.SessionOptions()
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if threads:
                so.intra_op_num_threads = threads
            t0 = time.time()
            s = ort.InferenceSession(str(MODEL_C), sess_options=so,
                                     providers=["CPUExecutionProvider"])
            return s, time.time() - t0

        def one_step(sess, u, s):
            ui = next(i.name for i in sess.get_inputs() if len(i.shape) == 4)
            si = next(i.name for i in sess.get_inputs() if len(i.shape) == 3)
            uo = next(o.name for o in sess.get_outputs() if len(o.shape) == 4)
            so_ = next(o.name for o in sess.get_outputs() if len(o.shape) == 3)
            t0 = time.time()
            ru, rs = sess.run([uo, so_], {ui: u.astype("float32"), si: s.astype("float32")})
            return ru, rs, time.time() - t0

        rows = []
        results = {}
        for label, threads in (("default (unpinned)", 0), ("pinned to %d" % PIN_THREADS, PIN_THREADS)):
            sess, t_load = build(threads)
            ru, rs, t_step = one_step(sess, upper, surface)
            eff = sess.get_session_options().intra_op_num_threads
            rows.append((label, eff, t_load, t_step))
            results[label] = (ru, rs)
            del sess
        say("| Configuration | intra_op_num_threads | session load | one 24 h step |")
        say("|---|---|---|---|")
        for label, eff, t_load, t_step in rows:
            say("| %s | %d | %.1f s | **%.1f s** |" % (label, eff, t_load, t_step))
        say("")

        (a_u, a_s), (b_u, b_s) = results[rows[0][0]], results[rows[1][0]]
        identical = bool(np.array_equal(a_u, b_u) and np.array_equal(a_s, b_s))
        max_abs = float(max(np.abs(a_u - b_u).max(), np.abs(a_s - b_s).max()))
        say("Same initial state through both sessions:")
        say("")
        say("- bitwise identical: **%s**" % ("yes" if identical else "NO"))
        say("- max absolute difference: **%.3e**" % max_abs)
        say("")
        if identical:
            say("Pinning is therefore not required for reproducibility on THIS machine "
                "at these two thread counts, but it costs nothing to set and removes "
                "the question on any other machine. Recommended for the run.")
        else:
            say("**Thread count changes the result.** The difference is tiny in "
                "absolute terms, but it means a byte-exact claim about the rebuilt "
                "extracts is only meaningful with the thread count pinned and "
                "recorded. Pinning is REQUIRED, not optional.")
        say("")
        step_s = rows[1][3]
        say("Projected extract clock at the pinned rate: 7 steps per date, %d dates, "
            "**%.1f h** of inference, plus about %.0f min of ERA5 fetch."
            % (len(bias_dates), 7 * step_s * len(bias_dates) / 3600.0,
               80 * t_fetch / 60.0))
        say("")
    else:
        say("Skipped: no initial state to run.")
        say("")

    say("## Verdict")
    say("")
    say("**R0 %s**" % ("CLEAN" if ok else "NOT CLEAN, do not start R1"))
    say("")
    say("Preflight wall time %.1f s." % (time.time() - t_all))

    text = "\n".join(L) + "\n"
    bad = sorted(set(c for c in text if ord(c) > 126))
    assert not bad, "non-ASCII in report: %r" % bad
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print("\nwrote %s" % OUT)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
