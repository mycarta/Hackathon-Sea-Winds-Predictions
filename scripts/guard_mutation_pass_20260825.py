"""
Guard mutation pass (audit design section 4.4), 2026-08-25.

For each high-consequence guard: inject a known-bad value upstream, confirm the
guard fires, and show the repo is untouched afterwards.

METHOD, and the one thing that makes this test worth anything. Every mutation
is applied to a SCRATCH COPY and the guard is exercised through the REAL code
path, by importing the actual module and repointing its path constants. No
guard logic is reimplemented here. A test that re-expresses the assertion it is
testing proves only that the test author can write the assertion twice.

The corollary: a mutation that fails to make the real guard fire is a FINDING.
Per the dispatch, findings are reported, never repaired in this pass.

Guards:
  a  Task 1 submission validator: row count, quantile ordering, direction range
     (scripts/validate_task1_submission.py)
  b  DST-aware ISP count in the TenneT pull: 92 on 2019-03-31, 100 on
     2019-10-27 (bidding_sim/fetch_tennet_settlement_2019.py)
  c  spend counter (bidding_sim/fetch_tennet_settlement_2019.py, _spent)
  d  downscaler SHA load-site assert
     (scripts/tier2_d7_build_submission.py:100-101)

Guard (a) note. The real submission.csv is 4,196,640 rows and 432 MB, and the
row-count guard is a comparison against a constant. Exercising the ordering and
range guards on a 432 MB file would make this pass take longer than it is worth
and would still not test them independently, because a file that fails the row
count fails the run regardless. So a small synthetic submission is built to the
real schema and EXPECTED_ROWS is repointed to its length. What is under test is
the guard logic, not the value of the constant; the constant is checked
separately by mutation a1, which moves the data and leaves the constant alone.

Deterministic. Synthetic data is seeded (numpy default_rng(42)).

Run:  conda run -n swnd python scripts/guard_mutation_pass_20260825.py
"""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(tempfile.gettempdir(), "guard_mutation_20260825")

RESULTS = []


def record(guard, mutation, fired, evidence):
    RESULTS.append({"guard": guard, "mutation": mutation,
                    "guard_fired": bool(fired), "evidence": evidence})
    print("  %-46s fired=%-5s  %s"
          % (mutation, fired, evidence.replace("\n", " ")[:90]))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Guard (a): Task 1 submission validator
# ---------------------------------------------------------------------------
COLS = ["type", "window", "region", "latitude", "longitude", "horizon", "hour",
        "level", "q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]


def synth_submission(n_points=20):
    """A small submission with the real schema, valid on every checked axis."""
    rng = np.random.default_rng(42)
    rows = []
    for w in range(8):
        for hz in (1, 7, 14):
            for hr in (0, 6, 12, 18):
                for p in range(n_points):
                    q50 = float(rng.uniform(5, 15))
                    # half is bounded by q50 so q05 stays non-negative: the
                    # control run has to pass EVERY check, not just the three
                    # under test, or the mutations are being read against a
                    # baseline that was already failing.
                    half = float(rng.uniform(0.5, min(4.0, q50 - 0.5)))
                    d50 = float(rng.uniform(10, 350))
                    dh = float(rng.uniform(5, 9))
                    rows.append(["grid", w, "north_sea",
                                 round(51.0 + 0.01 * p, 2),
                                 round(0.9 + 0.01 * p, 2), hz, hr, "125m",
                                 round(q50 - half, 3), round(q50, 3),
                                 round(q50 + half, 3),
                                 round(d50 - dh, 3), round(d50, 3),
                                 round(d50 + dh, 3)])
    return pd.DataFrame(rows, columns=COLS)


def run_validator(df, expected_rows):
    """Run the REAL validator against a scratch submission. Returns (ok, text)."""
    d = os.path.join(SCRATCH, "a")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    csv_p = os.path.join(d, "submission.csv")
    zip_p = os.path.join(d, "submission.zip")
    rep_p = os.path.join(d, "report.md")
    df.to_csv(csv_p, index=False)
    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_p, "submission.csv")

    mod = load_module(os.path.join(REPO, "scripts",
                                   "validate_task1_submission.py"), "v_t1")
    from pathlib import Path
    mod.CSV_PATH = Path(csv_p)
    mod.ZIP_PATH = Path(zip_p)
    mod.REPORT_PATH = Path(rep_p)
    mod.EXPECTED_ROWS = expected_rows

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            mod.main()
    except SystemExit:
        pass
    except Exception:
        buf.write("\nEXCEPTION\n" + traceback.format_exc())
    text = buf.getvalue()
    if os.path.isfile(rep_p):
        text += "\n" + open(rep_p, encoding="utf-8").read()
    return ("FAIL" not in text), text


def guard_a():
    print("\n(a) Task 1 submission validator")
    base = synth_submission()
    n = len(base)

    ok, text = run_validator(base, n)
    record("a", "baseline, unmutated (control)", not ok is False and ok,
           "clean run PASSes: %s" % ok)
    if not ok:
        print(text[-1500:])

    # a1: row count. Drop one row, leave EXPECTED_ROWS alone.
    m = base.iloc[:-1].copy()
    ok1, t1 = run_validator(m, n)
    line = [l for l in t1.splitlines() if l.startswith("- rows:")]
    record("a", "a1 row count: dropped 1 row of %d" % n, not ok1,
           line[0] if line else "row-count line not found in output")

    # a2: quantile ordering. Push q95 below q05 in one row.
    m = base.copy()
    m.loc[7, "q95"] = m.loc[7, "q05"] - 1.0
    ok2, t2 = run_validator(m, n)
    line = [l for l in t2.splitlines() if "q05 <= q50 <= q95" in l]
    record("a", "a2 quantile order: q95 := q05 - 1 on row 7", not ok2,
           line[0] if line else "ordering line not found in output")

    # a3: direction range. Put one direction outside [0, 360).
    m = base.copy()
    m.loc[11, "dir_95"] = 400.0
    ok3, t3 = run_validator(m, n)
    line = [l for l in t3.splitlines() if "directions in [0, 360)" in l]
    record("a", "a3 direction range: dir_95 := 400.0 on row 11", not ok3,
           line[0] if line else "direction-range line not found in output")


# ---------------------------------------------------------------------------
# Guard (b): DST-aware ISP count
# ---------------------------------------------------------------------------
def guard_b():
    print("\n(b) DST-aware ISP count in the TenneT pull")
    src_raw = os.path.join(REPO, "bidding_sim", "market_data_2019", "raw_tennet")
    if not os.path.isdir(src_raw):
        record("b", "cache copy", False,
               "SKIPPED: %s absent, cannot exercise offline" % src_raw)
        return

    d = os.path.join(SCRATCH, "b")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    raw = os.path.join(d, "raw_tennet")
    shutil.copytree(src_raw, raw)

    # Match the window that STARTS on 01-03. A bare "20190301" substring also
    # matches settlement_20190201_20190301.json, whose name ends on that date,
    # and listdir returns it first.
    march = [f for f in os.listdir(raw) if f.startswith("settlement_20190301_")]
    if not march:
        record("b", "locate March cache file", False,
               "SKIPPED: no March window in the cache")
        return
    mp = os.path.join(raw, march[0])
    payload = json.load(open(mp, encoding="utf-8"))

    # Find the list of points and drop 4 ISPs from the DST day, so the day
    # carries 88 instead of the 92 a 23 hour local day must have. 92 is the
    # value the guard exists to insist on, so 88 is the sharpest mutation:
    # it is also what a naive 96-per-day check would already have rejected.
    def points_of(payload):
        """Response -> TimeSeries[] -> Period -> Points, the shape the cached
        API responses actually carry. Verified against the March window."""
        out = []
        for ts in payload.get("Response", {}).get("TimeSeries", []) or []:
            pts_ = (ts.get("Period", {}) or {}).get("Points", []) or []
            out.append(pts_)
        return out

    lists = points_of(payload)
    pts = lists[0] if lists else None
    if not pts:
        record("b", "parse cached payload", False,
               "SKIPPED: cache shape not recognised, no mutation applied")
        return

    def is_dst_day(p):
        return str(p.get("timeInterval_start", "")).startswith("2019-03-31")

    dst_pts = [i for i, p in enumerate(pts) if is_dst_day(p)]
    if len(dst_pts) < 8:
        record("b", "locate 2019-03-31 points", False,
               "SKIPPED: found %d points on the DST day, expected 92"
               % len(dst_pts))
        return
    for i in sorted(dst_pts[-4:], reverse=True):
        pts.pop(i)
    json.dump(payload, open(mp, "w", encoding="utf-8"))

    mod = load_module(os.path.join(REPO, "bidding_sim",
                                   "fetch_tennet_settlement_2019.py"), "tn_f")
    mod.OUTDIR = d
    mod.RAWDIR = raw
    mod.OUT_PARQUET = os.path.join(d, "out.parquet")

    fired, evidence = False, ""
    buf = io.StringIO()
    old_argv = sys.argv[:]
    sys.argv = ["fetch_tennet_settlement_2019.py"]
    try:
        with redirect_stdout(buf):
            mod.main()
    except AssertionError as e:
        fired, evidence = True, "AssertionError: %s" % str(e).splitlines()[0]
    except SystemExit as e:
        out = buf.getvalue()
        fired = "DST signature" in out or "expected 92" in out
        evidence = "SystemExit(%s); %s" % (
            e.code, [l for l in out.splitlines() if "31-03" in l or "DST" in l][:2])
    except Exception as e:
        out = buf.getvalue()
        fired = "DST" in out or "92" in str(e)
        evidence = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:120])
    finally:
        sys.argv = old_argv
    if not fired and not evidence:
        evidence = "ran to completion with 88 ISPs on the DST day"
    record("b", "b1 dropped 4 ISPs from 2019-03-31 (92 -> 88)", fired, evidence)


# ---------------------------------------------------------------------------
# Guard (c): spend counter
# ---------------------------------------------------------------------------
def guard_c():
    """The spend counter.

    NAMING, reported rather than smoothed over. The dispatch calls this "the
    submission spend counter". There is no submission spend counter in the
    repo: the competition submission budget is tracked in prose, not in code.
    The only spend counter that exists is the TenneT API request counter
    (`_spent`, fetch_tennet_settlement_2019.py:121), which is what is tested
    here. See the report-back.

    The existing self-test, test_fetch_stop_path_20260820.py, cannot be
    re-run: its first assertion requires that no parquet and no cache exist,
    which stopped being true the moment the real pull succeeded on 08-21. So
    the counter is exercised directly instead, through the real module, with
    its paths repointed at an empty scratch directory and its HTTP call
    stubbed to a 401."""
    print("\n(c) spend counter")

    d = os.path.join(SCRATCH, "c")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "raw_tennet"))

    os.environ.setdefault("TENNET_API_KEY", "dummy-key-for-offline-test-0000")
    mod = load_module(os.path.join(REPO, "bidding_sim",
                                   "fetch_tennet_settlement_2019.py"), "tn_c")
    mod.OUTDIR = d
    mod.RAWDIR = os.path.join(d, "raw_tennet")
    mod.OUTFILE = os.path.join(d, "tennet_settlement_prices_2019.parquet")

    calls = {"n": 0}

    class FakeResp(object):
        status_code = 401
        text = "unauthorized"
        def json(self):
            return {}

    def fake_get(*a, **k):
        calls["n"] += 1
        return FakeResp()

    mod.requests.get = fake_get

    buf = io.StringIO()
    old_argv = sys.argv[:]
    sys.argv = ["fetch_tennet_settlement_2019.py"]
    code = None
    try:
        with redirect_stdout(buf):
            mod.main()
    except SystemExit as e:
        code = e.code
    except Exception as e:
        code = "%s: %s" % (type(e).__name__, e)
    finally:
        sys.argv = old_argv
    out = buf.getvalue()

    # Three things must all hold: exactly one request issued, the REPORTED
    # spend equal to it, and no parquet written on an incomplete run.
    reported = "requests spent this run : %d" % calls["n"] in out
    one_shot = calls["n"] == 1
    no_parquet = not os.path.isfile(mod.OUTFILE)
    fired = one_shot and reported and no_parquet
    record("c", "c1 first request returns 401 (real module, stubbed HTTP)",
           fired,
           "issued=%d reported_spend_matches=%s no_parquet=%s exit=%s"
           % (calls["n"], reported, no_parquet, code))

    record("c", "c2 existing self-test re-runnable?", False,
           "FINDING: test_fetch_stop_path_20260820.py asserts no parquet and "
           "no cache exist, so it cannot run again now that the 08-21 pull "
           "succeeded. One-shot test, not a standing guard.")


# ---------------------------------------------------------------------------
# Guard (d): downscaler SHA load-site assert
# ---------------------------------------------------------------------------
def guard_d():
    print("\n(d) downscaler SHA load-site assert")
    src = os.path.join(REPO, "scripts", "tier2_d7_build_submission.py")
    if not os.path.isfile(src):
        record("d", "locate the build script", False, "SKIPPED: %s absent" % src)
        return
    d = os.path.join(SCRATCH, "d")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    dst = os.path.join(d, "tier2_d7_build_submission_MUTATED.py")

    text = open(src, encoding="utf-8").read()
    real = "b3ae32c0bf4203351a03526a454030817f70adb588f55caefdb0b43b5a2d8703"
    bad = "0" * 64
    if real not in text:
        record("d", "locate DWN_CACHE_SHA constant", False,
               "SKIPPED: pinned constant not found in the script")
        return
    open(dst, "w", encoding="utf-8").write(text.replace(real, bad))

    # Run from the real scripts/ directory so sibling imports resolve; only the
    # mutated copy differs. The frozen file is never written to.
    # The mutated copy lives in scratch, so scripts/ must be on PYTHONPATH or
    # the run dies on sibling imports and never reaches the guard. That would
    # look identical to an unreachable guard while being purely an artifact of
    # how this harness stages the copy.
    env = dict(os.environ)
    env["PYTHONPATH"] = (os.path.join(REPO, "scripts") + os.pathsep
                         + env.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, dst], cwd=os.path.join(REPO, "scripts"),
                       capture_output=True, text=True, timeout=1800, env=env)
    out = p.stdout + p.stderr
    fired = "DOWNSCALER SHA MISMATCH" in out
    if fired:
        line = [l for l in out.splitlines() if "DOWNSCALER SHA MISMATCH" in l][0]
    else:
        # Distinguish "the guard is broken" from "the guard is unreachable".
        # They call for different responses and must not be reported alike.
        stale_root = os.path.join(REPO, "phase_2", "build", "phase2_dataset")
        missing = [l for l in out.splitlines()
                   if "FileNotFoundError" in l or "No such file" in l]
        if not os.path.isdir(stale_root) and missing:
            line = ("guard UNREACHABLE, not broken. The script dies before "
                    "main() runs: it reads the dataset root "
                    "phase_2/build/phase2_dataset/, which no longer exists "
                    "(the tree now lives under "
                    "phase_2/inference_2022/phase2_dataset_ship/). "
                    "Last error: " + missing[-1].strip()[:150])
        else:
            line = ("guard NOT reached. tail: "
                    + out.strip()[-200:].replace("\n", " | "))
    record("d", "d1 DWN_CACHE_SHA := 64 zeros (scratch copy)", fired, line)


def main():
    if not os.path.isdir(SCRATCH):
        os.makedirs(SCRATCH)
    print("guard mutation pass, 2026-08-25")
    print("scratch: %s" % SCRATCH)

    for fn in (guard_a, guard_b, guard_c, guard_d):
        try:
            fn()
        except Exception:
            record(fn.__name__[-1], "harness error", False,
                   traceback.format_exc().strip().splitlines()[-1])

    print("\n--- revert check ---")
    diff = subprocess.run(["git", "diff", "--stat"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    st = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                        capture_output=True, text=True).stdout
    mod = [l for l in st.splitlines() if l and not l.startswith("??")]
    print("git diff --stat: %s" % (diff if diff else "(empty)"))
    print("modified tracked paths: %d" % len(mod))
    for m in mod:
        print("  %s" % m)

    print("\n--- summary ---")
    fired = sum(1 for r in RESULTS if r["guard_fired"])
    print("%d of %d checks reported as expected" % (fired, len(RESULTS)))
    out = os.path.join(REPO, "reports", "guard_mutation_pass_20260825.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"date": "2026-08-25", "results": RESULTS,
                   "git_diff_stat": diff, "modified_tracked": mod},
                  fh, indent=2)
    print("written: reports/guard_mutation_pass_20260825.json")


if __name__ == "__main__":
    main()
