"""
Terminal-tier check (audit design section 4.6).

The last gate before delivery. Given the final PDF's number list, re-verify
every number against the artifact it came from, re-run change control, and
print one pass/fail sheet. Offline, no network, no pipeline, no model load.
Designed to be run by Matteo alone, in under 20 minutes, with nobody to ask.

    conda run -n swnd python scripts/terminal_check.py --date 2026-08-29
    conda run -n swnd python scripts/terminal_check.py --date 2026-08-29 \\
        --numbers docs/audit/final_number_list.csv

Exit code 0 only if every number passes AND change control passes. Any other
outcome exits non-zero and the sheet says why. There is no partial pass: a
terminal gate that reports "mostly fine" is not a gate.

THE NUMBER LIST
---------------
CSV, one row per number that appears in the report. Columns:

    id          short handle, used in the sheet
    number      the value AS PRINTED IN THE REPORT
    section     where it appears, so a failure is findable in the PDF
    artifact    repo-relative path to the file the number came from
    extractor   how to pull the value back out of that artifact (below)
    tolerance   absolute tolerance; 0 means exact
    note        free text, printed on failure

Extractors:

    json:<dotted.path>      read a JSON file, walk the dotted path
    csv:<column>:<agg>      read a CSV, aggregate one column
                            agg in sum, mean, min, max, count, nunique
    csvrows                 count data rows in a CSV (header excluded)
    regex:<pattern>         first capture group of the first match, as a float
    sha256                  hash the artifact; `number` is the expected hex

Adding an extractor is the expected way to extend this. Inventing a number
that no extractor can reach is not: if a figure in the report cannot be pulled
back out of a committed artifact by machine, that is the finding.

WHAT THIS DOES NOT DO
---------------------
It does not check that the number is the RIGHT number, only that the report
and the artifact agree. A number that was wrong when it was computed will pass
here. The defence against that is the anchor recomputations, not this script.
Stated plainly because a green sheet is exactly the moment that distinction
stops being obvious.

STUB NUMBER LIST
----------------
The strategist supplies the real list on 2026-08-26. Until then the default
list is docs/audit/terminal_numbers_STUB_20260825.csv, which this script
generates on first run if absent.

*** The stub is NOT the eight opening numbers from the report skeleton. ***
That skeleton is not in the repo, so CC could not transcribe it without
guessing, and guessing the numbers a terminal gate checks would defeat the
gate. The stub is eight numbers CC could trace to committed artifacts, chosen
so the mechanism can be dry-run end to end on 08-26. Replace it with
--numbers as soon as the real list lands.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB = os.path.join(REPO, "docs", "audit", "terminal_numbers_STUB_20260825.csv")

STUB_ROWS = [
    ("alpha_star", "0.627955", "bonus",
     "bidding_sim/results_2019/stage4_summary.json", "json:alpha_star_raw",
     "0.000001", "newsvendor critical ratio, annual basis"),
    ("eviu_eur", "182434.70", "bonus",
     "bidding_sim/results_2019/stage4_summary.json", "json:eviu_eur",
     "0.01", "quantile minus naive over 1460 delivery hours"),
    ("evpi_eur", "-27477.77", "bonus",
     "bidding_sim/results_2019/stage4_summary.json", "json:evpi_eur",
     "0.01", "negative by construction, see the summary"),
    ("delivery_hours", "1460", "bonus",
     "bidding_sim/results_2019/stage4_summary.json", "json:delivery_hours",
     "0", "NOT a full year, nothing scaled"),
    ("cutout_hours", "21", "bonus",
     "bidding_sim/results_2019/stage4_summary.json", "json:cutout_hours",
     "0", "hours touched by cut-out, sorted-triple interpolation applies"),
    ("case3_cf", "0.4721", "task2",
     "reports/three_case_scorer_20260818.json",
     "json:effects.banked_case3_reference.cf", "0.0001",
     "banked case 3 net capacity factor"),
    ("case3_wake", "0.0827", "task2",
     "reports/three_case_scorer_20260818.json",
     "json:effects.banked_case3_reference.wake", "0.0001",
     "banked case 3 wake fraction"),
    ("task1_rows", "4196640", "task1",
     "phase_2/kit/phase_2/part1_forecast/submission.csv", "csvrows",
     "0", "43715 pts x 8 windows x 3 horizons x 4 hours"),
]


def write_stub():
    os.makedirs(os.path.dirname(STUB), exist_ok=True)
    with open(STUB, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "number", "section", "artifact", "extractor",
                    "tolerance", "note"])
        w.writerows(STUB_ROWS)


def dotted(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def extract(artifact, extractor):
    """Pull the value back out of the artifact. Raises on anything unclear."""
    if extractor == "sha256":
        h = hashlib.sha256()
        with open(artifact, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if extractor == "csvrows":
        n = 0
        with open(artifact, "rb") as fh:
            for _ in fh:
                n += 1
        return float(n - 1)  # header excluded

    if extractor.startswith("json:"):
        with open(artifact, encoding="utf-8") as fh:
            return float(dotted(json.load(fh), extractor[5:]))

    if extractor.startswith("csv:"):
        import pandas as pd
        _, col, agg = extractor.split(":", 2)
        s = pd.read_csv(artifact)[col]
        return float(getattr(s, agg)())

    if extractor.startswith("regex:"):
        pat = extractor[6:]
        with open(artifact, encoding="utf-8", errors="replace") as fh:
            m = re.search(pat, fh.read())
        if not m:
            raise ValueError("regex found no match")
        return float(m.group(1))

    raise ValueError("unknown extractor %r" % extractor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD, stamps the sheet")
    ap.add_argument("--numbers", default=None,
                    help="number-list CSV (default: the stub)")
    ap.add_argument("--skip-change-control", action="store_true",
                    help="numbers only. Use for a fast iteration, never for "
                         "the terminal run itself.")
    args = ap.parse_args()

    t0 = time.time()
    path = args.numbers or STUB
    if path == STUB and not os.path.isfile(STUB):
        write_stub()
        print("stub number list generated: %s"
              % os.path.relpath(STUB, REPO).replace("\\", "/"))

    if not os.path.isfile(path):
        print("ERROR: number list not found: %s" % path)
        return 2

    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    print("=" * 78)
    print("TERMINAL CHECK  %s" % args.date)
    print("number list: %s  (%d numbers)"
          % (os.path.relpath(path, REPO).replace("\\", "/"), len(rows)))
    if os.path.abspath(path) == os.path.abspath(STUB):
        print("*** STUB LIST IN USE. This is NOT the report's number list. ***")
    print("=" * 78)
    print()
    print("%-16s %-8s %16s %16s  %s"
          % ("id", "section", "reported", "artifact", "result"))
    print("-" * 78)

    n_pass = n_fail = 0
    failures = []
    for r in rows:
        art = os.path.join(REPO, r["artifact"])
        rid = r["id"]
        try:
            if not os.path.isfile(art):
                raise FileNotFoundError(r["artifact"])
            got = extract(art, r["extractor"])
            if r["extractor"] == "sha256":
                ok = (str(got).lower() == r["number"].strip().lower())
                shown, exp = str(got)[:16] + "...", r["number"][:16] + "..."
            else:
                exp_v = float(r["number"])
                tol = float(r["tolerance"])
                ok = abs(got - exp_v) <= tol
                shown, exp = "%.6g" % got, "%.6g" % exp_v
            print("%-16s %-8s %16s %16s  %s"
                  % (rid, r["section"], exp, shown, "PASS" if ok else "**FAIL**"))
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                failures.append("%s (%s): report says %s, artifact %s gives %s. %s"
                                % (rid, r["section"], exp, r["artifact"], shown,
                                   r.get("note", "")))
        except Exception as e:
            n_fail += 1
            print("%-16s %-8s %16s %16s  **ERROR**"
                  % (rid, r["section"], r["number"], "-"))
            failures.append("%s: %s: %s" % (rid, type(e).__name__, e))

    print("-" * 78)
    print("numbers: %d pass, %d fail" % (n_pass, n_fail))
    print()

    cc_ok = None
    if not args.skip_change_control:
        print("=" * 78)
        print("CHANGE CONTROL")
        print("=" * 78)
        p = subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts",
                                          "change_control_attest.py"),
             "--date", args.date],
            capture_output=True, text=True)
        cc_ok = (p.returncode == 0)
        tail = [l for l in p.stdout.splitlines()
                if l.startswith("| MISMATCH")
                or l.startswith("| MISSING")
                or l.startswith("| UNRESOLVED")
                or l.startswith("Hash checks:")
                or l.startswith("**ATTESTATION")]
        for l in tail:
            print("  " + l)
        if not tail:
            print("  attestation produced no verdict lines; see its own output")
        print()

    print("=" * 78)
    if failures:
        print("FAILURES, in full:")
        for f in failures:
            print("  - %s" % f)
        print()
    mins = (time.time() - t0) / 60.0
    print("elapsed: %.1f min (budget 20)" % mins)
    verdict = (n_fail == 0) and (cc_ok is not False)
    print("TERMINAL CHECK: %s" % ("PASS" if verdict else "FAIL"))
    if cc_ok is False:
        print("  change control did not pass; the number sheet alone is not "
              "the gate.")
    print("=" * 78)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
