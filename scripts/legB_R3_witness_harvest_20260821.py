#!/usr/bin/env python3
"""Leg B R3: find witnesses for the rebuilt extracts, and say what they can gate.

The dispatch names three witness files. All three exist, and none of them can
gate the rebuilt bias table. Two independent reasons, either one sufficient:

  1. The bias table is built by `build_bias` in tier2_d7_score_blocks.py, which
     calls `downscaled()`, which uses the DOWNSCALER. The rebuilt downscaler is a
     new object by construction (the July pickle was fitted under unseeded code;
     random_state=42 arrived later, in 17baf59). So the rebuilt table cannot
     equal July's even if every extract is byte-identical. The dispatch's
     "byte-reproducible given extracts" is true only holding the downscaler
     fixed, and it is not fixed.
  2. Every number in tier2_d7_fourblock.json is a SCORE, computed over 112 block
     valid dates that read the 91 EVAL extracts. Those are not being regenerated;
     only the 80 bias dates are. Reproducing any of them would need the full 182.

So the named files are witnesses to the pipeline, not to the extracts.

A REAL witness exists elsewhere. The July smoke-test session logged, per issue
date, the `d7_fallback_frac` that `tier2_pangu_rollout.py` writes into each
extract's `meta`. That scalar is a pure function of the 7-step Pangu rollout and
the coupler: it does NOT touch the downscaler, the bias table or the scorer. It
is therefore the only recorded quantity that speaks about the extracts
themselves, and it is comparable date by date.

This script harvests those values from the archived transcript and writes them as
a witness file, so each rebuilt extract can be checked against July as it lands.

Source transcript (read-only, never resumed):
  <SESSION_ID>.jsonl

Run:  python scripts/legB_R3_witness_harvest_20260821.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from _publication_paths import ppath  # noqa: E402  (publication tree)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SESSION = str(ppath("<CLAUDE_SESSION_PATH>/<SESSION_ID>.jsonl",
                    must_exist=False))
DATEMAP = os.path.join(REPO, "scripts", "artifacts", "tier2_d7_datemap.json")
OUT = os.path.join(REPO, "scripts", "artifacts", "legB_R3_witnesses_20260821.json")

# [i/N] YYYYMMDD leads=[7] NNNs fb={'d7_fallback_frac': 0.123...}
LINE = re.compile(
    r"\[\d+/\d+\]\s+(?:extract_)?(\d{8})(?:_h\d{2}\.npz)?\s+leads=\[([^\]]*)\]"
    r"[^{]*fb=\{'d7_fallback_frac':\s*([0-9.eE+-]+)")


def texts():
    with open(SESSION, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("message")
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, str):
                    yield c
                elif isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and b.get("type") == "text":
                            yield b.get("text", "")
                        elif isinstance(b, dict) and b.get("type") == "tool_result":
                            cc = b.get("content")
                            if isinstance(cc, str):
                                yield cc
                            elif isinstance(cc, list):
                                for bb in cc:
                                    if isinstance(bb, dict) and bb.get("type") == "text":
                                        yield bb.get("text", "")
            tr = r.get("toolUseResult")
            if isinstance(tr, str):
                yield tr
            elif isinstance(tr, dict):
                yield json.dumps(tr)[:20000]


def main():
    if not os.path.isfile(SESSION):
        print("FATAL: transcript not found")
        sys.exit(2)

    found = {}
    conflicts = []
    for t in texts():
        for raw in t.replace("\\r", "\n").replace("\\n", "\n").splitlines():
            m = LINE.search(raw)
            if not m:
                continue
            d8, leads, val = m.group(1), m.group(2), float(m.group(3))
            # Only the 7-step-only rollouts are the same construction as the
            # rebuild. The 28 DJF extracts were harvested at [7, 14]; a 14-step
            # rollout visits the same step 7, so the value is still comparable,
            # but it is flagged rather than silently mixed.
            steps = [x.strip() for x in leads.split(",") if x.strip()]
            key = "%s-%s-%s" % (d8[:4], d8[4:6], d8[6:])
            rec = {"fallback_frac": val, "leads": steps}
            if key in found and found[key]["fallback_frac"] != val:
                conflicts.append((key, found[key]["fallback_frac"], val))
            found[key] = rec

    dm = json.load(open(DATEMAP))
    bias_dates = set(dm["bias_issue_to_validV"].keys())
    usable = {k: v for k, v in found.items() if k in bias_dates}

    print("recorded fallback fractions found in the transcript : %d" % len(found))
    print("of which fall on one of the 80 bias dates           : %d" % len(usable))
    print("value conflicts between duplicate log lines         : %d" % len(conflicts))
    for k, a, b in conflicts[:5]:
        print("   %s: %r vs %r" % (k, a, b))
    print()
    if usable:
        for k in sorted(usable)[:10]:
            print("  %s  %.16f  leads=%s"
                  % (k, usable[k]["fallback_frac"], usable[k]["leads"]))
        if len(usable) > 10:
            print("  ... %d more" % (len(usable) - 10))

    payload = {
        "source_session": os.path.basename(SESSION),
        "harvested": "2026-08-21",
        "what_this_is": (
            "d7_fallback_frac per issue date, as logged by tier2_pangu_rollout.py "
            "during the July 2026 batch. A pure function of the 7-step Pangu "
            "rollout and the coupler: it does not touch the downscaler, the bias "
            "table or the scorer. It is therefore the only recorded quantity that "
            "witnesses the EXTRACTS rather than the pipeline downstream of them."),
        "n_found_total": len(found),
        "n_on_bias_dates": len(usable),
        "conflicts": conflicts,
        "witnesses": usable,
        "all_found": found,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
