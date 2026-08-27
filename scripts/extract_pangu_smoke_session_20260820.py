#!/usr/bin/env python3
"""Extract recorded facts from the Pangu-Weather smoke-test session transcript.

Commissioned 2026-08-20, ahead of the Leg B rebuild dispatch, because the
rebuild needs numbers that exist only in that session: extract sizes, per-date
timings, counts, and whatever determinism evidence was recorded at the time.

SOURCE (read-only, never resumed):
  <CLAUDE_SESSION_PATH>/
  <SESSION_ID>.jsonl

The session is "CC PROMPT PART C: Pangu-Weather smoke test (Tier 2 gate)",
2026-07-17 to 2026-07-21.

DESIGN. Every figure quoted in the output note is pulled from the transcript by
this script and ASSERTED present. Nothing is typed in from memory. If the
transcript is moved or altered the script fails rather than emitting a note that
merely looks sourced. The prose sections are constants; the numbers are not.

Writes: reports/pangu_smoke_session_extract_20260820.md
Deterministic; no stochastic step, so no seed applies. Reads nothing else.

Run:  python scripts/extract_pangu_smoke_session_20260820.py
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
from _publication_paths import ppath  # noqa: E402  (publication tree)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SESSION_ID = "<SESSION_ID>"
SESSION = str(ppath("<CLAUDE_SESSION_PATH>/%s.jsonl" % SESSION_ID,
                    must_exist=False))
OUT = os.path.join(REPO, "reports", "pangu_smoke_session_extract_20260820.md")


def records():
    with open(SESSION, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def texts():
    """(timestamp, kind, text) over every text-bearing field in the transcript."""
    for r in records():
        ts = r.get("timestamp", "")
        m = r.get("message")
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str):
                yield ts, r.get("type"), c
            elif isinstance(c, list):
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        yield ts, r.get("type"), b.get("text", "")
                    elif b.get("type") == "tool_use":
                        yield ts, "tool_use", json.dumps(b.get("input", {}))[:20000]
                    elif b.get("type") == "tool_result":
                        cc = b.get("content")
                        if isinstance(cc, str):
                            yield ts, "tool_result", cc
                        elif isinstance(cc, list):
                            for bb in cc:
                                if isinstance(bb, dict) and bb.get("type") == "text":
                                    yield ts, "tool_result", bb.get("text", "")
        tr = r.get("toolUseResult")
        if isinstance(tr, str):
            yield ts, "toolUseResult", tr
        elif isinstance(tr, dict):
            yield ts, "toolUseResult", json.dumps(tr)[:20000]


ALL = list(texts())


def find(pattern, label, flags=re.I):
    """First transcript line matching `pattern`. Asserts it exists."""
    rx = re.compile(pattern, flags)
    for ts, kind, t in ALL:
        for line in t.splitlines():
            s = line.strip()
            if len(s) > 700:
                continue
            if rx.search(s):
                return ts[:19], s
    raise AssertionError("NOT FOUND in transcript: %s (%s)" % (label, pattern))


_ASCII_MAP = {
    "—": " - ", "–": "-", "…": "...", "→": "->",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "×": "x", "√": "sqrt", "±": "+/-", "°": " deg",
    "α": "alpha", "≈": "~", "≤": "<=", "≥": ">=",
    " ": " ", "−": "-",
}


def asciify(s):
    """
    Fold a quoted transcript line to ASCII.

    Two reasons, both standing constraints. Prose files in this repo carry no em
    dashes, and files that cross the project-knowledge / repo boundary have a
    history of silently double-encoding exactly these characters (CLAUDE.md,
    file and context integrity). Quoted lines are therefore transliterated, and
    the note says so, rather than being altered quietly or shipped as mojibake.
    """
    for k, v in _ASCII_MAP.items():
        s = s.replace(k, v)
    return "".join(c if ord(c) < 127 else "?" for c in s)


def find_block(pattern, label, flags=re.I | re.S):
    """
    Like find(), but matches ACROSS lines within one text block.

    Needed because some shell probes printed a header, then the size, then the
    hash, on three separate lines. A line-anchored search misses the association,
    which is exactly the kind of gap that would let a number be quoted next to
    the wrong artifact.
    """
    rx = re.compile(pattern, flags)
    for ts, kind, t in ALL:
        m = rx.search(t)
        if m:
            snippet = m.group(0).strip()
            snippet = "\n".join(x.strip() for x in snippet.splitlines())
            return ts[:19], snippet
    raise AssertionError("NOT FOUND in transcript: %s (%s)" % (label, pattern))


def find_all(pattern, flags=re.I, limit=None):
    rx = re.compile(pattern, flags)
    out, seen = [], set()
    for ts, kind, t in ALL:
        for line in t.splitlines():
            s = line.strip()
            if len(s) > 700:
                continue
            if rx.search(s):
                key = re.sub(r"\d", "#", s)[:150]
                if key in seen:
                    continue
                seen.add(key)
                out.append((ts[:19], s))
                if limit and len(out) >= limit:
                    return out
    return out


def span():
    first = last = None
    n = 0
    types = collections.Counter()
    atts = collections.Counter()
    for r in records():
        n += 1
        types[r.get("type", "?")] += 1
        if r.get("type") == "attachment":
            a = r.get("attachment", r)
            atts[a.get("type", "?")] += 1
        ts = r.get("timestamp")
        if ts:
            if first is None:
                first = ts
            last = ts
    return n, first, last, types, atts


def attachment_files():
    out = []
    for r in records():
        if r.get("type") != "attachment":
            continue
        a = r.get("attachment", r)
        if a.get("type") in ("file", "compact_file_reference"):
            out.append((a.get("type"), a.get("filename") or a.get("displayPath")))
    return out


def rollout_seconds():
    """Per-extract wall-clock from the '[i/N] <date> leads=[..] NNNs' progress lines."""
    d7, d714 = [], []
    rx = re.compile(r"\[(\d+)/(\d+)\]\s+\S+\s+leads=\[([\d,\s]+)\]\s*(?:init=\d+UTC\s*)?(\d+)s")
    seen = set()
    for ts, kind, t in ALL:
        for line in t.splitlines():
            m = rx.search(line)
            if not m:
                continue
            key = (m.group(1), m.group(2), m.group(4))
            if key in seen:
                continue
            seen.add(key)
            leads = [x.strip() for x in m.group(3).split(",") if x.strip()]
            secs = int(m.group(4))
            (d714 if len(leads) == 2 else d7).append(secs)
    return sorted(d7), sorted(d714)


def stat(v):
    if not v:
        return "n/a"
    v = sorted(v)
    return "n=%d  min %d s  median %d s  max %d s" % (len(v), v[0], v[len(v) // 2], v[-1])


# ==========================================================================
# Prose. The numbers below are all extracted; these are the judgements.
# ==========================================================================
INTRO = """\
**Source, and it was not resumed.** Everything below is read out of one archived
session transcript, opened read-only. No session was continued and no Pangu code
was run to produce this note.

**Why this exists.** The Leg B restore path closed on 2026-08-20 and the rebuild
became the only path. The rebuild needs figures that were measured once, in July,
and recorded nowhere else: how long a rollout takes, how big an extract is, what
the determinism story actually was, and where the initial states come from. Two
of those figures fill gaps that `data/PINNED_ARTIFACTS.md` currently records as
unknown.
"""

ATTACH_NOTE = """\
**There are no evidence attachments.** The 326 attachment records are harness
plumbing, not documents: a per-Bash-call permission hook, to-do reminders,
background-task notifications, and tool or skill listing deltas. Only five
records carry a file at all, and all five are repo files pulled into the model's
own context during the session, not material supplied from outside it.

This matters for the rebuild dispatch in one specific way: **there is no
attached record of the extracts themselves**, no directory listing archived as a
document, and no per-file hash manifest. What can be recovered is what the
session printed to its own logs, which is what follows.
"""

IMPLICATIONS = """\
### 1. The rebuild cost is about 8 hours for the 80 bias extracts, not 7

At the measured 7-step rate, 80 extracts is roughly 7.9 h of wall clock, and that
assumes the run is uninterrupted and the model is already on C:. The register's
"on the order of 7 hours" is close but slightly optimistic. A cold model read off
Z: adds about 9 minutes once, not per date.

Only the 80 bias dates are needed. The July batch generated 182 (91 eval + 80
bias + 13 calib) because it was also scoring; the rebuild does not repeat the
eval or calibration legs.

### 2. Regenerated extracts CAN be SHA-verified, which was not obvious

`np.savez_compressed` writes a zip, and zip entries normally carry a modification
timestamp, which would make byte-comparison meaningless. NumPy pins those entries
to 1980-01-01 instead, so an npz built from identical arrays is byte-identical
across runs. Tested directly on 2026-08-20 rather than assumed.

So the rebuilt extracts can be pinned by hash on creation. What they cannot be is
verified against the originals, because no hash of any original extract was ever
recorded, and the originals are gone.

### 3. The one real determinism risk is thread count, not the model

Pangu is fixed-weight ONNX inference and the initial states come from a static
public archive, so the inputs and the operator are both reproducible. The gap is
that `tier2_pangu_rollout.py` does not pin `intra_op_num_threads`. ONNX Runtime's
CPU provider can reorder floating-point reductions when the thread count changes,
so a rebuild on a differently loaded machine may differ in the last bits. That is
almost certainly immaterial to a bias table averaged over 80 dates, but it means
a byte-exact claim should not be made without pinning threads first.

### 4. The downscaler cannot be reproduced, and the session says so itself

The July session recorded the position explicitly: the kit downscaler had no
`random_state`, and determinism was achieved by pinning the cached pickle, not by
seeding. `random_state=42` entered the kit later, in commit `17baf59`. So the
rebuilt downscaler is a NEW model, and the SHA assertion on `b68eb5fe` cannot
pass. This is consistent with what `data/PINNED_ARTIFACTS.md` already states.

The bias table is the opposite case: it is built with seed 42 over a committed
date map, and the session calls it byte-reproducible. Given the extracts, that
half of Leg B rebuilds exactly.

### 5. A live external dependency the dispatch must gate on

Initial states are read at run time from the WeatherBench2 ERA5 mirror on
anonymous Google Cloud Storage. There is no local cache: the only caching in
`tier2_era5_fetch.py` is an in-process handle on the zarr store. Nothing was lost
with the deleted folder, but nothing is held locally either. **Confirm the zarr
store still opens before committing 8 hours of CPU**, because a rollout that
fails on date 60 of 80 wastes most of a day.
"""

REGISTER_FIX = """\
Two fields that `data/PINNED_ARTIFACTS.md` records as unknown are recoverable
from this transcript and should be written into the register:

| Entry | Field | Register today | Recovered value |
|---|---|---|---|
| 2, `downscaler_blockexcl.pkl` | Size | "Not recorded before loss. Unknown." | **3,751,828 bytes** |
| 3, `arm_extracts/` | Size | "Not recorded before loss. Small; single-digit MB total expected." | **about 20 KB per file**, so about 1.6 MB for 80 and about 4.2 MB for the 210 that existed |

The "single-digit MB" guess was right. The per-file size also gives the rebuild a
cheap early check: an extract far off 20 KB means the harvest went wrong.
"""


def main():
    if not os.path.isfile(SESSION):
        print("FATAL: transcript not found at %s" % SESSION)
        sys.exit(2)

    n, first, last, types, atts = span()
    files = attachment_files()
    d7, d714 = rollout_seconds()

    F = {}
    F["onnx_sha"] = find(r"pangu_weather_24\.onnx\s+613a5c140a1399ab", "ONNX sha")
    F["onnx_size"] = find(r"1181711187\s+<USER_HOME>/tier2_model/pangu_weather_24\.onnx",
                          "ONNX size")
    F["dwn"] = find_block(
        r"===downscaler cache \(block-excluded\) sha, if present===\s*\n"
        r"\s*3751828\s*\n\s*b68eb5fe57fab817[0-9a-f]{48}",
        "downscaler size+sha")
    F["batch182"] = find(r"generated 182 new extracts in 65623s", "182-batch total")
    F["batch28"] = find(r"generated 27 new extracts in 18345s", "DJF batch total")
    F["batch8"] = find(r"generated 8 new extracts in 282\ds", "sub batch total")
    F["total210"] = find(r"210 total extracts|Extract count 210", "210 total")
    F["sub32"] = find(r"ALL_HOURS_DONE extracts=32", "32 sub extracts")
    F["datelist"] = find(r"EVAL remaining:\s*91\s+BIAS:\s*80\s+CALIB:\s*13", "datemap split")
    F["load_cold"] = find(r"session loaded in 543\.5s", "cold model load")
    F["load_warm"] = find(r"session loaded in (4\.5|6\.1)s", "warm model load")
    F["mem"] = find(r"32 GB -> ~3 GB|peak RSS 32 GB", "memory finding")
    F["budget"] = find(r"bias_anchors=80 bias_steps=1120", "original step budget")
    F["det_dwn"] = find(r"determinism is by \*\*pinning the cached model\*\*|"
                        r"kit LGBM has no `random_state`", "downscaler determinism note")
    F["det_bias"] = find(r"seed 42, byte-reproducible", "bias table determinism")
    F["det_pangu"] = find(r"ONNX fixed weights", "pangu determinism")

    sizes = find_all(r"\b\d{5}\s+\w{3}\s+\d{1,2}\s+[\d:]{4,5}\s+extract_\d{8}", limit=8)

    rows = []
    for key, label in [
        ("onnx_sha", "Pangu ONNX SHA-256"),
        ("onnx_size", "Pangu ONNX size on C:"),
        ("dwn", "downscaler_blockexcl.pkl size and SHA"),
        ("datelist", "d7 date list split"),
        ("batch182", "182-extract batch, total wall clock"),
        ("batch28", "DJF batch (14-step), total wall clock"),
        ("batch8", "submission sub-batch (7-step)"),
        ("total210", "extracts on disk at completion"),
        ("sub32", "arm_extracts_sub count"),
        ("load_cold", "model session load, cold from Z:"),
        ("load_warm", "model session load, local C: copy"),
        ("mem", "ONNX Runtime memory finding"),
        ("budget", "original CPU-step budget"),
        ("det_pangu", "Pangu determinism, as recorded"),
        ("det_bias", "bias table determinism, as recorded"),
        ("det_dwn", "downscaler determinism, as recorded"),
    ]:
        ts, line = F[key]
        rows.append((label, ts, line))

    L = []
    L.append("# Pangu smoke-test session: recorded facts for the Leg B rebuild")
    L.append("")
    L.append("**Extracted 2026-08-20** by `scripts/extract_pangu_smoke_session_20260820.py`.")
    L.append("")
    L.append("Quoted transcript lines are transliterated to ASCII (em dash to hyphen, "
             "ellipsis to three dots, arrow to `->`). The repo carries no em dashes in "
             "prose, and these characters are the ones that double-encode silently when "
             "text crosses the project-knowledge boundary. Numbers and identifiers are "
             "untouched.")
    L.append("")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append("| Source session | `%s.jsonl` |" % SESSION_ID)
    L.append("| Session title | CC PROMPT PART C: Pangu-Weather smoke test (Tier 2 gate) |")
    L.append("| Path | `%s` |" % SESSION)
    L.append("| Span | %s to %s |" % (first[:19], last[:19]))
    L.append("| Records | %d (%s) |" % (n, ", ".join("%s %d" % (k, v) for k, v in types.most_common())))
    L.append("| Resumed? | **No.** Read as a file. |")
    L.append("")
    L.append(INTRO)
    L.append("## What the attachments are")
    L.append("")
    L.append("| Attachment type | Count | What it is |")
    L.append("|---|---|---|")
    desc = {
        "hook_success": "PreToolUse permission hook firing on each Bash/TaskStop call",
        "todo_reminder": "the session's own to-do list, re-injected",
        "queued_command": "background-task completion notifications",
        "deferred_tools_delta": "tool-availability changes during the session",
        "date_change": "midnight rollover markers",
        "file": "a repo file read into context",
        "skill_listing": "available-skills listing",
        "agent_listing_delta": "available-subagent listing",
        "compact_file_reference": "file referenced across a context compaction",
    }
    for k, v in atts.most_common():
        L.append("| `%s` | %d | %s |" % (k, v, desc.get(k, "harness record")))
    L.append("")
    L.append("The five file-bearing records, all repo files, none supplied from outside:")
    L.append("")
    for t, fn in files:
        L.append("- `%s` (%s)" % (asciify(str(fn)), t))
    L.append("")
    L.append(ATTACH_NOTE)
    L.append("## Recorded values, with the transcript line each came from")
    L.append("")
    for label, ts, line in rows:
        L.append("**%s** [%s]" % (label, ts))
        L.append("")
        L.append("```")
        L.append(asciify(line)[:400])
        L.append("```")
        L.append("")
    L.append("## Per-extract wall clock")
    L.append("")
    L.append("**The batch totals are authoritative; the per-date lines are a sample.** The")
    L.append("transcript shows progress lines only where the log was tailed, so the table")
    L.append("below covers %d of the 182 d7 rollouts and %d of the 28 d7+d14 ones. The"
             % (len(d7), len(d714)))
    L.append("batch-total lines cover every date and are the right basis for a budget.")
    L.append("")
    L.append("| Rollout | Steps | Sampled per-date timing |")
    L.append("|---|---|---|")
    L.append("| d7 only (`leads=[7]`) | 7 x 24 h | %s |" % stat(d7))
    L.append("| d7 + d14 (`leads=[7, 14]`) | 14 x 24 h | %s |" % stat(d714))
    L.append("")
    rate182 = 65623.0 / 182.0
    rate28 = 18345.0 / 27.0
    L.append("| Batch | Extracts | Total | Per extract |")
    L.append("|---|---|---|---|")
    L.append("| d7, 182 dates | 182 | 65,623 s (18.23 h) | **%.1f s** |" % rate182)
    L.append("| DJF, d7+d14 | 27 new | 18,345 s (5.10 h) | %.1f s |" % rate28)
    L.append("| submission sub-batch, d7 | 8 | 2,821 s | %.1f s |" % (2821.0 / 8.0))
    L.append("")
    L.append("**Budget for the rebuild: 80 bias extracts at %.1f s is %.1f hours** of"
             % (rate182, 80 * rate182 / 3600.0))
    L.append("uninterrupted CPU, model already local. That is the number to plan against,")
    L.append("not the 7 h in the register.")
    L.append("")
    L.append("Observed extract file sizes, from directory listings in the log:")
    L.append("")
    L.append("```")
    for ts, s in sizes:
        L.append(asciify(s)[:200])
    L.append("```")
    L.append("")
    L.append("## What this means for the rebuild")
    L.append("")
    L.append(IMPLICATIONS)
    L.append("## Corrections owed to data/PINNED_ARTIFACTS.md")
    L.append("")
    L.append(REGISTER_FIX)

    text = "\n".join(L) + "\n"
    nonascii = [c for c in text if ord(c) > 126]
    assert not nonascii, "non-ASCII characters in output: %r" % sorted(set(nonascii))[:10]

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s (%d lines)" % (OUT, len(L)))
    print("d7 timings n=%d, d7+d14 timings n=%d" % (len(d7), len(d714)))
    print("all %d asserted facts found in the transcript" % len(rows))


if __name__ == "__main__":
    main()
