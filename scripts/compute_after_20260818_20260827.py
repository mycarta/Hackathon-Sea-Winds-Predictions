"""Compute and carbon accounting for everything after 2026-08-18.

CC dispatch 2026-08-27, item 3. `compute_accounting/compute_table.md` and report
Table 5 cover runs logged to 2026-08-18. The selected submission was rebuilt
after that date and the audit recomputations ran after it, so the table has an
open tail. This script produces the "after 08-18" row.

**Method is unchanged from compute_table.md**, deliberately, so the new row can
be appended without touching any method statement:
  - the same five categories (Pangu inference, LightGBM training, downscaler
    training, scoring/diagnostics, data processing);
  - the same TDP-anchored bracket, 37 W lower bound (idle-corrected package,
    45 - 8) and 67.5 W upper bound (1.5 x PL1, whole-system);
  - the same grid intensity, Nova Scotia 660 gCO2e/kWh (CER Provincial /
    Territorial Energy Profiles, citing ECCC NIR 1990-2022).

**Provenance is carried per line, not asserted in aggregate.** Every line
declares one of three sources:
  DISCLOSED  a wall-clock figure stated in an LLM_AGENT_LOG.md compute
             disclosure, cited by line number.
  MEASURED   no disclosure exists; the figure is recomputed here from the
             artifacts the run produced (file mtime span plus one median
             inter-file gap, since the first file is written only after the
             first unit of work completes). The method is validated against the
             one Pangu block that IS disclosed: see VALIDATION below.
  BOUND      the log states an upper bound in words ("well under one
             CPU-minute") rather than a number; the bound is used, which
             over-states rather than under-states the footprint.

**VALIDATION of the MEASURED method.** The 2026-08-21 R1 block is disclosed at
26,883 s (LLM_AGENT_LOG.md:4039-4040) AND left 80 artifacts on disk. Applying
the mtime method to those artifacts reproduces the disclosed figure; the
agreement is printed at run time and gated. Only after that gate passes is the
same method applied to the two undisclosed R4a blocks.

**FINDING, reported not repaired (dispatch standing instruction).** Two
substantial Pangu inference blocks after 08-18 carry no compute disclosure at
all: the 2026-08-22 R4a submission-window extracts for 2021 and for 2022, 32
rollouts x 7 steps each. Contract §5.3 requires a compute disclosure with every
substantial run. They are quantified here from artifacts and flagged; the log is
not back-filled by this script. Several smaller runs are also undisclosed and
are listed in `gaps` with no number attached rather than silently estimated.

**SECOND FINDING.** `LLM_AGENT_LOG.md` ends at 2026-08-22. The audit
recomputations of 2026-08-24 to 2026-08-27 -- which the dispatch explicitly says
ran after the table's cutoff -- are therefore not in the log at all. They are
accounted separately below, under `post_log_audit_block`, and are NOT folded
into the "after 08-18" row, because that row is defined by the dispatch as a sum
over the log.

Deterministic: a fixed ledger plus filesystem stat calls. No stochastic step, so
no seed applies.

Reads : LLM_AGENT_LOG.md (line references only, for the record)
        data/arm_extracts_20260821/*.npz              (validation)
        data/arm_extracts_sub_2021_20260822/*.npz     (measured)
        data/arm_extracts_sub_2022_20260822/*.npz     (measured)
        reports/audit_*.json                          (post-log block)
Writes: compute_accounting/compute_after_20260818.md
        compute_accounting/compute_after_20260818.json
"""
from __future__ import annotations

import glob
import json
import os
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_MD = REPO_ROOT / "compute_accounting" / "compute_after_20260818.md"
OUT_JSON = REPO_ROOT / "compute_accounting" / "compute_after_20260818.json"

# Conversion anchors, identical to compute_table.md Part C.
W_LOWER = 37.0          # idle-corrected package, 45 - 8
W_UPPER = 67.5          # 1.5 x PL1, whole-system
GRID_G_PER_KWH = 660.0  # Nova Scotia

CATEGORIES = ["Pangu inference", "LightGBM training", "Downscaler training",
              "Scoring / diagnostics", "Data processing"]

# ── The ledger ──────────────────────────────────────────────────────────
# (date, category, description, seconds, provenance, source)
# Only entries dated strictly AFTER 2026-08-18. The 2026-08-18 S0/S2 entry is
# excluded: compute_table.md already covers "runs logged to 2026-08-18".
LEDGER = [
    # ---- 2026-08-19 -------------------------------------------------------
    ("2026-08-19", "Scoring / diagnostics",
     "S1.1 Zenodo 20874645 pin gate + 8-pair structure gate",
     60.0, "BOUND", "LLM_AGENT_LOG.md:3216-3220 ('well under one CPU-minute')"),
    ("2026-08-19", "Scoring / diagnostics",
     "S3 three-case like-for-like scorer (198 s) + one failed 3 s run in the wrong env",
     201.0, "DISCLOSED", "LLM_AGENT_LOG.md:3309-3313"),
    ("2026-08-19", "LightGBM training",
     "Leg A Guard B, two full MOS fits (107 s + 101 s)",
     208.0, "DISCLOSED", "LLM_AGENT_LOG.md:3459"),
    ("2026-08-19", "LightGBM training",
     "Leg A A1 notebook 1, one full MOS fit, 8 windows, all 2022 issue dates",
     98.0, "DISCLOSED", "LLM_AGENT_LOG.md:3460"),
    ("2026-08-19", "Downscaler training",
     "Leg A A2 notebook 2, downscaler on 74 days, 4,196,640 rows",
     253.0, "DISCLOSED", "LLM_AGENT_LOG.md:3461"),
    ("2026-08-19", "Data processing",
     "Leg A A3 d14 climatology, 1,398,880 rows across 32 blocks",
     47.0, "DISCLOSED", "LLM_AGENT_LOG.md:3462"),
    ("2026-08-19", "Scoring / diagnostics",
     "Leg A Guard A (40 s), A4 direction residual apply (61 s), "
     "A5 d7 bias correction (46 s), 2022 assertion (57 s), validation gate (8 s)",
     212.0, "DISCLOSED", "LLM_AGENT_LOG.md:3458,3463-3468"),
    ("2026-08-19", "Data processing",
     "Leg A A6 dir 360 wrap (44 s) + archive and zip (16 s)",
     60.0, "DISCLOSED", "LLM_AGENT_LOG.md:3464-3465"),
    # ---- 2026-08-20 -------------------------------------------------------
    ("2026-08-20", "Data processing",
     "ENTSO-E NL 2019 pulls, two runs of 24 monthly requests (217 s + 193 s)",
     410.0, "DISCLOSED", "LLM_AGENT_LOG.md:3659-3660"),
    # ---- 2026-08-22 -------------------------------------------------------
    ("2026-08-22", "Downscaler training",
     "Leg B R2 block-excluded downscaler refit, 63 days x 4 hours, n_jobs=-1",
     384.0, "DISCLOSED", "LLM_AGENT_LOG.md:4257-4259"),
    ("2026-08-22", "Scoring / diagnostics",
     "Leg B R3 bias table, 80 extract reads + 80 downscaler applications, "
     "run twice for a determinism check (48 s x 2)",
     96.0, "DISCLOSED", "LLM_AGENT_LOG.md:4259-4260"),
    ("2026-08-22", "Scoring / diagnostics",
     "Leg B R5 build, 32 extract reads + 32 downscaler applications, "
     "4.2 M-row stream with byte-identity verification",
     95.0, "DISCLOSED", "LLM_AGENT_LOG.md:4386-4387"),
]

# Pangu blocks handled separately: one disclosed (validation), two measured.
PANGU_DISCLOSED_S = 26883.0
PANGU_DISCLOSED_REF = "LLM_AGENT_LOG.md:4039-4040"
EXTRACT_DIRS = {
    "validation_R1_80_rollouts_20260821":
        ("data/arm_extracts_20260821", 80, 7, PANGU_DISCLOSED_S),
    "R4a_sub_extracts_2021_20260822":
        ("data/arm_extracts_sub_2021_20260822", 32, 7, None),
    "R4a_sub_extracts_2022_20260822":
        ("data/arm_extracts_sub_2022_20260822", 32, 7, None),
}
VALIDATION_TOL_FRAC = 0.05   # measured must land within 5 % of the disclosed

# Runs after 08-18 with NO wall-clock anywhere. Listed, never estimated.
GAPS = [
    ("2026-08-20", "TenneT Aether probing: five probe scripts and two failed "
     "fetch runs, roughly 170-195 HTTPS GETs, ~3 MB payload",
     "LLM_AGENT_LOG.md:3794-3797 (request count and payload only, no wall-clock)"),
    ("2026-08-21", "Bonus Stage 2 production: full PyWake per timestep through "
     "the Task 2 scorer replica, seven series x 1,460 hours",
     "LLM_AGENT_LOG.md:3941-3944 (workload described, no wall-clock)"),
    ("2026-08-21", "Bonus Stages 4 and 5: closed-form settlement arithmetic "
     "plus one seeded bootstrap, 20,000 resamples, seed 42",
     "LLM_AGENT_LOG.md:3943-3945 (no wall-clock)"),
    ("2026-08-22", "Leg B R4b diagnostic and G1-G3 gate runs",
     "no compute disclosure in the 2026-08-22 entries"),
    ("2026-08-22", "Leg B dependency sweep: static AST parse plus filesystem stats",
     "LLM_AGENT_LOG.md:4260-4261 ('seconds', no figure)"),
    ("2026-08-22", "R5 repair pass, two Guard B runs, and several full SHA-256 "
     "hashes over 433 MB CSVs (only the 95 s build itself is disclosed)",
     "LLM_AGENT_LOG.md:4386-4389"),
]

# 2026-08-24 to 2026-08-27, absent from LLM_AGENT_LOG.md entirely.
POST_LOG = [
    ("2026-08-25", "Scoring / diagnostics",
     "Audit anchor 6: S3 case-1 recompute in a clean environment "
     "(load 92.5 s + physics 2.1 s)",
     100.5, "MEASURED", "reports/audit_anchor6_s3_case1_20260825.json:wall_time_s.total"),
    ("2026-08-27", "Scoring / diagnostics",
     "Item 0/0b audit recompute, final run (2 AROME loads + 4 PyWake cases)",
     188.7, "MEASURED", "reports/audit_lcoe_row2_and_curve_20260827.json:wall_clock_s"),
    ("2026-08-27", "Scoring / diagnostics",
     "Item 0/0b audit, two earlier runs of the same script: one aborted at the "
     "depth sampler after both AROME loads, one complete before the "
     "gross-vs-gross field was added. Not separately instrumented; charged at "
     "the measured cost of the final run",
     377.4, "ESTIMATE", "2 x the 188.7 s measured final run"),
    ("2026-08-27", "Scoring / diagnostics",
     "Item 5 submission row check, 4,196,640 rows scanned from inside the zip "
     "plus three full SHA-256 hashes over 433 MB / 80 MB artifacts",
     11.0, "MEASURED", "reports/audit_submission_rowcheck_20260827.json:wall_clock_s"),
    ("2026-08-27", "Data processing",
     "Published IEA-22-280-RWT curve extraction and item 6 submission build "
     "with kit validate_layout",
     15.0, "BOUND", "both scripts returned in under 15 s wall-clock"),
]


def measure_dir(rel: str) -> dict:
    """Wall-clock of a run from the artifacts it left, plus one median gap.

    The first artifact appears only after the first unit of work finishes, so
    the raw first-to-last span systematically under-counts by about one unit.
    Adding the median inter-file gap corrects for that.
    """
    files = glob.glob(str(REPO_ROOT / rel / "*.npz"))
    if not files:
        return {"path": rel, "n_files": 0, "seconds": None,
                "note": "no artifacts found"}
    ts = sorted(os.path.getmtime(f) for f in files)
    span = ts[-1] - ts[0]
    gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    med = statistics.median(gaps) if gaps else 0.0
    return {
        "path": rel, "n_files": len(files),
        "first_mtime": ts[0], "last_mtime": ts[-1],
        "span_s": span, "median_gap_s": med,
        "seconds": span + med,
    }


def convert(seconds: float) -> dict:
    h = seconds / 3600.0
    kwh_lo = h * W_LOWER / 1000.0
    kwh_hi = h * W_UPPER / 1000.0
    return {
        "seconds": seconds, "hours": h,
        "kwh_low": kwh_lo, "kwh_high": kwh_hi,
        "kgco2e_low": kwh_lo * GRID_G_PER_KWH / 1000.0,
        "kgco2e_high": kwh_hi * GRID_G_PER_KWH / 1000.0,
    }


def main() -> None:
    # ── Measure the Pangu blocks, validating the method first ───────────
    measured = {k: measure_dir(v[0]) for k, v in EXTRACT_DIRS.items()}
    vkey = "validation_R1_80_rollouts_20260821"
    v = measured[vkey]
    if v["seconds"] is None:
        raise SystemExit("cannot validate the mtime method: R1 artifacts absent")
    err = abs(v["seconds"] - PANGU_DISCLOSED_S) / PANGU_DISCLOSED_S
    print("mtime-method validation against the one disclosed Pangu block")
    print(f"  disclosed : {PANGU_DISCLOSED_S:,.0f} s  ({PANGU_DISCLOSED_REF})")
    print(f"  measured  : {v['seconds']:,.0f} s  "
          f"(span {v['span_s']:,.0f} s + median gap {v['median_gap_s']:.0f} s, "
          f"{v['n_files']} artifacts)")
    print(f"  error     : {err*100:.2f} %  (tolerance {VALIDATION_TOL_FRAC*100:.0f} %)")
    if err > VALIDATION_TOL_FRAC:
        raise SystemExit("mtime method does not reproduce the disclosed figure; "
                         "refusing to apply it to the undisclosed blocks")
    print("  -> PASS, method applied to the two undisclosed R4a blocks\n")

    ledger = list(LEDGER)
    ledger.append(("2026-08-21", "Pangu inference",
                   "Leg B R1 arm extracts: 80 rollouts x 7 x 24 h = 560 inference "
                   "steps, ONNX Runtime 1.27.0 CPU, intra_op_num_threads=16",
                   PANGU_DISCLOSED_S, "DISCLOSED", PANGU_DISCLOSED_REF))
    for key in ("R4a_sub_extracts_2021_20260822", "R4a_sub_extracts_2022_20260822"):
        m = measured[key]
        rel, n_roll, steps, _ = EXTRACT_DIRS[key]
        year = key.split("_")[3]
        ledger.append((
            "2026-08-22", "Pangu inference",
            f"Leg B R4a submission-window extracts, {year}: {n_roll} rollouts "
            f"x {steps} = {n_roll * steps} inference steps "
            f"(NO COMPUTE DISCLOSURE IN THE LOG)",
            m["seconds"], "MEASURED",
            f"{rel}: {m['n_files']} artifact mtimes, span {m['span_s']:,.0f} s "
            f"+ median gap {m['median_gap_s']:.0f} s"))

    # ── Aggregate ───────────────────────────────────────────────────────
    by_cat = {c: 0.0 for c in CATEGORIES}
    for _, cat, _, secs, _, _ in ledger:
        by_cat[cat] += secs
    total_s = sum(by_cat.values())

    disclosed_s = sum(s for _, _, _, s, p, _ in ledger if p == "DISCLOSED")
    measured_s = sum(s for _, _, _, s, p, _ in ledger if p == "MEASURED")
    bound_s = sum(s for _, _, _, s, p, _ in ledger if p == "BOUND")

    post_by_cat = {c: 0.0 for c in CATEGORIES}
    for _, cat, _, secs, _, _ in POST_LOG:
        post_by_cat[cat] += secs
    post_total_s = sum(post_by_cat.values())

    out = {
        "generated_by": "scripts/compute_after_20260818_20260827.py",
        "dispatch": "CC dispatch 2026-08-27, item 3",
        "scope": "LLM_AGENT_LOG.md entries dated strictly after 2026-08-18",
        "method": "unchanged from compute_accounting/compute_table.md Part C",
        "anchors": {"w_lower": W_LOWER, "w_upper": W_UPPER,
                    "grid_gco2e_per_kwh": GRID_G_PER_KWH,
                    "grid_source": "Nova Scotia, CER Provincial/Territorial "
                                   "Energy Profiles citing ECCC NIR 1990-2022"},
        "mtime_method_validation": {
            "disclosed_s": PANGU_DISCLOSED_S, "disclosed_ref": PANGU_DISCLOSED_REF,
            "measured_s": v["seconds"], "relative_error": err,
            "tolerance": VALIDATION_TOL_FRAC, "pass": True,
        },
        "ledger": [
            {"date": d, "category": c, "description": desc, "seconds": s,
             "provenance": p, "source": src}
            for d, c, desc, s, p, src in ledger
        ],
        "by_category": {c: convert(s) for c, s in by_cat.items()},
        "total_after_20260818": convert(total_s),
        "provenance_split": {
            "disclosed_s": disclosed_s, "measured_s": measured_s,
            "bound_s": bound_s,
            "disclosed_fraction": disclosed_s / total_s if total_s else 0.0,
        },
        "gaps_no_wall_clock_anywhere": [
            {"date": d, "run": r, "source": s} for d, r, s in GAPS
        ],
        "post_log_audit_block": {
            "note": ("2026-08-24 to 2026-08-27. LLM_AGENT_LOG.md ends at "
                     "2026-08-22, so none of this is in the log. Reported "
                     "separately and NOT folded into the after-08-18 row, "
                     "which the dispatch defines as a sum over the log."),
            "ledger": [
                {"date": d, "category": c, "description": desc, "seconds": s,
                 "provenance": p, "source": src}
                for d, c, desc, s, p, src in POST_LOG
            ],
            "by_category": {c: convert(s) for c, s in post_by_cat.items() if s},
            "total": convert(post_total_s),
        },
        "combined_after_20260818_incl_post_log": convert(total_s + post_total_s),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ── Markdown ────────────────────────────────────────────────────────
    L = []
    L.append("# Compute & carbon, runs after 2026-08-18")
    L.append("")
    L.append("Generated by `scripts/compute_after_20260818_20260827.py` "
             "(CC dispatch 2026-08-27, item 3).")
    L.append("")
    L.append("Method, anchors and categories are identical to "
             "`compute_accounting/compute_table.md`; this table is an "
             "**append**, and no method statement changes.")
    L.append("")
    L.append("## The after-08-18 row (sum over `LLM_AGENT_LOG.md`)")
    L.append("")
    L.append("| Category | Wall-clock | CPU-h | Energy (kWh) | kgCO2e @ NS 660 |")
    L.append("|---|---:|---:|---:|---:|")
    for c in CATEGORIES:
        k = convert(by_cat[c])
        L.append(f"| {c} | {by_cat[c]:,.0f} s | {k['hours']:.2f} | "
                 f"{k['kwh_low']:.3f} – {k['kwh_high']:.3f} | "
                 f"{k['kgco2e_low']:.3f} – {k['kgco2e_high']:.3f} |")
    t = convert(total_s)
    L.append(f"| **TOTAL (after 08-18)** | **{total_s:,.0f} s** | "
             f"**{t['hours']:.2f}** | **{t['kwh_low']:.2f} – {t['kwh_high']:.2f}** | "
             f"**{t['kgco2e_low']:.2f} – {t['kgco2e_high']:.2f}** |")
    L.append("")
    L.append(f"Provenance of the {t['hours']:.2f} CPU-h: "
             f"{disclosed_s/3600:.2f} h disclosed in the log, "
             f"{measured_s/3600:.2f} h measured from artifacts because no "
             f"disclosure exists, {bound_s/3600:.3f} h charged at a stated "
             f"verbal bound.")
    L.append("")
    L.append("## Line items")
    L.append("")
    L.append("| Date | Category | Run | Wall-clock | Provenance | Source |")
    L.append("|---|---|---|---:|---|---|")
    for d, c, desc, s, p, src in sorted(ledger, key=lambda r: (r[0], r[1])):
        L.append(f"| {d} | {c} | {desc} | {s:,.0f} s | {p} | `{src}` |")
    L.append("")
    L.append("## FINDING — two undisclosed Pangu blocks")
    L.append("")
    L.append("The 2026-08-22 R4a submission-window extract runs, 32 rollouts x 7 "
             "steps for 2021 and again for 2022 (448 inference steps in total), "
             "carry **no compute disclosure** in `LLM_AGENT_LOG.md`. Contract "
             "§5.3 requires one per substantial run. Together they are the "
             "second-largest compute block after 08-18. Quantified here from "
             "artifact mtimes and flagged; the log is not back-filled by this "
             "script.")
    L.append("")
    L.append(f"The mtime method is validated first against the one Pangu block "
             f"that IS disclosed: R1's 80 rollouts, disclosed at "
             f"{PANGU_DISCLOSED_S:,.0f} s, measure at {v['seconds']:,.0f} s, "
             f"an error of {err*100:.2f} %.")
    L.append("")
    L.append("## Runs after 08-18 with no wall-clock anywhere")
    L.append("")
    L.append("Listed rather than estimated. Each would add to the row above.")
    L.append("")
    L.append("| Date | Run | Where the gap is |")
    L.append("|---|---|---|")
    for d, r, s in GAPS:
        L.append(f"| {d} | {r} | `{s}` |")
    L.append("")
    L.append("## FINDING — the log stops before the audit block")
    L.append("")
    L.append("`LLM_AGENT_LOG.md` ends at 2026-08-22. The 2026-08-24 to "
             "2026-08-27 audit recomputations are absent from it entirely, so a "
             "sum over the log cannot include them. They are accounted here "
             "separately.")
    L.append("")
    L.append("| Date | Category | Run | Wall-clock | Provenance |")
    L.append("|---|---|---|---:|---|")
    for d, c, desc, s, p, src in POST_LOG:
        L.append(f"| {d} | {c} | {desc} | {s:,.0f} s | {p} |")
    pt = convert(post_total_s)
    L.append(f"| | | **subtotal** | **{post_total_s:,.0f} s "
             f"({pt['hours']:.2f} h)** | |")
    L.append("")
    ct = convert(total_s + post_total_s)
    L.append(f"**Combined after-08-18 including the post-log audit block:** "
             f"{ct['hours']:.2f} CPU-h, {ct['kwh_low']:.2f} – "
             f"{ct['kwh_high']:.2f} kWh, {ct['kgco2e_low']:.2f} – "
             f"{ct['kgco2e_high']:.2f} kgCO2e.")
    L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    print("=" * 72)
    print("AFTER 2026-08-18 (sum over LLM_AGENT_LOG.md)")
    for c in CATEGORIES:
        k = convert(by_cat[c])
        print(f"  {c:24s} {k['hours']:7.2f} CPU-h")
    print(f"  {'TOTAL':24s} {t['hours']:7.2f} CPU-h   "
          f"{t['kwh_low']:.2f}-{t['kwh_high']:.2f} kWh   "
          f"{t['kgco2e_low']:.2f}-{t['kgco2e_high']:.2f} kgCO2e")
    print("-" * 72)
    print(f"  post-log audit block (08-24..08-27, not in the log): "
          f"{pt['hours']:.2f} CPU-h")
    print(f"  combined: {ct['hours']:.2f} CPU-h, "
          f"{ct['kwh_low']:.2f}-{ct['kwh_high']:.2f} kWh, "
          f"{ct['kgco2e_low']:.2f}-{ct['kgco2e_high']:.2f} kgCO2e")
    print("=" * 72)
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
