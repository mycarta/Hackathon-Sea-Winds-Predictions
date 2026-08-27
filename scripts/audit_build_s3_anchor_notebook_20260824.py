"""Generate notebooks/audit_s3_anchor_scaffold.ipynb.

Audit v2, Tier B anchor 5 (rung-2 protocol). The notebook is an artifact and
therefore comes from a named, committed script rather than from hand-editing
(CLAUDE.md build discipline, the "file created or modified without a named,
committed script" line of the cross list).

**Scaffold contract.** Cells 1, 2, 3 and 5 are complete and executable. Cell 4
is EMPTY except for the comment block the dispatch fixes verbatim: the
principal writes the simulation there himself, from Phase_2.pdf, with no repo
code consulted. This script therefore contains no simulation configuration of
any kind, and neither does the notebook it emits. That absence is the
instrument; do not fill it in.

Re-runnable cold: rewrites the notebook from scratch, no manual step. The
expected SHA-256 literals below were taken from the files on 2026-08-24 and
are asserted by cell 1 against a fresh hash at run time.

Deterministic: string assembly and JSON serialisation. No stochastic step, so
no seed applies.

Output: notebooks/audit_s3_anchor_scaffold.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_IPYNB = REPO_ROOT / "notebooks" / "audit_s3_anchor_scaffold.ipynb"

CELL_1 = r'''# -- Cell 1 . Manifest ------------------------------------------------
# Audit v2, Tier B anchor 5: independent recomputation of the S3
# winner-layout case (case3_submitted_cell63, three-case scorer run of
# 2026-08-19).
#
# Every input this notebook reads is hashed here and asserted against a
# literal recorded on 2026-08-24. If an assertion fires, the input moved or
# changed and the recomputation is not comparable to the stored result.
#
# TWO HASHES PER FILE, and the reason. This repo is checked out with
# core.autocrlf=true, so three of the four files below sit on disk as CRLF
# while git stores them as LF. A raw-byte hash of the working copy therefore
# depends on the machine, not on the content, and would fail on any Linux
# clone. The GATE is `sha256_lf`, the hash of the content with CRLF collapsed
# to LF: platform-independent, and equal to the hash of the bytes git actually
# stores. `sha256_asis` is the raw working-copy hash recorded on the build
# machine; it is REPORTED and flagged on difference, never gated.
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

INPUTS = {
    "layout_coordinates": {
        "path": REPO_ROOT / "data" / "task2_layout_winner.json",
        "sha256_lf": "4ae465e0e3593842308b14f129bd434cfed2478be8475ed91c2f86828280b4b2",
        "bytes_lf": 1889,
        "sha256_asis": "c9f618034ebeeab9cbbf163d8c4006e9d1eb1f9e7f9bc6fd4e43752ab60f6f1d",
        "note": "banked Stage 3 layout, 2026-07-13",
    },
    "wind_resource_input": {
        "path": REPO_ROOT / "data" / "audit_s3_wind_input_cell63_20260824.csv",
        "sha256_lf": "05c6cd34ad66cd60b8e3eabcdb6c6f0128842acd20fe59b4b5b486c2aea19577",
        "bytes_lf": 844086,
        "sha256_asis": "05c6cd34ad66cd60b8e3eabcdb6c6f0128842acd20fe59b4b5b486c2aea19577",
        "note": ("the S3 series materialised as one file by "
                 "scripts/audit_export_s3_wind_input_20260824.py; native "
                 "level as shipped, unadjusted. LF-pinned in .gitattributes, "
                 "so both hashes coincide"),
    },
    "iea22mw_power_ct_table": {
        "path": REPO_ROOT / "data" / "iea22mw_power_ct.csv",
        "sha256_lf": "bb9eb2102d1ee0fb7b3fcc02d467f22d497c8c41a9794d2147da01491424e36a",
        "bytes_lf": 1393,
        "sha256_asis": "d25096a2ea14200c2814939e5fb3b5b75d3aac172842119feb2ea23b1718cb55",
        "note": "power/Ct table as used by the S3 run",
    },
}

# Reference values to match. Read from the artifact, not transcribed, so the
# comparison cannot drift from its source.
REFERENCE_SOURCE = REPO_ROOT / "reports" / "three_case_scorer_20260818.json"
REFERENCE_SOURCE_SHA_LF = "6e54bcb402c4759a2a12a318af358d04df362dc0a3cc38accfd0ce06b45fbb69"
REFERENCE_SOURCE_SHA_ASIS = "5b5e32b7c4c261af9fa3561dbbd6132f70d46acd61e651121750481141d9e140"
REFERENCE_CASE = "case3_submitted_cell63"

# Lower-precision duplicate of the same three numbers, banked independently at
# layout-selection time. Printed as a cross-check on the reference source.
SECONDARY_SOURCE = INPUTS["layout_coordinates"]["path"]


def hashes_of(path):
    """(sha256 of raw bytes, sha256 of CRLF-collapsed bytes, len of each)."""
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return (hashlib.sha256(raw).hexdigest(), hashlib.sha256(lf).hexdigest(),
            len(raw), len(lf))


def report_hashes(label, path, sha_lf_expected, bytes_lf_expected,
                  sha_asis_expected):
    assert Path(path).exists(), f"{label}: missing artifact {path}"
    asis, lf, n_asis, n_lf = hashes_of(path)
    print(f"{label}")
    print(f"  path        {path}")
    print(f"  sha256 LF   {lf}   ({n_lf} B)   <- gated")
    print(f"  sha256 asis {asis}   ({n_asis} B)")
    assert lf == sha_lf_expected, (
        f"{label}: LF-normalised SHA-256 mismatch -- the CONTENT changed\n"
        f"  expected {sha_lf_expected}\n  fresh    {lf}")
    assert n_lf == bytes_lf_expected, (
        f"{label}: LF-normalised size {n_lf} != recorded {bytes_lf_expected}")
    if asis == sha_asis_expected:
        print("  [OK]        content matches, and the working copy is "
              "byte-identical to the build machine")
    else:
        print("  [FLAG]      content matches; raw bytes differ from the build "
              "machine (line endings).")
        print(f"              recorded asis {sha_asis_expected}")
        print("              Not gated. Record it with the anchor result.")


print("INPUT ARTIFACTS")
print("=" * 78)
for key, spec in INPUTS.items():
    report_hashes(key, spec["path"], spec["sha256_lf"], spec["bytes_lf"],
                  spec["sha256_asis"])
    print(f"  note        {spec['note']}")
print()

print("STORED REFERENCE VALUES")
print("=" * 78)
report_hashes("reference_source", REFERENCE_SOURCE, REFERENCE_SOURCE_SHA_LF,
              4156, REFERENCE_SOURCE_SHA_ASIS)

with open(REFERENCE_SOURCE) as f:
    _ref_doc = json.load(f)
_case = next(c for c in _ref_doc["cases"] if c["case"] == REFERENCE_CASE)

CF_STORED = float(_case["cf_net"])
AEP_STORED_GWH = float(_case["aep_gwh"])
WAKE_STORED = float(_case["wake_loss_fraction"])

print(f"case     {REFERENCE_CASE}")
print(f"  CF (net, fraction)        {CF_STORED!r}")
print(f"  AEP (GWh/yr)              {AEP_STORED_GWH!r}")
print(f"  wake fraction             {WAKE_STORED!r}")
print(f"  all three from            {REFERENCE_SOURCE}")

with open(SECONDARY_SOURCE) as f:
    _secondary = json.load(f)["reported"]
print(f"\nsecondary (lower precision) from {SECONDARY_SOURCE}")
print(f"  CF   {_secondary['capacity_factor']}   "
      f"AEP {_secondary['aep_gwh']}   wake {_secondary['wake_loss_fraction']}")
'''

CELL_2 = r'''# -- Cell 2 . Loaders -------------------------------------------------
# Plain arrays and dataframes only. No simulation objects are constructed
# here; that is cell 4's job and cell 4's alone.
import numpy as np
import pandas as pd

# 1. Layout coordinates -------------------------------------------------
with open(INPUTS["layout_coordinates"]["path"]) as f:
    _layout_doc = json.load(f)

layout_x_m = np.asarray(_layout_doc["layout_x_m"], dtype=float)
layout_y_m = np.asarray(_layout_doc["layout_y_m"], dtype=float)
farm_centre_lat = float(_layout_doc["farm_centre_lat"])
farm_centre_lon = float(_layout_doc["farm_centre_lon"])
n_turbines = layout_x_m.size

print("LAYOUT")
print("=" * 78)
print(f"shapes            x {layout_x_m.shape}  y {layout_y_m.shape}")
print(f"turbine count     {n_turbines}")
print(f"farm centre       ({farm_centre_lat}, {farm_centre_lon})")
print(f"x bounds (m)      {layout_x_m.min():.1f} .. {layout_x_m.max():.1f}"
      f"   extent {layout_x_m.max() - layout_x_m.min():.1f}")
print(f"y bounds (m)      {layout_y_m.min():.1f} .. {layout_y_m.max():.1f}"
      f"   extent {layout_y_m.max() - layout_y_m.min():.1f}")
print("metadata          " + ", ".join(
    f"{k}={v}" for k, v in _layout_doc.items()
    if not isinstance(v, (list, dict))))
print("first 5 positions (x_m, y_m)")
for i in range(5):
    print(f"  {layout_x_m[i]:10.1f} {layout_y_m[i]:10.1f}")
print()

# 2. Wind resource input ------------------------------------------------
wind = pd.read_csv(INPUTS["wind_resource_input"]["path"], parse_dates=["time"])

print("WIND RESOURCE INPUT")
print("=" * 78)
print(f"shape             {wind.shape}")
print(f"columns           {list(wind.columns)}")
print(f"time bounds       {wind['time'].min()} .. {wind['time'].max()}")
print(f"years present     {sorted(wind['time'].dt.year.unique().tolist())}")
print(f"ws bounds         {wind['ws_native_ms'].min():.6f} .. "
      f"{wind['ws_native_ms'].max():.6f}   mean {wind['ws_native_ms'].mean():.6f}")
print(f"wd bounds (deg)   {wind['wd_deg'].min():.6f} .. {wind['wd_deg'].max():.6f}")
print(f"non-finite        ws {int((~np.isfinite(wind['ws_native_ms'])).sum())}"
      f"  wd {int((~np.isfinite(wind['wd_deg'])).sum())}")
print("head")
print(wind.head().to_string(index=False))
print()

# 3. Power / Ct table ---------------------------------------------------
power_ct = pd.read_csv(INPUTS["iea22mw_power_ct_table"]["path"])

print("POWER / CT TABLE")
print("=" * 78)
print(f"shape             {power_ct.shape}")
print(f"columns           {list(power_ct.columns)}")
print(f"monotone in ws    {bool(power_ct['wind_speed_ms'].is_monotonic_increasing)}")
print(f"ws bounds         {power_ct['wind_speed_ms'].min()} .. "
      f"{power_ct['wind_speed_ms'].max()}")
print("head")
print(power_ct.head().to_string(index=False))
print("tail")
print(power_ct.tail().to_string(index=False))
'''

CELL_3 = r'''# -- Cell 3 . Environment ---------------------------------------------
# The repo declares a FLOOR, not a pin: phase_2/kit/requirements.txt:33 reads
# "py_wake>=2.6". The version the S3 run actually executed under is recorded
# in reports/task2_scorer_bathymetry_ranking_20260710.md and LLM_AGENT_LOG.md
# as 2.6.17. Because the repo does not pin, this cell prints and FLAGS rather
# than asserting: a version difference is reportable, not fatal, and belongs
# in the anchor's write-up.
import sys

import py_wake

REPO_DECLARED = "py_wake>=2.6  (phase_2/kit/requirements.txt:33 - a floor, not a pin)"
S3_RECORDED_VERSION = "2.6.17"

installed = py_wake.__version__

print("ENVIRONMENT")
print("=" * 78)
print(f"python            {sys.version.split()[0]}")
print(f"executable        {sys.executable}")
print(f"py_wake installed {installed}")
print(f"repo declares     {REPO_DECLARED}")
print(f"S3 run recorded   {S3_RECORDED_VERSION}")

if installed == S3_RECORDED_VERSION:
    print("\n[OK] py_wake matches the version the S3 run executed under.")
else:
    print(f"\n[FLAG] py_wake {installed} != {S3_RECORDED_VERSION} recorded for the "
          f"S3 run.\n       The repo pins no exact version, so this is not "
          f"asserted. Record the\n       difference with the anchor result: a "
          f"mismatch here is a candidate\n       explanation for any discrepancy "
          f"cell 5 reports.")
'''

# Fixed verbatim by the dispatch. Nothing may be added to this cell.
CELL_4 = (
    "# Principal writes this cell from Phase_2.pdf p.5-6: wind farm model,\n"
    "# wake deficit model, superposition, shear, turbine object from the\n"
    "# loaded table, simulation call producing AEP and CF. No repo code\n"
    "# consulted."
)

CELL_5 = r'''# -- Cell 5 . Assertions against cell 4 -------------------------------
# Interface this cell consumes from cell 4. Define these three names there:
#
#   cf_recomputed              net capacity factor, a fraction in [0, 1]
#   aep_recomputed_gwh         annual energy production, GWh per year
#   wake_fraction_recomputed   wake loss, a fraction in [0, 1]
#
# Gated:    turbine count, minimum spacing, CF against the stored value.
# Reported: AEP and wake fraction differences, printed, not asserted.
_defined = set(globals())   # not dir(): a comprehension has its own scope
_missing = [name for name in
            ("cf_recomputed", "aep_recomputed_gwh", "wake_fraction_recomputed")
            if name not in _defined]
assert not _missing, (
    "cell 4 has not defined: " + ", ".join(_missing) +
    "\nRun cell 4 first; see the interface comment at the top of this cell.")

CF_TOLERANCE_PP = 0.1
MIN_SPACING_M = 1420.0      # Phase_2.pdf p.5 siting constraint
EXPECT_N_TURBINES = 55      # Phase_2.pdf p.5 farm definition

print("GATED ASSERTIONS")
print("=" * 78)

# 1. Turbine count ------------------------------------------------------
print(f"turbine count           {n_turbines}  (required {EXPECT_N_TURBINES})")
assert n_turbines == EXPECT_N_TURBINES, (
    f"turbine count {n_turbines} != {EXPECT_N_TURBINES}")
assert layout_y_m.size == EXPECT_N_TURBINES, (
    f"y array holds {layout_y_m.size} positions, x holds {n_turbines}")
print("  [PASS]")

# 2. Minimum pairwise spacing ------------------------------------------
_d = np.hypot(layout_x_m[:, None] - layout_x_m[None, :],
              layout_y_m[:, None] - layout_y_m[None, :])
np.fill_diagonal(_d, np.inf)
min_spacing_m = float(_d.min())
_i, _j = np.unravel_index(np.argmin(_d), _d.shape)
print(f"min pairwise spacing    {min_spacing_m!r} m  (required >= {MIN_SPACING_M})")
print(f"  closest pair          turbines {_i} and {_j}")
print(f"  margin                {min_spacing_m - MIN_SPACING_M:+.6f} m")
assert min_spacing_m >= MIN_SPACING_M, (
    f"min pairwise spacing {min_spacing_m} m < {MIN_SPACING_M} m")
print("  [PASS]")

# 3. Capacity factor against the stored value ---------------------------
cf_diff_pp = (float(cf_recomputed) - CF_STORED) * 100.0
print(f"CF recomputed           {float(cf_recomputed)!r}")
print(f"CF stored               {CF_STORED!r}")
print(f"  difference            {cf_diff_pp:+.6f} pp  "
      f"(tolerance +/- {CF_TOLERANCE_PP} pp)")
assert abs(cf_diff_pp) <= CF_TOLERANCE_PP, (
    f"CF differs by {cf_diff_pp:+.6f} pp, outside +/- {CF_TOLERANCE_PP} pp")
print("  [PASS]")

print()
print("REPORTED, NOT GATED")
print("=" * 78)

aep_diff = float(aep_recomputed_gwh) - AEP_STORED_GWH
aep_rel_pct = 100.0 * aep_diff / AEP_STORED_GWH
print(f"AEP recomputed          {float(aep_recomputed_gwh)!r} GWh")
print(f"AEP stored              {AEP_STORED_GWH!r} GWh")
print(f"  difference            {aep_diff:+.6f} GWh  ({aep_rel_pct:+.6f} %)")

wake_diff_pp = (float(wake_fraction_recomputed) - WAKE_STORED) * 100.0
print(f"wake recomputed         {float(wake_fraction_recomputed)!r}")
print(f"wake stored             {WAKE_STORED!r}")
print(f"  difference            {wake_diff_pp:+.6f} pp")

print()
print("=" * 78)
print("All gated assertions passed. Record the three reported differences, the")
print("py_wake version line from cell 3, and cell 4's source verbatim with the")
print("anchor result.")
'''


def code_cell(source: str) -> dict:
    lines = source.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }


def main() -> None:
    nb = {
        "cells": [code_cell(c) for c in (CELL_1, CELL_2, CELL_3, CELL_4, CELL_5)],
        "metadata": {
            "kernelspec": {
                "display_name": "pywake",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_IPYNB.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_IPYNB, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
        f.write("\n")
    print(f"wrote {OUT_IPYNB}  ({OUT_IPYNB.stat().st_size} bytes, "
          f"{len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
