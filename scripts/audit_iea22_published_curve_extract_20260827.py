"""Extract the published IEA 22 MW (IEA-22-280-RWT) WISDEM rotor-performance
curve from the upstream tabular workbook into a power/Ct CSV.

CC dispatch 2026-08-27, item 0b. Read-only with respect to the pipeline: this
script only converts a downloaded upstream workbook into the same three-column
CSV schema that `data/iea22mw_power_ct.csv` (the kit's synthetic fallback ramp)
already uses, so the two curves can be swapped at one call site in
`scripts/task2_scorer_replica.py:build_turbine`.

**Why this exists.** The scorer replica currently drives the wake model with the
kit's *synthetic* generic cubic ramp (`turbines_catalog._generic_power_ct`),
which is rated at 12.0 m/s because that is the module's hardcoded `rated_ws`
fallback, not because the IEA 22 MW machine is rated there. The published
reference turbine is rated at 11.13 m/s. Item 0b tests whether that difference
accounts for part of the gross-CF gap against the organizers' corrected
baseline (CF 53.2 % net, wake 7.1 %, AEP 5,635 GWh, LCOE 83.1 EUR/MWh --
organizer correction on the Codabench board, 2026-08-10).

**Source, pinned (contract A1 / amendment v2.1 §5a).**
  Repository : github.com/IEAWindSystems/IEA-22-280-RWT
  Path       : Documentation/IEA-22-280-RWT_tabular.xlsx
  Branch     : main
  Git blob   : 4d7b02b09abb93258755e1a9961b0cf7495c4710
  Last commit touching the path: b7a9b7e6b50eef9d6897b5bd8f6cbea1a97cf158
  Retrieved  : 2026-08-27
  Size       : 663,185 bytes
  SHA-256    : 8cf9a552a955982b210e8bc77702adc85c8ee0f1d1e10b5f717b3d29e95b4f84
  Citation   : Zahle et al. (2024), IEA Wind TCP Task 55 reference wind turbine.

The workbook sheet used is "Rotor Performance - WISDEM" (the steady-state
WISDEM/CCBlade solution), NOT the HAWC2 or OpenFAST sheets, because the
dispatch names the WISDEM curve and because WISDEM's steady-state solution is
the one directly comparable to a PyWake tabular power/Ct curve.

Deterministic: a pure format conversion of a SHA-pinned input. No stochastic
step, so no seed applies.

Reads : data/iea22_280_rwt/IEA-22-280-RWT_tabular.xlsx  (SHA asserted below)
Writes: data/iea22_280_rwt/iea22_280_rwt_wisdem_power_ct.csv
        data/iea22_280_rwt/iea22_280_rwt_wisdem_power_ct.provenance.json
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
XLSX = REPO_ROOT / "data" / "iea22_280_rwt" / "IEA-22-280-RWT_tabular.xlsx"
OUT_CSV = REPO_ROOT / "data" / "iea22_280_rwt" / "iea22_280_rwt_wisdem_power_ct.csv"
OUT_PROV = REPO_ROOT / "data" / "iea22_280_rwt" / "iea22_280_rwt_wisdem_power_ct.provenance.json"

# Pinned upstream identity -- asserted, not assumed (amendment v2.1 §5a).
XLSX_SHA256 = "8cf9a552a955982b210e8bc77702adc85c8ee0f1d1e10b5f717b3d29e95b4f84"
XLSX_SIZE = 663185
UPSTREAM = {
    "repository": "https://github.com/IEAWindSystems/IEA-22-280-RWT",
    "path": "Documentation/IEA-22-280-RWT_tabular.xlsx",
    "branch": "main",
    "git_blob_sha1": "4d7b02b09abb93258755e1a9961b0cf7495c4710",
    "last_commit_sha1": "b7a9b7e6b50eef9d6897b5bd8f6cbea1a97cf158",
    "retrieved_utc_date": "2026-08-27",
    "sheet": "Rotor Performance - WISDEM",
    "citation": "Zahle, F. et al. (2024). IEA Wind TCP Task 55: "
                "IEA-22-280-RWT reference wind turbine.",
}

SHEET_NAME = "Rotor Performance - WISDEM"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# Column headers we need, matched verbatim against row 1 of the sheet.
COL_WS = "Wind [m/s]"
COL_POWER_MW = "Power [MW]"
COL_CT = "Thrust Coefficient [-]"

RATED_MW = 22.0
RATED_TOL_MW = 0.01


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(NS + "t"))
            for si in root.findall(NS + "si")]


def _sheet_path(z: zipfile.ZipFile, sheet_name: str) -> str:
    """Resolve a sheet display name to its worksheet XML part."""
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rid = None
    for sh in wb.iter(NS + "sheet"):
        if sh.get("name") == sheet_name:
            rid = sh.get(RNS + "id")
            break
    if rid is None:
        raise RuntimeError(f"sheet {sheet_name!r} not in workbook")
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    targets = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
    tgt = targets[rid].lstrip("/")
    return tgt if tgt.startswith("xl/") else "xl/" + tgt


def _read_sheet(z: zipfile.ZipFile, part: str, ss: list[str]) -> dict[int, dict[str, str]]:
    """Return {row_number: {column_letter: value}}. Keyed by cell reference so a
    sparse or reordered sheet cannot silently shift a column."""
    root = ET.fromstring(z.read(part))
    out: dict[int, dict[str, str]] = {}
    for row in root.iter(NS + "row"):
        rn = int(row.get("r"))
        cells: dict[str, str] = {}
        for c in row.findall(NS + "c"):
            ref = c.get("r") or ""
            col = "".join(ch for ch in ref if ch.isalpha())
            v = c.find(NS + "v")
            if v is None or v.text is None:
                continue
            if c.get("t") == "s":
                cells[col] = ss[int(v.text)]
            else:
                cells[col] = v.text
        out[rn] = cells
    return out


def main() -> None:
    if not XLSX.exists():
        raise SystemExit(f"missing pinned input: {XLSX}")
    actual_sha = sha256_of(XLSX)
    actual_size = XLSX.stat().st_size
    if actual_sha != XLSX_SHA256 or actual_size != XLSX_SIZE:
        raise SystemExit(
            "pinned-artifact mismatch, refusing to proceed:\n"
            f"  expected sha256={XLSX_SHA256} size={XLSX_SIZE}\n"
            f"  actual   sha256={actual_sha} size={actual_size}"
        )
    print(f"pinned input verified: {XLSX.name} sha256={actual_sha[:12]}... "
          f"size={actual_size}")

    with zipfile.ZipFile(XLSX) as z:
        ss = _shared_strings(z)
        part = _sheet_path(z, SHEET_NAME)
        rows = _read_sheet(z, part, ss)

    header_row = min(rows)
    header = rows[header_row]
    col_of = {name: col for col, name in header.items()}
    for want in (COL_WS, COL_POWER_MW, COL_CT):
        if want not in col_of:
            raise SystemExit(f"column {want!r} not found; header row was {header}")
    c_ws, c_pw, c_ct = col_of[COL_WS], col_of[COL_POWER_MW], col_of[COL_CT]
    print(f"sheet {SHEET_NAME!r}: header row {header_row}, "
          f"ws={c_ws} power={c_pw} ct={c_ct}")

    records = []
    for rn in sorted(rows):
        if rn == header_row:
            continue
        cells = rows[rn]
        if c_ws not in cells or c_pw not in cells or c_ct not in cells:
            continue
        ws = float(cells[c_ws])
        power_w = float(cells[c_pw]) * 1e6
        ct = float(cells[c_ct])
        records.append((ws, power_w, ct))

    records.sort(key=lambda r: r[0])
    if not records:
        raise SystemExit("no data rows parsed")

    # ── Assertions on the extracted curve ───────────────────────────────
    ws_vals = [r[0] for r in records]
    pw_vals = [r[1] for r in records]
    if len(set(ws_vals)) != len(ws_vals):
        raise SystemExit("duplicate wind speeds in extracted curve")
    if any(p < 0 for p in pw_vals):
        raise SystemExit("negative power in extracted curve")
    peak_mw = max(pw_vals) / 1e6
    if abs(peak_mw - RATED_MW) > RATED_TOL_MW:
        raise SystemExit(f"peak power {peak_mw:.4f} MW != rated {RATED_MW} MW")

    # Rated wind speed = first speed at which power reaches rated (within tol).
    rated_ws = next(w for w, p, _ in records
                    if p / 1e6 >= RATED_MW - RATED_TOL_MW)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        fh.write("wind_speed_ms,power_w,ct\n")
        for ws, pw, ct in records:
            fh.write(f"{ws!r},{pw!r},{ct!r}\n")

    prov = {
        "generated_by": "scripts/audit_iea22_published_curve_extract_20260827.py",
        "upstream": UPSTREAM,
        "source_file_sha256": actual_sha,
        "source_file_size_bytes": actual_size,
        "n_points": len(records),
        "ws_min_ms": ws_vals[0],
        "ws_max_ms": ws_vals[-1],
        "rated_power_mw": peak_mw,
        "rated_wind_speed_ms": rated_ws,
        "output_csv": str(OUT_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
        "output_csv_sha256": None,   # filled below
    }
    prov["output_csv_sha256"] = sha256_of(OUT_CSV)
    with open(OUT_PROV, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)

    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)}: {len(records)} points, "
          f"{ws_vals[0]:.2f}..{ws_vals[-1]:.2f} m/s")
    print(f"  rated power       : {peak_mw:.4f} MW")
    print(f"  rated wind speed  : {rated_ws:.4f} m/s "
          f"(kit synthetic ramp: 12.0 m/s)")
    print(f"  csv sha256        : {prov['output_csv_sha256']}")


if __name__ == "__main__":
    main()
