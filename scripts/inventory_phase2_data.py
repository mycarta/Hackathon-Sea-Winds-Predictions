#!/usr/bin/env python3
"""
Phase 2 dataset inventory — read-only.
NaN scan strategy: stratified sample (1 file per month, year 2016 only).
If any variable shows NaN > 0, it is flagged for a full targeted scan.

Outputs (both written incrementally):
  reports/checkpoint_phase2_inv.json  — updated after each component
  reports/phase2_data_review.md       — final human-readable report
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "phase_2" / "phase2_dataset_ship"
ZIP_PATH = ROOT / "phase_2" / "phase2_dataset.zip"
MANIFEST_PATH = ROOT / "phase_2" / "MANIFEST_zenodo_20335351.md"
REPORTS_DIR = ROOT / "reports"
CHECKPOINT = REPORTS_DIR / "checkpoint_phase2_inv.json"
REPORT_PATH = REPORTS_DIR / "phase2_data_review.md"

EXPECTED_SHA256 = "f4b60b3de16adc161f32b4bdd652cfab102910bee32fef3e6cd097d77ab5982d"
EXPECTED_MD5 = "96988634e1dbce27cb5369b4809de964"

# ── Brief figures (from prompt — do NOT read the PDF) ──────────────────────────

BRIEF = {
    "training_years": list(range(2016, 2021)),
    "footprint_points": 43_715,
    "siting_cells": 159,
    "inference_year": 2021,
    "n_windows": 8,
    "n_horizons": 3,
    "n_hours": 4,
    "submission_rows": 4_196_640,
}

# ── Global state (serialised to checkpoint after each component) ───────────────

state: dict = {}


def _save() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(state, indent=2, default=str))
    print("  → checkpoint saved", flush=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def _disk_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _nc_summary(path: Path, load_values: bool = True) -> dict:
    """Open one NetCDF and return dims, coords, variable metadata, and NaN counts."""
    ds = xr.open_dataset(path, engine="netcdf4")
    dims = dict(ds.sizes)
    coords_info = {
        k: {"dtype": str(v.dtype), "shape": list(v.shape)}
        for k, v in ds.coords.items()
    }
    global_attrs = dict(ds.attrs)

    var_info: dict = {}
    for vname, var in ds.data_vars.items():
        shape = list(var.shape)
        dtype = str(var.dtype)
        units = var.attrs.get("units", "?")
        long_name = var.attrs.get("long_name", "")
        estimated_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize if shape else 0

        nan_count = None
        nan_frac = None
        if load_values:
            try:
                arr = var.values
                nan_count = int(np.isnan(arr.astype(float)).sum()) if np.issubdtype(arr.dtype, np.number) else 0
                total = int(arr.size)
                nan_frac = round(nan_count / total, 8) if total > 0 else None
            except Exception as exc:
                nan_count = f"ERROR: {exc}"
                nan_frac = None

        var_info[vname] = {
            "dtype": dtype,
            "shape": shape,
            "dims": list(var.dims),
            "units": units,
            "long_name": long_name,
            "estimated_mem_bytes": estimated_bytes,
            "nan_count": nan_count,
            "nan_frac": nan_frac,
        }

    ds.close()
    return {
        "dims": dims,
        "coords": coords_info,
        "global_attrs": global_attrs,
        "variables": var_info,
    }


def _accum_nan(paths: list, prev: dict | None = None) -> dict:
    """Accumulate NaN counts over a list of NC files (one pass per file)."""
    cum: dict = prev or {}
    for p in paths:
        ds = xr.open_dataset(p, engine="netcdf4")
        for vname, var in ds.data_vars.items():
            try:
                arr = var.values
                if not np.issubdtype(arr.dtype, np.number):
                    continue
                nans = int(np.isnan(arr.astype(float)).sum())
                total = int(arr.size)
            except Exception:
                continue
            if vname not in cum:
                cum[vname] = {"nan_count": 0, "total": 0}
            cum[vname]["nan_count"] += nans
            cum[vname]["total"] += total
        ds.close()

    return {
        k: {
            "nan_frac": round(v["nan_count"] / v["total"], 8) if v["total"] > 0 else None,
            "nan_count": v["nan_count"],
            "total_elements": v["total"],
        }
        for k, v in cum.items()
    }


def _monthly_sample(year_dir: Path, year: int, pattern: str = "*.nc") -> list:
    """Return one file per calendar month from year_dir (first alphabetically)."""
    all_files = sorted(year_dir.glob(pattern))
    samples = []
    for m in range(1, 13):
        prefix = f"{year}{m:02d}"
        hits = [f for f in all_files if prefix in f.name]
        if hits:
            samples.append(hits[0])
    return samples


# ── Step 0: Checksum verification ──────────────────────────────────────────────

def step0_checksums() -> None:
    print("\n=== STEP 0: Checksum verification ===", flush=True)

    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found — STOP.", flush=True)
        sys.exit(1)

    zip_size = ZIP_PATH.stat().st_size
    print(f"  File: {ZIP_PATH.name}  ({_fmt(zip_size)}, {zip_size:,} bytes)", flush=True)
    print(f"  Computing SHA-256 and MD5 in a single pass …", flush=True)

    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    block = 1 << 20          # 1 MiB read chunks
    progress_interval = 512  # print every 512 MiB
    read = 0

    with open(ZIP_PATH, "rb") as fh:
        while chunk := fh.read(block):
            sha256.update(chunk)
            md5.update(chunk)
            read += len(chunk)
            if (read // block) % progress_interval == 0:
                pct = 100.0 * read / zip_size
                print(f"    {pct:5.1f}%  ({_fmt(read)} read)", flush=True)

    sha256_got = sha256.hexdigest()
    md5_got = md5.hexdigest()
    sha256_ok = sha256_got == EXPECTED_SHA256
    md5_ok = md5_got == EXPECTED_MD5

    print(f"  SHA-256  expected : {EXPECTED_SHA256}", flush=True)
    print(f"  SHA-256  computed : {sha256_got}", flush=True)
    print(f"  SHA-256  match    : {'PASS ✓' if sha256_ok else 'FAIL ✗'}", flush=True)
    print(f"  MD5      expected : {EXPECTED_MD5}", flush=True)
    print(f"  MD5      computed : {md5_got}", flush=True)
    print(f"  MD5      match    : {'PASS ✓' if md5_ok else 'FAIL ✗'}", flush=True)

    state["step0"] = {
        "zip_path": str(ZIP_PATH),
        "zip_size_bytes": zip_size,
        "sha256_expected": EXPECTED_SHA256,
        "sha256_computed": sha256_got,
        "sha256_match": sha256_ok,
        "md5_expected": EXPECTED_MD5,
        "md5_computed": md5_got,
        "md5_match": md5_ok,
        "status": "PASS" if (sha256_ok and md5_ok) else "FAIL",
    }
    _save()

    if not (sha256_ok and md5_ok):
        print("\nERROR: checksum mismatch — STOP. Do not proceed.", flush=True)
        sys.exit(1)

    print("  Step 0 PASSED.", flush=True)


# ── Component inventories ──────────────────────────────────────────────────────

def inv_static() -> None:
    print("\n=== Static grids ===", flush=True)
    base = DATA / "static"
    result: dict = {}

    # footprint_points.parquet — row count is the primary discrepancy check
    fp_path = base / "footprint_points.parquet"
    df = pd.read_parquet(fp_path)
    nan_counts = df.isna().sum().to_dict()
    result["footprint_points"] = {
        "path": str(fp_path.relative_to(ROOT)),
        "size_bytes": fp_path.stat().st_size,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "nan_counts": {k: int(v) for k, v in nan_counts.items()},
        "sample_head": df.head(3).to_dict(orient="list"),
    }
    brief_match = len(df) == BRIEF["footprint_points"]
    print(f"  footprint_points.parquet : {len(df):,} rows  "
          f"(brief={BRIEF['footprint_points']:,})  "
          f"{'✓' if brief_match else '⚠ MISMATCH'}", flush=True)

    # arome_static.nc
    for nc_name in ("arome_static.nc", "reanalysis_static.nc"):
        nc_path = base / nc_name
        s = _nc_summary(nc_path, load_values=True)
        s["path"] = str(nc_path.relative_to(ROOT))
        s["size_bytes"] = nc_path.stat().st_size
        key = nc_name.replace(".", "_")
        result[key] = s
        print(f"  {nc_name} : dims={s['dims']}  vars={list(s['variables'].keys())}", flush=True)

    # bathymetry
    bathy_path = base / "bathymetry" / "emodnet_northsea_1km.nc"
    s = _nc_summary(bathy_path, load_values=True)
    s["path"] = str(bathy_path.relative_to(ROOT))
    s["size_bytes"] = bathy_path.stat().st_size
    result["bathymetry"] = s
    print(f"  emodnet_northsea_1km.nc : dims={s['dims']}  vars={list(s['variables'].keys())}", flush=True)

    # Duplicate check: arome_static also lives in train/arome/
    dup_path = DATA / "train" / "arome" / "arome_static.nc"
    result["arome_static_train_duplicate"] = dup_path.exists()
    if dup_path.exists():
        print(f"  NOTE: arome_static.nc also present in train/arome/ root", flush=True)

    state["static"] = result
    _save()


def inv_arome() -> None:
    print("\n=== AROME target (high-res) ===", flush=True)
    base = DATA / "train" / "arome"

    year_counts: dict = {}
    for yr in range(2016, 2021):
        yr_dir = base / str(yr)
        year_counts[yr] = len(sorted(yr_dir.glob("*.nc"))) if yr_dir.exists() else None

    static_in_root = [f.name for f in base.glob("*.nc")]
    total_daily = sum(c for c in year_counts.values() if c is not None)
    total_size = _disk_size(base)

    print(f"  Year counts : {year_counts}", flush=True)
    print(f"  Static files in root : {static_in_root}", flush=True)
    print(f"  Total daily files : {total_daily}  on disk : {_fmt(total_size)}", flush=True)

    # Structural sample — one file from 2016
    sample_yr = 2016
    sample_files = sorted((base / str(sample_yr)).glob("*.nc"))
    sample_file = sample_files[0]
    summary = _nc_summary(sample_file, load_values=True)
    print(f"  Sample ({sample_file.name}) : dims={summary['dims']}  "
          f"vars={list(summary['variables'].keys())}", flush=True)

    # Estimate single-file and full-corpus in-memory footprint
    total_elements = 1
    for d in summary["dims"].values():
        total_elements *= d
    n_vars = len(summary["variables"])
    # Determine dtype; default float32 if mixed or unknown
    dtypes = [v["dtype"] for v in summary["variables"].values()]
    dominant_dtype = dtypes[0] if len(set(dtypes)) == 1 else "float32"
    try:
        bytes_per_elem = np.dtype(dominant_dtype).itemsize
    except Exception:
        bytes_per_elem = 4
    one_file_mem = total_elements * n_vars * bytes_per_elem
    corpus_mem = one_file_mem * total_daily

    # Stratified NaN sample: 1 file per month from sample_yr
    monthly = _monthly_sample(base / str(sample_yr), sample_yr)
    print(f"  NaN sample : {len(monthly)} files (1/month, year={sample_yr})", flush=True)
    nan_result = _accum_nan(monthly)
    flagged = [v for v, info in nan_result.items() if (info["nan_frac"] or 0) > 0]
    if flagged:
        print(f"  ⚠ NaN detected in : {flagged}", flush=True)
    else:
        print(f"  NaN scan : all clean in sample", flush=True)

    state["arome"] = {
        "year_counts": year_counts,
        "static_in_root": static_in_root,
        "total_daily_files": total_daily,
        "total_size_bytes": total_size,
        "sample_file": str(sample_file.relative_to(ROOT)),
        "sample_structure": summary,
        "one_file_mem_estimate_bytes": one_file_mem,
        "corpus_mem_estimate_bytes": corpus_mem,
        "nan_sample_year": sample_yr,
        "nan_sample_n": len(monthly),
        "nan_by_variable": nan_result,
        "nan_flagged_variables": flagged,
    }
    _save()


def inv_arome_coarse() -> None:
    print("\n=== AROME coarse (arome_coarse125) ===", flush=True)
    base = DATA / "train" / "arome_coarse125"

    year_counts: dict = {}
    for yr in range(2016, 2021):
        yr_dir = base / str(yr)
        year_counts[yr] = len(sorted(yr_dir.glob("*.nc"))) if yr_dir.exists() else None

    static_in_root = [f.name for f in base.glob("*.nc")]
    total_daily = sum(c for c in year_counts.values() if c is not None)
    total_size = _disk_size(base)

    print(f"  Year counts : {year_counts}", flush=True)
    print(f"  Static in root : {static_in_root}", flush=True)

    sample_yr = 2016
    sample_files = sorted((base / str(sample_yr)).glob("*.nc"))
    sample_file = sample_files[0]
    summary = _nc_summary(sample_file, load_values=True)
    print(f"  Sample dims={summary['dims']}  vars={list(summary['variables'].keys())}", flush=True)

    monthly = _monthly_sample(base / str(sample_yr), sample_yr)
    print(f"  NaN sample : {len(monthly)} files", flush=True)
    nan_result = _accum_nan(monthly)
    flagged = [v for v, info in nan_result.items() if (info["nan_frac"] or 0) > 0]

    state["arome_coarse"] = {
        "year_counts": year_counts,
        "static_in_root": static_in_root,
        "total_daily_files": total_daily,
        "total_size_bytes": total_size,
        "sample_file": str(sample_file.relative_to(ROOT)),
        "sample_structure": summary,
        "nan_sample_year": sample_yr,
        "nan_sample_n": len(monthly),
        "nan_by_variable": nan_result,
        "nan_flagged_variables": flagged,
    }
    _save()


def inv_hres() -> None:
    print("\n=== HRES (ECMWF) ===", flush=True)
    hres_dir = DATA / "train" / "hres"
    files = sorted(hres_dir.glob("*.parquet"))
    result: dict = {"files": [], "total_size_bytes": _disk_size(hres_dir)}

    for fp in files:
        df = pd.read_parquet(fp)
        nan_counts = {k: int(v) for k, v in df.isna().sum().items()}
        nan_fracs = {k: round(v / len(df), 8) if len(df) > 0 else None for k, v in nan_counts.items()}

        time_range: dict = {}
        for col in df.columns:
            if any(t in col.lower() for t in ("time", "date", "step", "lead")):
                try:
                    time_range[col] = {"min": str(df[col].min()), "max": str(df[col].max())}
                except Exception:
                    pass

        file_info = {
            "name": fp.name,
            "size_bytes": fp.stat().st_size,
            "rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "nan_counts": nan_counts,
            "nan_fracs": nan_fracs,
            "time_range": time_range,
        }
        result["files"].append(file_info)
        print(f"  {fp.name} : {len(df):,} rows × {len(df.columns)} cols", flush=True)
        print(f"    time_range : {time_range}", flush=True)

    state["hres"] = result
    _save()


def inv_reanalysis() -> None:
    print("\n=== Reanalysis (surface, daily) ===", flush=True)
    base = DATA / "train" / "reanalysis"

    year_counts: dict = {}
    for yr in range(2016, 2021):
        yr_dir = base / str(yr)
        year_counts[yr] = len(sorted(yr_dir.glob("*.nc"))) if yr_dir.exists() else None

    total_daily = sum(c for c in year_counts.values() if c is not None)
    total_size = _disk_size(base)
    print(f"  Year counts : {year_counts}", flush=True)

    sample_yr = 2016
    sample_files = sorted((base / str(sample_yr)).glob("*.nc"))
    sample_file = sample_files[0]
    summary = _nc_summary(sample_file, load_values=True)
    print(f"  Sample dims={summary['dims']}  vars={list(summary['variables'].keys())}", flush=True)

    monthly = _monthly_sample(base / str(sample_yr), sample_yr)
    print(f"  NaN sample : {len(monthly)} files (1/month, year={sample_yr})", flush=True)
    nan_result = _accum_nan(monthly)
    flagged = [v for v, info in nan_result.items() if (info["nan_frac"] or 0) > 0]
    if flagged:
        print(f"  ⚠ NaN detected in : {flagged}", flush=True)

    state["reanalysis"] = {
        "year_counts": year_counts,
        "total_daily_files": total_daily,
        "total_size_bytes": total_size,
        "sample_file": str(sample_file.relative_to(ROOT)),
        "sample_structure": summary,
        "nan_sample_year": sample_yr,
        "nan_sample_n": len(monthly),
        "nan_by_variable": nan_result,
        "nan_flagged_variables": flagged,
    }
    _save()


def inv_reanalysis_extra() -> None:
    print("\n=== Reanalysis extra (pressure-level / surface-level) ===", flush=True)
    base = DATA / "train" / "reanalysis_extra"
    all_files = sorted(base.glob("*.nc"))
    pl_files = [f for f in all_files if "_pl_" in f.name]
    sl_files = [f for f in all_files if "_sl_" in f.name]
    total_size = _disk_size(base)

    def _extract_years(files: list) -> list:
        years = set()
        for f in files:
            parts = f.stem.split("_")
            # e.g. reanalysis_pl_2016_01 → parts[2]='2016'
            #      reanalysis_sl_2016    → parts[2]='2016'
            if len(parts) >= 3:
                try:
                    years.add(int(parts[2]))
                except ValueError:
                    pass
        return sorted(years)

    pl_years = _extract_years(pl_files)
    sl_years = _extract_years(sl_files)
    print(f"  PL files : {len(pl_files)}  years={pl_years}", flush=True)
    print(f"  SL files : {len(sl_files)}  years={sl_years}", flush=True)
    print(f"  Total size : {_fmt(total_size)}", flush=True)

    pl_summary = _nc_summary(pl_files[0], load_values=True) if pl_files else {}
    sl_summary = _nc_summary(sl_files[0], load_values=True) if sl_files else {}

    if pl_summary:
        print(f"  PL sample dims={pl_summary['dims']}  vars={list(pl_summary['variables'].keys())}", flush=True)
    if sl_summary:
        print(f"  SL sample dims={sl_summary['dims']}  vars={list(sl_summary['variables'].keys())}", flush=True)

    # NaN scan: first file of each type (already small — light enough to load fully)
    pl_nan = _accum_nan([pl_files[0]]) if pl_files else {}
    sl_nan = _accum_nan([sl_files[0]]) if sl_files else {}

    state["reanalysis_extra"] = {
        "pl_count": len(pl_files),
        "sl_count": len(sl_files),
        "pl_years_covered": pl_years,
        "sl_years_covered": sl_years,
        "pl_files": [f.name for f in pl_files],
        "sl_files": [f.name for f in sl_files],
        "total_size_bytes": total_size,
        "pl_sample_structure": pl_summary,
        "sl_sample_structure": sl_summary,
        "pl_sample_nan": pl_nan,
        "sl_sample_nan": sl_nan,
    }
    _save()


def inv_inference() -> None:
    print("\n=== Inference windows ===", flush=True)
    base = DATA / "inference"
    windows = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("window_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    result: dict = {}

    for w in windows:
        meta_path = w / "metadata.json"
        hres_path = w / "context_hres_north_sea.parquet"
        ra_path = w / "context_reanalysis_north_sea.parquet"

        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        def _parquet_info(p: Path) -> dict:
            if not p.exists():
                return {"exists": False}
            df = pd.read_parquet(p)
            tr: dict = {}
            for col in df.columns:
                if any(t in col.lower() for t in ("time", "date", "step", "lead")):
                    try:
                        tr[col] = {"min": str(df[col].min()), "max": str(df[col].max())}
                    except Exception:
                        pass
            return {
                "exists": True,
                "rows": len(df),
                "n_columns": len(df.columns),
                "columns": list(df.columns),
                "time_range": tr,
                "nan_counts": {k: int(v) for k, v in df.isna().sum().items()},
            }

        hres_info = _parquet_info(hres_path)
        ra_info = _parquet_info(ra_path)

        result[w.name] = {"metadata": meta, "hres": hres_info, "reanalysis": ra_info}
        print(f"  {w.name} : meta={meta}  "
              f"hres={hres_info.get('rows','?')} rows  "
              f"ra={ra_info.get('rows','?')} rows", flush=True)

    state["inference"] = {
        "n_windows": len(windows),
        "windows": result,
    }
    _save()


# ── Discrepancy check ──────────────────────────────────────────────────────────

def check_discrepancies() -> None:
    print("\n=== Discrepancy check vs. brief ===", flush=True)
    discs: list = []

    def flag(msg: str) -> None:
        discs.append(msg)
        print(f"  ⚠  {msg}", flush=True)

    def ok(msg: str) -> None:
        print(f"  ✓  {msg}", flush=True)

    # Training years (AROME coverage)
    if "arome" in state:
        found = sorted(yr for yr, cnt in state["arome"]["year_counts"].items() if cnt)
        if found != BRIEF["training_years"]:
            flag(f"AROME training years: expected {BRIEF['training_years']}, found {found}")
        else:
            ok(f"AROME training years: {found}")

    # Footprint points
    if "static" in state and "footprint_points" in state["static"]:
        rows = state["static"]["footprint_points"]["rows"]
        if rows != BRIEF["footprint_points"]:
            flag(f"footprint_points.parquet rows: expected {BRIEF['footprint_points']:,}, found {rows:,}")
        else:
            ok(f"footprint_points: {rows:,}")

    # Siting cells — check footprint columns for a 'cell' or 'site' column
    if "static" in state and "footprint_points" in state["static"]:
        cols = state["static"]["footprint_points"]["columns"]
        cell_cols = [c for c in cols if any(t in c.lower() for t in ("cell", "site", "siting"))]
        if not cell_cols:
            flag(f"siting_cells ({BRIEF['siting_cells']}) cannot be verified — "
                 f"no 'cell'/'site' column found in footprint_points.parquet "
                 f"(columns: {cols})")
        else:
            # Try to count unique cells
            try:
                df = pd.read_parquet(DATA / "static" / "footprint_points.parquet")
                unique_cells = df[cell_cols[0]].nunique()
                if unique_cells != BRIEF["siting_cells"]:
                    flag(f"siting_cells: expected {BRIEF['siting_cells']}, "
                         f"found {unique_cells} unique in column '{cell_cols[0]}'")
                else:
                    ok(f"siting_cells: {unique_cells} (column '{cell_cols[0]}')")
            except Exception as exc:
                flag(f"siting_cells check failed: {exc}")

    # HRES coverage
    if "hres" in state:
        for f in state["hres"]["files"]:
            fname = f["name"]
            if "2019" not in fname or "2020" not in fname:
                flag(f"HRES '{fname}' appears to cover only 2016–2018; 2019–2020 absent")

    # Reanalysis extra coverage
    if "reanalysis_extra" in state:
        re = state["reanalysis_extra"]
        if re["pl_years_covered"] and max(re["pl_years_covered"]) < 2020:
            flag(f"reanalysis_extra pressure-level covers {re['pl_years_covered']}; 2019–2020 absent")
        if re["sl_years_covered"] and max(re["sl_years_covered"]) < 2020:
            flag(f"reanalysis_extra surface-level covers {re['sl_years_covered']}; 2019–2020 absent")

    # Inference windows count
    if "inference" in state:
        n = state["inference"]["n_windows"]
        if n != BRIEF["n_windows"]:
            flag(f"Inference windows: expected {BRIEF['n_windows']}, found {n}")
        else:
            ok(f"Inference windows: {n}")

        # All windows should be in 2021
        for wname, wdata in state["inference"]["windows"].items():
            meta = wdata.get("metadata", {})
            for key, val in meta.items():
                if val and "2021" not in str(val) and any(t in key.lower() for t in ("start", "end", "date", "year")):
                    flag(f"{wname} metadata key '{key}'={val} does not reference 2021")

    # Submission shape arithmetic
    computed = (BRIEF["footprint_points"] * BRIEF["n_windows"]
                * BRIEF["n_horizons"] * BRIEF["n_hours"])
    if computed != BRIEF["submission_rows"]:
        flag(f"Submission shape arithmetic: "
             f"{BRIEF['footprint_points']}×{BRIEF['n_windows']}×"
             f"{BRIEF['n_horizons']}×{BRIEF['n_hours']}={computed:,} "
             f"≠ brief's {BRIEF['submission_rows']:,}")
    else:
        ok(f"Submission arithmetic: {BRIEF['footprint_points']}×{BRIEF['n_windows']}×"
           f"{BRIEF['n_horizons']}×{BRIEF['n_hours']} = {computed:,}")

    # NaN summary — flag any variable with NaN > 0 across all components
    nan_flagged: list = []
    for comp in ("arome", "arome_coarse", "reanalysis"):
        if comp in state:
            flagged_vars = state[comp].get("nan_flagged_variables", [])
            for v in flagged_vars:
                nan_flagged.append(f"{comp}/{v}")
    if nan_flagged:
        flag(f"NaN detected in sample — consider full scan for: {nan_flagged}")
    else:
        ok("NaN scan clean across AROME, AROME-coarse, reanalysis sample")

    state["discrepancies"] = discs
    _save()
    print(f"\n  Total discrepancies: {len(discs)}", flush=True)


# ── Report generation ──────────────────────────────────────────────────────────

def _var_table(var_info: dict) -> list:
    lines = ["| Variable | dims | shape | units | nan_frac |",
             "|----------|------|-------|-------|----------|"]
    for vname, info in var_info.items():
        frac = info.get("nan_frac")
        flag = " ⚠" if frac and frac > 0 else ""
        lines.append(f"| `{vname}` | {info['dims']} | {info['shape']} | "
                     f"{info['units']} | {frac}{flag} |")
    return lines


def write_report() -> None:
    print("\n=== Writing report ===", flush=True)
    L: list = []

    def h(text: str, level: int = 2) -> None:
        L.append(f"\n{'#' * level} {text}\n")

    def p(*args) -> None:
        L.append(" ".join(str(a) for a in args))

    L.append("# Phase 2 Dataset Inventory\n")
    L.append(f"**Generated:** 2026-07-02  ")
    L.append(f"**Script:** `scripts/inventory_phase2_data.py`  ")
    L.append(f"**Checkpoint:** `reports/checkpoint_phase2_inv.json`  ")
    L.append(f"**NaN strategy:** stratified sample — 1 file/month, year 2016\n")

    # Step 0
    h("Step 0 — Provenance Verification")
    s0 = state.get("step0", {})
    L.append("| Item | Value |")
    L.append("|------|-------|")
    L.append(f"| File | `{ZIP_PATH.name}` |")
    L.append(f"| Size | {_fmt(s0.get('zip_size_bytes', 0))} ({s0.get('zip_size_bytes', 0):,} bytes) |")
    L.append(f"| SHA-256 computed | `{s0.get('sha256_computed', '?')}` |")
    L.append(f"| SHA-256 match | {'PASS ✓' if s0.get('sha256_match') else 'FAIL ✗'} |")
    L.append(f"| MD5 computed | `{s0.get('md5_computed', '?')}` |")
    L.append(f"| MD5 match | {'PASS ✓' if s0.get('md5_match') else 'FAIL ✗'} |")
    L.append(f"| Manifest | `phase_2/MANIFEST_zenodo_20335351.md` |")
    L.append(f"| Status | **{s0.get('status', '?')}** |")

    # Static
    h("Static Grids")
    if "static" in state:
        st = state["static"]
        if "footprint_points" in st:
            fp = st["footprint_points"]
            brief_rows = BRIEF["footprint_points"]
            match = "✓" if fp["rows"] == brief_rows else "⚠ MISMATCH"
            h("footprint_points.parquet", 3)
            L.append(f"- **Rows:** {fp['rows']:,}  (brief: {brief_rows:,}) {match}")
            L.append(f"- Size on disk: {_fmt(fp['size_bytes'])}")
            L.append(f"- Columns: `{fp['columns']}`")

        for key, label in [("arome_static_nc", "static/arome_static.nc"),
                            ("reanalysis_static_nc", "static/reanalysis_static.nc"),
                            ("bathymetry", "static/bathymetry/emodnet_northsea_1km.nc")]:
            if key in st:
                s = st[key]
                h(label, 3)
                L.append(f"- Size: {_fmt(s.get('size_bytes', 0))}")
                L.append(f"- Dimensions: `{s.get('dims', {})}`")
                L.extend(_var_table(s.get("variables", {})))

        if st.get("arome_static_train_duplicate"):
            L.append("\n> NOTE: `arome_static.nc` also present in `train/arome/` root (duplicate).")

    # AROME high-res
    h("AROME Target (High-Resolution)")
    if "arome" in state:
        a = state["arome"]
        expected_days = sum(366 if yr in (2016, 2020) else 365 for yr in range(2016, 2021))
        match = "✓" if a["total_daily_files"] == expected_days else f"⚠ expected {expected_days}"
        L.append(f"- **Total daily files:** {a['total_daily_files']}  ({match})")
        L.append(f"- **Size on disk:** {_fmt(a['total_size_bytes'])}")
        L.append(f"- **Est. single-file in-memory footprint:** {_fmt(a.get('one_file_mem_estimate_bytes', 0))}")
        L.append(f"- **Est. full corpus in-memory footprint:** {_fmt(a.get('corpus_mem_estimate_bytes', 0))}")
        L.append(f"- Files per year: `{a['year_counts']}`")
        if a["static_in_root"]:
            L.append(f"- Static files in root of `train/arome/`: `{a['static_in_root']}`")

        h("Sample structure (one file)", 3)
        L.append(f"File: `{a['sample_file']}`")
        L.append(f"Dimensions: `{a['sample_structure'].get('dims', {})}`")
        L.extend(_var_table(a["sample_structure"].get("variables", {})))

        h(f"NaN scan (sample: {a['nan_sample_n']} files, year={a['nan_sample_year']})", 3)
        for vname, info in a["nan_by_variable"].items():
            frac = info.get("nan_frac", 0) or 0
            flag = " **⚠ flag for full scan**" if frac > 0 else ""
            L.append(f"- `{vname}`: nan_frac={frac}{flag}")

    # AROME coarse
    h("AROME Coarse (arome_coarse125)")
    if "arome_coarse" in state:
        ac = state["arome_coarse"]
        L.append(f"- **Total files:** {ac['total_daily_files']}  "
                 f"(vs AROME: {state.get('arome', {}).get('total_daily_files', '?')})")
        L.append(f"- **Size:** {_fmt(ac['total_size_bytes'])}")
        L.append(f"- Files per year: `{ac['year_counts']}`")
        L.append(f"- Sample dims: `{ac['sample_structure'].get('dims', {})}`")
        L.append(f"- Variables: `{list(ac['sample_structure'].get('variables', {}).keys())}`")
        h("NaN scan", 3)
        for vname, info in ac["nan_by_variable"].items():
            frac = info.get("nan_frac", 0) or 0
            flag = " **⚠**" if frac > 0 else ""
            L.append(f"- `{vname}`: {frac}{flag}")

    # HRES
    h("HRES (ECMWF Forecast)")
    if "hres" in state:
        L.append(f"> ⚠ **COVERAGE GAP: HRES covers only 2016–2018** "
                 f"(filename: `north_sea_hres_2016_2018.parquet`). 2019–2020 absent.\n")
        for f in state["hres"]["files"]:
            h(f["name"], 3)
            L.append(f"- Rows: {f['rows']:,}")
            L.append(f"- Columns ({f['n_columns']}): `{f['columns']}`")
            L.append(f"- Size: {_fmt(f['size_bytes'])}")
            L.append(f"- Time range: `{f.get('time_range', {})}`")
            nans = {k: v for k, v in f["nan_counts"].items() if v > 0}
            L.append(f"- NaN counts (non-zero only): `{nans if nans else 'none'}`")

    # Reanalysis
    h("Reanalysis (Surface, Daily)")
    if "reanalysis" in state:
        ra = state["reanalysis"]
        L.append(f"- **Total daily files:** {ra['total_daily_files']}")
        L.append(f"- **Size:** {_fmt(ra['total_size_bytes'])}")
        L.append(f"- Files per year: `{ra['year_counts']}`")
        L.append(f"- Sample dims: `{ra['sample_structure'].get('dims', {})}`")
        L.extend(_var_table(ra["sample_structure"].get("variables", {})))

    # Reanalysis extra
    h("Reanalysis Extra (Pressure-Level / Surface-Level)")
    if "reanalysis_extra" in state:
        re = state["reanalysis_extra"]
        L.append(f"> ⚠ **COVERAGE GAP: reanalysis_extra covers only 2016–2018.** 2019–2020 absent.\n")
        L.append(f"- PL files: {re['pl_count']} ({re['pl_years_covered']})")
        L.append(f"- SL files: {re['sl_count']} ({re['sl_years_covered']})")
        L.append(f"- Total size: {_fmt(re['total_size_bytes'])}")
        if re["pl_sample_structure"]:
            L.append(f"- PL sample dims: `{re['pl_sample_structure'].get('dims', {})}`")
            L.append(f"- PL variables: `{list(re['pl_sample_structure'].get('variables', {}).keys())}`")
        if re["sl_sample_structure"]:
            L.append(f"- SL sample dims: `{re['sl_sample_structure'].get('dims', {})}`")
            L.append(f"- SL variables: `{list(re['sl_sample_structure'].get('variables', {}).keys())}`")

    # Inference windows
    h("Inference Windows")
    if "inference" in state:
        inf = state["inference"]
        brief_n = BRIEF["n_windows"]
        match = "✓" if inf["n_windows"] == brief_n else f"⚠ expected {brief_n}"
        L.append(f"- **Count:** {inf['n_windows']}  (brief: {brief_n}) {match}")
        for wname, wdata in inf["windows"].items():
            meta = wdata.get("metadata", {})
            h(wname, 3)
            L.append(f"- Metadata: `{meta}`")
            hres = wdata.get("hres", {})
            ra_w = wdata.get("reanalysis", {})
            if hres.get("exists"):
                L.append(f"- HRES context: {hres.get('rows', '?'):,} rows  "
                         f"cols={hres.get('n_columns', '?')}  "
                         f"time={hres.get('time_range', {})}")
            if ra_w.get("exists"):
                L.append(f"- Reanalysis context: {ra_w.get('rows', '?'):,} rows  "
                         f"time={ra_w.get('time_range', {})}")

    # Discrepancies
    h("Discrepancy Summary")
    discs = state.get("discrepancies", [])
    if discs:
        for d in discs:
            L.append(f"- ⚠ {d}")
    else:
        L.append("No discrepancies found against brief figures.")

    # Open questions
    h("Open Questions")
    L.append("1. **HRES gap (2019–2020):** Is this intentional? Are HRES features dropped or "
             "extrapolated for training years 2019–2020?")
    L.append("2. **reanalysis_extra gap (2019–2020):** Pressure-level and surface-level "
             "supplemental data only runs to 2018. Are these features required for training "
             "on 2019–2020, or are they optional / not used in those years?")
    L.append("3. **arome_coarse125 role:** Does this serve as the low-resolution AROME proxy "
             "at inference time (since there is no high-res target at inference)? "
             "Confirm role vs. the main AROME.")
    L.append("4. **arome_static.nc duplicate:** File appears in both `static/` and "
             "`train/arome/` root. Are they identical? Intended?")
    L.append("5. **siting_cells (159):** Could not be directly confirmed — depends on "
             "whether a siting-cell column exists in footprint_points.parquet or a "
             "separate file. Script will have reported this; act on result.")
    L.append("6. **NaN follow-up:** If the sample scan flagged any variable with NaN > 0, "
             "a targeted full scan for that variable across all 1827 files is recommended.")
    L.append("")

    REPORT_PATH.write_text("\n".join(L))
    state["report_path"] = str(REPORT_PATH)
    print(f"  Report written → {REPORT_PATH}", flush=True)
    _save()


# ── Final re-checksum (integrity gate) ────────────────────────────────────────

def final_rechecksum() -> None:
    print("\n=== Final re-checksum of zip ===", flush=True)
    sha256 = hashlib.sha256()
    block = 1 << 20
    with open(ZIP_PATH, "rb") as fh:
        while chunk := fh.read(block):
            sha256.update(chunk)
    got = sha256.hexdigest()
    ok = got == EXPECTED_SHA256
    print(f"  SHA-256: {got}  {'✓ matches — raw data intact' if ok else '✗ MISMATCH — FILE CHANGED'}", flush=True)
    state["final_rechecksum"] = {"sha256": got, "match": ok}
    _save()
    if not ok:
        print("ERROR: zip SHA-256 changed during inventory run — raw data integrity violation.", flush=True)
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    step0_checksums()
    inv_static()
    inv_arome()
    inv_arome_coarse()
    inv_hres()
    inv_reanalysis()
    inv_reanalysis_extra()
    inv_inference()
    check_discrepancies()
    write_report()
    final_rechecksum()

    print("\n✓ Inventory complete.", flush=True)
    print(f"  Checkpoint : {CHECKPOINT}", flush=True)
    print(f"  Report     : {REPORT_PATH}", flush=True)
