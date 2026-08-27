"""Fetch and pin the published IEA-22-280-RWT tabular workbook.

CC dispatch 2026-08-27, item 0b. This script exists so that the workbook has a
named, committed producer like every other artifact in the repository
(contract §4.2), and so that the chain is re-runnable cold from nothing
(contract §4.4) rather than depending on someone having typed the right `curl`.

The file is already committed at the destination path, so a normal run of this
script is a VERIFY, not a download: if the file is present and its SHA-256
matches, nothing is fetched. `--force` re-downloads and re-verifies, which is
the swap-week / cold-clone path.

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
  Licence    : the upstream repository is Apache-2.0. This is third-party
               reference data, redistributed under that licence, and is NOT
               competition-kit material.

The download URL is pinned to the COMMIT SHA-1 above rather than to `main`, so
a future upstream revision cannot silently change what this script fetches.
`main` is kept only as a fallback, and the SHA-256 gate is what makes that
fallback safe: a moved `main` fails the gate instead of being accepted.

Deterministic: one HTTPS GET of a commit-pinned path, then a SHA gate. No
stochastic step, so no seed applies.

Writes: data/iea22_280_rwt/IEA-22-280-RWT_tabular.xlsx  (only if absent or --force)

Downstream: scripts/audit_iea22_published_curve_extract_20260827.py converts
this workbook's "Rotor Performance - WISDEM" sheet into the power/Ct CSV that
scripts/audit_lcoe_row2_and_curve_20260827.py feeds to PyWake.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "iea22_280_rwt" / "IEA-22-280-RWT_tabular.xlsx"

SHA256 = "8cf9a552a955982b210e8bc77702adc85c8ee0f1d1e10b5f717b3d29e95b4f84"
SIZE = 663185
BLOB_SHA1 = "4d7b02b09abb93258755e1a9961b0cf7495c4710"      # provenance only, see URL note
COMMIT_SHA1 = "b7a9b7e6b50eef9d6897b5bd8f6cbea1a97cf158"

# Pinned to the COMMIT that last touched the path, not to `main`, so a future
# upstream revision cannot silently change what this fetches. raw.githubusercontent
# serves commit SHAs and refs; it does NOT serve blob SHA-1s, so BLOB_SHA1 above is
# recorded for provenance only and is not usable as a URL component (verified
# 2026-08-27: the blob-SHA form returns HTTP 404).
URL = ("https://raw.githubusercontent.com/IEAWindSystems/IEA-22-280-RWT/"
       f"{COMMIT_SHA1}/Documentation/IEA-22-280-RWT_tabular.xlsx")
# Fallback only. `main` can move; the SHA gate below is what makes that safe.
URL_BRANCH = ("https://raw.githubusercontent.com/IEAWindSystems/IEA-22-280-RWT/"
              "main/Documentation/IEA-22-280-RWT_tabular.xlsx")

TIMEOUT_S = 120


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> tuple[bool, str, int]:
    actual = sha256_of(path)
    size = path.stat().st_size
    return (actual == SHA256 and size == SIZE), actual, size


def download(url: str, dest: Path) -> None:
    print(f"  GET {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)          # atomic: an interrupted fetch leaves no half file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file is present and verifies")
    args = ap.parse_args()

    if DEST.exists() and not args.force:
        ok, actual, size = verify(DEST)
        if ok:
            print(f"present and verified, no download needed:\n"
                  f"  {DEST.relative_to(REPO_ROOT)}\n"
                  f"  sha256 {actual}\n  size   {size:,} B")
            return
        print(f"present but DOES NOT VERIFY (sha256 {actual}, size {size:,} B); "
              f"re-downloading")

    last_err: Exception | None = None
    for url in (URL, URL_BRANCH):
        try:
            download(url, DEST)
            break
        except Exception as exc:                    # noqa: BLE001 - reported below
            print(f"  failed: {type(exc).__name__}: {exc}")
            last_err = exc
    else:
        raise SystemExit(f"could not fetch the workbook: {last_err}")

    ok, actual, size = verify(DEST)
    if not ok:
        raise SystemExit(
            "downloaded file does not match the pin, refusing to keep it:\n"
            f"  expected sha256={SHA256} size={SIZE}\n"
            f"  actual   sha256={actual} size={size}"
        )
    print(f"downloaded and verified:\n  {DEST.relative_to(REPO_ROOT)}\n"
          f"  sha256 {actual}\n  size   {size:,} B")


if __name__ == "__main__":
    main()
