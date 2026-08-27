"""Download and pin the final-evaluation inference set, Zenodo record 20874645.

CC dispatch 2026-08-18, block S1.1. Contract A1 pin gate: nothing in the
pipeline may read this data until its manifest entry exists with a verified
checksum. This script is that gate, and it is the only sanctioned path by which
`inference_2022.zip` enters the repository.

**Swap-week rule (data/MANIFEST_zenodo_20335351.md).** Any re-download must use
the VERSION-SPECIFIC DOI, never the concept record, or a future version may be
pulled silently. This script therefore hardcodes the version-specific record id
20874645 and refuses to follow the concept record 19538993.

**What this pins.** Only `inference_2022.zip` (21,742,784 B). The other three
files in the record are deliberately NOT downloaded:
  - `phase2_dataset.zip` MD5 96988634e1dbce27cb5369b4809de964 is BYTE-IDENTICAL
    to the v3 file already pinned in data/MANIFEST_zenodo_20335351.md, so the
    training inputs are unchanged and re-downloading 12 GB would be waste.
  - `phase1_dataset.zip` and `mini_challenge_dataset.zip` are out of scope for
    the Task 1 swap re-run.

Deterministic: a fixed URL, a byte-exact size assertion, and two checksums. No
stochastic step, so no seed applies. Idempotent: if the zip is already present
and verifies, it is not re-downloaded.

Usage:
    conda run -n swnd python scripts/fetch_pin_inference_2022.py
    conda run -n swnd python scripts/fetch_pin_inference_2022.py --extract
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Version-specific record, per the swap-week rule. Not the concept record. ──
RECORD_ID = "20874645"
CONCEPT_RECID = "19538993"          # refused on purpose, see module docstring
RECORD_DOI = "10.5281/zenodo.20874645"
FILE_KEY = "inference_2022.zip"
FILE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files/{FILE_KEY}/content"

# ── Expected values, read from the Zenodo API 2026-08-19 ────────────────────
EXPECTED_SIZE_B = 21_742_784
EXPECTED_MD5 = "07bedad96f60de39134e01780be54a9d"

DEST_ZIP = REPO_ROOT / "phase_2" / "inference_2022" / FILE_KEY
EXTRACT_DIR = REPO_ROOT / "phase_2" / "inference_2022"


def hashes(path: Path) -> tuple[str, str]:
    """Return (md5, sha256) of a file, streamed."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def download() -> None:
    DEST_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if DEST_ZIP.exists():
        md5, _sha = hashes(DEST_ZIP)
        if md5 == EXPECTED_MD5:
            print(f"already present and MD5-verified: {DEST_ZIP}")
            return
        raise RuntimeError(
            f"{DEST_ZIP} exists but MD5 {md5} != expected {EXPECTED_MD5}. "
            "Refusing to overwrite; investigate before re-downloading."
        )
    print(f"downloading {FILE_URL}")
    urllib.request.urlretrieve(FILE_URL, DEST_ZIP)
    print(f"wrote {DEST_ZIP} ({DEST_ZIP.stat().st_size:,} B)")


def verify() -> dict:
    size = DEST_ZIP.stat().st_size
    md5, sha = hashes(DEST_ZIP)
    ok_size = size == EXPECTED_SIZE_B
    ok_md5 = md5 == EXPECTED_MD5
    print(f"size   : {size:,} B   expected {EXPECTED_SIZE_B:,}   {'OK' if ok_size else 'MISMATCH'}")
    print(f"md5    : {md5}   {'OK' if ok_md5 else 'MISMATCH'}")
    print(f"sha256 : {sha}")
    if not (ok_size and ok_md5):
        raise RuntimeError("PIN GATE FAILED: downloaded file does not match the record listing.")
    return {"size_b": size, "md5": md5, "sha256": sha}


def inspect() -> list[dict]:
    """List the archive contents without extracting."""
    with zipfile.ZipFile(DEST_ZIP) as z:
        return [{"name": i.filename, "size": i.file_size, "crc": f"{i.CRC:08x}"}
                for i in z.infolist()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true",
                    help="extract into phase_2/inference_2022/ after verifying")
    args = ap.parse_args()

    assert RECORD_ID != CONCEPT_RECID, "swap-week rule: never resolve the concept record"

    download()
    pin = verify()
    entries = inspect()
    print(f"\narchive holds {len(entries)} entries")

    out = {
        "record_id": RECORD_ID, "record_doi": RECORD_DOI,
        "concept_recid_refused": CONCEPT_RECID,
        "file_key": FILE_KEY, "url": FILE_URL,
        "pin": pin, "entries": entries,
    }
    manifest_json = EXTRACT_DIR / "pin_inference_2022.json"
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {manifest_json}")

    if args.extract:
        with zipfile.ZipFile(DEST_ZIP) as z:
            z.extractall(EXTRACT_DIR)
        print(f"extracted into {EXTRACT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
