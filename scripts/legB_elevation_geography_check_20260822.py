#!/usr/bin/env python3
"""Is `elevation_m` a phantom feature, or degenerate by geography?

Follow-up to the 2026-08-22 dependency sweep, which found that the downscaler's
elevation DEM has never existed in this repo, so `elevation_m` (FEATURES[3]) is
identically zero. The open question is whether that is an accident with no
consequence or an accident that happens to coincide with the correct value.

The claim to test: **every location the downscaler is trained on and applied to
is a sea point**, in which case surface elevation is zero by definition and the
loader's zero fallback returned the right answer for the wrong reason.

Checks, all read-only:

  1. `train_downscaler` restricts rows to `sea = seamask > 0.5` (code), and
     `downscale` returns NaN off-sea (code). Verified here by re-deriving the
     same mask and counting.
  2. The 43,715 submission footprint cells: are they a subset of the sea mask?
  3. Do any footprint cells sit on a land pixel of the target grid?
  4. `dist_shore_km`, the sibling terrain feature, is NOT degenerate. Reported
     as a contrast so "terrain features are useless here" is not over-claimed
     from one zero column.

Output: `reports/legB_elevation_geography_20260822.md`, ASCII-asserted.

Run:  python scripts/legB_elevation_geography_check_20260822.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))

import downscaling as dn                 # noqa: E402
import tier2_d7_score_blocks as S        # noqa: E402
import terrain_features as tf            # noqa: E402

REPORT = REPO / "reports" / "legB_elevation_geography_20260822.md"


def main():
    st = dn._static()
    seamask = np.asarray(st.seamask)
    sea = seamask > 0.5
    lat = np.asarray(st.lat)
    lon = np.asarray(st.lon)

    n_tot = sea.size
    n_sea = int(sea.sum())
    n_land = n_tot - n_sea

    print("target grid %s = %d cells" % (sea.shape, n_tot))
    print("  sea  %d (%.2f%%)" % (n_sea, 100.0 * n_sea / n_tot))
    print("  land %d (%.2f%%)" % (n_land, 100.0 * n_land / n_tot))

    # ---- 1. the DEM really is absent, and the fallback really is zeros ----
    dem = tf._load_elevation_dem()
    print("\n_load_elevation_dem() -> %r" % (dem,))
    assert dem is None, "a DEM is present; this whole analysis is moot, re-check"
    terr = dn._terrain()
    elev = np.asarray(terr["elevation_m"])
    print("elevation_m: shape %s, min %.6f, max %.6f, unique %d"
          % (elev.shape, float(elev.min()), float(elev.max()),
             int(np.unique(elev).size)))
    assert np.all(elev == 0.0), "elevation is not identically zero"

    # ---- 2. footprint cells vs the sea mask ----
    S._prep()
    ys, xs = S._YS, S._XS
    fp_sea = sea[ys, xs]
    n_fp = ys.size
    n_fp_sea = int(fp_sea.sum())
    n_fp_land = n_fp - n_fp_sea
    print("\nfootprint cells: %d" % n_fp)
    print("  on sea  %d" % n_fp_sea)
    print("  on land %d" % n_fp_land)

    # ---- 3. training rows are sea-only, by the same mask the trainer uses ----
    sea_flat = dn._sea_flat()
    assert sea_flat.sum() == n_sea, "sea mask disagrees between paths"
    print("\ntrain_downscaler row filter: keep = sea & finite(...)")
    print("  sea pixels available per snapshot: %d of %d" % (n_sea, n_tot))
    print("  land pixels ever used in training: 0 (excluded by the mask)")

    # ---- 4. the sibling feature, for contrast ----
    dsh = np.asarray(terr["dist_shore_km"])
    dsh_sea = dsh[sea]
    print("\ndist_shore_km over sea cells: min %.3f, max %.3f, mean %.3f, unique %d"
          % (float(dsh_sea.min()), float(dsh_sea.max()), float(dsh_sea.mean()),
             int(np.unique(dsh_sea).size)))

    fp_dsh = dsh[ys, xs]
    fp_lat = lat[ys, xs]
    fp_lon = lon[ys, xs]

    verdict = "DEGENERATE BY GEOGRAPHY" if n_fp_land == 0 else "NOT PURELY OFFSHORE"

    L = []
    L.append("# Is `elevation_m` a phantom feature? 2026-08-22")
    L.append("")
    L.append("Produced by `scripts/legB_elevation_geography_check_20260822.py`.")
    L.append("Read-only: no fit, no pipeline run, no network.")
    L.append("")
    L.append("## Verdict: **%s**" % verdict)
    L.append("")
    L.append("## 1. The zero is real")
    L.append("")
    L.append("`terrain_features._load_elevation_dem()` returns `None` (the DEM has")
    L.append("never existed in this repo), so `elevation_m` is identically zero:")
    L.append("min %.1f, max %.1f, %d unique value across all %d target cells."
             % (float(elev.min()), float(elev.max()), int(np.unique(elev).size), n_tot))
    L.append("Both asserted by this script.")
    L.append("")
    L.append("## 2. Where the downscaler actually lives")
    L.append("")
    L.append("| Population | Cells | On sea | On land |")
    L.append("|---|---|---|---|")
    L.append("| Target grid %s | %d | %d (%.2f%%) | %d (%.2f%%) |"
             % (str(sea.shape), n_tot, n_sea, 100.0 * n_sea / n_tot,
                n_land, 100.0 * n_land / n_tot))
    L.append("| Training rows | sea-only by construction | %d per snapshot | **0** |" % n_sea)
    L.append("| Submission footprint | %d | %d | **%d** |" % (n_fp, n_fp_sea, n_fp_land))
    L.append("")
    L.append("`train_downscaler` (`downscaling.py:129,140`) builds")
    L.append("`keep = sea & isfinite(...)` and keeps only those rows, and")
    L.append("`downscale` returns NaN at every non-sea pixel. So land cells exist on")
    L.append("the grid but are excluded from BOTH training and inference.")
    L.append("")
    L.append("Footprint geography: lat %.3f to %.3f, lon %.3f to %.3f."
             % (float(fp_lat.min()), float(fp_lat.max()),
                float(fp_lon.min()), float(fp_lon.max())))
    L.append("Distance to shore over the footprint: min %.2f km, median %.2f km,"
             % (float(fp_dsh.min()), float(np.median(fp_dsh))))
    L.append("max %.2f km." % float(fp_dsh.max()))
    L.append("")
    if n_fp_land == 0:
        L.append("## 3. What this changes about the finding")
        L.append("")
        L.append("Every location the downscaler is trained on and applied to is a sea")
        L.append("point. **Surface elevation at a sea point is zero.** So the loader's")
        L.append("zero fallback returned the correct value, and the missing DEM had no")
        L.append("consequence for this model.")
        L.append("")
        L.append("The right description is therefore **degenerate by geography**, not")
        L.append("phantom: the feature is inherited from the Phase 1 station design,")
        L.append("where targets sit at coastal and land stations and elevation varies")
        L.append("and matters. Carried into a fully offshore Phase 2 target, its")
        L.append("correct value is a constant zero, and a constant column yields no")
        L.append("LightGBM split gain. The loader fallback masked a missing input")
        L.append("without changing a single prediction.")
        L.append("")
        L.append("**What does NOT change.** The mechanism is still the")
        L.append("silent-degradation class: `return None` handled by a quiet default")
        L.append("rather than a loud stop. It was harmless HERE because geography made")
        L.append("the default correct, which is luck about this deployment and not a")
        L.append("property of the code. Had the same loader been pointed at a coastal")
        L.append("target it would have zeroed a live feature just as silently.")
    else:
        L.append("## 3. The upgrade does NOT apply")
        L.append("")
        L.append("%d footprint cells sit on land pixels, so elevation is not" % n_fp_land)
        L.append("uniformly zero-by-definition over the applied population. The")
        L.append("original 'phantom feature' wording stands and the missing DEM is a")
        L.append("real information loss at those cells.")
    L.append("")
    L.append("## 4. The sibling feature is not degenerate")
    L.append("")
    L.append("Stated so that one zero column does not get generalised into 'the")
    L.append("terrain features do nothing here'. `dist_shore_km` is computed from the")
    L.append("sea mask, not from the DEM, and over sea cells it spans %.2f to %.2f km"
             % (float(dsh_sea.min()), float(dsh_sea.max())))
    L.append("with %d distinct values. It is live, and it is the feature that carries"
             % int(np.unique(dsh_sea).size))
    L.append("the coastal-gradient information `elevation_m` would have carried on a")
    L.append("land-touching target.")
    L.append("")

    text = "\n".join(L) + "\n"
    bad = sorted(set(c for c in text if ord(c) > 126))
    assert not bad, "non-ASCII in report: %r" % bad
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    print("\nwrote %s" % REPORT.relative_to(REPO).as_posix())

    print("\n" + json.dumps({
        "verdict": verdict,
        "grid_cells": int(n_tot), "grid_sea": n_sea, "grid_land": n_land,
        "footprint_cells": int(n_fp), "footprint_sea": n_fp_sea,
        "footprint_land": n_fp_land,
        "elevation_unique_values": int(np.unique(elev).size),
        "dist_shore_km_sea": {"min": round(float(dsh_sea.min()), 3),
                              "max": round(float(dsh_sea.max()), 3),
                              "unique": int(np.unique(dsh_sea).size)},
    }, indent=2))


if __name__ == "__main__":
    main()
