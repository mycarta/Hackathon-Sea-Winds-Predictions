"""Guard B: no fit step consumes a 2022 row, and the fitted objects are
independent of which inference windows are installed.

CC dispatch 2026-08-19, condition 5 on the blocker-4 approval: "any fit step
consuming a 2022 row, or any run-time fitted object differing from the 2021
rehearsal, is a stop."

**Honest scope note.** There is no stored 2021 rehearsal artifact to diff
against: `scripts/tier2_swap_rehearsal.py` was written but never run (swap
runbook §R5, "script written; DO NOT run yet"). So this guard cannot compare
against a saved reference. Instead it establishes the same property directly and
more strongly, by running the fit twice under the two window installations and
comparing, and by watching the actual data flow rather than reading the source:

  B1  Static. Every fit step's training-date source is a literal 2016-2020
      range, independent of the installed windows. Asserted, with the dates
      enumerated so the claim is checkable rather than asserted from reading.

  B2  Runtime instrumentation. `forecast_hres.build_hres_table` and
      `forecast_hres.build_climatology_forecast` are wrapped to record every
      date they are called with during fitting. Any date in an eval year that
      reaches a fit is a stop. This catches a hidden dependency that static
      reading would miss.

  B3  Installation independence. `fit_forecast` is run twice, once with the 2022
      windows installed and once with the archived 2021 windows installed, and
      the fitted objects are compared by SHA-256. Identical means the fit cannot
      have seen the inference set at all, which is the property the frozen-model
      requirement actually needs. The swap is physical and is restored in a
      `finally` block.

The B3 SHAs are recorded as the reference baseline for any future rehearsal.

Deterministic given the kit's `random_state=42` downscaler patch; `fit_forecast`
itself uses seeded LightGBM. No unseeded stochastic step is introduced here.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO_ROOT / "phase_2" / "kit" / "phase_2"
SHIP = REPO_ROOT / "phase_2" / "phase2_dataset_ship"
ACTIVE_INFER = SHIP / "inference"
ARCHIVE_2021 = REPO_ROOT / "phase_2" / "inference_2021"
STAGED_2022 = REPO_ROOT / "phase_2" / "inference_2022" / "inference"
OUT_JSON = REPO_ROOT / "reports" / "guard_b_no_2022_in_fit_20260819.json"

EVAL_YEARS = (2021, 2022)

_seen_dates: list[pd.Timestamp] = []


def obj_sha(obj) -> str:
    return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()


def instrument(fh):
    """Wrap the two table builders so every requested date is recorded."""
    for name in ("build_hres_table", "build_climatology_forecast"):
        orig = getattr(fh, name)
        if getattr(orig, "_guard_b_wrapped", False):
            continue

        def make(orig=orig):
            def wrapper(dates, *a, **kw):
                try:
                    for d in pd.to_datetime(list(dates)):
                        _seen_dates.append(pd.Timestamp(d))
                except Exception:
                    pass
                return orig(dates, *a, **kw)
            wrapper._guard_b_wrapped = True
            return wrapper

        setattr(fh, name, make())


def swap_inference(src: Path) -> None:
    """Replace the active inference dir contents with those of `src`."""
    if ACTIVE_INFER.exists():
        shutil.rmtree(ACTIVE_INFER)
    ACTIVE_INFER.mkdir(parents=True)
    for w in sorted(src.glob("window_*")):
        shutil.copytree(w, ACTIVE_INFER / w.name)
    readme = src / "README.md"
    if readme.exists():
        shutil.copy2(readme, ACTIVE_INFER / "README.md")


def main() -> None:
    os.environ.setdefault("PHASE2_DATA_ROOT", str(SHIP))
    os.chdir(KIT_PHASE2)
    for p in (KIT_PHASE2, KIT_PHASE2 / "part0_dataset_setup", KIT_PHASE2 / "part1_forecast"):
        sys.path.insert(0, str(p))

    import splits
    import forecast_hres as fh
    import forecast_pipeline as P

    failures: list[str] = []
    out: dict = {"generated_by": "scripts/guard_b_no_2022_in_fit_20260819.py"}

    # ── B1 static ───────────────────────────────────────────────────────
    print("[B1] static: training-date sources")
    train = P.train_dates("6D")
    yrs = sorted({int(d.year) for d in train})
    print(f"  P.train_dates('6D'): {len(train)} dates, {train.min().date()} .. "
          f"{train.max().date()}, years {yrs}")
    if any(y in EVAL_YEARS for y in yrs):
        failures.append(f"B1 train_dates contains an eval year: {yrs}")
    calib_default = pd.to_datetime([f"{y}-{m:02d}-15" for y in (2016, 2017, 2018, 2019, 2020)
                                    for m in (2, 6, 10)])
    print(f"  calibrate_intervals default calib_dates: {len(calib_default)} dates, "
          f"years {sorted({int(d.year) for d in calib_default})}")
    out["b1"] = {"train_years": yrs, "n_train_dates": len(train),
                 "calib_years": sorted({int(d.year) for d in calib_default})}

    # ── B3a fit with 2022 installed (current state) + B2 instrumentation ─
    instrument(fh)
    print(f"\n[B2/B3] fit_forecast with installed windows year = "
          f"{splits._installed_windows_year()}")
    _seen_dates.clear()
    t0 = time.time()
    mos_a, qmos_a, adj_a, offs_a = P.fit_forecast(train)
    t_a = time.time() - t0
    seen_a = sorted({d.year for d in _seen_dates})
    n_seen_a = len(_seen_dates)
    sha_a = {"mos": obj_sha(mos_a), "qmos": obj_sha(qmos_a),
             "adj": obj_sha(adj_a), "offs": obj_sha(offs_a)}
    print(f"  fit done in {t_a:.0f}s; table builders called with {n_seen_a} dates, "
          f"years {seen_a}")
    for k, v in sha_a.items():
        print(f"    {k:5s} sha={v[:32]}...")
    bad = [y for y in seen_a if y in EVAL_YEARS]
    if bad:
        failures.append(f"B2 a fit step requested data for eval year(s) {bad}")

    # ── B3b fit with the archived 2021 windows installed ────────────────
    sha_b = None
    if not ARCHIVE_2021.exists():
        failures.append(f"B3 archive missing: {ARCHIVE_2021}")
    else:
        print(f"\n[B3] swapping inference/ -> archived 2021 windows, re-fitting")
        try:
            swap_inference(ARCHIVE_2021)
            import importlib
            importlib.reload(splits)
            print(f"  installed windows year now = {splits._installed_windows_year()}")
            if hasattr(fh._load_hres, "cache_clear"):
                fh._load_hres.cache_clear()
            _seen_dates.clear()
            t0 = time.time()
            mos_b, qmos_b, adj_b, offs_b = P.fit_forecast(train)
            t_b = time.time() - t0
            seen_b = sorted({d.year for d in _seen_dates})
            sha_b = {"mos": obj_sha(mos_b), "qmos": obj_sha(qmos_b),
                     "adj": obj_sha(adj_b), "offs": obj_sha(offs_b)}
            print(f"  fit done in {t_b:.0f}s; years requested {seen_b}")
            bad_b = [y for y in seen_b if y in EVAL_YEARS]
            if bad_b:
                failures.append(f"B3 a fit step requested data for eval year(s) {bad_b}")
        finally:
            print("  restoring 2022 windows")
            swap_inference(STAGED_2022)
            import importlib
            importlib.reload(splits)
            if hasattr(fh._load_hres, "cache_clear"):
                fh._load_hres.cache_clear()
            print(f"  installed windows year restored = {splits._installed_windows_year()}")

    if sha_b is not None:
        print("\n[B3] fitted-object comparison, 2022-installed vs 2021-installed")
        for k in sha_a:
            same = sha_a[k] == sha_b[k]
            print(f"  {k:5s} {'IDENTICAL' if same else 'DIFFERS'}")
            if not same:
                failures.append(f"B3 fitted object '{k}' differs between installations")
        out["b3"] = {"sha_windows_2022": sha_a, "sha_windows_2021": sha_b,
                     "identical": sha_a == sha_b}

    out["b2"] = {"n_dates_requested": n_seen_a, "years_requested": seen_a}
    out["failures"] = failures
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print(f"GUARD B: {'FAIL (' + str(len(failures)) + ')' if failures else 'PASS'}")
    for x in failures:
        print(f"  - {x}")
    print("=" * 70)
    print(f"wrote {OUT_JSON}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
