#!/usr/bin/env python3
"""Leg B Stage R2: refit the block-excluded downscaler, seeded, and pin on creation.

The July pickle (`downscaler_blockexcl.pkl`, SHA b68eb5fe...) was lost. It was
fitted under the OLD unseeded downscaler code; `random_state=42` entered the kit
later, in commit 17baf59. So this refit produces a NEW artifact and **cannot and
will not** match b68eb5fe. That is expected, authorized, and the reason the
frozen file's SHA constant has to be re-pinned.

TRAINING CONSTRUCTION, identical to `tier2_f2_d14_precheck.get_downscaler()`:

    excl      = ec.exclusion_set()                    # blocks +/- 14 d buffer
    d2020     = every 5th 2020 date in the target root
    d2020_red = d2020 minus excl                      # 63 days in July
    dwn       = dn.train_downscaler(d2020_red, hours=ec.HOURS)   # (0, 6, 12, 18)

`_LGBM` in downscaling.py carries `random_state=42` (n_estimators=300,
max_depth=8, learning_rate=0.05, num_leaves=63, subsample=0.8,
colsample_bytree=0.8). The subsample and colsample draws are what made the
unseeded original irreproducible; with the seed present the refit is
reproducible from here forward, which is the actual gain.

OUTPUT, dual custody per the pattern approved 2026-08-21:
  repo      data/downscaler_blockexcl_20260822.pkl   (version controlled)
  working   <PROTECTED_ARTIFACTS>/downscaler_blockexcl_20260822.pkl

Both hash-verified after writing. The SHA is printed for the frozen-file re-pin
and for the PINNED_ARTIFACTS.md promotion entry.

Run:  python scripts/legB_R2_refit_downscaler_20260822.py
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

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part1_forecast"))
sys.path.insert(0, str(REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup"))

import tier2_eval_common as ec            # noqa: E402
import downscaling as dn                  # noqa: E402
import config                             # noqa: E402
import target_loader                      # noqa: E402
from _publication_paths import ppath  # noqa: E402  (publication tree)

OUT_REPO = REPO / "data" / "downscaler_blockexcl_20260822.pkl"
OUT_WORK = ppath("<PROTECTED_ARTIFACTS>/downscaler_blockexcl_20260822.pkl",
                 must_exist=False)   # output
REPORT = REPO / "reports" / "legB_R2_refit_20260822.md"

OLD_SHA = "b68eb5fe57fab817364f3df2feb4f4bd77a6658cf91642301b9159e9d0fa8e0a"
JULY_N_DAYS = 63          # "[F2] training downscaler on 63 block-excluded 2020 days"
JULY_FIT_S = 103          # "[F2] downscaler trained in 103s"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    t_all = time.time()
    print("=" * 74)
    print("Leg B R2: block-excluded downscaler refit, seeded")
    print("=" * 74)

    # ---- reproduce the training date construction exactly -----------------
    excl = ec.exclusion_set()
    d2020 = [d for d in target_loader.list_dates(config.target_root()) if d.year == 2020][::5]
    d2020_red = [d for d in d2020 if d not in excl]

    print("\ntraining construction")
    print("  2020 dates in target root, every 5th : %d" % len(d2020))
    print("  excluded (blocks +/- 14 d buffer)    : %d of those"
          % (len(d2020) - len(d2020_red)))
    print("  block-excluded training days         : %d" % len(d2020_red))
    print("  hours                                : %s" % (ec.HOURS,))
    print("  July used                            : %d days" % JULY_N_DAYS)
    if len(d2020_red) != JULY_N_DAYS:
        print("\n  DAY COUNT DIFFERS FROM JULY (%d vs %d). The training construction is"
              % (len(d2020_red), JULY_N_DAYS))
        print("  supposed to be identical. Stop and inspect before using this object.")
        sys.exit(3)
    print("  day count matches July exactly")

    # ---- seed check, before spending the fit ------------------------------
    seed = dn._LGBM.get("random_state")
    print("\nLightGBM params: %s" % json.dumps(dn._LGBM, sort_keys=True))
    assert seed == 42, "random_state is %r, expected 42; the refit would be unseeded" % seed
    print("  random_state = 42 present. The refit is reproducible from here forward;")
    print("  the July original was not, which is why it cannot be reproduced.")

    # ---- fit --------------------------------------------------------------
    print("\nfitting on %d days x %d hours ..." % (len(d2020_red), len(ec.HOURS)))
    t0 = time.time()
    dwn = dn.train_downscaler(d2020_red, hours=ec.HOURS)
    fit_s = time.time() - t0
    print("  trained in %.0f s (July: %d s)" % (fit_s, JULY_FIT_S))
    assert set(dwn.keys()) == {"u", "v"}, "unexpected model keys: %s" % sorted(dwn)

    # ---- write, dual custody, hash-verified -------------------------------
    OUT_REPO.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPO, "wb") as fh:
        pickle.dump(dwn, fh)
    new_sha = sha256(OUT_REPO)
    size = os.path.getsize(OUT_REPO)

    OUT_WORK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_REPO, OUT_WORK)
    assert sha256(OUT_WORK) == new_sha, "working copy differs from the repo copy"

    print("\nwrote")
    print("  repo    %s" % OUT_REPO.relative_to(REPO))
    print("  working %s" % OUT_WORK)
    print("  size    %d bytes" % size)
    print("  SHA-256 %s" % new_sha)
    assert new_sha != OLD_SHA, (
        "The refit reproduced b68eb5fe exactly. That should be impossible under "
        "an unseeded original; stop and work out why before trusting anything.")
    print("\n  differs from the lost b68eb5fe as expected (unseeded original)")

    # ---- round-trip the pickle -------------------------------------------
    with open(OUT_REPO, "rb") as fh:
        back = pickle.load(fh)
    assert set(back.keys()) == {"u", "v"}
    print("  pickle round-trips and carries both component models")

    L = []
    L.append("# Leg B R2: downscaler refit, 2026-08-22")
    L.append("")
    L.append("Produced by `scripts/legB_R2_refit_downscaler_20260822.py`.")
    L.append("")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append("| New SHA-256 | `%s` |" % new_sha)
    L.append("| Size | %d bytes |" % size)
    L.append("| Supersedes | `%s` (LOST) |" % OLD_SHA)
    L.append("| Fit date | 2026-08-22 |")
    L.append("| Seed | `random_state=42` (`downscaling.py` `_LGBM`, present since `17baf59`) |")
    L.append("| Training days | %d block-excluded 2020 days, every 5th date minus blocks +/- 14 d |" % len(d2020_red))
    L.append("| Hours | %s |" % (ec.HOURS,))
    L.append("| Fit wall time | %.0f s (July: %d s) |" % (fit_s, JULY_FIT_S))
    L.append("| Repo copy | `%s` |" % OUT_REPO.relative_to(REPO).as_posix())
    L.append("| Working copy | `%s` |" % OUT_WORK.as_posix())
    L.append("")
    L.append("## Why this cannot match the lost pickle")
    L.append("")
    L.append("The July object was fitted under the OLD unseeded downscaler code.")
    L.append("`random_state=42` entered the kit later, in `17baf59` (2026-07-20), after")
    L.append("that fit. `_LGBM` draws `subsample=0.8` and `colsample_bytree=0.8`, so an")
    L.append("unseeded fit is not reproducible even from identical data. The refit is a")
    L.append("NEW artifact, pinned on creation, and the frozen file's SHA constant is")
    L.append("re-pinned to it under the authorization of 2026-08-20.")
    L.append("")
    L.append("The gain is forward-looking: this object IS reproducible from its seed and")
    L.append("its committed training construction, which the original never was.")
    L.append("")
    L.append("## Day count")
    L.append("")
    L.append("%d block-excluded training days, matching July's %d exactly. The script"
             % (len(d2020_red), JULY_N_DAYS))
    L.append("asserts this before fitting: a differing count would mean the training")
    L.append("construction had drifted, which would make the refit a different experiment")
    L.append("rather than a re-execution.")
    L.append("")

    text = "\n".join(L) + "\n"
    bad = sorted(set(c for c in text if ord(c) > 126))
    assert not bad, "non-ASCII in report: %r" % bad
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8", newline="\n")

    print("\nwrote %s" % REPORT.relative_to(REPO))
    print("\n" + json.dumps({
        "new_sha256": new_sha, "size_bytes": size, "supersedes": OLD_SHA,
        "training_days": len(d2020_red), "hours": list(ec.HOURS),
        "fit_s": round(fit_s, 1), "elapsed_s": round(time.time() - t_all, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
