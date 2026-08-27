#!/usr/bin/env python3
"""TASK 1 — WS d7 split structure & leakage quantification (2026-07-16).

Documents the discount on every WS-d7 holdout number. Quotes the split source
and measures how leaky the d7 holdout is. No fix; documentation only.

splits.train_val_dates (part0_dataset_setup/splits.py:174-189), verbatim:

    def train_val_dates(seed: int = 42, val_fraction: float = 0.2):
        \"\"\"Deterministic random train/val split over the TRAIN_YEARS target dates.
        Never includes any hidden year. Returns (train_dates, val_dates).\"\"\"
        import numpy as np
        import target_loader
        dates = [d for d in target_loader.list_dates(config.target_root())
                 if d.year in TRAIN_YEARS]
        rng = np.random.default_rng(seed)
        n = len(dates)
        n_val = max(1, round(n * val_fraction))
        val_idx = set(rng.choice(n, size=n_val, replace=False).tolist())
        train = [d for i, d in enumerate(dates) if i not in val_idx]
        val = [d for i, d in enumerate(dates) if i in val_idx]
        return train, val

=> val dates are a SEEDED-RANDOM SCATTER (rng.choice over the pooled 2016-2020
target dates), NOT contiguous temporal blocks.
"""
import sys
import bisect
from collections import Counter
from datetime import timedelta

# Was a hardcoded absolute repo root. The file knows where it lives.
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT + r"\phase_2\kit\phase_2")
sys.path.insert(0, ROOT + r"\phase_2\kit\phase_2\part0_dataset_setup")
import pandas as pd
import splits            # noqa: E402
from pathlib import Path  # noqa: E402  (publication tree)

# The pipeline's actual fit set (forecast_pipeline.train_dates('6D')).
fit_days = sorted(pd.to_datetime(pd.date_range("2016-01-01", "2020-12-31", freq="6D")).date)

train, val = splits.train_val_dates(seed=42)
val = sorted(val)
print(f"n_train={len(train)}  n_val={len(val)}  val_fraction={len(val)/(len(train)+len(val)):.4f}")

# contiguous vs scattered: run-lengths of consecutive val days
runs, cur = [], 1
for i in range(1, len(val)):
    cur = cur + 1 if (val[i] - val[i - 1]).days == 1 else (runs.append(cur) or 1)
runs.append(cur)
print(f"val consecutive-day run-length histogram: {dict(sorted(Counter(runs).items()))}")
print(f"max run length = {max(runs)}  => SCATTERED (a block split would show few long runs)")

# d7 leakage: fraction of val dates within 7 days of a fit date
def frac_within_7d(val_dates, ref_days):
    ref = sorted(ref_days)
    hit = 0
    for v in val_dates:
        lo, hi = v - timedelta(days=7), v + timedelta(days=7)
        i = bisect.bisect_left(ref, lo)
        hit += int(i < len(ref) and ref[i] <= hi)
    return hit, hit / len(val_dates)

h, f = frac_within_7d(val, fit_days)
print(f"\nd7 LEAKAGE: val dates within 7 days of a fit date "
      f"(train_dates('6D')): {h}/{len(val)} = {f:.4f}")
h2, f2 = frac_within_7d(val, [d for d in train])
print(f"(cross-ref, vs the split's own train list: {h2}/{len(val)} = {f2:.4f})")
