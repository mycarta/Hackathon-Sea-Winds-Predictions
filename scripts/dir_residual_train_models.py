#!/usr/bin/env python3
"""
Step 2 of the d1/d7 direction-residual fix: train 6 LightGBM quantile
models (q05/q50/q95 x lead 1/7) on the unwrapped direction residual
(scripts/artifacts/dir_residual_training.parquet, built by
dir_residual_build_training_data.py).

Features (prompt-specified, HRES + static only - no AROME-derived features,
since AROME is never available at inference):
  hres_dir_sin, hres_dir_cos, hres_speed, hour_sin, hour_cos,
  season (DJF=0,MAM=1,JJA=2,SON=3), month_sin, month_cos, lat, lon

Hyperparameters exactly as specified in the task prompt (2026-07-07):
  n_estimators=300, max_depth=7, learning_rate=0.05, num_leaves=63,
  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
Deterministic given random_state=42 (LightGBM's own internal row/feature
subsampling is seeded by this).

Train/val split: the 'split' column already tagged by
splits.py::train_val_dates(seed=42) in the training-data build (day-level,
no leakage across hours of the same day).

Parameter gate (per prompt): reports training residual distribution and
validation MAE/coverage per horizon. STOPS (raises, does not save models or
proceed) if d1 val MAE > 30 deg or d7 val MAE > 60 deg - that would signal
a likely convention mismatch or data bug, not just weak fit.

Output: models/dir_residual_{horizon}_q{quantile}.lgb (LightGBM native
Booster text format, 6 files), scripts/artifacts/dir_residual_train_report.md
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TRAIN_PATH = ARTIFACTS_DIR / "dir_residual_training.parquet"
MODELS_DIR = ROOT / "models"
REPORT_PATH = ARTIFACTS_DIR / "dir_residual_train_report.md"

LEADS = (1, 7)
QUANTILES = (0.05, 0.50, 0.95)
SEASON_CODE = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}

PARAMS = dict(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

FEATURES = ["hres_dir_sin", "hres_dir_cos", "hres_speed", "hour_sin", "hour_cos",
            "season_code", "month_sin", "month_cos", "lat", "lon"]

STOP_THRESHOLDS = {1: 30.0, 7: 60.0}


def engineer_features(df):
    df = df.copy()
    df["hres_dir_sin"] = np.sin(np.radians(df["hres_dir"]))
    df["hres_dir_cos"] = np.cos(np.radians(df["hres_dir"]))
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["season_code"] = df["season"].map(SEASON_CODE)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def circular_mae(pred_dir_deg, true_dir_deg):
    d = np.abs((pred_dir_deg - true_dir_deg + 180) % 360 - 180)
    return d.mean()


def main():
    t_start = time.time()
    print(f"Loading {TRAIN_PATH} ...")
    full = pd.read_parquet(TRAIN_PATH)
    print(f"Loaded {len(full)} rows in {time.time() - t_start:.1f}s")

    full = engineer_features(full)

    report_lines = ["# d1/d7 direction residual model - training report (2026-07-07)\n"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    stop_triggered = False
    all_metrics = {}

    for L in LEADS:
        sub = full[full["lead"] == L]
        tr = sub[sub["split"] == "train"]
        va = sub[sub["split"] == "val"]
        print(f"\n{'=' * 60}\nLEAD {L}: train={len(tr)} rows, val={len(va)} rows\n{'=' * 60}")

        resid = tr["residual"].to_numpy()
        r_mean, r_std = resid.mean(), resid.std()
        r_p05, r_p95 = np.percentile(resid, [5, 95])
        print(f"Training residual distribution: mean={r_mean:.3f} std={r_std:.3f} "
              f"p05={r_p05:.3f} p95={r_p95:.3f}")
        report_lines.append(f"## Lead {L}\n")
        report_lines.append(f"- Training residual distribution: mean={r_mean:.3f}, "
                             f"std={r_std:.3f}, p05={r_p05:.3f}, p95={r_p95:.3f} "
                             f"(expected: {'tight ~+-15deg' if L == 1 else 'wider ~+-40deg'})")

        X_tr, y_tr = tr[FEATURES], tr["residual"]
        X_va, y_va = va[FEATURES], va["residual"]

        preds_va = {}
        for q in QUANTILES:
            t0 = time.time()
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
            model.fit(X_tr, y_tr)
            elapsed = time.time() - t0
            print(f"  q{int(q * 100):02d} trained in {elapsed:.1f}s")

            model_path = MODELS_DIR / f"dir_residual_d{L}_q{int(q * 100):02d}.lgb"
            model.booster_.save_model(str(model_path))
            print(f"  saved -> {model_path}")

            preds_va[q] = model.predict(X_va)

        q_sorted = sorted(QUANTILES)
        stacked = np.vstack([preds_va[q] for q in q_sorted])
        stacked = np.sort(stacked, axis=0)  # enforce non-crossing
        pred_q05, pred_q50, pred_q95 = stacked[0], stacked[1], stacked[2]

        mae = np.abs(pred_q50 - y_va.to_numpy()).mean()
        coverage = ((y_va.to_numpy() >= pred_q05) & (y_va.to_numpy() <= pred_q95)).mean()
        print(f"\nValidation (lead {L}): MAE(q50)={mae:.3f} deg, "
              f"coverage(q05-q95)={coverage * 100:.1f}%")
        report_lines.append(f"- Validation MAE (q50 vs truth residual): {mae:.3f} deg")
        report_lines.append(f"- Validation coverage (fraction of val residuals within "
                             f"predicted q05-q95): {coverage * 100:.1f}% (target ~90%)")

        # reconstruct actual direction MAE too (circular, on the real HRES+residual chain)
        recon_dir = (va["hres_dir"].to_numpy() + pred_q50) % 360
        dir_mae = circular_mae(recon_dir, va["arome_dir"].to_numpy())
        print(f"Reconstructed direction circular MAE (lead {L}): {dir_mae:.3f} deg")
        report_lines.append(f"- Reconstructed direction circular MAE: {dir_mae:.3f} deg")

        all_metrics[L] = dict(mae=mae, coverage=coverage, dir_mae=dir_mae,
                               r_mean=r_mean, r_std=r_std, r_p05=r_p05, r_p95=r_p95)

        threshold = STOP_THRESHOLDS[L]
        if mae > threshold:
            stop_triggered = True
            msg = (f"STOP AND REPORT: lead {L} val MAE {mae:.3f} > threshold {threshold} deg "
                   f"- likely convention mismatch or data bug.")
            print(f"\n{'!' * 60}\n{msg}\n{'!' * 60}")
            report_lines.append(f"\n**{msg}**")

    report_lines.append(f"\nTotal training wall-clock: {time.time() - t_start:.1f}s")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")

    if stop_triggered:
        raise SystemExit("Parameter gate FAILED - see STOP AND REPORT message(s) above. "
                          "Models were still saved for inspection; do NOT proceed to Step 3.")

    print("\nParameter gate PASSED for both leads - safe to proceed to Step 3.")
    return all_metrics


if __name__ == "__main__":
    main()
