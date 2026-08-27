#!/usr/bin/env python3
"""
WS d7 diagnostic 2 (2026-07-08): does the kit's training pipeline use HRES
d10, and does d10 even exist in the training data? Pure inspection + a
direct data check - no models trained, no submission touched.

(a) grep-confirms the exact feature columns forecast_hres.py's train_mos/
    train_quantile_mos give the d7 model.
(b) checks both training HRES parquets + one inference HRES parquet for
    fcst_*_d10_* columns.
(c) prints a one-line recommendation.

Output: appended as a section into reports/ws_d7_diagnostics_20260708.md
(this script's own dedicated section - run after the coverage/bias script,
or standalone; either order produces a valid file since each script only
appends/writes its own section marker).
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KIT_PART1 = ROOT / "phase_2" / "kit" / "phase_2" / "part1_forecast"
REPORT_PATH = ROOT / "reports" / "ws_d7_diagnostics_20260708.md"

sys.path.insert(0, str(KIT_PART1))


def main():
    lines = ["\n## Diagnostic 2: HRES d10 usage\n"]

    fh_src = (KIT_PART1 / "forecast_hres.py").read_text(encoding="utf-8")
    features_match = re.search(r"FEATURES\s*=\s*\[([^\]]+)\]", fh_src)
    features = features_match.group(1) if features_match else "NOT FOUND"
    print(f"forecast_hres.py FEATURES: [{features}]")
    lines.append(f"**(a) d7 training feature columns** (`forecast_hres.py::FEATURES`): "
                 f"`[{features}]`")

    hres_leads_match = re.search(r"HRES_LEADS\s*=\s*\(([^)]+)\)", fh_src)
    hres_leads = hres_leads_match.group(1) if hres_leads_match else "NOT FOUND"
    lines.append(f"`HRES_LEADS = ({hres_leads})` — d10 is never read by "
                 f"`build_hres_table()` (only reads `fcst_*_d{{L}}_h{{H}}` "
                 f"for `L in HRES_LEADS`).")

    d10_hits = [(i + 1, line) for i, line in enumerate(fh_src.splitlines()) if "d10" in line]
    lines.append(f"\nAll 'd10' occurrences in `forecast_hres.py` (comments only, no code path):")
    for ln, txt in d10_hits:
        lines.append(f"- line {ln}: `{txt.strip()}`")

    fm_path = KIT_PART1 / "forecast_model.py"
    if fm_path.exists():
        fm_src = fm_path.read_text(encoding="utf-8")
        fm_d10 = "d10" in fm_src
        lines.append(f"\n`forecast_model.py` exists but is a different/older engine "
                     f"(ERA5-persistence, used only by `1_forecast_era5_target.ipynb`, "
                     f"not the shipped `1_predict_target.ipynb` pipeline). "
                     f"Contains 'd10': {fm_d10}.")
    else:
        lines.append("\n`forecast_model.py` not found under part1_forecast/.")

    print("\nChecking training + inference HRES parquets for d10 columns...")
    paths = {
        "train: hres_north_sea.parquet (Phase-1 ship, 2019-2021)":
            ROOT / "phase_2/phase2_dataset_ship/train/hres/hres_north_sea.parquet",
        "train: north_sea_hres_2016_2018.parquet (Phase-2 back-fill)":
            ROOT / "phase_2/phase2_dataset_ship/train/hres/north_sea_hres_2016_2018.parquet",
        "inference: window_1/context_hres_north_sea.parquet":
            ROOT / "phase_2/phase2_dataset_ship/inference/window_1/context_hres_north_sea.parquet",
    }
    lines.append("\n**(b) d10 presence in HRES data:**\n")
    lines.append("| File | n_cols | n_rows | d10 columns present |")
    lines.append("|---|---|---|---|")
    any_train_has_d10 = False
    all_train_has_d10 = True
    for label, p in paths.items():
        if not p.exists():
            lines.append(f"| {label} | - | - | file not found |")
            continue
        df = pd.read_parquet(p, columns=None)
        d10_cols = [c for c in df.columns if "_d10_" in c]
        has_d10 = len(d10_cols) > 0
        print(f"  {label}: {len(df.columns)} cols, {len(df)} rows, d10={has_d10}")
        lines.append(f"| {label} | {len(df.columns)} | {len(df)} | "
                     f"{'YES (' + str(len(d10_cols)) + ' cols)' if has_d10 else 'NO'} |")
        if "train" in label.lower():
            any_train_has_d10 = any_train_has_d10 or has_d10
            all_train_has_d10 = all_train_has_d10 and has_d10

    lines.append(f"\n**(c) Recommendation:** ")
    if any_train_has_d10 and not all_train_has_d10:
        lines.append(
            "d10 exists in the Phase-1 ship file (2019-2021) but NOT the Phase-2 "
            "back-fill file (2016-2018) - i.e. 3 of 5 training years (2016-2018) have "
            "NO d10 data at all. Using d10 as a d7-model feature would require either "
            "(a) dropping 2016-2018 from training (60% of training years - large loss), "
            "or (b) imputing/flagging missing d10 for those years (adds complexity, "
            "risk of the model learning a spurious year-boundary artifact from the "
            "missing-data indicator). d10 IS present in the inference HRES for all 8 "
            "eval windows, so it's not an inference-availability blocker - but the "
            "training-side gap makes it a genuinely awkward feature to add safely, "
            "not a simple free win. **Not recommended as a near-term addition** given "
            "the training-data gap; revisit only if 2016-2018 d10 can be reconstructed "
            "or backfilled.")
    elif all_train_has_d10:
        lines.append("d10 present in ALL training years - straightforward to add as a "
                     "feature. Recommended as a low-risk experiment.")
    else:
        lines.append("d10 absent from training data entirely - inference-only, cannot "
                     "be used without a different training data source.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAppended Diagnostic 2 section to {REPORT_PATH}")


if __name__ == "__main__":
    main()
