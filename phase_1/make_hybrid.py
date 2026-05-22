"""Build hybrid submission: speed from Heavy, direction from Light."""
from pathlib import Path
import zipfile
import pandas as pd

HERE = Path(__file__).parent
heavy_path = HERE / "predictions_heavy.csv"
light_path = HERE / "predictions_light.csv"
out_csv = HERE / "predictions_hybrid.csv"
out_zip = HERE / "submission_hybrid.zip"

KEY_COLS = ["type", "window", "region", "latitude", "longitude",
            "station", "horizon", "hour", "level"]
SPEED_COLS = ["q05", "q50", "q95"]
DIR_COLS = ["dir_05", "dir_50", "dir_95"]

print("Loading source CSVs...")
heavy = pd.read_csv(heavy_path)
light = pd.read_csv(light_path)
print(f"  heavy: {len(heavy):,} rows, cols={list(heavy.columns)}")
print(f"  light: {len(light):,} rows, cols={list(light.columns)}")

# --- Verify alignment ---
assert len(heavy) == len(light), f"Row count mismatch: heavy={len(heavy)} light={len(light)}"

missing_heavy = [c for c in KEY_COLS + SPEED_COLS if c not in heavy.columns]
missing_light = [c for c in KEY_COLS + DIR_COLS if c not in light.columns]
assert not missing_heavy, f"Heavy missing columns: {missing_heavy}"
assert not missing_light, f"Light missing columns: {missing_light}"

print("\nVerifying key columns match row-by-row...")
for col in KEY_COLS:
    if not heavy[col].equals(light[col]):
        # Try tolerant numeric compare for floats (lat/lon)
        try:
            diff_mask = ~(heavy[col].astype(float).round(6)
                          == light[col].astype(float).round(6))
            n_diff = int(diff_mask.sum())
        except Exception:
            n_diff = int((heavy[col] != light[col]).sum())
        raise AssertionError(f"Column '{col}' differs in {n_diff:,} rows")
    print(f"  ok: {col}")

# --- Build hybrid (preserve heavy's column order) ---
hybrid = heavy.copy()
for c in DIR_COLS:
    hybrid[c] = light[c].values
# Ensure final column order matches the original heavy file
hybrid = hybrid[list(heavy.columns)]

print(f"\nHybrid: {len(hybrid):,} rows, cols={list(hybrid.columns)}")

# --- Sanity checks: speed comes from heavy, dir comes from light ---
for c in SPEED_COLS:
    assert hybrid[c].equals(heavy[c]), f"speed col {c} should match heavy"
for c in DIR_COLS:
    assert (hybrid[c].values == light[c].values).all(), f"dir col {c} should match light"
print("Sanity checks passed: speed=heavy, direction=light.")

# --- Show samples from each source for the same 3 rows ---
print("\n--- Sample rows ---")
sample_idx = [0, len(hybrid) // 2, len(hybrid) - 1]
show_cols = KEY_COLS + SPEED_COLS + DIR_COLS
print("\nHEAVY (source of speed):")
print(heavy.loc[sample_idx, show_cols].to_string(index=False))
print("\nLIGHT (source of direction):")
print(light.loc[sample_idx, show_cols].to_string(index=False))
print("\nHYBRID:")
print(hybrid.loc[sample_idx, show_cols].to_string(index=False))

# --- Save CSV and ZIP ---
print(f"\nWriting {out_csv.name}...")
hybrid.to_csv(out_csv, index=False)

print(f"Writing {out_zip.name} (containing predictions.csv)...")
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(out_csv, arcname="predictions.csv")

# Confirm zip contents
with zipfile.ZipFile(out_zip) as zf:
    print(f"  zip contents: {zf.namelist()}")

print(f"\nDone. Saved: {out_csv.name} ({len(hybrid):,} rows)")
print(f"Ready to upload to Codabench: {out_zip.name}")
