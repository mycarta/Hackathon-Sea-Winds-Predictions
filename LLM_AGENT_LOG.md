# LLM Agent Log

Additions only. Every substantial LLM-agent run gets one dated entry
(contract v2.0, Logging / Dim 5). Backfill of pre-2026-07-06 sessions is
parked, not in scope for any single entry below.

## 2026-07-06 — Task 1 baseline: kit pull (CSV format fix), rerun, package submission

**Agent:** CC (Claude Code), role = builder/executor per contract §2.
**Task scope:** `phase_2/kit/` and `part1_forecast` outputs only. No
`phase_1/`, no report material, no checkpoints, no contract docs, no
model/calibration code, no parameter changes.

**Trigger:** Organizers changed the Task 1 submission format from parquet to
CSV (forum, capinventlab, 2026-07-05): "the forecast notebook now writes
submission.csv (plus a ready-to-upload submission.zip)." Our clone was
pinned at `090c930` (2026-07-02), stale.

**Kit pull:**
- `090c930684742d5a63620bc83a22b0d610de7ab8` -> `16071bf0a77f4db987b0b063ce2763d3222289ee`
  (`git fetch && git pull origin phase_2`, fast-forward).
- Diff inspected before trusting (`git diff --stat 090c930..16071bf`): 2 files
  changed, 12 insertions(+), 4 deletions(-) — confined to
  `build_forecast_submission.py::write_submission` (CSV branch: round
  lat/lon 2dp, q05/q50/q95/dir_05/dir_50/dir_95 3dp, then `to_csv`) and
  `2_downscale_to_target.ipynb`'s output cell (writes `submission.csv` +
  zips to `submission.zip`, markdown updated to say upload the zip). No
  changes to `forecast_hres.py`, `forecast_model.py`, `forecast_pipeline.py`,
  `downscaling.py`, `forecast_features.py`, `terrain_features.py`. Did not
  stop/escalate — diff confirmed cosmetic to submission writing only.
- Provenance re-pinned: `data/KIT_PROVENANCE.md` "Update — 2026-07-06"
  section appended (additions only).

**Rerun (env: `swnd` conda env, `jupyter nbconvert --execute --inplace`):**

| Notebook | Wall-clock |
|---|---|
| `1_predict_target.ipynb` | ~120.7 s |
| `2_downscale_to_target.ipynb` | ~296.7 s |

Both exit 0. Phase-1 HRES (`hres_north_sea.parquet`, 2019-2021) was already
in place at `phase_2/phase2_dataset_ship/train/hres/` for this run — unlike
the 2026-07-02 baseline run (memory: `phase2-kit-baseline-run`), which was
degenerate for d1/d7 due to missing HRES.

**Validation gate** (`scripts/validate_task1_submission.py`, new named
script, committed): full report at
`reports/task1_submission_validation_20260706.md`.

- Rows: 4,196,640 exact. NaN: 0. window/horizon/hour/type/region/level: all
  PASS. q05<=q50<=q95: PASS. speeds >= 0: PASS. `submission.zip` present and
  contains `submission.csv`: PASS.
- **FAIL** — directions in [0, 360): 24 / 12,589,920 direction cells
  (0.00019%) are exactly `360.000` instead of `0.000`. Root cause traced to
  `build_forecast_submission.py:60-62` (`dir_0X % 360`, unchanged by this
  kit update) combined with the *new* 3dp rounding step: a pre-round value
  like `359.9997` survives `% 360` unchanged (already in `[0,360)`) but
  `round(x, 3)` pushes it to `360.000`. This is a kit-code cosmetic artifact
  introduced by the organizers' new rounding step, not a forecasting or
  calibration defect on our side. Not patched — out of scope (kit code, "no
  parameter changes anywhere" per task prompt). Flagged for Matteo/organizer
  awareness before Codabench upload, since some scorers reject values >= 360
  strictly.
- Calibration metrics (circular convention matching
  `forecast_pipeline.py:38`):

  | Horizon | speed width (q95-q05) | dir half-width (deg) | q50 median (m/s) |
  |---|---|---|---|
  | d1 | 7.330 | 42.000 | 8.491 |
  | d7 | 23.217 | 40.700 | 5.786 |
  | d14 | 36.517 | 14.800 | 4.159 |

  d1 and d7 median q50 both land in [5,15] m/s — no STOP condition hit. d14
  median (~4.16 m/s) matches the known kit vector-mean climatology artifact
  (CLAUDE.md); reported as-is, not fixed.

**Compute disclosure:** 2 notebook executions (`nbconvert --execute`),
single machine, `swnd` conda env, ~7 min total wall-clock. No GPU. No
external API calls.

**Output:** `phase_2/kit/phase_2/part1_forecast/submission.zip` (contains
`submission.csv`, 4,196,640 rows). Not submitted to Codabench — Matteo
submits.

**Report sync:** no report-worthy checkpoint entries this session requiring
`[REPORT: pending]` sync (task was code-side kit rerun, not a report-facing
result).

**Follow-up (same day):** the 360.000 direction-wrap artifact above was fixed
via new named script `scripts/fix_dir_360_wrap.py` (360.000 -> 0.000,
circularly equivalent; exactly 24 values replaced across
dir_05/dir_50/dir_95 as expected), `submission.csv`/`submission.zip`
regenerated, gate re-run with the strict `< 360` check and now PASSes
clean (see addendum in `reports/task1_submission_validation_20260706.md`).

## 2026-07-06 — Task 1 d14 climatological fix (speed + direction)

**Agent:** CC, role = builder/executor per contract §2. **Scope:** new
scripts under `scripts/`, `submission.csv` d14 rows only. d1/d7 rows,
model training/calibration, Phase 1 files, report material, contract docs:
untouched.

**Problem:** the kit's d14 forecast (`forecast_hres.py::_climatology`)
averages u/v components (vector-mean) before taking magnitude, which lets
direction variability cancel the wind vectors — d14 speed q50 was ≈4.16 m/s
vs a real AROME median ≈9.5 m/s. Fix: replace d14 speed AND direction with
scalar/circular quantiles computed directly from training AROME native-grid
data (2016-2020), stratified by 3-month season × hour, per footprint cell.

**Step 1 parameter gate (approved in full, this session):** bin-size check
(`scripts/d14_climatology_bin_sizes.py`) found 1,826/1,827 training days
present (`2018-08-07` missing, matches known gap), 0 NaN in a 10-file
spot-check. Season stratification (DJF/MAM/JJA/SON × 4 hours): 452-460
days/bin. Monthly (12 × 4 hours): 142-155 days/bin. Recommended season
(≈3x more samples/bin for stable q05/q95 tails, matches Phase 1 precedent)
— approved.

**Step 2-3 build:**
- `scripts/compute_d14_climatology.py` — for each of 43,715 footprint cells
  × 4 seasons × 4 hours (16 bins), computes scalar speed q05/q50/q95 from
  `sqrt(u125m²+v125m²)` per timestep (never averaged u/v), and circular
  direction dir_50 (circular mean of per-timestep unit-vector directions,
  using the kit's own `degrees(atan2(-u,-v))%360` convention for byte-
  compatible convention with d1/d7) plus dir_05/dir_95 (endpoints of the
  shortest arc containing 90% of directions, vectorized sliding-window over
  the 360°-duplicated sorted array). Deterministic, no stochastic step, no
  seed needed. Output: `scripts/artifacts/d14_climatology_season.parquet`
  (699,440 rows; gitignored — 21.6MB, regenerable from the committed script
  + raw AROME data, not committed to keep the repo lean).
- `scripts/apply_d14_climatology.py` — replaces d14 rows in
  `submission.csv`. Window→season resolved from the kit's own
  `splits.py::eval_windows(2021)` `score_days['d14']` (not hardcoded):
  window 0→DJF, 1-2→MAM, 3-5→JJA, 6-7→SON. Row↔cell correspondence is
  **positional** (block-index), not a (lat,lon) value join — a value-join
  was tried first and rejected: 11/43,715 cells' rounded latitude sits
  exactly on a `.xx5` boundary where the kit's float32 round-then-cast and
  an independent float64 round pick opposite round-half-to-even directions,
  breaking dict/merge joins on rounded floats (real, harmless artifact).
  Verified instead that `submission.csv` is 96 perfectly contiguous
  43,715-row blocks (8 windows × 3 horizons × 4 hours) whose row order
  matches `footprint.footprint_mask()`'s `np.where` order exactly — used
  that directly. Checkpointed the pre-fix `submission.csv` to
  `scripts/artifacts/submission_pre_d14fix_backup.csv` before any write
  (build discipline #5; also gitignored, 331MB, exceeds GitHub's 100MB
  limit).

**Bug found and fixed mid-task (own code, not kit code):** rounding
`dir_05/dir_50/dir_95` to 3dp reintroduced the same 360.000-boundary
artifact as the earlier kit-CSV-writer bug, this time from the new
climatology values (10 cells this time, vs 24 previously). Fixed at the
source in `apply_d14_climatology.py` (wrap ≥360.0 → 0.0 after rounding) and
patched the then-current file with the existing `scripts/fix_dir_360_wrap.py`
(generalized — no longer asserts an exact expected count, since the count
legitimately varies by which rounding step produced it).

**Second bug found and fixed (in `scripts/validate_task1_submission.py`,
this session's own tooling):** the calibration-metric "direction
half-width" used `circular_dist(dir_95,dir_05)/2` (shortest distance
between the two endpoints), which silently reports the *complementary*
arc whenever the true constructed interval exceeds 180° — verified this
happens for **100% of d7 rows** and **100% of d14 bins**. Fixed to the
correct forward-arc formula `((dir_95-dir_05)%360)/2`, matching how the
intervals are actually constructed (dir_05→dir_95, forward mod 360) for
both the kit's MOS leads and this fix's climatology.

**Before/after (corrected metrics, from the pre-fix backup vs current
`submission.csv`):**

| Metric | Before | After | Target |
|---|---|---|---|
| d14 speed q50 median | 4.159 m/s | 9.215 m/s | 8-11 m/s |
| d14 speed conformal width (mean q95-q05) | 36.517 m/s | 14.652 m/s | narrower/saner |
| d14 direction half-width (mean, correct arc metric) | 165.200° | 139.383° | — |
| d7 direction half-width (unchanged, for reference) | 139.300° | 139.300° | — |

d14 q50 median lands in the 8-11 m/s target. d14 half-width (139.383°) is
technically above d7's (139.300°) by 0.083° — monotonicity criterion met,
but see the flagged finding below on why this margin is not meaningful.

**Flagged finding, out of scope, not fixed:** d1 and d7 direction intervals
in the current submission are **flat global constants**, not per-cell/
per-time calibrated — every single d1 row is `dir_50 ± 42.000°` (std=0
across 1,398,880 rows) and every single d7 row is `dir_50 ± 139.300°`
(std=0), i.e. d7's "90% interval" spans 278.6° of the 360° compass for
every row, everywhere, regardless of cell or window. This is a pre-existing
kit property (`forecast_pipeline.py` calibration), not introduced or
touched by this fix. The task prompt's stated baseline ("currently d14 80°
< d7 98.9°") was measured with the same buggy `circular_dist` metric this
session found and fixed in `validate_task1_submission.py` — confirmed with
Matteo (2026-07-06) that these reference numbers came from the same run,
not a different kit/submission state, so the metric bug invalidates them;
not a separate discrepancy to chase. Left as a flagged item for
Matteo/Opus: worth understanding why d7's calibrated offset is a flat
constant (not data-driven per cell) and unusually wide, independent of this
task.

**Validation gate:** `scripts/validate_task1_submission.py` re-run after
all fixes above — 4,196,640 rows, 0 NaN, all schema/range/monotonicity
checks PASS (including strict `<360` direction range), `submission.zip`
present and contains the csv. Independently verified from disk (not just
the apply script's internal check): d1/d7 rows byte-identical between
`scripts/artifacts/submission_pre_d14fix_backup.csv` and the current
`submission.csv` (2,797,760 rows compared, exact match); d14 rows confirmed
changed.

**Compute disclosure:** `compute_d14_climatology.py` processed 1,826 daily
AROME NetCDF files (2016-2020) at the 43,715-point footprint, single
machine, `swnd` conda env, season-by-season (peak RAM ≈1GB). No GPU. No
external API calls.

**Output:** `phase_2/kit/phase_2/part1_forecast/submission.zip` (updated,
contains fixed `submission.csv`). Not submitted to Codabench — Matteo
submits.

**Report sync:** no `phase2_report_material.md` entries touched this
session (that file was mid-update by Matteo outside this session — left
untouched throughout, per its Opus-stewarded status).

**Comms-watch:** forum thread noted — tsgoodie submission failure (ID 828970, parquet scorer crash) → organizer fix (capinventlab, 2026-07-05 23:59): submission format changed parquet → CSV, kit updated accordingly.
**Forum bug report posted:** "Bug: CSV writer rounds directions to 360.000 (spec requires < 360)" — 24/12.59M direction cells round up past wrap due to new 3dp step; reported to organizers, 2026-07-06.

## 2026-07-07 — Task 1 direction residual model (d1+d7)

**Agent:** CC, role = builder/executor per contract §2. **Scope:** new
scripts under `scripts/`, `submission.csv` direction columns (dir_05/
dir_50/dir_95) for horizon 1 and 7 rows only. Speed columns (all
horizons), d14 rows, Phase 1 files, report material, contract docs:
untouched.

**Problem:** d1/d7 direction intervals in the kit's own submission were
flat, non-data-driven constants (flagged in the 2026-07-06 d14 entry) —
every d1 row `dir_50 ± 42.000°`, every d7 row `dir_50 ± 139.300°`
(std=0 across 1,398,880 rows each), because `forecast_pipeline.py`'s
`fit_forecast()`/`calibrate_intervals()` compute the interval half-width as
a single `np.nanpercentile` pooled over *all* cells/hours/dates per lead,
collapsing all spatial/temporal variation. Fix: the Phase 1 "unwrapped
residual" transfer — predict `residual = ((truth_dir - hres_dir + 180) %
360) - 180` (a linear, non-circular target) with LightGBM quantile
regression, instead of predicting direction directly.

**Architecture decision (parameter gate, approved in full 2026-07-07):**
HRES only exists on the coarse 45×57 (0.25°, 2,565-cell) grid — no native
fine-resolution HRES. Two designs were possible: (A) train on the coarse
grid directly (matches the kit's own `train_mos()` architecture, ~18.7M
rows, but requires reusing the kit's u/v terrain downscaler to reach the
fine submission grid — coupling into speed-adjacent code out of scope for
this task), or (B) train on the fine 43,715-cell footprint grid with HRES
features broadcast (nearest-neighbor) from each cell's containing coarse
cell, self-contained and isolated from the speed/downscaler pipeline.
**(B) was approved**, with a seeded 5,000-cell subsample (full 43,715 ×
~1,826 days × 4 hours ≈ 319M rows is intractable) and an explicit follow-up
check: after sampling, report how many of the ~159 "allowed cells" domain
are represented, switching to stratified-by-coarse-cell sampling if gaps
appear.

**Bonus finding (resolves an open CLAUDE.md question):** "distinct 0.25°
reanalysis cells over `footprint_points.parquet` == 159?" — **No.** The
full 43,715-point footprint touches **322** distinct coarse cells, not
159. The 159-cell domain must come from additional siting-specific
filtering (e.g. depth/box constraints) beyond simple footprint↔coarse-cell
membership — flagged for whoever picks up that question next, not chased
further here (out of scope).

**Cell subsample** (`scripts/dir_residual_select_cells.py` → gitignored
`scripts/artifacts/dir_residual_cells.parquet`, seed=42): plain uniform
sampling of 5,000/43,715 cells covered only 307/322 distinct coarse cells
(15 gaps) → switched to stratified-by-coarse-cell sampling (same seed) →
full 322/322 coverage achieved.

**Step 0 convention check** (in `scripts/dir_residual_build_training_data.py`):
confirmed AROME truth (`target_loader.py`'s `(270-atan2(v,u))%360`) and
HRES-derived direction (`forecast_hres.py`'s `atan2(-u,-v)%360`, used
throughout `forecast_pipeline.py` for the submission's own dir_50) are
algebraically identical (max diff 2.84e-14° on real data) — same
convention, no correction needed. HRES stores direction directly
(`fcst_dir_d{L}_h{H}` columns), not as u/v — round-trip through
`_uv_from_speed_dir` → `atan2(-u,-v)%360` reproduces the original value
exactly (diff 2.84e-14°).

**Training data build**: one row per (cell, valid_date, hour, lead) for
lead in (1,7), joined via `splits.py::eval_windows`-style issue-date →
valid-date arithmetic (D = V − lead), HRES broadcast from the containing
coarse cell, AROME truth from native `u125m/v125m` at the fine cell.
Train/val split from `splits.py::train_val_dates(seed=42)` (day-level, no
leakage across hours of the same day), applied to the valid (truth) date.
1,825 AROME files read (1,826 calendar days in TRAIN_YEARS, 0 missing —
the 2018-08-07 gap is excluded upstream by `target_loader.list_dates()`).
**72,880,000 total rows** (lead=1: 36,500,000 [29.2M train / 7.3M val];
lead=7: 36,380,000 [29.08M train / 7.3M val]) — larger than initially
estimated (5,000 cells was approved; the ~1,826-day population was not
separately reduced). Output: gitignored
`scripts/artifacts/dir_residual_training.parquet` (9.03 GB in memory).

**Bugs found and fixed during the build (own code, not kit code):** (1)
`.iterrows()` on a mixed-dtype pandas row upcasts int columns to float64,
breaking `u_all[row['y'], row['x']]` fancy indexing — fixed with explicit
`int(...)` casts. (2) A string-key join mismatch: `f"{x:.3f}"` formatting
vs pandas' `.astype(str)` produce different strings for the same float
(`"51.000"` vs `"51.0"`), breaking the coarse-cell HRES lookup — fixed by
using the same `.map(lambda x: f"{x:.3f}")` formatting on both sides of
every join in both `dir_residual_build_training_data.py` and
`dir_residual_apply.py`.

**Step 2 - model training** (`scripts/dir_residual_train_models.py`):
features = `hres_dir_sin, hres_dir_cos, hres_speed, hour_sin, hour_cos,
season_code (DJF=0/MAM=1/JJA=2/SON=3), month_sin, month_cos, lat, lon`
(HRES + static only, no AROME-derived features, since AROME is never
available at inference). Hyperparameters exactly as specified in the task
prompt: `n_estimators=300, max_depth=7, learning_rate=0.05, num_leaves=63,
subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1`. 6
LightGBM quantile models (q05/q50/q95 × lead 1/7), ~130-151s each, ~14 min
total wall-clock. Saved to `models/dir_residual_d{1,7}_q{05,50,95}.lgb`
(LightGBM native Booster format).

**Parameter gate (PASSED both leads):**

| Lead | Residual mean/std | Residual p05/p95 | Val MAE (q50) | Coverage (q05-q95) | STOP threshold |
|---|---|---|---|---|---|
| d1 | 2.312 / 27.087 | -31.030 / 38.390 | 15.159° | 88.6% | 30° (not triggered) |
| d7 | 1.448 / 72.365 | -133.126 / 133.575 | 55.866° | 86.6% | 60° (not triggered, but close) |

d7's MAE (55.866°) sits closer to its 60° stop threshold than d1's does to
30° — flagged as a genuine finding (large lead-7 residual variance), not a
bug: matches the Phase 1 notebook's own documented experience that d+7
residual variance is large and coverage matters more than width at long
lead.

**Step 3/4 - apply** (`scripts/dir_residual_apply.py`): predicted on all
43,715 footprint cells (full grid, not just the 5,000-cell training
subsample) across the 8 2021 eval windows × 2 leads × 4 hours, using the
kit's pre-sliced per-window inference HRES files
(`phase_2/phase2_dataset_ship/inference/window_{1..8}/context_hres_north_sea.parquet`).
Residual quantile predictions sorted per-row (enforce non-crossing) before
reconstruction; verified every reconstructed arc spans < 360°. Replaced
`submission.csv` positionally (same validated 96-contiguous-block approach
as the d14 fix). Checkpointed pre-fix state to gitignored
`scripts/artifacts/submission_pre_dir_residual_fix_backup.csv` before
writing (build discipline #5) - this is a separate checkpoint from the
existing `submission_pre_d14fix_backup.csv`, which is left untouched.

**Before/after direction half-width** (forward-arc metric,
`scripts/validate_task1_submission.py`):

| Horizon | Before | After | Target (prompt) | Met? |
|---|---|---|---|---|
| d1 | 42.000° | 37.646° | 15-35° | Meaningfully improved, but above the optimistic target range |
| d7 | 139.300° | 116.487° | 50-120° | Yes |
| d14 | 139.383° | 139.383° (unchanged) | n/a | n/a |

Monotonicity restored and verified: d1 (37.646°) < d7 (116.487°) < d14
(139.383°).

**Validation gate:** `scripts/validate_task1_submission.py` — 4,196,640
rows, 0 NaN, all schema/range checks PASS including strict `<360`
direction range (no new rounding-boundary artifacts this time — wrap
handled at the source in `dir_residual_apply.py`, same pattern as the d14
fix). Independently verified from disk (numeric tolerance, not `.equals()`
— CSV round-trip introduces ~1e-14 deg float64 repr noise even for
untouched values): speed columns byte-identical for every horizon (max
diff 0), d14 direction columns unchanged (max diff 2.84e-14°, floating-
point noise), d1/d7 direction columns changed substantially (mean abs diff
58.538°, confirming the fix took effect), all non-value columns
(window/horizon/hour/lat/lon/type/region/level) byte-identical throughout.

**Compute disclosure:** training-data build ~10 min (1,825 AROME file
reads + HRES joins, `swnd` env, single machine); model training ~14 min (6
LightGBM models, `n_jobs=-1`, peak RAM ~22GB); apply step ~1 min (8
windows × 2 leads × 4 hours × 43,715-cell prediction). No GPU. No external
API calls.

**Output:** `phase_2/kit/phase_2/part1_forecast/submission.zip` (updated).
Not submitted to Codabench — Matteo submits.

**Report sync:** no `phase2_report_material_MERGED_*.md` entries touched
this session (Opus-stewarded, out of scope).

## 2026-07-08 — Task 1 WS d7 diagnostics (3 checks, no model changes)

**Scope:** diagnostic scripts + a report only, per explicit task prompt.
No submission, model, d14-fix, direction-fix, Phase 1, or report-material
files touched. Deliverable: `reports/ws_d7_diagnostics_20260708.md`.

**Diagnostic 1 — per-horizon speed coverage / alpha-sweep.** Script
`scripts/ws_d7_diagnostic_coverage_bias.py`. Evaluated on
`splits.train_val_dates(seed=42)` val split (365 dates) using the shipped
speed pipeline (`forecast_pipeline.fit_forecast` + `downscaling.
train_downscaler` on 2020[::5] + `forecast_pipeline.calibrate_intervals`),
refit in-memory only — no models saved or modified. Swept
`alpha ∈ [0.8, 1.1]` applied as `q95_new = q50 + α×(q95-q50)`,
`q05_new = q50 - α×(q50-q05)`; reported coverage, mean width, and
estimated Winkler (alpha_level=0.10) per horizon × alpha. Transparency
check: overlap between the 365 val dates and the model's own
training/calib dates is small (mos-train=65, conformal-calib=9,
downscaler-train=21) — not excluded, noted for the record. d14 numbers
carry an unavoidable self-leakage caveat (climatology pooled over all of
2016-2020 per week-of-year × hour, so TRAIN_YEARS val dates leak into
their own bin) — d1/d7 are the reliable comparison.

Winkler-optimal alpha: **d1=0.95** (Winkler 10.400 vs 10.402 at current
α=1.0, essentially flat — current calibration already near-optimal),
**d7=0.8** (Winkler 31.442 vs 35.287 at α=1.0 — current intervals are
over-wide relative to Winkler-optimal), **d14=0.8** (same direction,
caveat above applies). Full table in the report.

**Diagnostic 2 — HRES d10 usage.** Script
`scripts/ws_d7_diagnostic_hres_d10.py`. Confirmed by source inspection
(`forecast_hres.py::FEATURES`, `HRES_LEADS=(1,7)`) that d10 is never read
by the shipped pipeline — only comment mentions, no code path. Direct
parquet check: d10 columns exist in the Phase-1 ship HRES file
(2019-2021) and all inference-window HRES files, but are **absent from
the Phase-2 back-fill file (2016-2018)** — i.e. 3 of 5 training years have
no d10 data at all. **Recommendation: not worth pursuing near-term** —
adding d10 as a d7 feature would mean either dropping 60% of training
years or imputing/flagging missing d10 for them (risk of a spurious
year-boundary artifact), for a lead that already has good HRES coverage
at d1/d7. Not an inference-side blocker, purely a training-data gap.

**Diagnostic 3 — per-cell d7 speed bias.** Same script as Diagnostic 1
(shares the expensive per-issue-date forecast+downscale loop). Computed
`bias = q50_pred - arome_truth` per footprint cell (all 43,715 cells,
pooled over ~884 issue dates / ~1,460 samples per cell) and per season.
Result: **mean bias -3.3273 m/s, std across cells 0.1816 m/s**, seasonal
spread 0.93 m/s (JJA worst at -3.89, DJF best at -2.95) — a systematic,
near-uniform d7 underprediction.

This result looked suspiciously tight for 43,715 spatially/
environmentally diverse cells, so it was independently spot-checked before
trusting it: a standalone single-date script (issue date 2017-06-01,
lead=7, hour=0, valid 2017-06-08) compared `pred_q50` directly against
AROME truth speed for that one forecast. Single-date bias: mean=-3.591,
std=4.040 across cells — much noisier per-date (expected, day-to-day
weather varies), but same sign and comparable magnitude to the pooled
result. The pooled std (0.18) is consistent with sampling convergence: at
~1,460 samples/cell and ~4 m/s per-sample std, expected standard error of
a per-cell mean is ~4/√1460≈0.10 m/s, close to the observed 0.18 —
meaning the tight cross-cell spread is mostly averaging convergence on top
of a genuinely near-uniform bias, not a computational artifact. **Verdict:
real finding, not a diagnostic-script bug.** Bug found and fixed during
development: initial script called `fh.fit_forecast(train)` —
`fit_forecast` lives in `forecast_pipeline.py` (`P`), not `forecast_hres.py`
(`fh`); fixed before the reported run.

**Recommendation:** per-cell or global d7 speed bias correction is worth
pursuing (mean -3.33 m/s is large relative to typical d7 wind speeds).

**Compute disclosure:** Diagnostics 1+3 combined run ~65 min wall-clock
(884 issue dates × forecast+downscale+truth-compare, `swnd` env, single
machine, no GPU). Diagnostic 2 <1 min (pure inspection + 3 parquet reads).
Single-date sanity-check script ~few min. No external API calls.

**Output:** `reports/ws_d7_diagnostics_20260708.md` (all 3 diagnostics).
No submission, model, or other files changed — numbers only, for a future
recalibration/bias-correction decision.

**Report sync:** no `phase2_report_material_MERGED_*.md` entries touched
this session (Opus-stewarded, out of scope). These are diagnostic
findings, not yet written into the report — mark `[REPORT: pending]` in
the checkpoint.

## 2026-07-08 — Task 1 d7 speed bias correction + alpha recalibration

**Scope:** new scripts in `scripts/`, modified `submission.csv`/`.zip` d7
speed columns only. d1/d14 speed, ALL direction columns (every horizon),
models, Phase 1 files, and report material untouched. Driven directly by
`reports/ws_d7_diagnostics_20260708.md`'s two findings: a systematic
-3.3 m/s d7 speed underprediction bias, and an over-wide interval
(pre-correction Winkler-optimal alpha=0.8, coverage 90.5%).

**Step 1 — per-cell x per-season bias table**
(`scripts/ws_d7_bias_correction_build_table.py`). Per-cell x per-season
(not just per-cell) because the diagnostics report's seasonal spread
(0.934 m/s) exceeded the task prompt's 0.5 m/s threshold for that choice.
Refit the shipped speed pipeline in-memory (same methodology as the
diagnostics script), evaluated on the `train_val_dates(seed=42)` holdout,
365 d7-relevant issue dates (~63.8M cell-samples). Overall global mean
bias reproduced exactly: -3.3273 m/s (matches Diagnostic 3), confirming
internal consistency between the two scripts. Per-season bias: DJF
-2.954, MAM -3.271, JJA -3.888, SON -3.173.

Shrinkage: `bias_shrunk(cell,season) = w*bias_raw(cell,season) +
(1-w)*season_global_mean_bias`, `w = n_blocks_season/(n_blocks_season+30)`.
**Finding not anticipated by the task prompt:** because truth for each
holdout sample is a single full-grid AROME snapshot, every footprint cell
gets an identical sample count within a season (no per-cell sparsity like
the direction-residual fix's subsampled training set) — so w is constant
across cells within a season (0.920-0.928 across the 4 seasons), not
cell-varying as the prompt's formula implied. The per-cell bias *values*
still vary spatially and are preserved; only the shrinkage weight is
season-uniform. Documented in the script docstring and report rather than
silently assumed. With 344-388 holdout dates per season, w is close to 1
throughout — shrinkage has limited effect here but is applied as
specified.

**Step 3 — alpha re-sweep on bias-corrected holdout predictions.** Initial
sweep (the task prompt's suggested grid [0.7, 0.75, 0.8, 0.85, 0.9]) found
Winkler monotonically decreasing to the grid's edge (alpha=0.7, coverage
78.4%) with no interior minimum — a bigger post-correction shift than the
prompt anticipated ("optimal alpha may shift slightly"). **Escalated to
Matteo rather than applying the edge value** (per build discipline —
results outside expected range, a real parameter decision). Extended
sweep to [0.1..0.9] in 0.05 steps (rerun of the same ~30 min pipeline,
this time also checkpointing the raw stacked holdout arrays to
`scripts/artifacts/ws_d7_bias_correction_stacked.npz` before the resweep,
so any further alpha exploration is a cheap array reload, not a pipeline
rerun) — found a genuine interior minimum at alpha=0.65 (Winkler=30.626,
coverage 76.1%), inside a shallow bowl [0.60, 0.75] where Winkler stays
within 0.9% of the minimum. **Matteo approved alpha=0.70** (Winkler=30.676,
coverage 78.4%) — a conservative pick above the auto-selected minimum, to
hedge against holdout overfitting. Recorded via a small named script
(`scripts/ws_d7_set_chosen_alpha.py`) that overrides
`ws_d7_bias_correction_params.json`'s `chosen_alpha` from the
already-computed resweep table (no recomputation), documenting the
auto-selected value, Matteo's override, and the reasoning.

**Steps 2/4 — apply + validate**
(`scripts/ws_d7_apply_bias_correction.py`). Checkpointed pre-fix
`submission.csv` to `scripts/artifacts/submission_pre_d7_bias_correction_backup.csv`
before writing (build discipline #5). Applied bias shift then alpha=0.70
tightening to horizon==7 q05/q50/q95 only, using positional block indexing
(96 contiguous 43,715-row blocks, same validated approach as the d14 and
direction-residual fixes) — only the 32 horizon==7 blocks (8 windows x 4
hours) touched. Season per window resolved from `splits.eval_windows(2021)`
(each window's single d7 valid date determines its season).

**Before/after (full submission, all windows/hours; also cross-checked via
`scripts/validate_task1_submission.py`):**

| Metric | Before | After |
|---|---|---|
| d7 q50 median | 5.786 m/s | 9.353 m/s |
| d7 mean interval width (q95-q05) | 23.217 m/s | 16.252 m/s (30.0% narrower) |
| d7 holdout Winkler (estimated) | 35.287 | 30.676 (13.1% better) |

d7 q50 median lands in the task's 8-11 m/s target range. Verified
byte-identical: d1/d14 speed columns, ALL direction columns (every
horizon), all non-value columns on d7 rows (window/hour/lat/lon/etc.).
`scripts/validate_task1_submission.py` gate: 4,196,640 rows, 0 NaN, all
schema/range checks PASS, q05<=q50<=q95 and speeds>=0 verified throughout.
`reports/task1_submission_validation_20260706.md` addendum 4 appended
(full history restored — the validator regenerates that file from scratch
each run, losing prior addenda, so addenda 1-3 were recovered from commit
`e74c542` before appending addendum 4).

**Compute disclosure:** Step 1+3 build-table script run twice (~30 min
each, `swnd` env, single machine, no GPU, peak RAM ~2.3GB): once with the
prompt's suggested alpha grid, once more with the extended grid after the
edge-case escalation. Apply script (Steps 2/4) <1 min. No new models
trained — pipeline refit in-memory only, matching the diagnostics
methodology. No external API calls.

**Output:** `phase_2/kit/phase_2/part1_forecast/submission.zip` (updated).
Not submitted to Codabench — Matteo submits.

**Report sync:** no `phase2_report_material_MERGED_*.md` entries touched
this session (Opus-stewarded, out of scope). Report-worthy: the d7 bias
correction result and the shrinkage/alpha-sweep methodology, both
currently `[REPORT: §dir-residual-ws-d7-fix-20260708 (shrunk bias table + α recalibration), 20260708]`.

## 2026-07-08 — Task 1 d7 alpha revision (0.70 -> 0.90)

**Scope:** re-run of the same fix, revised parameter only. Same
constraints as above (d1/d14 speed, all direction columns, models, Phase
1, report material untouched).

Matteo revised the approved d7 alpha from 0.70 to 0.90 (more
conservative, further from the Winkler-minimum at 0.65, trading Winkler
for more coverage margin) with an explicit safety instruction: re-run
from the pre-bias-correction backup, not the already-corrected
submission.csv, to avoid double-applying the bias shift.

**Bug-shaped risk caught before it happened:** the original
`scripts/ws_d7_apply_bias_correction.py` read its input from the live
`submission.csv` (already bias-corrected from the alpha=0.70 run). A
naive re-run would have subtracted `bias_shrunk` a second time — doubling
the ~+3.3 m/s center shift. **Fixed the script to always read from
`BACKUP_PATH`** (the pristine, pre-correction checkpoint, created once
and never overwritten) rather than `CSV_PATH`, making it safe to re-run
with any future alpha value: every run now applies bias-correction +
alpha in one pass from the same untouched baseline.

`scripts/ws_d7_set_chosen_alpha.py` generalized similarly: preserves the
true auto-selected (Winkler-minimum) baseline across repeated overrides
(reads `auto_selected_alpha` if already set, rather than re-deriving it
from a previous override's `chosen_alpha`, which would have drifted the
reference point on a second revision) and appends a dated revision note
to the report rather than silently replacing the prior selection.

**Result after re-running with alpha=0.90:** d7 q50 median unchanged at
9.353 m/s (confirms the bias was applied exactly once, not double
-applied). d7 mean interval width: 23.217 -> 20.896 m/s (10.0% narrower,
vs 30.0% narrower at the previous alpha=0.70). `scripts/validate_task1_submission.py`
gate re-PASSed clean. `reports/task1_submission_validation_20260706.md`
addendum 5 appended (addendum 4 marked superseded, not deleted).

**Output:** `phase_2/kit/phase_2/part1_forecast/submission.zip` (updated,
alpha=0.90 version). Not submitted to Codabench — Matteo submits.

**Report sync:** no `phase2_report_material_MERGED_*.md` entries touched
this session (Opus-stewarded, out of scope).

## 2026-07-08 — Task 1 d7 alpha revision (0.90 -> 1.0, bias correction only)

**Scope:** same fix, revised parameter only. Same constraints as the two
prior entries (d1/d14 speed, all direction columns, models, Phase 1,
report material untouched).

Matteo revised the approved d7 alpha to 1.0 — keep the bias-corrected
center, apply **no interval tightening** (the widest, most conservative
option, deliberately trading Winkler optimality for maximum coverage
margin). alpha=1.0 was outside the originally-swept [0.1..0.9] grid.

**New script:** `scripts/ws_d7_alpha_sweep_from_checkpoint.py` computes
additional alpha values directly from the raw stacked holdout arrays
checkpointed by the Step 1 script
(`scripts/artifacts/ws_d7_bias_correction_stacked.npz`) — seconds, not
the ~30 min forecast+downscale pipeline. This is exactly what that
checkpoint was saved for. Result: alpha=1.0 gives coverage=88.6%,
width=29.928 m/s (identical to the pre-correction width, as expected —
alpha=1.0 is a pure re-centering with no scaling), Winkler=34.139 (only
3.3% better than the 35.287 pre-correction baseline, vs 13.2% better at
the true Winkler-minimum alpha=0.65). Inserted into
`ws_d7_bias_correction_params.json`'s `alpha_resweep` table (sorted, no
duplicates) rather than overwriting it, preserving the full sweep record.

`scripts/ws_d7_set_chosen_alpha.py`'s decision-history docstring extended
with this third revision; `scripts/ws_d7_apply_bias_correction.py` needed
no further changes — its existing pristine-backup-sourcing design (fixed
in the previous entry) already made this re-run safe by construction.

**Result:** d7 q50 median unchanged at 9.353 m/s (confirms the bias was
applied exactly once). d7 mean interval width: 23.217 -> 23.217 m/s (0.0%
narrower — unchanged, as expected for alpha=1.0).
`scripts/validate_task1_submission.py` gate re-PASSed clean.
`reports/task1_submission_validation_20260706.md` addendum 6 appended
(addenda 4-5 marked superseded, not deleted).

**Output:** `phase_2/kit/phase_2/part1_forecast/submission.zip` (updated,
alpha=1.0 version — bias-corrected center, original interval width). Not
submitted to Codabench — Matteo submits.

**Compute disclosure:** alpha extension <5s (vectorized numpy over the
cached ~592MB checkpoint, no pipeline rerun). Apply script <1 min. No
new models trained. No external API calls.

**Report sync:** no `phase2_report_material_MERGED_*.md` entries touched
this session (Opus-stewarded, out of scope).

## 2026-07-08 — Task 1 d1 speed bias diagnostic

**Scope:** diagnostic only, no submission/model changes.
`scripts/ws_d1_bias_diagnostic.py` — same per-cell bias methodology as
Diagnostic 3 (`scripts/ws_d7_diagnostic_coverage_bias.py`), applied to d1
instead of d7, same `splits.train_val_dates(seed=42)` holdout, same
shipped-pipeline reproduction (refit in-memory, no models saved).

**Result:** mean bias -0.2446 m/s, std 0.2791 m/s, p05/p95
[-0.7073, 0.1899] m/s. Per-season: DJF -0.0918, MAM -0.4517, JJA -0.5101,
SON +0.0647 (sign flips across seasons, seasonal spread 0.5748 m/s).
Below the |mean|>0.3/std>0.5 correctable-structure threshold used for d7.
**Recommendation: d1 MOS is already well-centered; bias correction
unlikely to help** — a sharp contrast with d7's -3.33 m/s bias, expected
given HRES skill degrades substantially between +1d and +7d lead time.

**Compute disclosure:** ~30 min wall-clock (365 d1-relevant issue dates x
4 hours, `swnd` env, single machine, no GPU). No new models trained. No
external API calls.

**Output:** `reports/ws_d1_bias_diagnostic_20260708.md`. No submission
changes.

**Report sync:** no `phase2_report_material_MERGED_*.md` entries touched
this session (Opus-stewarded, out of scope).

## 2026-07-08 — Leaderboard results: d7 alpha selection outcome

**Leaderboard selection:** submission 835026 (2026-07-08).

**Three alpha test submissions** (real Codabench leaderboard scores, not
holdout estimates): alpha=0.70 -> 28.37, alpha=0.90 -> **27.14 (selected)**,
alpha=1.0 -> 27.65. Note this ranking differs from the holdout-estimated
Winkler ordering earlier in this log (where alpha=0.65-0.70 scored best
and 0.90/1.0 scored progressively worse) - the leaderboard-actual optimum
landed on the more conservative alpha=0.90, not the holdout minimum.
Consistent with Matteo's original instinct to pick alpha above the
holdout-estimated minimum "to hedge against holdout overfitting."

**d1 bias diagnostic:** negligible (mean -0.245 m/s, std 0.279 m/s), no
correction applied - matches the recommendation in the entry above.

Full leaderboard tracking in `phase_2/hunt_record_task1.md` (Matteo-
maintained).

## 2026-07-08 — Task 1 interval symmetry diagnostic

**Scope:** diagnostic only, no submission/model changes.
`scripts/ws_symmetry_diagnostic.py`.

**Scope correction before starting:** the task prompt as given described
CatBoost quantile regression, per-region alpha scaling (NS 1.005/ECS
0.995), and other details that don't match Task 1's actual pipeline.
Verified via an Explore agent before writing any code: those claims all
trace to Phase 1 (`phase_2/kit/phase_1/utils.py`'s CatBoost option, and
narrative text in `phase2_report_material_MERGED_20260708.md`, a Phase-1
retrospective) — Task 1 uses LightGBM throughout and has a single
`north_sea` region. Matteo confirmed: adapt the methodology to the real
Task 1 pipeline (raw LightGBM quantile output -> conformal_adjust ->
spd_infl calibration -> d7 bias correction -> d14 climatological
replacement), drop the Phase-1-specific references.

**Step 1 (code audit):** read `forecast_hres.py`, `forecast_pipeline.py`,
`compute_d14_climatology.py` in full. Found `conformal_adjust` (CQR,
Romano/Patterson/Candès 2019) applies the SAME additive scalar `Q_L` to
both `q05` (subtract) and `q95` (add) — a genuine symmetrizing operation
(proof: `(upper+Q_L)/(lower+Q_L) -> 1` as `Q_L` grows). By contrast,
`spd_infl`'s `k`-scaling and our own alpha-tightening apply the same
multiplicative factor to each side's own distance from center — this
cancels in the ratio, ratio-invariant *in the unclamped regime*. Our own
bias correction is a pure translation, ratio-neutral by construction.
d14's climatological replacement uses genuine empirical quantiles
(`np.quantile` on raw samples), no post-hoc adjustment.

**Step 2 (asymmetry measurement, holdout, 646 issue dates):** confirmed
conformal_adjust's symmetrizing effect empirically at the coarse stage
(d1: 1.1096->1.0888, d7: 1.0335->1.0272 - both move toward 1, small).
**But the fine-grid d7 ratio (3.4143) massively exceeds what conformal
alone explains.**

**Bug found and fixed during development:** initial script crashed
(`ValueError: operands could not be broadcast together with shapes
(479,433) (43715,)`) — `downscale()`/`interp_coarse_to_target()` return
the full target grid, but truth/bias arrays are already flattened to the
43,715 footprint cells; fixed by indexing `[ys,xs]` on all downscaled
arrays before comparison, consistent with every other script's convention.

**Critical finding (caught before handoff, changed the report's headline
conclusion):** the analytical proof said spd_infl's k-scaling should be
ratio-invariant, but the empirical d7 result showed a huge, unexplained
drift (0.8905). Investigated rather than accepting the contradiction:
checked `scripts/artifacts/submission_pre_d7_bias_correction_backup.csv`
directly — **100% of d7 rows (1,398,880/1,398,880) had `q05` exactly
clamped to 0.000 m/s** before our bias-correction fix. `spd_infl=4.747`
is large enough that the `max(0, spd50-k*(spd50-q05))` floor saturates
for essentially every cell (pre-k lower distance up to `0.7*spd50`,
`4.747*0.7=3.32` >> 1). This floor-clamp - not conformal_adjust - is what
dominates d7's 3.41 fine-grid asymmetry ratio. Checked the live
`submission.csv`: our bias correction (shifting the whole interval up by
~+3.3 m/s) incidentally rescues q05 off the zero floor (0.0% clamped now,
mean q05=3.361 m/s) - a side effect, not by design. Corrected the report
(originally attributed the fine-grid asymmetry mainly to conformal
symmetrization) before treating it as final - the report's Step 1 table,
symmetrization-flag section, and Recommendation were all revised to
reflect the floor-clamp finding as primary and reprioritize accordingly.

**Step 3 (Winkler cost, same holdout, spd_infl held fixed):** symmetric
conformal vs. an asymmetric split-conformal alternative (separate
one-sided margins, each targeting alpha/2 one-sided miss rate) - d1: cost
small (0.27% worse with symmetric). d7: cost material (6.64% worse with
symmetric) - but this isolates conformal's own contribution only, holding
the already bias-inflated spd_infl fixed; not a full re-optimization.

**Step 4 (ordering check):** two orderings exist at different layers.
Our own d7 fix (bias-correct then alpha-tighten) is correctly ordered
throughout its git history, never reversed. But the kit's own `spd_infl`
calibration predates any bias correction and was derived on a
bias-uncorrected center - so part of why `k=4.747` is so large is that it
was absorbing the -3.3 m/s bias via width, not just genuine dispersion.
Not fixed here (diagnostic only).

**Recommendation, priority order:** (1) recalibrate `spd_infl` on
bias-corrected residuals (re-run `calibrate_intervals`'s binary search) -
should shrink `k` and reduce/eliminate the floor-clamp saturation,
higher-leverage than #2; (2) asymmetric split-conformal for d7 if #1
doesn't fully resolve it; (3) not worth pursuing for d1 alone.

**Compute disclosure:** ~29 min wall-clock (646 issue dates, d1+d7
combined, 3 interval variants each, `swnd` env, single machine, no GPU,
peak RAM ~1.9GB). Plus two <5s direct CSV checks (pre-fix backup and live
submission q05 clamp rates). No new models trained. No external API calls.

**Output:** `reports/ws_interval_symmetry_diagnostic_20260708.md`. No
submission changes.

**Report sync:** no `phase2_report_material_MERGED_*.md` entries touched
this session (Opus-stewarded, out of scope). Report-worthy: the floor-
clamp finding and its connection to the bias-then-calibrate ordering
principle, currently `[REPORT: §pipeline-ordering-and-interval-geometry-20260709 (floor-clamp + ordering), 20260709]`.

## 2026-07-08 — Task 1 spd_infl recalibration + asymmetric conformal (Fix #1 + Fix #2)

**Scope:** diagnostic + holdout validation only, no submission/model
changes. `scripts/ws_d7_spdinfl_recal_and_asym_conformal.py`. Two
sequential fixes per the task prompt (#1 must complete before #2, since
#1 changes the residuals #2 operates on) - implemented as one combined
script for efficiency (a single ~31 min per-date pass captures every
PRE-WIDENING quantity both fixes need, avoiding a second expensive
pipeline rerun), with the logical dependency honored via computation
order: Fix #1's sweep completes and picks `k_new` before Fix #2's
Winkler comparison runs, which uses `k_new` throughout.

**Fix #1 — spd_infl recalibration on bias-corrected residuals.** Method:
"center -> decouple -> calibrate" done properly this time, inside the
calibration itself rather than as a downstream patch - captured the
ratio-based PRE-WIDENING (spd_infl=1.0-equivalent) fine-grid interval
once per holdout sample, re-centered it on the bias-corrected center
(`spd50_bc = spd50_raw - bias_shrunk(cell,season)`), then swept `k`
(coarse grid `[1.0..4.747]` per the prompt, then a finer grid auto-
generated around the coarse optimum) purely via vectorized numpy over
the cached arrays - no further pipeline reruns needed even for the fine
sweep.

**Result: d7's Winkler-optimal spd_infl drops from the stale 4.747 to
1.7** - a 28.1% Winkler improvement on the same holdout (35.436 ->
25.485), and the q05 floor-clamp rate collapses from 94.7% (at old k)
down to 0.2% (at new k=1.7), confirming the interval symmetry
diagnostic's hypothesis directly: most of the old inflation was
compensating for the -3.3 m/s center bias via width, not genuine
dispersion. d1's recalibrated k (1.05 vs old 1.067) is a negligible
change, as expected (no bias correction adopted for d1). d14 confirmed
not applicable (climatological replacement bypasses spd_infl entirely).

**Coverage-margin flag:** the Winkler-optimal k=1.7 gives only 82.0%
coverage (below the ~88% sanity anchor), in a flat bowl (k=1.9: only
+0.75% Winkler for +3.4pp coverage) - same shape as the earlier d7 alpha
decision, where the real leaderboard validated a MORE conservative pick
over the holdout minimum. Flagged for Matteo's decision, not resolved
unilaterally - a k in the 1.8-2.0 range is a plausible conservative
alternative.

**Fix #2 — asymmetric split-conformal (d7 only), using Fix #1's new k.**
Implemented Matteo's exact formula (distinct from the symmetric-score
construction used in the interval symmetry diagnostic): separate
`score_upper = truth-q95` / `score_lower = q05-truth` computed only on
violated calibration rows, each corrected by its own `(1-alpha)`
order-statistic quantile (same finite-sample rank adjustment as the
shipped `conformal_adjust`, since the prompt didn't fully specify this
detail - documented as a reasonable completion, not assumed silently).

**Result: the asymmetric-conformal advantage REVERSES once Fix #1 is
applied.** The original symmetry diagnostic found asymmetric beating
symmetric by 6.64% (at the stale, heavily floor-clamped k=4.747). At the
recalibrated k=1.7, symmetric conformal (Winkler 25.485) now BEATS
asymmetric (26.699) by ~4.5% - strong evidence the earlier apparent
advantage was a floor-clamp interaction artifact, not a genuine
asymmetric-conformal benefit. **Recommendation: do not adopt asymmetric
conformal; keep the shipped symmetric `conformal_adjust` as-is.**

**Compute disclosure:** ~31 min wall-clock (646 issue dates, d1+d7
combined, symmetric+asymmetric coarse variants for d7, `swnd` env, single
machine, no GPU). Checkpointed raw stacked holdout arrays to
`scripts/artifacts/ws_spdinfl_recal_stacked.npz` (gitignored) for any
further re-sweeps without a pipeline rerun. No new models trained. No
external API calls.

**Output:** `reports/ws_spdinfl_recal_asym_conformal_20260708.md`. No
submission changes - new spd_infl value reported, not applied.

**Report sync:** no `phase2_report_material_MERGED_*.md` entries touched
this session (Opus-stewarded, out of scope). Report-worthy: the spd_infl
recalibration result (28% Winkler improvement, largest single lever
found in the WS d7 diagnostic sequence) and the Fix #2 reversal finding,
currently `[REPORT: §pipeline-ordering-and-interval-geometry-20260709 (spd_infl −28.1% + Fix #2 reversal), 20260709]`.

## 2026-07-08 — Task 1 full submission rebuild: d7 spd_infl=1.9 (single change vs. 835026)

**Scope:** this DOES modify submission.csv/zip (a build step, not a
diagnostic) — but NOT submitted to Codabench, and explicitly NOT to be
selected on the leaderboard: "Do NOT select on the leaderboard — we
review the board result at the WS d7 sketch gate tomorrow before
selecting." Matteo submits and reports the submission ID + per-dimension
Winkler scores back once scored — not yet available, to be logged in a
follow-up entry.

**Method:** `scripts/ws_d7_build_submission_spdinfl19.py`. Unlike every
prior d7 fix this session (which patched the existing submission.csv),
this is a full rebuild from the kit's own base pipeline, since spd_infl
is applied INSIDE `downscale_window`/`_speed_interval`, before any of our
patches, and the pre-widening intermediate values aren't recoverable from
the already-widened CSV (verified: q05 was clamped to exactly 0.000 for
100% of pre-fix d7 rows at the old k=4.747, an irreversible clamp —
confirmed in the interval symmetry diagnostic).

Fidelity to "everything else unchanged," verified before writing any
code (Explore-agent audit of `1_predict_target.ipynb`/
`2_downscale_to_target.ipynb`): `mos`/`qmos`/`adj`/`offs` and the 8
windows' coarse fields are loaded VERBATIM from
`part1_forecast/cache/coarse_forecasts.pkl` — the exact same cache
underlying submission 835026, not refit. Only `spd_infl[7]` is
overridden (4.747 -> 1.9), injected between `calibrate_intervals()` and
`downscale_window()`, exactly where the kit's own notebook would apply
it. The downscaler `dwn` is retrained fresh — confirmed via source read
to be a pre-existing, unavoidable kit limitation (never cached by the
notebooks, no `random_state` anywhere in `train_mos`/`train_quantile_mos`/
`train_downscaler`), not something this rebuild introduces. d14
climatological replacement, the direction residual model, and the d7
bias-shrunk table are all independent of spd_infl by construction (q50/
center comes from the deterministic MOS model, untouched by k-scaling;
the bias table corrects q50 and shifts q05/q95 by the same amount
regardless of k) — reused as-is against the fresh base. d7 alpha stays
at 0.90 (matching 835026, deliberately NOT re-optimized for the new k —
an isolated single-variable test, not a joint re-optimization).

**Bug found and fixed during development:** `sub` is built fresh
in-memory via `bfs.assemble(blocks)`, whose columns are explicitly
`float32` (per `field_to_rows`'s cast) — unlike every prior apply script,
which operates on a CSV-round-tripped (`pd.read_csv`) DataFrame, default
`float64`. Assigning float64-computed values into a float32 column hit
pandas' `LossySetitemError` (a newer, stricter dtype-casting behavior).
Fixed by explicitly casting all computed value/direction arrays to
`float32` before assignment (`round3`/`round3_wrap360` helpers).

**Result:** d7 mean interval width: 23.217 -> 11.609 m/s (50.0% narrower
than the original submission, narrower than any prior alpha-only
revision since the base width itself shrank this time, not just the
tightening factor). d1 (q50 median 8.491 m/s) and d14 (9.215 m/s)
medians identical to every prior run this session, confirming they were
untouched. Checkpointed the pre-rebuild submission.csv (the alpha=1.0
experimental state, itself never submitted) to
`scripts/artifacts/submission_pre_spdinfl19_rebuild_backup.csv` before
overwriting. `scripts/validate_task1_submission.py` gate re-PASSed clean.
`reports/task1_submission_validation_20260706.md` addendum 7 appended
(addenda 4-6 marked superseded/local-only, not deleted — addendum 5's
lineage is specifically noted as matching submission 835026 for future
reference).

**Compute disclosure:** ~3 min wall-clock (only the 8 real 2021 eval
windows processed, not a holdout sweep — `swnd` env, single machine, no
GPU). `mos`/`qmos`/`adj` reused from cache, not refit; `dwn` retrained
fresh (~74 days). No new models trained. No external API calls.

**Output:** `phase_2/kit/phase_2/part1_forecast/submission.zip` (updated,
spd_infl=1.9 version). NOT submitted to Codabench, NOT to be selected on
the leaderboard until reviewed at tomorrow's WS d7 sketch gate. Once
Matteo reports the submission ID and per-dimension Winkler scores, they
will be logged in a follow-up entry per his explicit instruction.

**Report sync:** no `phase2_report_material_MERGED_*.md` entries touched
this session (Opus-stewarded, out of scope).

## 2026-07-09 — WS d7 spd_infl=1.9 leaderboard result: NOT selected

**Follow-up to the 2026-07-08 rebuild entry above** (commit `81a6c6f`),
per Matteo's instruction to log the result once scored. Source:
`checkpoint_update_20260709.md` and `phase_2/hunt_record_task1.md`
(both Matteo-authored, committed this entry alongside).

**Board result: 27.65, vs submission 835026's 27.14 — 0.51 Winkler
points WORSE.** Despite a large holdout improvement (-28.1% Winkler,
floor-clamp rate 94.7%->0.2%, interval width 23.2->11.6 m/s), the
recalibrated spd_infl=1.9 underperformed on the real 2021 eval year.
**Submission NOT selected** — 835026 (spd_infl=4.747, alpha=0.90)
remains the current best and the leaderboard-standing entry.

**Diagnosis (Matteo's, recorded in the checkpoint):** same transfer-
failure shape as the earlier alpha finding, but in reverse — a single-
year (2020) holdout optimum doesn't reliably survive to the hidden eval
year. Here, the stale k=4.747 was apparently accidentally over-covering
in a way that happened to help on the 2021 board, so "fixing" the
floor-clamp pathology (a real, verified defect) still cost Winkler
points in practice.

**Conclusion: WS d7 post-processing is now confirmed exhausted.**
spd_infl recalibration was the largest single lever found across the
whole WS d7 diagnostic sequence (this session's biggest holdout Winkler
win) and it moved the board score the wrong way. The remaining +5.78
gap to Printemps (21.36) is model-architecture territory, not further
post-processing tuning - matches `hunt_record_task1.md`'s "Blocked by:
architecture sketch gate (tomorrow, Fri 10th)."

**No further action from CC on WS d7 post-processing** pending the
sketch gate's architecture decision.

## 2026-07-10 — Task 2 opening: scorer pin, bathymetry map, cell ranking

Per the CC prompt "Task 2 opening: pin the scorer, map the bathymetry,
rank the cells" (Fable Task 2 session). Data-side groundwork only, no
design decision taken, report at
`reports/task2_scorer_bathymetry_ranking_20260710.md`. Compute disclosure:
5 named scripts (`scripts/task2_export_turbine_curve.py`,
`task2_depth_sampler.py`, `task2_bathymetry_map.py`,
`task2_rank_cells_by_resource.py`, `task2_shallow_box_feasibility.py`),
deterministic (no unseeded stochastic step), longest run 119 s (1826
AROME daily files, swnd env).

**Key findings requiring escalation before any Task 2 design decision:**

1. **The organiser scorer (`internal/simulator/score_phase2.py`, named in
   `part2_siting/SUBMISSION.md`) has never existed in this repo** —
   verified against the full git history of the nested kit repo. Every
   Task A finding below is pinned from the kit's *mirror* implementation
   (`wind_farm_simulator.py`/`cost_model.py`/`optimization.py`), not the
   real scorer; we have no way to verify equivalence.
2. Superposition model resolves to PyWake's default `LinearSum` (not
   `SquaredSum`) — confirmed against installed `py_wake==2.6.17` source,
   since the kit never passes `superpositionModel` explicitly.
3. **No real IEA 22 MW power/Ct curve is shipped anywhere** (repo or
   Zenodo Phase-2 dataset) — `turbines_catalog.py`'s
   `_TURBINES_DIR` doesn't exist, so the kit silently falls back to a
   synthetic cubic-ramp curve for all four catalog turbines
   (`real_curve=False` confirmed for all). `data/iea22mw_power_ct.csv`
   is that synthetic fallback, not a real curve.
3b. Derived per-sector TI feeds the `XRSite` but `BastankhahGaussianDeficit.k`
   is a fixed constant (not TI-dependent) and no `turbulenceModel` is
   set — the TI appears to have no effect on the wake calc. Flagged as
   an observation, not fully traced through `EngineeringWindFarmModel`.
4. **Depth threshold discrepancy, flagged not resolved**: module defaults
   (`bathymetry.py`, `cost_model.py`, EMODnet file attrs) all say 60 m;
   the actual siting notebooks (`3a`/`3b_farm_optimization_*.ipynb`) and
   `SUBMISSION.md`/CLAUDE.md all say 50 m and the notebooks explicitly
   override to 50 m at the call site. 50 m governs the reference
   pipeline; 60 m is dead-default.
5. **`bathymetry.py`'s hardcoded EMODnet search paths do not match the
   ship data layout** — `bathymetry.available()` returns `False` under
   `PHASE2_DATA_ROOT=.../phase2_dataset_ship` (empirically confirmed),
   so depth checks silently no-op (return eligible=True) rather than
   erroring. `scripts/task2_depth_sampler.py` reads the file directly
   and does not have this gap.
6. **No lat/lon↔local-metres projection exists anywhere in the kit** —
   confirmed by exhaustive grep. Per-turbine depth checking is
   structurally impossible with shipped code; `depth_at_xy` in the new
   depth-sampler module introduces our own (documented) equirectangular
   projection to make it possible.
7. Two documents referenced in the originating prompt
   (`task2_siting_constraints_and_decisions.md`,
   `scotian_shelf_layout_comparison.ipynb`) **do not exist anywhere in
   this repo's history** — likely cross-project reference error
   (Scotian Shelf / Middle Bank don't match this project's North-Sea/
   IEA-22MW domain). Flagged, not fabricated.
8. **159-cell count independently verified** (`zone.py::_grid`, exactly
   159), matching the CLAUDE.md open check.
9. Resource ranking (AROME 2016-2020, 125 m, mean ws³): the top-10
   wind-resource cells all sit in 57-78 m water (north of the domain,
   lat 54.25-55.25°N) — **none is close to all-shallow under the 50 m
   fixed-bottom threshold** (best case only 38% of a 15×15 km box under
   50 m). The kit's own baseline centre (53.5°N, 1.5°E, 26 m depth)
   ranks only 58th of 159 by wind resource. This is a real,
   quantified wind-vs-depth tradeoff for the siting decision — surfaced
   for Matteo/Opus, not resolved here.

**Report sync:** none of this is yet in
`phase2_report_material_MERGED_*.md` (Opus-stewarded, out of scope for
this entry) — `[REPORT: §10 + §11, 20260713-14 (scorer, depth, resource-vs-depth landed); projection/datum remains open via forum, tracked in checkpoints — not report material until answered]` should be applied to report-worthy
items (esp. finding 1, 3, 4, 5, 9) once reviewed.

## 2026-07-13 — Task 2 continued: shallow-box retest, projection, well-tie

Per the CC prompt "Task 2 continued: shallow-box retest, projection,
well-tie" (Opus session, Monday), predecessor to the 2026-07-10 entry
above. Report: `reports/task2_retest_welltie_20260713.md`. Compute
disclosure: 4 named scripts (`scripts/task2_shallow_box_retest.py`,
`task2_projection.py`, `task2_scorer_replica.py`,
`task2_welltie_kit_crosscheck.py`), longest run ~150s (AROME nearest-pixel
load, pywake env). Deterministic except the kit's own `seed=0` bootstrap
(reused verbatim, not re-seeded by us).

**Task D (shallow-box retest, two thresholds, top-20 shallow-centre
cells):** a materially better picture than the 2026-07-10 top-10 run —
filtering to centre depth ≤50 m FIRST, then ranking by wind resource,
finds candidates the pure-resource ranking never surfaced. All three
priority cells (63, 119, 72) are 100% shallow at both 50 m and 60 m.
13/20 fully all-shallow at 50 m, 19/20 at 60 m (vs 0/10 previously). Only
2/20 cells are materially sensitive to the still-open 50-vs-60 m
question. Cell 112 has a real (not artifact) 82 m channel cutting through
an otherwise-28.6 m-deep box — a genuine per-turbine placement hazard,
correctly captured in its exported mask.

**Task E (projection module):** extracted the equirectangular tangent-
plane projection (already used inline in `task2_depth_sampler.py` since
2026-07-10) into its own module, `scripts/task2_projection.py`, as single
source of truth; `depth_sampler` now imports it instead of duplicating.
No behavior change (verified: `depth_at(53.5,1.5)` unchanged).

**Task F (scorer replica + well-tie) — GATE FAILS, but root-caused, not a
physics bug:**

1. Before running, found the well-tie's own TARGET is compromised two
   ways: (a) the kit's reference pipeline (`3b_farm_optimization_refined.ipynb`,
   which `SUBMISSION.md` cites) runs on a **synthetic, seeded AR-sampled
   wind year**, not real AROME/reanalysis; (b) the only notebook that
   actually executes a plain-grid `simulate_year` and prints CF/wake
   (`2_simulator_intro.ipynb`) does so at **Dogger Bank (54.5N,2.0E)**,
   not the 53.5N/1.5E `SUBMISSION.md` attributes the 53.8%/5.9% figures
   to — **we could not find any code path in the current kit that
   actually produces those numbers.**
2. Caught and avoided a real methodological error: the prompt's literal
   "10m->170m" shear instruction, if applied to AROME's already-125m
   data, would have inflated wind speed by 1.366x instead of the correct
   1.034x (125m->170m) — a 32-point error, not a rounding difference.
   Used 125m->170m, documented why.
3. Ran 3 cases (real AROME @ baseline, kit's own synthetic wind @ Dogger
   through OUR independent replica, real AROME @ Dogger). All 3 fail
   tolerance (CF -4.2 to -7.9pp, wake +3.6 to +3.8pp low/high respectively).
4. **Decisive diagnostic**: ran the kit's OWN `wind_farm_simulator.simulate_year`
   directly (not our replica) on the same synthetic-Dogger wind as case 2
   — got CF=45.92%, wake=9.71%, AEP=4868 GWh, matching our independent
   replica EXACTLY. This proves the replica's pinned-physics construction
   is correct; the gap is not a bug in our code, it's that
   `SUBMISSION.md`'s target number doesn't reproduce from any code we
   have access to.
5. Hypothesis (unverified): wake loss is consistently ~9.5-9.7% across
   all 3 very different wind inputs, suggesting the 5.9% target may
   belong to an optimised (rotated/wider) layout, not the plain 7D grid.

**Gate status: FAIL. Per the prompt's own rule, no downstream layout
optimization or submission work has been started.** Escalating the
target-provenance question to Matteo/Opus — three options laid out in the
report (accept 45-50% CF as the real baseline order of magnitude, source
a corrected target, or test the optimised-layout hypothesis via `3b`).

**Report sync:** not yet in `phase2_report_material_MERGED_*.md` —
`[REPORT: §11.10 QA-and-provenance (gate fail + provenance escalation), 20260714]`, especially the gate-fail finding and its diagnosis.

## 2026-07-13 — Task 2 Step 3: layout optimization at cell 63

Per the CC prompt "Task 2 Step 3: Layout optimization at cell 63" (Opus
session, afternoon). Predecessor: the well-tie gate above was redefined
by Matteo and passed (per his 2026-07-13 checkpoint - replica matches
kit's own `simulate_year()` to displayed precision; `checkpoint_update_20260713.md`
itself was not found in the repo, proceeded on Matteo's direct statement
+ the prior session's own F.4 diagnosis, which already established the
same match). Report: `reports/task2_layout_optimization_20260713.md`.
Scripts: `scripts/task2_layout_search.py` (Stages 1-3, fast screen),
`scripts/task2_layout_validate.py` (full-series validation + robustness
+ winner selection). Deterministic throughout (Stage 4 hill-climb, the
only stage that would need a seed, was NOT run - no sketch-gate approval
obtained, per the prompt's explicit instruction).

**Headline: winning layout is Stage 3 (boundary-loaded: 40 perimeter +
15 interior turbines), CF=47.21%, AEP=5,003.8 GWh, wake=8.27% — not the
Stage 1 grid baseline the prompt frames as "everything after must beat."
It did, by ~3.6 CF points, in every one of the 5 training years
individually.**

**Two real bugs caught by sanity checks before they could bias the
result, both worth remembering for any future PyWake screening work:**

1. **A naive "energy-equivalent" per-sector representative speed
   (cbrt(mean(ws^3)), matching the Task C ranking philosophy) is WRONG
   for feeding a saturating turbine power curve.** First attempt gave
   Stage 1 a fast-screen CF of 80.8% - physically impossible. Root
   cause: cube-mean speed sits above the ~12 m/s rated threshold for the
   dominant sectors purely from variance (Jensen/power-mean inequality),
   so a single representative point gets pushed into rated/saturated
   output far more than the real distribution warrants. Fixed by binning
   each sector into 5 equal-probability quantile bins (arithmetic mean
   per bin) instead of one point.
2. **Even after that fix, the fast 16-sector rose screen completely
   REVERSED the real Stage 1-3 ranking.** Fast screen ranked: Stage1
   (49.1% CF, best) > Stage2 (48.3%) > Stage3 (46.8%, worst). Full
   14,608-step replica ranked: Stage3 (47.2% CF, BEST, wake 8.3%) >>
   Stage1/2 (~43.4-43.6% CF, wake ~15.3-15.9% - nearly DOUBLE the fast
   screen's wake estimate). Diagnosed cause (plausible, not fully
   proven): discretizing continuous wind direction into 16 fixed 22.5°
   sectors resonates strongly with a REGULAR GRID's sharp repeating
   row/column alignment angles (over/under-counting exact-alignment wake
   exposure depending on where those angles fall relative to the fixed
   sector centres), but an IRREGULAR perimeter layout has no such
   alignment structure to resonate with, so its fast estimate was much
   closer to real (0.4pp gap vs 5.7-6.7pp for the grids). **Lesson for
   any future coarse-rose screening: don't trust a binned-direction
   fast screen to rank a regular grid against an irregular layout -
   always validate the top candidates (not just the presumed winner) on
   the real series before finalizing.** This is exactly what the
   prompt's own staged design (cheap screen -> real validation) is for,
   and skipping the validation step here would have shipped the wrong
   layout.

Also caught (geometry-only, no ranking impact): a naive constant-arclength
perimeter walk gave 1010m corner spacing against a 1420m requirement
(triangle-inequality shortening across a 90-degree turn); a full 0.5x
brick-pattern stagger offset pushed an already-tight grid config out of
the box. Both fixed before the real run (see report for details).

Also: re-validated the top-10 Stage-1 fast-screen candidates on the full
series (not just the presumed #1) - found a small (0.2pp) real reordering
even within Stage 1's own pool, used the corrected pick. Confirms the
"always re-check, don't trust the fast screen's exact ranking" lesson
applies even among similar candidates, not just across dissimilar stage
types.

Per-year robustness: Stage 3 wins in EVERY year (44.7-48.6%), not just
pooled - no stability trade-off, it dominates on both mean and worst-case.
Sanity anchors both pass (CF 47.21% in [45,55]%; wake 8.27% well below
Stage 1's 15.3%+).

**Not done, per explicit instruction**: Stage 4 (hill-climb) - needs a
sketch-gate approval not sought/given this session. No scorer-physics
tuning (only the NEW fast-screen approximation was touched/fixed;
`task2_scorer_replica.py`'s well-tie-verified construction was imported
unchanged). No submission. No external optimizer.

**Report sync:** not yet in `phase2_report_material_MERGED_*.md` —
`[REPORT: §11.2 reversal table, 20260714 — landed in §11, not §10 as originally expected]`. Section 10 ("Siting Rationale") should now be
updatable with real PyWake numbers instead of its 2026-07-13-morning
placeholder (≈51% vs ≈54%) - real winning-layout CF is 47.21%.

## 2026-07-13 — Task 2 QA: verify Stage 3 layout (3 checks)

Per the CC prompt "Task 2 QA: verify Stage 3 layout (3 checks)", scoped
verification-only (no optimization, no modification, no submission) on
the Stage 3 boundary-loaded winner from the prior entry above. Report:
`reports/task2_qa_verification_20260713.md`. Scripts:
`scripts/task2_qa_geometry_check.py` (Check 1),
`scripts/task2_qa_kit_crosscheck.py` (Check 2, writes
`data/task2_qa_kit_crosscheck.csv`). Check 3 is a read-only code trace.
Deterministic throughout (no stochastic step).

**Compute disclosure:** Check 2 ran real AROME loading (14,608 steps,
~90s) + one full PyWake `simulate_year()` pass in the `pywake` conda env
(not `swnd` - `swnd` lacks `py_wake`, confirmed this session; noted for
future runs). Check 1/3 are pure Python/pandas, seconds.

**Check 1 (geometry): FAIL by strict tolerance, not a real design bug.**
55 turbines, box bounds, and duplicates all pass. Pairwise spacing: 6 of
1,485 pairs (turbines 40-54, the 15-point interior sub-grid) sit at
1419.993 m, exactly 7 mm under the 1420 m (5D) minimum, and by exactly
the same 7 mm on every violating pair. Root cause: `task2_layout_validate.py:205-206`
writes `layout_x_m`/`layout_y_m` via `.round(2)` - 2-decimal coordinate
rounding can shift a pairwise distance by up to ~1 cm at this geometry,
which is what ate the (presumably exact-or-near-exact) 1420 m construction
margin. Independently cross-validated: the kit's own `validate_layout()`
(called inside the Check 2 script) flags the identical 6 pairs. **Not
escalating as a design problem** - it's a serialization-precision issue,
not a siting-constraint violation - but flagging because the JSON file as
it stands today does not pass a literal `>=1420m` check, and any
downstream code that re-validates strictly (as the kit's own
`validate_layout` does) will reject it.

**Check 2 (kit cross-check): PASS.** Kit's own `wind_farm_simulator.simulate_year()`
(not `task2_scorer_replica.py`) on the Stage 3 layout, real AROME cell 63
(52.50N,3.00E), all training years: CF=47.2074% (reported 47.21%, diff
-0.0026pp), wake=8.2721% (reported 8.27%, diff +0.0021pp), AEP=5,003.79 GWh,
n_steps=14,608 - both metrics inside the +/-0.5pp tolerance by two orders
of magnitude. Confirms the prior entry's reported Stage 3 numbers are not
an artifact of the independent replica construction.

**Check 3 (code-path identity): PASS.** Traced `scripts/task2_layout_validate.py`:
Stage 1/2/3 share one wind-load call (`rep.load_arome_series`, line 106,
called once before the per-stage loop), one turbine object
(`rep.build_turbine()`, line 111, same Python object passed by reference
into every `run_case` call), and the same `run_case()` function
(`task2_scorer_replica.py:196-239`, no stage-conditional branching found
in `build_site`/`build_wake_model`/`run_case`). Cross-validated from data,
not just code: `mean_ws_hub`, `duration_hours`, and `n_steps` are
bit-identical across all three stages' pooled rows in
`data/task2_layout_robustness.csv`. The winner's reported CF/wake come
directly from the same per-stage-loop dict (`winner_result =
pooled_results[best_label]`, line 187) - no separately-constructed final
run.

**Flagged, out of this task's scope:** the standing `grep -n "REPORT:
pending" checkpoint_update_*.md` gate (CLAUDE.md "Report sync") returns
10 hits, all in `checkpoint_update_20260708.md` (5 days older than this
session). Pre-existing, unrelated to this QA task; fixing requires
writing into `phase2_report_material_MERGED_*.md` (Opus-stewarded report
prose), outside CC's remit without Opus review. Not actioned here -
surfaced to Matteo instead.

**Report sync:** this entry's findings are QA-verification results on
the already-`[REPORT: pending]` Stage 3 layout from the entry above, not
new report-worthy numbers in their own right - no separate
`[REPORT: pending]` tag added. The geometry-rounding finding (Check 1)
should be considered before `data/task2_layout_winner.json` is ever
copied into a scored submission file, whenever that step happens.

## 2026-07-13 — Task 2 QA continued: spread-grid + spacing fix

Per the CC prompt "Task 2 QA continued: spread-grid + spacing fix",
follow-up to the entry directly above. Two pieces: Check 4 (spread-grid
full-series validation, closing a gap the predecessor report left open)
and a fix for Check 1's spacing finding. Report appended to
`reports/task2_qa_verification_20260713.md`. Scripts:
`scripts/task2_qa_spread_grid_check.py` (Check 4, writes
`data/task2_qa_spread_grid.csv`); fix applied in
`scripts/task2_layout_search.py`. Deterministic throughout.

**Compute disclosure:** four real-AROME full-series runs this session
(~90s AROME load each + PyWake sim time), all in the `pywake` conda env:
Check 4 (two runs - first attempt found `uniform_7D_baseline` invalid at
the prescribed 10 deg, so the script was corrected and re-run), the
`task2_layout_search.py` fast-screen re-run (Stage 1-3), the
`task2_layout_validate.py` full-series re-run (Stage 1-3 + top-10 Stage-1
recheck, ~28 `run_case` calls), and the Check-2 kit-crosscheck re-run on
the fixed layout. Total wall time this session roughly 12-15 min.

**Check 4 (spread-grid): Stage 3 still wins, by 0.7-1.5 CF points over
every uniform-grid variant tested.** Ran the kit's stock 8x7 uniform-7D
grid (`uniform_7D_baseline`, `task2_layout_search.py`'s own Stage 1
candidate-pool control) plus two box-maximal uniform grids (8x7 and 7x8,
isotropic spacing bisected to the largest value that fits 55 turbines in
the box) through the full 14,608-step real AROME replica at cell 63 -
the same `rep.run_case()` code path as Stage 1-3. Results: baseline
(7.00D, 0deg) CF=46.49%/wake=9.67%; 8x7 max (6.57D, 10deg)
CF=45.75%/wake=11.09%; 7x8 max (6.78D, 10deg) CF=46.11%/wake=10.40% - all
below Stage 3's 47.21%/8.27%.

**Real geometric finding, not a bug:** `uniform_7D_baseline` is
`out_of_box` at the prescribed 10deg orientation (that angle was tuned
for Stage 1's elongated 6x10 winning shape, not this near-square 8x7
grid) - confirmed against the regenerated candidates CSV, valid ONLY at
0/90deg out of the full 0-170deg sweep. Ran it at 0deg instead (also the
kit's own `grid_layout` default). Same mechanism explains why the two
"max spacing" configs came out TIGHTER (6.57-6.78D) than the 7D baseline
despite being explicitly maximized: the bisection maximizes spacing AT
the fixed 10deg angle, which fits this grid shape worse than 0deg does -
an artifact of forcing all configs onto one shared orientation for a
clean comparison, not a bug in the bisection itself.

**Spacing fix: CONFIRMED.** Root cause was already correctly diagnosed in
the entry above (`.round(2)` JSON export eating a true ~1420.000m
construction margin - rotation preserves distance exactly, so the
shortfall was pure serialization noise). Fix: added
`INTERIOR_SPACING_M = MIN_SPACING_M + 1.0` (1421m) in
`task2_layout_search.py`, changed Stage 3's interior 3x5 sub-grid to
build at that spacing instead of the bare 1420m floor - a 1m margin that
survives 2-decimal rounding regardless of orientation. Perimeter spacing
(1500m) and the hard 5D=1420m validity floor used everywhere else are
unchanged. Cold-reran the full chain (no manual CSV/JSON patching):
`task2_layout_search.py` -> `task2_layout_validate.py` -> re-verify.
Result: min pairwise distance 1419.9930m -> **1420.9975m**, 0 violating
pairs (was 6); the kit's own `validate_layout()` now returns `True, []`
where it previously flagged the same 6 pairs. CF/wake reconfirmed via a
fresh kit-crosscheck run on the regenerated layout: 47.21%/8.27% (kit:
47.2086%/8.2698%), within 0.0014pp/0.0002pp of the pre-fix values - "1m
change on 1420m is noise" held exactly as expected.

**`data/task2_layout_winner.json` is now clean**: passes strict >=1420m
geometry (own check + kit's own validator agree) with CF/AEP/wake
unchanged from the values already cited in the predecessor report and
`[REPORT: pending]` in the entry above. No open geometry issues remain on
this layout.

**Report sync:** no new `[REPORT: pending]` tag - this entry finalizes
(fix + additional validation) the already-pending Stage 3 numbers from
the two entries above, doesn't introduce new report-worthy figures beyond
confirming the existing 47.21%/8.27%/5,003.9 GWh values are now backed by
a clean, strictly-valid layout file plus a full-series-validated grid
comparison.

## 2026-07-13 — Report sync: Task 2 layout findings now in phase2_report_material

Per Matteo (2026-07-13, same day): `phase_2/phase2_report_material_MERGED_20260713.md`
has been updated by Opus with this session's Task 2 layout results.
Confirmed by reading the file directly (not just taking the statement on
faith, per standing practice of verifying before recommending/relying on
a claim): Section 11 ("LAYOUT SELECTION AND THE DIRECTION-BIN ALIASING
FINDING", lines 1016-1117) covers the Stage 1-3 fast-screen reversal, the
direction-bin-aliasing mechanism (with the Sickler et al. 2023
citation), the per-year robustness table, AND explicitly cites this
session's Check 4 spread-grid result ("§11.4 ... validated the max-spread
uniform grid (~7D, the kit's own 8x7 control configuration)... wins by
0.7-1.5 CF points over all three uniform-grid variants tested"), tagged
`[REPORT: §fast-screen-reversal-and-aliasing, 20260713]`. A related
siting-rationale section is tagged `[REPORT: §siting-rationale-narrative,
20260713]`.

**Resolving, by this addition (not by editing the original entries -
LLM_AGENT_LOG.md is additions-only):**
- The "2026-07-13 - Task 2 Step 3: layout optimization at cell 63" entry
  above (this file, "Report sync" paragraph): its `[REPORT: pending]`
  tag on the Stage 3 winner numbers (CF=47.21%, wake=8.27%,
  AEP=5,003.8 GWh) is now SYNCED -> `phase2_report_material_MERGED_20260713.md`
  §11 (anchor `fast-screen-reversal-and-aliasing`, dated 20260713). Note:
  that entry referenced "Section 10 (Siting Rationale)" as the expected
  landing spot; the actual content landed in the newly-added Section 11
  instead (section numbering shifted as report content grew) - same
  underlying numbers, different section number than originally guessed.
- `checkpoint_update_20260713.md`'s two `[REPORT: pending]` tags (Check 2
  kit-cross-check confirmation, Check 1 spacing finding): the Check 2
  finding's underlying number (CF/wake matching the kit's own code) is
  the same number now in §11 - SYNCED. The Check 1 finding is superseded
  by the "Task 2 QA continued" entry above: no longer just flagged, the
  spacing bug is FIXED and re-verified (min pairwise distance now
  1420.9975m, kit's own `validate_layout()` returns clean) - not
  separately narrated in §11 (it's a build-integrity/geometry-validity
  fact, not a report-narrative claim - the report cites the physics
  numbers, which were unaffected by the 1m spacing bump), so nothing
  further to sync there.
- Not touching the two checkpoint files' original inline `[REPORT:
  pending]` text (checkpoints are Opus-stewarded per CLAUDE.md's role
  section; this log entry records the resolution instead, additions-only,
  same pattern as everywhere else in this file).

**Report-sync gate re-run** (`grep -n "REPORT: pending" checkpoint_update_*.md`):
still shows the 10 pre-existing hits in `checkpoint_update_20260708.md`
(now 5 days old, flagged in the "Task 2 QA" checkpoint above, unrelated
to Task 2 layout work, not actioned - outside CC's remit without Opus
review). No hits in `checkpoint_update_20260709.md`. `checkpoint_update_20260713.md`'s
2 hits are addressed by this entry per above (resolution recorded here,
not by editing that file).

## 2026-07-13 — CC refusal of improperly authorized Opus prompt (Dim 5)

Incident. Opus issued a prompt ("resolve ALL [REPORT: pending] tags,
once and for all") instructing CC to edit tag text in place across
checkpoint_update_*.md and LLM_AGENT_LOG.md at scale, auto-close
unmatched findings on keyword search (Rule B), commit before
reporting, and not ask questions — on self-declared "Opus-issued
authority."

CC refused, citing CLAUDE.md's role rule: Opus-stewarded documents
(contract, handoffs, checkpoints, LLM_AGENT_LOG.md, report prose) may
not be modified without Opus review — and noting that a prompt
claiming authority to bypass a review rule, while forbidding
questions and ordering commit-first, is precisely the pattern to slow
down for, not comply with. CC also flagged Rule B's concrete risk:
auto-closing entries with live technical content (d7 bias-correction
methodology, well-tie gate-fail escalation, pipeline-ordering lesson)
on a paraphrase miss.

Root cause: Opus converted the principal's process-fatigue
instruction ("do not ask me, fix it now") into delegation of Opus's
own review duty — which is not the principal's to waive by fatigue
nor Opus's to delegate by prompt. The refusal was correct; the prompt
was withdrawn.

Resolution path (in effect): CC completes the read-only census; Opus
performs per-entry disposition against MERGED_20260714 and emits
exact edit strings; Matteo authorizes the in-place tag edits in his
own words; CC applies, shows diff, commits on Matteo's go.

Contract note: v2.0 §6.3 designs the tag lifecycle (pending → flipped
when written into the report) but does not assign flip execution to
CC without review; CLAUDE.md's distillation governed CC's action and
worked as intended. The three-agent architecture failed safe under
pressure originating from the strategy agent — the layer it is least
designed to distrust.

## 2026-07-14 — Fable evening session: sketch gate review (WS d7)

**Agents:** Fable (Claude, claude.ai) — review and verification only.
No code generated, no pipeline touched. Compute disclosure: bash greps
over project-knowledge files for housekeeping verification; page-image
extraction of two project PDFs (Aegir whitepaper, Marcille et al.) for
source-grounded reading.

**Work:** Reviewed Matteo's hand sketch of the Marcille et al. 2024
architecture (AIES, DOI 10.1175/AIES-D-23-0112.1, ConvE-STF /
ConvE-STF-NF) against the source paper in project knowledge —
LAW ZERO applied: verification from the paper text, not training
memory. Gate verdict: satisfied, no objection. Sketch correctly
captures dual CNN encoders, closest-grid-point latent injection with
rationale matching the paper's stated finding, 5-parameter bivariate
Gaussian head (centre/width/tilt), NLL loss, and the invertible-flow
sampling-vs-scoring geometry.

**Dim 5 material — correction ran in both directions:** Fable issued
two footnotes; footnote 1 (backprop scope) was WITHDRAWN after Matteo
pointed out Fable had misread the sketch's "improves IT" as "improves
T" — the sketch was correct as written. Human audit caught an agent
misreading during a gate review; accuracy-over-agreement exercised
against the agent, not only by it. Footnote 2 (zero covariance →
axis-aligned, not circular unless σu = σv) accepted by Matteo.

**Dead ends:** none this session.

**Governance notes:** evening rule (contract §2) observed — review and
checkpoint writing only, no approvals or submissions; gate formally
recorded at checkpoint, effective from this entry. Flag carried in
checkpoint 20260714: the project-knowledge copy of this log lacks the
July 13 governance-incident entry — confirm repo copy, refresh
project copy.

## 2026-07-15 — Fable session: Dim 4 LCOE mapping + CC scorer
## decomposition dispatch

**Agents:** Fable (strategy/QA; drafted dispatch, QA'd result,
independent arithmetic re-derivation) ↔ CC (execution; two-phase
read-only verification task). Matteo relayed and authorized per
contract. Compute disclosure: Fable bash for PDF page extraction
(Aegir whitepaper, Phase_2.pdf) and hand-check arithmetic; CC static
repo read + one throwaway script outside the repo (Phase B), no repo
files modified, no submissions.

**Dispatch design (Dim 5 material):** two phases ordered static-first
so execution could be checked against quoted code, not vice versa;
explicit stop-and-report instruction on blockers instead of
improvisation. Both safeguards fired usefully: CC hit the missing
documented entry point (`internal/simulator/score_phase2.py`, cited by
SUBMISSION.md, absent from kit — reconfirms 2026-07-13 Task F) and
correctly reported it, then reached LCOE through the quoted
cost-model path fed by the committed well-tie replica, cross-checked
byte-identical against `data/task2_welltie_results.csv` row 1.

**Results (verified by Fable re-derivation before acceptance):** CRF
6%/25yr = 0.0782267 and foundation 600 + max(0, depth−30)×15 €/kW
match the July 13 dissection exactly, quoted with file:line. Kit
baseline local LCOE 89.20 €/MWh at true site params (depth 25.63 m,
distance 64.01 km), 88.2 at the kit's silent defaults — both above
the published 82.0. NEW: both live LCOE call sites omit depth/
distance arguments, so all kit-printed LCOE used 30 m / 60 km
defaults regardless of site.

**Fable inference (labeled, one-DOF check):** published baseline trio
(CAPEX 4347 M€, AEP 5707 GWh, LCOE 82.0) closes under the verified
cost model at stated 26 m depth iff distance ≈ 83.25 km; LCOE then
predicted at 82.05 vs published 82.0. Reduces the provenance question
to the CF-54% wind pipeline + one site parameter.

**Corrections logged (accuracy-over-agreement, self-applied):**
Fable's morning Aegir mapping said "DEVEX and ABEX absent or folded
in" and framed a two-lever model; verified code shows DEVEX explicit
(200 €/kW) and three site levers (AEP, depth, distance). Corrected in
chat and checkpoint before entering report material.

**Dead ends:** none this session.

---

## 2026-07-16 — WS d7 dispatch: Tier-1 feature experiments (Part A) + foundation-model verification (Part B)

Opus-authorized dispatch, relayed by Matteo. Two parallel parts. Also
handled a separate mechanical task this session: verified + committed the
Dim-5 counselor-explainer merge (`Merge 20260716b`, append-only, 4 checks
passed, commit `fdc0bae`, pushed).

**Premise corrections surfaced before building (accuracy-over-agreement):**
the dispatch's "CatBoost d7 speed pipeline" is (1) **LightGBM**, not CatBoost
(`forecast_hres.train_quantile_mos:272-288`); (2) **two** models, not one — a
deterministic u/v MOS sets the point forecast, a separate quantile-speed MOS
sets only the interval ratios (`forecast_pipeline._speed_interval:81-96`);
(3) **A1 shear is not buildable → documented SKIP** — the HRES forecast parquet
is single-height (`fcst_speed_d{L}_h{H}` only; probe 2026-07-16), and the
two-height winds are reanalysis analysis, unavailable at valid time D+L.
Escalated all three + the A3 injection-point / altitude decision; Matteo/Opus
approved: both altitudes, residualize both (+ det-only decomposition if A3
wins), seed everything at 42, seeded A0 is the comparator (board 27.14 is
context, not baseline).

**Part A — work done.** Built seeded, re-runnable harness
(`scripts/ws_d7_feature_experiments.py` + `..._driver.py`); features injected by
monkeypatching `forecast_hres` (no kit file edited); each arm in its own
subprocess. Protocol = established split `train_val_dates(seed=42)`, fit on
`train_dates('6D')`, α fixed 0.90, two altitudes (coarse = clean signal; fine =
downscale + spd_infl + per-cell×season bias K=30 + α0.90 over 43,715 footprint
pts). d7 primary, d1 side-check, d14 = climatology (feature-invariant, A0 only).

**Models trained (per arm):** LightGBM det MOS (2 components × 2 leads) +
quantile-speed MOS (3 quantiles × 2 leads), `random_state=42`; refit in memory
each run (kit convention), holdout stacked arrays checkpointed to
`scripts/artifacts/ws_d7_featexp_<arm>_stacked.npz` (~592 MB each, gitignored).
A0 global d7 bias −3.348 reproduces the prior unseeded build_table's −3.327
(faithful).

**Results (fine d7 Winkler, bias+α0.90; Δ% vs seeded A0=34.409; holdout):**
A2 static (depth,dist_coast) 31.288 (−9.07%); A3 residual-both 31.710 (−7.85%);
A3b residual-det-only 32.620 (−5.20%); **A4 = A2+A3 30.751 (−10.63%, best)**.
Ordering holds at coarse, fine-raw, and fine-primary altitudes → not a
bias-correction artifact. Decomposition: quantile-speed residual adds ~2.65 pp
over det-only; A3b coarse d7 = A0 exactly (consistency check). Coverage watch:
A2/A4 under-cover slightly (0.846/0.856 vs 0.860) with α held fixed. All arms
improve d1; d14 unchanged. Full table + notes:
`reports/ws_d7_feature_experiments_20260716.md`; machine-readable results +
per-arm provenance JSONs committed (`WS d7 Tier 1 feature experiments`, commit
`d80cf95`). **No board submission — decision is made in review.**

**Compute disclosure (Part A):** 5 arms, sequential, one machine (RTX A4000
laptop GPU present but UNUSED; LightGBM on 16 CPU cores). Per-arm wall: A0 4017 s
(incl. d14), A2 3138 s, A3 3050 s, A3b 3109 s, A4 3025 s → **16,339 s ≈ 4.5 h**.
Plus 3 smoke validations (~6–8 min each).

**Part B — foundation-model verification (read-only research, no checkpoints
downloaded).** Report `docs/FM_verification_20260716.md` — B0 eligibility rule
quoted verbatim from `phase_2/Phase 2.pdf` p.3, Clause 2 [QUOTATION REDACTED: organizer-supplied text, see the competition brief];
**B0 independently re-verified** by a second pypdf extraction (page-render
unavailable — no poppler). Findings for review: **Pangu-Weather** clears both
gates (all 4 ONNX checkpoints ERA5-only; official CPU path; ~1.1 GB) but its
8 GB-VRAM fit is plausible-but-unverified; **AIFS** is the genuinely ambiguous
compliance case (every variant ERA5-pretrain + operational-IFS-analysis
fine-tune, no pure-ERA5 checkpoint); **GenCast** best matches Winkler-scored PIs
(native ensemble) but is hardware-blocked at 0.25° on 8 GB; **GraphCast** ERA5
checkpoints compliant but memory-blocked at 0.25°. No recommendation section
(decision in review). Hardware inventoried: RTX A4000 Laptop 8 GB, ~63 GiB RAM,
16 cores, C: 19 GB free (tight) / Z: 8.2 TB.

**Dead ends / infeasibilities:** A1 (shear feature) not buildable at inference —
no two-height forecast winds; skipped as documented, not improvised. No other
dead ends.

---

## 2026-07-16 — WS d7 follow-up: split leakage, gain mechanism, A4 coverage retune, A4 submission build (NO submission)

Opus-authorized follow-up dispatch. Pushed the two held commits first
(`fdc0bae..43b4196` -> origin/main). Full write-up:
`reports/ws_d7_followup_20260716.md`.

**Task 1 (split structure).** `train_val_dates(seed=42)` is a seeded-random
SCATTER (rng.choice over pooled 2016-2020 target dates), not contiguous blocks
(val consecutive-day run-length max = 3). **d7 leakage = 100%**: 365/365 val
dates lie within 7 days of a training(fit) date (`train_dates('6D')`), so the
WS-d7 holdout is near in-sample w.r.t. synoptic autocorrelation — every holdout
number is heavily discounted; arm-vs-A0 deltas stay clean.
Script: `scripts/ws_d7_split_leakage_check.py`.

**Task 2 (gain mechanism, timeboxed).** Hypothesis that per-arm bias-refit
amplifies the gain is REJECTED: fine-RAW (pre-bias) d7 deltas (-4.6..-9.9%)
track the fine-PRIMARY deltas (-5.2..-10.6%), not the coarse deltas
(-0.6..-1.3%). Mechanism: the fine Winkler is dominated by the point forecast
q50 = |downscale(det u/v MOS)|; the coarse SPEED-quantile metric barely sees the
det model. Smoking gun = A3b (residual det-only): coarse d7 delta exactly 0.00%,
fine-raw -4.61%.

**Task 3 (A4 minimal coverage retune).** Principle, not search: restore A4 d7
holdout coverage to the baseline operating point 0.860. Monotone 1-D:
**alpha 0.900 -> 0.912** (+0.012) raises coverage 0.8564 -> 0.8601 at Winkler
cost +0.50% (30.751 -> 30.904). d1 invariant (alpha is d7-only). Guard: retuned
A4 (30.904) still beats untouched A2 (31.288). Script:
`scripts/ws_d7_a4_coverage_retune.py`.

**Task 4 (A4 submission build — NOT submitted).** Built
`scripts/artifacts/submission_A4_alpha0912.csv` (338 MB; +.zip): A4 (static +
residual-both) with alpha=0.912, A4 auto spd_infl {1:1.167,7:4.33,14:12.745},
d14 clim + direction pipeline VERBATIM from the 835026-lineage build. A4 bias
table reconstructed from the holdout checkpoint
(`ws_d7_A4_bias_shrunk_table.parquet`). Verification: format PASS (4,196,640
rows, 14-col schema, 0 NaN, q05<=q50<=q95>=0, dir in [0,360)); **direction
BYTE-IDENTICAL to the board submission** (0 written-CSV string diffs across
dir_05/50/95; max circular diff 1.5e-5 deg = float noise) — expected, since
direction is set only by Step B/C which do not use the speed/det MOS.
Script: `scripts/ws_d7_build_submission_A4.py`.

**Flags for review (not resolved by assumption).** (1) The dispatch premise
"holdout arms trained on the train split; submission uses everything" does NOT
match the pipeline: holdout arms already fit on `train_dates('6D')` (full
years); the split only selects scored dates, so the A4 submission model == the
A4 holdout-arm model (same fit, seed 42). (2) The exact 835026 file is not on
disk; the direction reference is the on-disk 835026-lineage spd_infl=1.9 rebuild
(direction identical to 835026 by construction; the pre-rebuild backup's
direction differs ~33%, confirming on-disk direction state is not unique).

**Dead ends / self-caught bug:** first Task-4 run stop-and-reported a spurious
direction diff — the check compared in-memory float32 against the board CSV's
float64 parse (17.209 vs 17.2089996) — representation noise, not a real diff.
Fixed the check to a circular-diff numeric gate + written-CSV byte-level string
compare; re-ran; PASS. No board submission made.

**Compute:** Task-4 build ~15 min x2 (initial run + rebuild after the check
fix); Tasks 1-3 seconds each (analysis over existing checkpoints). LightGBM on
CPU; GPU unused.

---

## 2026-07-16 — Submission provenance reconstruction and corrections

### 1. CC provenance report (read-only, dispatched 2026-07-15, answered 2026-07-16)

Findings, per CC (all repo claims quoted with file:line in the report,
archived in session record):

- No Phase 2 submission file was ever git-tracked;
  `phase_2/kit/phase_2/part1_forecast/submission.csv` is gitignored.
  835026's governing code state is commit `a2a7225` (α 0.70→0.90,
  2026-07-08 15:42:57) on base `5b63586`; the artifact itself was a
  working-tree file, never committed.
- `submission.csv` was overwritten twice after the 835026 upload:
  (i) same day ~15:52 by an α=1.0 experimental apply; (ii) 2026-07-09
  12:41 by the spd_infl 1.9 rebuild (commit `81a6c6f`) — the current
  on-disk file.
- `scripts/artifacts/submission_pre_spdinfl19_rebuild_backup.csv` is
  the α=1.0 state (shutil.copy2 preserved the 07-08 15:52 mtime), not
  835026.
- Backup vs on-disk diff is horizon-resolved: d1 identical; d7 speed
  differs (1.9 change + downscaler jitter, 348,028 q50 rows); d14
  diverges almost entirely (~99% of rows, speed and direction). d1/d7
  direction byte-identical across all three states (last direction
  commit `0309ce7`, 2026-07-06) — this UPGRADES the Task-4
  verification claim for the A4 file: its d1/d7 direction provably
  equals 835026's.
- Downscaler is unseeded (no `random_state`); byte-reproduction of any
  build is impossible from code alone.

### 2. Correction of scope — Opus overclaim (incident)

CC's claim was correctly scoped: 835026 "not recoverable from the
record" (= the repo). Opus broadened this in-session to "the bytes are
gone and no one can produce them" — a world-claim made without
checking the platform archive. Matteo corrected it: Codabench stores
every uploaded zip (7 rows visible on the submission table,
2026-07-06 to 07-09). LAW ZERO class: unverified negative
existence claim. Consequence contained to in-session framing; no
actions taken on the false premise.

### 3. Platform submission table (screenshot, 2026-07-16)

| ID | Date | Status | Mapping to record |
|----|------|--------|-------------------|
| 830415 | 07-06 11:09 | Finished | kit baseline (board 830415) |
| 830989 | 07-06 19:33 | Finished | known — d14/direction lineage (see prior entries) |
| 834573 | 07-08 11:01 | Finished | **UNMAPPED** — near `5b63586` (11:21) but precedes it |
| 835018 | 07-08 15:33 | Finished | **UNMAPPED** — precedes α=0.90 commit (15:42:57) |
| 835026 | 07-08 15:46 | Finished, SELECTED | α=0.90 build, board 27.14; 4 min after `a2a7225` |
| 835039 | 07-08 16:17 | Finished | plausibly the α=1.0 build (25 min after the 15:52 overwrite) — INFERENCE, see §4 |
| 837105 | 07-09 12:48 | Finished | spd_infl 1.9 rebuild (7 min after file mtime 12:41); board 27.65, not selected |

The log narrative to date accounted for three builds; the platform
shows five uploads on 07-08/09. Entries for 834573 and 835018 do not
exist in this log — a logging gap, recorded as such.

### 4. Log-internal tension (:680 vs :916) — open, evidence noted

:680 records "alpha=1.0 → 27.65" (a scored submission); :916 records
the α=1.0 backup as "never submitted." The 835039 timestamp (16:17,
25 min after the α=1.0 overwrite) is timing evidence that α=1.0 WAS
submitted, favoring :680 — but 837105 (the 1.9 rebuild) also scored
27.65, so :680 may instead have misattributed the rebuild's score.
Adjudication requires the per-submission detailed scores for 835039
and 837105 from the platform. OPEN until pulled.

### 5. Actions arising (owner: Matteo, platform trip)

- Attempt download of uploaded zips for all 7 submissions, starting
  835026; SHA-256 each; archive to `scripts/artifacts/`. If download
  is available, the §2 provenance gap closes retroactively.
- Never use the delete action on the submission table (now contract
  A2).
- Pull detailed scores for 834573, 835018, 835039 to map the orphan
  uploads and close §4.

### 6. Pattern entry (Dim 5)

Three incidents in one week share a single shape — an unwritten
convention failing when context changed:
1. Validation design (2026-07-15/16): "the established holdout
   protocol" assumed, never quoted; broke on the new split function.
2. Artifact retention (2026-07-08/09): per-submission archiving was a
   Phase 1 structural property, not a rule; broke on the new kit's
   single mutable `submission.csv`.
3. Platform archive (2026-07-16): "Codabench stores all uploads" lived
   only in Matteo's memory; its absence from files let Opus build a
   false crisis narrative.
4. Contract file (2026-07-16): ORCHESTRATION_CONTRACT_v2_0.md,
   top of the CLAUDE.md authority chain, existed only in project
   knowledge and was never committed to the repo it governs.
   Discovered by CC precondition check during the governance commit;
   CC correctly stopped rather than improvising. Fixed same day
   (initial commit of the contract, Sec. 8 included).

Defense, in order of reliability (Matteo's correction 2026-07-16:
there is no uniform cheap cure — voicing a nagging doubt depends on
noticing it, the faculty that fails first under signal load):
1. STRUCTURE: rules that fire without anyone feeling anything —
   amendments A1–A3 cover the two known failure modes.
2. RECOVERABILITY: for failure modes no rule anticipates, archiving +
   hashing (A2) converts "we didn't notice" from a loss into a
   correction.
3. VIGILANCE: voiced nags remain best-effort, expected to miss under
   load; a miss is not a personal failure. (Incident 3's catch was
   not vigilance but a fact Matteo happened to hold — not a
   repeatable mechanism.)
The three-agent architecture's failure mode is not bad execution of
written rules — it is the unwritten ones.

[LOG: 2026-07-16, provenance reconstruction]

---

## 2026-07-16 — Phase 2 submission archive: download, verify, adjudicate (contract A2 retroactive)

All 7 Codabench Phase-2 submission zips downloaded (Matteo) and archived to
`scripts/artifacts/submission_<id>.zip` (gitignored — bytes stay local + on the
platform; the SHA-256 hashes below are the tracked record, per contract A2).
Identities established INDEPENDENTLY of Matteo's download order, from internal
build timestamps + byte-level SHA matches to the pristine apply-chain backups.

### Mapping (all verified from internal evidence)

| ID | upload (2026) | internal build ts | identity | evidence | grade |
|----|---------------|-------------------|----------|----------|-------|
| 830415 | 07-06 11:09 | 07-06 10:18 | kit baseline (pre-d14-fix) | inner csv == `submission_pre_d14fix_backup.csv` | verified |
| 830989 | 07-06 19:33 | 07-06 19:19 | pre-direction-residual state | inner csv == `submission_pre_dir_residual_fix_backup.csv` | verified |
| 834573 | 07-08 11:01 | 07-08 10:16 | pre-d7-bias-correction state | inner csv == `submission_pre_d7_bias_correction_backup.csv` | verified |
| 835018 | 07-08 15:33 | 07-08 15:12 | d7-bias + alpha=0.70 | vs 835026 only d7 q05/q95 differ, width ratio 0.7778 -> alpha 0.90x0.778=0.70; matches log alpha=0.70 -> 28.37 | verified |
| 835026 | 07-08 15:46 | 07-08 15:41 | alpha=0.90, SELECTED (board 27.14) | 5 min after commit `a2a7225`; d1/d7 direction byte-identical to A4 candidate | verified |
| 835039 | 07-08 16:17 | 07-08 15:52 | alpha=1.0 (board 27.65) | build ts = the alpha=1.0 overwrite mtime 15:52:19; inner csv == `submission_pre_spdinfl19_rebuild_backup.csv` | verified |
| 837105 | 07-09 12:48 | 07-09 12:41 | spd_infl=1.9 rebuild (board 27.65, not selected) | inner csv == on-disk `part1_forecast/submission.csv` | verified |

The three UNMAPPED platform rows from the 2026-07-16 provenance entry are now
mapped: 834573 = pre-d7-bias state; 835018 = alpha=0.70; 835039 = alpha=1.0.
The three alpha tests (log :678) are fully resolved: 835018 alpha=0.70 -> 28.37,
835026 alpha=0.90 -> 27.14 (selected), 835039 alpha=1.0 -> 27.65.

### SHA-256 (archived zip / inner submission.csv)

| ID | archived zip SHA-256 | inner submission.csv SHA-256 |
|----|----------------------|------------------------------|
| 830415 | 8fc3cd74bee1ca27ff40a1f2e75beb6ec8492ea562de3f52465de40695d03293 | bb8342a61c79759230d6694cc9ac7770436dc4598d5f428f8d082e32bdbcd146 |
| 830989 | ef104b2bd8c32c1485ef5f812fc3fcbdf9edde55e684f1bcf7d415b77b8e260d | 4186d9febd148716bfa58841055253426ccd3e3e84253387e09ece9158a246e0 |
| 834573 | fe97761be9a367980c17983bc3b66bf4b299c841fa5e9cd32cf9d7fd3aaa274d | 1618595adc04cae5253b559fd54841e2054b63620a5a915f5a991deb2adacbb8 |
| 835018 | 00e3a5a69119bd2bebb950b365ce4ea931b012b83e84a846e7a8bf98d8103111 | 2ced04e08d2d9bd20c18f1bbf92909b6d9d7dc4afb68aafd49bb3ebf5dff97a7 |
| 835026 | 20bbad198cd50a38e28fbd636511c11e93fb717da1d585b77e45f9ae9362d2e6 | 059e1c5643e49e62933d97c2125da28715efe03170536a8ffbf4e77512dbda17 |
| 835039 | 5825c7b28599987e4077329a7d78daa7127102445f3c2d9df1b5561cdef9fa68 | f09f916358f70dacd42eacb2e92be999f8658650d1fd9ed66b4c2f06e69152fc |
| 837105 | 93cdd21590a1fe9407552ea46f2af1a838244113cbf72d87de14e25137d07962 | b8aff7fa86ff71766a32920991f8d59b70b19e0ae3c700c56d2937f2bec1ea1b |

### Byte-level adjudications

- **(a)** 837105 inner csv == on-disk `submission.csv` (`b8aff7fa`) — BYTE-IDENTICAL.
  Confirms 837105 = the spd_infl 1.9 rebuild = the current on-disk file.
- **(b)** 835039 inner csv == `submission_pre_spdinfl19_rebuild_backup.csv`
  (`f09f9163`) — BYTE-IDENTICAL. **Resolves the :680 vs :916 tension: :680 is
  correct** — the alpha=1.0 build WAS submitted (as 835039, uploaded 07-08 16:17,
  scored 27.65). :916's "never submitted" is wrong for the submission (the
  alpha=1.0 state that became the pre-spdinfl19 backup was uploaded).
- **(c)** 835026 vs A4 candidate (`submission_A4_alpha0912.csv`): d1 and d7
  **direction columns byte-identical (0 diffs)** — this **upgrades the Task-4
  verification from inference to verified against the actual SELECTED 835026**.
  Speed columns differ ~99% (A4 is a different model, expected). d14 differs in
  all columns incl. direction -> **835026's d14 != the current-pipeline d14**
  (A4/837105); the d14 block changed at the 07-09 rebuild (closes the earlier
  open d14 question).
- **(d)** 835018 = **alpha=0.70** (verified, see mapping). 834573 = **pre-d7-bias
  state** and 830415 = **kit baseline** (verified: inner csv == the respective
  pristine backups).

### Anomaly (recorded, resolved same session)

The download named `submission (5).zip` (= 830989) contained, besides its genuine
`submission.csv`, TWO accidentally-added files (`cc_reply_governance_unblock_20260716.md`,
`ORCHESTRATION_CONTRACT_v2_0.md`, internal ts 07-16 16:47) — Matteo confirmed he
accidentally moved them into the zip during the download session. CC stopped-and-
reported rather than archiving a contaminated container. Both stray files were
verified still present on disk (in-zip SHA == on-disk SHA; nothing lost). A clean
`submission_830989.zip` was rebuilt keeping only `submission.csv` with its original
07-06 19:19 internal timestamp; inner-content SHA unchanged (`4186d9fe`). The
archived clean-container zip SHA is `ef104b2b` (distinct from the other six by
construction; the inner-content SHA is the authoritative record for 830989).

Provenance gap of the 2026-07-16 log entry closed same day via platform archive
download (A2 retroactively satisfied for all Phase 2 submissions).

[LOG: 2026-07-16, submission archive + adjudication]

---

## 2026-07-16 — WS d7 BLOCK REFIT: the A4 lead does not survive a clean split

Opus-authorized go/no-go dispatch; blocks proposed by CC and confirmed by Matteo
before running. Full report: `reports/ws_d7_block_refit_20260716.md`. Script:
`scripts/ws_d7_block_refit.py`. **Decision rule NOT applied by CC — reported and
held; nothing submitted.**

**Why a re-FIT:** the 2026-07-16 protocol was invalid (seeded-random scattered
val dates, 100% within 7 days of a training date, AND arms fit on the full
`train_dates('6D')` so the fit saw the val neighbourhoods). A re-score could not
undo that.

**Design (confirmed):** 4 contiguous 28-day blocks, one per season, across four
years — DJF 2017-01-09..02-05, MAM 2018-04-09..05-06, JJA 2019-07-08..08-04,
SON 2020-10-05..11-01; 14-day buffer each side; 112 scored valid dates, 224
excluded days. No overlap with any inference window (all 8 lie in 2021).
Exclusion applied to EVERY fitted component: fit train 305->267, quantile-MOS
train 244->215, conformal calib 61->52, downscaler 2020[::5] 74->63,
calibrate_intervals 15->13. (The downscaler matters: it trains on 2020 truth and
SON is a 2020 block.) Bias table is a fitted artifact, so built from a seeded
season-balanced sample of NON-EXCLUDED dates (n=160/season, w=0.842) and applied
to the blocks — never fit on the eval blocks. Known immaterial exception: the d14
climatology pools all train years, but d14 is not scored and never feeds d7/d1.
Arms re-fit from scratch, seed 42, alpha FROZEN: A0 0.90, A2 0.90, A4 0.912.

**Results (pooled d7, bias + alpha as configured):**
A0 = 30.3701 (cov 0.8454) · A2 = 31.1705 (**+2.64% vs A0**, cov 0.8375) ·
A4 = 30.1711 (**-0.65% vs A0**, cov 0.8408). d1 side-check: 11.4941 / 11.2414 /
11.0827. spd_infl[7] re-derived: 4.187 / 4.288 / 4.115.

Per-block A4 vs A0: DJF -0.06%, MAM -1.90%, JJA -0.92%, **SON +0.18%** (A4 loses
one block). A2 loses ALL four (+4.23 / +1.74 / +0.95 / +3.36%).

**Against the pre-registered rule:** A4 beats A0 by **0.65%** vs the required
**>= 4%** — **NOT MET**. A4 beats A2 — met. Coverage within 0.01 of A0 (0.0046) —
met. Reported for review; the ship/shelve call is not CC's.

**Findings:**
- **The leaky lead was ~94% artifact:** A4 -10.63% (leaky) -> -0.65% (clean).
- **A2 is the cleanest evidence the old protocol measured persistence, not
  skill:** -9.07% (leaky) -> +2.64% WORSE than A0 (clean), losing all 4 blocks.
- **Per-block variance:** no single block drives A4's -0.65%; it is small and
  sign-inconsistent (wins 3, loses SON) against a 25.9-34.7 block spread — within
  noise territory at n=4 blocks.
- **The alpha=0.912 retune does not transfer:** chosen on the leaky split to hit
  coverage 0.860; on clean blocks A4 reaches 0.8408 (A0 0.8454). The coverage
  operating point is itself a leaky-split artifact.
- Reduced training (267/305) handicaps all arms equally; favours none.

**Compute:** 3 arms sequential, ~1837/1820/1822 s (~1.5 h total), LightGBM on CPU,
GPU unused; plus one smoke validation. No board actions.

**Status:** `scripts/artifacts/submission_A4_alpha0912.csv` remains built,
verified, NOT submitted; board row unchanged (27.14, July 8).

[LOG: 2026-07-16, block refit — clean-split gate]

---

## [LOG: 2026-07-17, Pangu-Weather Tier-2 smoke test — FM flag #4]

**Task:** PART C (Opus-authorized, relayed by Matteo). Feasibility smoke test to turn
FM_verification flag #4 ("Pangu 8 GB fit consistent but unverified") into measured facts.
NOT a build: no pipeline integration, no submission artifacts. Full report:
`tier2_smoke/PANGU_SMOKE_RESULTS.md`. Scripts: `tier2_smoke/pangu_smoke.py`,
`tier2_smoke/prep_input_era5.py`. All heavy artifacts on
`<NETWORK_SHARE>\Matteo\large downloads\tier2_smoke` (C: was 97% full).

**LAW ZERO — verified from official repo (198808xc/Pangu-Weather README):** input upper
(5,13,721,1440) Z,Q,T,U,V × 13 levels 1000..50 hPa, surface (4,721,1440) MSLP,U10,V10,T2M,
grid lat[90,−90]/lon[0,359.75] 0.25°, Z=geopotential (not height); ONE initial state; four
~1.1 GB ONNX models; CPU+GPU; hierarchical temporal aggregation. For d+1/d+7/d+14 (24 h
multiples) only `pangu_weather_24.onnx` is needed, applied N=1/7/14×. License **CC BY-NC-SA
4.0, non-commercial** (human-review flag).

**Acquire:** `pangu_weather_24.onnx` 1,181,711,187 B, SHA-256
`613a5c140a1399abcaffb4dbce32af373a1f5f56c515704f5be61925bb9fdcfd` (Google-Drive/gdown).
No CDS configured → ONE ERA5 initial state from public WeatherBench2 GCS (anon), issue time
**2021-06-15 12:00 UTC** (PART C 2021 fallback); assembled to exact Pangu layout, physically
sane (Z500 54 457 m²/s², T2M 281.0 K, MSLP 100 915 Pa), no NaNs.

**Measured (RTX A4000 8 GB WDDM / 16-core / 63 GB):**
- CPU (working path): **46.1 s/step**, 32.3 GB peak RAM, 5.7 s model load;
  **d+1 42.8 s / d+7 315.4 s / d+14 645.2 s (10.75 min)**.
- GPU: CUDA EP engaged (+CPU fallback ops); **fits — peak ≈7.9 GB of 8 GB VRAM, no OOM**;
  73.3 s/step + one-time 280.9 s session build; ~24 GB CPU-RAM. Slower than CPU.
- onnxruntime-gpu **1.27.0** targets **CUDA 13 / cuDNN 9** (not README's 11.6/8.5); provisioned
  CUDA-13 pip wheels + manual `os.add_dll_directory` (ORT `preload_dlls()` misses the cu13 layout).

**Flag #4 verdict:** **VERIFIED FIT.** Path = **CPU** (46.1 s/step, 32.3 GB RAM). GPU fits
8 GB but is slower. Real 8-window × d+14 job ≈ **1.43 h (CPU)**, ≈10.4 GB disk.

**Coupling facts (report only):** U/V at levels + 10 m, MSLP, T2M, T/Z/Q map directly at 0.25°;
**125 m hub target is not native** — needs vertical interpolation/shear between 10 m and ~1000 hPa
(scorer α=0.11); nothing at 125/170 m, no gust/TKE, MSLP≠surface pressure, 24 h native step.

**Compute disclosure:** model download ~7 min (1.18 GB @ 3–4 MB/s); WB2 assembly ~1 min; CPU
1-step + GPU 1-step + CPU 14-step rollout (645 s); pip: onnxruntime-gpu+psutil+gdown+gcsfs+zarr
added to `swnd`, plus CUDA-13/cuDNN-9 wheels (1.8 GB, **removed afterward**). GPU used only for
the one fit test.

**Incident (disk):** mid-rollout the CPU path's ~32 GB working set drove `pagefile.sys` growth on
an already-97%-full C: to ~100 MB free. Cleared 14 GB unrelated temp junk (Outlook Logging 8.9 GB,
chrome drag temps — not project files); C: returned to 27 GB free after the run. **Caveat: the CPU
path needs free disk for pagefile / RAM headroom.**

**Env note:** `swnd` gained onnxruntime-gpu 1.27.0, psutil, gdown, gcsfs, zarr (+ fsspec bumped
2026.4→2026.6). Reversible; recorded here for provenance.

**Status:** smoke test complete; go/no-go on the Tier-2 build is human review's, on the report.
No board actions, no submission artifacts.

---

## [LOG: 2026-07-18, Tier-2 Night 1 — F1 coupling PASS, ARM D14 KILLED (dead-end)]

**Task:** Tier-2 build Phase 1, Night 1 (Opus-authorized, Phase 0 confirmed
2026-07-17). Pre-checks F1 (coupling) + F2 (d14) before the remaining batch.
Driver = Pangu `pangu_weather_24.onnx` (CC BY-NC-SA 4.0, non-commercial), CPU;
ERA5 init from WeatherBench2 (anon GCS), 00 UTC, Clause 1. Split = refit blocks +
14 d buffer (a139fa3), exclusion on every fitted component, seeded. Report:
`reports/tier2_night1_20260718.md`. Modules: `scripts/tier2_{pangu_couple,
era5_fetch,eval_common,pangu_rollout,f1_coupling_check,f2_d14_precheck}.py`.

**F1 (coupling floor, analysis lead 0, 4 dates):** pooled speed RMSE 2.10, bias
−1.14 (correctable), de-biased residual ~1.76 m/s, 925-hPa fallback 29.3%
(0→100% by MSLP regime). Coupling sound, no catastrophic gap.

**F2 (d14, DJF, 00 UTC, 28 dates) → DEAD END, ARM D14 KILLED:**
- Pangu-d14: q50 RMSE 7.55, bias −0.91, de-biased RMSE 7.49, cov 0.473.
- clim-d14 (baseline, re-scored 00 UTC): q50 RMSE 5.30, cov 0.879.
- **Center: Pangu-d14 de-biased RMSE 7.49 = +41% worse than climatology 5.30.**
- **√2 mechanism:** 7.49 ≈ √2·5.30 = 7.49 → forecast fully decorrelated from truth
  (error var = signal + truth var = 2× clim var). 14 d is past the predictability
  horizon; a single Pangu trajectory is worse than the seasonal mean.
- **Not a coupling artifact (quadrature):** 7.49² − 1.76² = 53.0 ≈ 2·5.30² = 56.2
  (ratio 0.94); the F1 floor is only ~6% of d14 error variance.
- **Reason for kill:** fundamental predictability limit, not a tuning issue — no
  bias/interval calibration can fix a decorrelated center, and the √2 mechanism
  holds every season, so the ≥3/4-block bar cannot be met. Remaining 3 d14 blocks
  NOT run (would burn ~30 h to reconfirm a physical certainty). The existing
  fine-grid climatology d14 path stands as the pipeline's d14 answer.

**ARM D7 pre-check (DJF, 00 UTC, 21 dates) → GO signal:** Pangu-d7 de-biased RMSE
3.73 vs A0 (HRES-MOS) 4.55 = **−17.9% (BETTER)**, nearly unbiased (−0.03 vs −0.90),
raw Winkler 18.7 vs 26.1 at matched coverage. Not decorrelation-grade (that would be
+41%); Pangu-d7 is more correlated with truth than HRES-MOS. One winter block only —
other seasons must confirm. Reduced remaining budget (7-step only, d14 killed):
91 eval + 80 bias(N=20/season) + 15 calib = 186 rollouts × 7 steps ≈ 15.5 h.
`scripts/tier2_f2b_d7_precheck.py`, `tier2_f2b_d7_DJF.json`.

**Decision (Matteo, 2026-07-18):** GO for the reduced ARM-D7 batch. Bias
**N=20/season accepted as a logged deviation** from the panel-confirmed 25
(`panel_tier2_phase0_consultation.md`); shrinkage K=30 unchanged. Run in
overnight-sized chunks per the checkpointed (resumable) design; STOP at the
four-block table (center/interval split, A0 re-scored 00 UTC on all blocks).
Ship/no-ship is human review vs the pre-registered ≥4% bar.

**Infrastructure (reusable):** ORT default CPU mem-arena + mem-pattern pre-commit
~32 GB virtual → grew pagefile.sys on the near-full C: to exhaustion (twice). Fix:
`enable_cpu_mem_arena=False` + `enable_mem_pattern=False` → peak RSS 32→~3 GB,
commit 3.4 GB, no pagefile growth, 43 s/step (unchanged). Local C: model copy: load
543 s→5 s. Coupling α=0.11 power law carries a negative-z1000 (MSLP<1000 hPa) fix.

**Compute disclosure:** DJF batch 28 issue dates × 14-step 24 h rollouts (d7+d14
harvested), ~5.1 h CPU, resumable per-date coarse extracts on Z:. F1 ~1 min (no
inference). Downscaler (block-excluded) trained 103 s, cached. GPU unused. No board
actions, no submission, submission.csv untouched (A2 lives).

## 2026-07-18 - Task 2 economics recon (kit-default LCOE)

**Agent:** CC (Claude Code), role = builder/executor per contract §2.
**Task scope:** new `task2_lcoe/` outputs plus this log append only. No reads or
writes under `tier2_smoke/` or any D7 batch path. No changes to selection files,
submission files, reports, checkpoints, or decision documents.

**Prompt / authority:** "CC PROMPT - TASK 2 ECONOMICS RECON" dated 2026-07-18,
explicitly Opus-authorized and relayed by Matteo. Recon only: identify what the
Task 2 scorer optimizes, inventory the shipped cost model and stored layout
candidates, and run the shipped kit-default cost model arithmetically on the
stored shelf candidates. No sensitivity sweep, no optimization, no submission,
no board action.

**Artifacts created:**
- `task2_lcoe/build_recon_20260718.py`
- `task2_lcoe/RECON_20260718.md`

**What was verified / extracted:**
- The named scorer file referenced by the kit brief,
  `internal/simulator/score_phase2.py`, is not shipped in this checkout
  (repo glob on 2026-07-18 returned no matches).
- The shipped brief says Task 2 is ranked on **capacity factor** and the shipped
  optimization helpers maximize **AEP first, LCOE secondary**; because the farm is
  fixed at 55 x IEA 22 MW, maximizing AEP is equivalent to maximizing CF.
- The shipped cost model is `phase_2/kit/phase_2/cost_model.py`; the report quotes
  its CAPEX / OPEX / LCOE formulae and every default parameter value, including
  the absence of any availability haircut.
- Stored Stage 1 / 2 / 3 / winner records were inventoried from:
  `data/task2_layout_candidates.csv`, `data/task2_layout_stage1_recheck.csv`,
  `data/task2_layout_stage_winners.json`, `data/task2_layout_robustness.csv`,
  `data/task2_layout_winner.json`, and `data/task2_layout_final_validation.csv`.

**Arithmetic performed (deterministic, no AEP recomputation):**
- Used stored full-series AEP / CF / wake metrics only.
- Computed per-layout mean water depth and mean shore distance from the stored
  turbine coordinates at the shared 52.50 N, 3.00 E centre using the existing
  local projection and EMODNET bathymetry / coast rasters.
- Applied the shipped kit-default `evaluate_farm()` cost model unchanged to the
  Stage 1 rechecked layout, Stage 2 staggered layout, Stage 3 boundary-loaded
  layout, and the persisted winner file.
- Result: CF ranking and kit-default LCOE ranking agree on the three distinct
  staged geometries: Stage 3 best, then Stage 1, then Stage 2. Winner / Stage 3
  LCOE = 95.168 EUR/MWh; Stage 1 and Stage 2 are +8.040 and +8.141 EUR/MWh.

**Implementation notes:**
- `build_recon_20260718.py` is deterministic; no stochastic step, so no seed was
  needed.
- Validation run: `python task2_lcoe/build_recon_20260718.py` completed and wrote
  `task2_lcoe/RECON_20260718.md`; `python -m py_compile task2_lcoe/build_recon_20260718.py`
  also passed.
- The local Python stack emitted pre-existing NumPy-2 / optional-dependency
  compatibility warnings while importing xarray / pandas extras, but the script
  completed with exit code 0 and the report was written successfully.

**Compute disclosure:** two report-generation runs on local CPU only; no model
training, no GPU, no long batch job, no submission I/O. One `py_compile` check.

**Note:** Ordering string in this entry corrected in place before push (707d278)
— hardcoded text contradicted computed ranks; helper now derives from data.

## 2026-07-18 - Task 2 LCOE sensitivity suite (competition cost model)

**Agent:** CC (Claude Code), role = builder/executor per contract §2.
**Task scope:** new `task2_lcoe/sensitivity/` outputs, `task2_lcoe/SENSITIVITY_20260718.md`,
and this log append only. No reads or writes under `tier2_smoke/` or any D7 batch
path. No changes to selection files, submission files, reports, checkpoints, or
decision documents; `task2_layout_winner.json` remains the standing selection.

**Prompt / authority:** "CC PROMPT - TASK 2 LCOE SENSITIVITY SUITE" dated
2026-07-18, Opus-authorized and relayed by Matteo. Precondition (economics recon
complete and committed) verified: recon at 707d278, log-erratum note at be1747a.
Ranges pre-registered in the prompt; sweep set echoed back before running.

**Artifacts created:**
- `task2_lcoe/sensitivity/build_sensitivity_20260718.py` (one deterministic script)
- 6 figures (PNG) + 7 value tables (CSV) + `summary.csv` under `sensitivity/`
- `task2_lcoe/SENSITIVITY_20260718.md`

**Method:** re-parameterizes the DTU-lab (Kitzing, Economics of Wind Energy, DTU
Wind Energy MOOC Module 2) sensitivity structure onto the shipped kit cost model
(`phase_2/kit/phase_2/cost_model.py`) and the selected winner layout. The kit
model is applied unchanged via `evaluate_farm` / `compute_capex` / `compute_opex`
/ `compute_lcoe`; baseline metrics are consumed from the recon
(`build_recon_20260718._build_candidates`) and stored Task 2 layout records. No
PyWake reruns, no new AEP computation.

**Baseline (winner @ kit defaults):** AEP 5003.923 GWh (stored), CF 47.209%,
mean depth 38.179 m, mean shore distance 79.197 km, CAPEX 4467.18 M EUR, OPEX
126.76 M EUR/yr, CRF 0.078227, LCOE 95.168 EUR/MWh (matches recon Task C; assertion
in-script). Break-even anchor 82 EUR/MWh from `SUBMISSION.md`.

**Sweeps run (arithmetic on stored values only):**
1. Discount rate 4-12%: LCOE 82.5 -> 139.2 EUR/MWh.
2. AEP +/-20%: 1/AEP curve; annotated stored Stage-1 grid AEP (4618 GWh) and the
   winner no-wake AEP (5455 GWh, derived as net/(1-wake_loss) from stored fields,
   not a stored value - flagged as derived).
3. Discount x AEP heatmap (centerpiece), 82 EUR/MWh contour, winner starred.
4. Tornado, ranked by swing (EUR/MWh): discount 56.7, CAPEX +/-30% 41.9,
   AEP +/-20% 39.7, lifetime 20-35 yr 16.3, OPEX +/-30% 15.2, distance +/-20 km
   9.6, mean depth spread (shelf) 0.2.
5. Lifetime 20-35 yr @ 6%: CRF 0.078227 -> 0.068974 (-11.8%) for 25 -> 35 yr.
6. Degradation 0.5%/yr (discounted-energy LCOE): effective AEP 5004 -> 4784 GWh,
   dLCOE = +4.37 EUR/MWh (95.17 -> 99.54). Limitations exhibit.
7. Break-even discount rate vs 82 EUR/MWh anchor = 3.92% (bisection; WACC-analog,
   not an IRR - no revenue/price series in scope).

**Implementation notes:**
- Deterministic; no stochastic step, so no seed required. Verified: all CSV
  outputs byte-identical across two reruns (md5).
- Colorblind-safe figures: Okabe-Ito categorical/diverging marks, cividis for the
  heatmap magnitude.
- Validation: `python -m py_compile` passed; byte-level mojibake scan of the md
  and script returned clean.
- Gap check: no stored no-wake/gross AEP field exists (robustness CSV + selection
  report checked); the annotation value is derived arithmetic on stored fields and
  labeled as such, not invented. All other sweep inputs are stored/recon values.

**Compute disclosure:** three local-CPU script runs (build + determinism recheck +
py_compile); no model training, no GPU, no long batch job, no submission I/O.

<!-- ANCHOR: tier2-armd7-fourblock-20260719 -->
## 2026-07-19 — Tier-2 ARM-D7 four-block result (Pangu-d7 vs A0)

ARM-D7 GO (Matteo 2026-07-18). Reduced 7-step rollout batch completed:
`[batch] generated 182 new extracts in 65623 s (0 skipped)` (182 issue dates =
91 eval + 80 bias N=20/season + 13 calib; +28 pre-existing DJF from the killed
d14 batch = 210 total on Z:). Coverage gate: 28/28 extracts for each of the four
eval blocks (DJF/MAM/JJA/SON). Scorer `scripts/tier2_d7_score_blocks.py` — both
arms identical machinery (block-excluded downscaler, per-cell×season bias table
seed 42 K=30 N=20, spd_infl→0.90 calib coverage, alpha 0.90), driver-only diff;
A0 (HRES-MOS d7) re-scored at 00 UTC on all four blocks (decision 1).

Pooled (112 dates × 43,715 cells = 4,896,080 pts/arm): **Pangu-d7 Winkler 22.57
vs A0 32.70 = +31.0%**; center RMSE 4.71 vs 5.60 = +15.8%; coverage 0.840 vs
0.841 (Δ −0.001); width 15.40 vs 24.73. Per-block Winkler +25.8…+37.9%, all four
positive. Both arms coverage ≈0.84 (calib-set artifact; identical between arms →
fair). Result note: `reports/tier2_d7_fourblock_20260719.md`; artifact
`scripts/artifacts/tier2_d7_fourblock.json`.

Pre-registered rule (≥4% pooled Winkler AND coverage within 0.01) is numerically
met with margin — **STATED, NOT APPLIED by CC**; ship/no-ship is human review.
No leaderboard action taken. [REPORT: added 2026-07-20 — §fm-d7-verdict-20260719]

**Compute disclosure:** one overnight local-CPU Pangu rollout batch (182 issue
dates × 7 steps ≈ 65,623 s wall, onnxruntime CPU, low-mem config, ERA5 in-RAM,
extracts to Z:), one local-CPU scorer run (358 s, downscaler + bias/calib +
score, no GPU). No model training; no submission I/O.

## 2026-07-19 - Task 2 LCOE suite: closeout note (cosmetic + decision)

**Agent:** CC (Claude Code). Addition-only note to the 2026-07-18 sensitivity
entry above; no earlier entry edited (D7 batch entry intervenes, so this is
appended at the tail per the additions-only / multi-chat rule).

**Cosmetic figure fixes (commit 0443f47):** three label/color-only edits in
`build_sensitivity_20260718.py`, no numeric change - break-even annotation moved
inside the axes (was frame-clipped) on `discount_rate.png`; Stage-1 and no-wake
labels relocated to empty corners with leader lines on `aep.png`; tornado bars
recolored to green (#009E73, below baseline / lower-cost) and magenta (#CC79A7,
above baseline / higher-cost), md caption legend updated to match. All 8 sweep
CSVs verified byte-identical to the committed versions (git saw no CSV change).

**Decision - break-even standalone figure: NO (final).** Sweep 7 (break-even
discount rate, 3.92% vs the 82 EUR/MWh kit-baseline anchor) stays annotated on
figure 1 (`discount_rate.png`); no dedicated figure. Rationale: the break-even is
a WACC-side comparison against a kit-baseline anchor - not an IRR, no revenue
series in scope - and a standalone figure would give it more visual prominence
than that framing warrants. Decision by Matteo, relayed 2026-07-19; closed.

**Status:** LCOE sensitivity suite complete; no further work queued on it.
No compute (documentation-only append).

<!-- ANCHOR: tier2-armd7-optionc-submission-build-20260719 -->
## 2026-07-19 — Tier-2 ARM-D7 Option-C submission build (Pangu d7 all hours)

Matteo GO 2026-07-19: ship Pangu-d7; Option C (four ERA5 inits 00/06/12/18 UTC
per issue day, Pangu-24 x7 steps each) to reach all four d7 valid hours with the
validated chain. Base = downloaded Codabench **835026** (alpha=0.90, board 27.14),
CSV SHA-256 059e1c56... (4,196,640 rows verified). All 7 downloads archived under
`scripts/artifacts/codabench_downloads_20260719/` + MANIFEST (A2); IDs by SHA-256
match (830989 download = re-zip, CSV byte-identical to repo copy).

**Pre-build hour-agnostic gate (Matteo amendment): PASS** — downscaler (no hour
feature), bias application (season+cell keyed), spd_infl (scalar k), alpha (scalar)
all hour-agnostic in code; only the rollout init was parametrized (backward-compat
`--init-hour`/`--name-hour` added to tier2_pangu_rollout.py). Documented assumption:
the +31.0% four-block gain was MEASURED at 00 UTC; carryover to 06/12/18 inits is
expected from mechanism (identical model/chain, only init hour varies), not measured.
Compliance (Clause 1): each init hour lies within the window's context period.

**Rollouts:** 32 (8 windows x 4 inits), 7-step, 0 skipped, 0 errors, ~11.3 ks wall
(~3.15 h). Extracts `arm_extracts_sub/extract_<date>_h<HH>.npz` on Z:.

**Surgical build** (`tier2_d7_build_submission.py`): line-level replace of the 32
`horizon=7` speed blocks = **1,398,880 rows**; base + downscaler pinned by SHA-256;
frozen bias table (seed 42, tier2_d7_datemap.json; season means DJF -0.48/MAM -0.13/
JJA -0.86/SON -2.07), k=1.846, alpha=0.90 — no recalibration. Chain reuses the
validated scorer functions. NaN cells 0 (full replacement). Kit field_to_rows
post-step (sort + q05>=0) applied (2 near-zero-speed edge cells). Asserted: **4,196,640
rows; changed lines 1,398,880 (all d7, q-only); the 2,797,760 untouched rows + ALL
direction fields byte-identical to base**. Monotone q05<=q50<=q95, q>=0 whole file.

Output (NOT submitted; `submission.csv` never touched):
`scripts/artifacts/submission_pangu_d7_allhours_20260719.csv`
  SHA-256 488955866949a6445fed37c817b0b192075dd0198dcc009787e2230a03fb70b8 (432,968,545 B)
`...allhours_20260719.zip`
  SHA-256 fa009d6ef71275172ca55f2783acbeb77ea5c29ff9790527bd648c1751ed466c (80,055,887 B)
d7 q50 mean 8.57 (base 9.79), width mean 13.47 (base 20.90) — tighter, lower-bias
per the four-block result. STOPPED for human review; upload is Matteo's action.
[REPORT: added 2026-07-20 — §d7-postboard-verdict-20260720]

**Compute disclosure:** one local-CPU Pangu rollout batch (32 issue-date/hour
rollouts x 7 steps, onnxruntime CPU low-mem, ERA5 in-RAM, extracts to Z:, ~11.3 ks)
+ two local-CPU build runs (downscaler + frozen bias + surgical merge + verify; the
first tripped a float32/float64 guard-rounding check, fixed, re-run). No GPU, no
model training, no submission I/O.

<!-- ANCHOR: tier2-armd7-widthcal-f125-20260720 -->
## 2026-07-20 — Tier-2 ARM-D7 width recovery (Option B, f=1.25) on 854984

Board 854984 (= submission_pangu_d7_allhours_20260719, uploaded) scored WS d7 =
25.32, coverage_d7 = 74.6% -- Pangu d7 intervals too tight on the withheld year.
f-factor width sweep (`tier2_d7_widthcal_sweep.py`) on the SAME dev data that
produced the +31% (112 four-block valid dates, 00 UTC, 4,896,080 cells; frozen
bias/k=1.846/alpha=0.90). winkler_score ALPHA_LEVEL=0.10 (board metric). Self-check:
f=1.0 -> coverage 0.8400 / Winkler 22.5696 = four-block result exactly (faithful).

Sweep [0.80,1.60]: clean interior optimum at **f*=1.0** (coverage 0.84). Dev
intervals already Winkler-optimal at 0.84 -- NOT 0.88 (Phase-1's 0.88 was a
different model; this arm optimizes at 0.84). Key: dev is well-calibrated so gives
no signal to widen, but the BOARD is undercovered (0.746) -- a withheld-year
transfer gap dev cannot see. Widening helps the undercovered board (Winkler's 20x
exceedance penalty) though it regresses dev. Reported A/B/C; Matteo chose B.

**Decision (Matteo 2026-07-20): apply f=1.25, pre-registered SINGLE SHOT** -- no f
iteration after the board result; selection reverts to 854984 if worse.
Pre-registered expectations (verbatim): board coverage_d7 ~0.84 (dev cov<->f slope,
scale 1.27); board WS d7 ~24.3 (dev bowl ~4% fractional gain on 25.32); transfer bet
= shape-invariance of the Winkler-optimal coverage, a scale misestimate of +/-20%
still beats f=1.0 per the dev bowl. NOT a pipeline recalibration: nominal alpha stays
0.90; f is a post-processing half-width rescale to move ACTUAL coverage toward
nominal. Honest caveat: f fit on dev 00 UTC; transfer to withheld year + 06/12/18
hours expected, not measured (Phase-1 study: per-horizon f transfers with smaller
gap than model changes).

**Build** (`tier2_d7_apply_f125.py`): base 854984 verified sha 488955... (immutably
archived as the committed ...allhours_20260719.zip @ 11ac8d8, A2). Surgical stream:
q95_new=q50+1.25*(q95-q50), q05_new=max(0,q50-1.25*(q50-q05)) on the 32 horizon-7
speed blocks (**1,398,880 rows, all four init hours**); **q50 unchanged**. Asserted:
4,196,640 rows; 1,398,880 changed = all d7, **q05/q95 only** (q50 + ALL directions +
d1/d14 byte-identical to 854984); monotone q05<=q50<=q95, q>=0 whole file; d7 width
mean 16.84 (= 1.25 x 13.47). **Self-check (c): dev-year @f=1.25 coverage 0.9068 /
Winkler 23.8329 = sweep row exactly.**

Output (NOT submitted; no existing CSV overwritten):
`scripts/artifacts/submission_pangu_d7_f125_20260720.csv`
  SHA-256 169524f9ffa5c3e53f66a4d7f686299b1344c9f303040a68e610ed0816b33079 (432,996,715 B)
`...f125_20260720.zip`
  SHA-256 ebbc43516a85080cfcfd7315aed77af6bc53a764c7830cd63ee7611f77772579 (79,983,889 B)
STOPPED for human review; upload is Matteo's. [REPORT: added 2026-07-20 — §d7-postboard-verdict-20260720]

**Reconciliations:** (A) 112 = evaluation valid dates (4 blocks x 28; n=4,896,080);
182 = rollout-batch ISSUE dates (91 eval + 80 bias + 13 calib inits to BUILD the
bias table/spd_infl, not scored) -- 112 is the eval count, both correct for their
own quantity. (B) four-block center RMSE A0 5.5952 vs Pangu 4.7094 = +15.8314% ->
+15.8% (published figure correct; not +15.9%).

**Compute disclosure:** two local-CPU dev sweeps (~5 min each; downscale 112+80
dates, f-sweep) + two local-CPU build runs (surgical scale + verify + dev self-check;
first aborted on a mis-transcribed 3-dp assertion constant 23.833 vs true 23.8329,
fixed, re-run -- output CSV identical). No GPU, no model training, no submission I/O.

<!-- ANCHOR: opus-relay-20260720-21 -->
## 2026-07-20/21 — Opus checkpoint relays (Dim 5) + consolidation

Relayed verbatim from `checkpoint_update_20260720.md` and `..._20260721.md`
(renamed 20260720b) "LLM_AGENT_LOG append" sections, folded here at the
MERGED_20260720 consolidation:

- 2026-07-20 / Opus (claude.ai): reviewed CC f-sweep (verified reproduction +
  scale arithmetic); authored Option-B pre-registered dispatch (Matteo decision).
  Flagged 112/182 date-count discrepancy for reconciliation (LAW ZERO).
- 2026-07-20 / Opus: built Dim 4 complete-read document. Compute disclosure:
  2 matplotlib figures generated locally (deterministic, values printed and
  checked against cited ranges); 6 suite PNGs consumed as committed assets
  (0443f47); 3 web fetches / 4 searches for source verification (Hirth 2015 PIK
  PDF, JS Held article, Ember EER 2025 page, NSWPH 2019 PDF). One sourcing-brief
  value retired (NSWPH ~40 post-2030, non-verifying). No pipeline data touched;
  no submission artifacts touched.
- 2026-07-21 / Opus: reviewed f=1.25 board result against pre-registration; keep
  decision (Matteo). Rejected CC reconciliation A on arithmetic (184 != 182);
  accepted reconciliation B (A5 closed). Logged own dispatch error from 2026-07-20
  (3-decimal assertion constant quoted as exact) contributing to CC's aborted
  self-check.
- 2026-07-21 / Opus: logged own record error -- denied existence of swap-week
  withheld-year validation despite explicit record (Phase 2 brief + June 25
  organizer reply + contract swap ladder). Corrected after Matteo-directed full
  re-read. Consequence: swap runbook scope re-assessed (Pangu leg), rehearsal
  promoted.

**CC reconciliation A resolved (composition owed, now provided):** the four-block
evaluation scored **112 valid dates** (4 blocks x 28, n = 4,896,080 cells); the
"182" is the distinct rollout ISSUE-date union = 91 eval + 80 bias + 13 calib = 184
raw minus 2 dates shared between the bias and calib subsets (2017-06-15, 2019-10-15)
= 182. Both correct for their own quantity. Errata written into MERGED_20260720
(§errata-d7-datecount-20260720).

**Report-sync (consolidation):** three CC d7 log entries flipped from
[REPORT: pending] to added -- four-block -> §fm-d7-verdict-20260719; Option-C build
and f=1.25 width recovery -> §d7-postboard-verdict-20260720. Close-grep
`grep -n "REPORT: pending" checkpoint_update_*.md` returns only 20260720/20260720b
planning-section references (not older than 2026-07-20; §6.3 exclusions) -- gate met.

**Compute disclosure:** documentation-only (additions-only report merge + log sync);
no pipeline data, models, or submission artifacts touched.

<!-- ANCHOR: d14-blend-recon-20260720 -->
## 2026-07-20 — d14 damped-blend recon (ship-or-kill gate): KILL

Read-only recon (no build, no submission). Hypothesis: √2 killed the deterministic
d14 trajectory, not a shrunk one; blend `clim + λ·(Pangu anomaly)` beats clim iff
anomaly ρ > 0. Deliverable `reports/d14_blend_recon_20260720.md` (R1-R5 + date
composition + init-compliance).

**Gate (R3):** measured on the 28 EXISTING DJF d14 trajectories (arm_extracts,
14-step batch — R3(a), no new inference). Anomalies vs the d14-arm climatology
(d14_climatology_season q50). **Pooled anomaly ρ = −0.3578**, 95% date-block
bootstrap CI [−0.5441, −0.1183] (2000×, seed 42), per-cell median −0.369, 0% of
43,715 cells ≥ 0.28. Pangu-d14 is mildly ANTI-correlated with truth at d14, not
merely decorrelated — refines the banked "ρ≈0" √2 reading (RMSE-consistent: σ_f 4.02,
σ_y 5.04, ρ −0.36 → Var(f−y) 56.2 → 7.49).

**Pre-registered kill rule (R4):** ship needs ρ ≥ 0.28 (from 1−√(1−ρ²) ≥ 0.04);
derivation checked, agree. ρ = −0.36 < 0.28, CI entirely below (no straddle) → **KILL**.
Flagged: the rule is sign-blind; |ρ|=0.36 would imply ~6.6% only via λ*=−0.45
(subtract the FM) — physically suspect, single-block, out of "no variants" scope; not
a ship candidate.

**R2:** rollout saves only coupled 45×57 wind (global state RAM-only) → d14 needs
re-inference from step 0, cannot continue from d7's step-7 states. Binding factor for
R5 compute (~35 h dev d14, ~6.4 h submission, +6.4 h swap increment; all moot given
KILL). **A:** 112 eval valid dates (= 91 new + 21 pre-harvested DJF issue dates);
182 distinct rollout issue dates = 91+80+13=184 raw − 2 bias∩calib shared
(2017-06-15, 2019-10-15). **B:** init ≤ context_end compliant by construction
(issue=context_end, hours 00-18 < predict_start) but NO explicit runtime assert —
recommended one-liner for the swap runbook.

Held at the report; no blend fitting (Matteo's verdict pending, though the mechanical
gate is a clean KILL). Task-1 freeze may proceed on 855076.

**Compute disclosure:** one local-CPU read-only computation (~2 min: downscale 28
existing DJF d14 extracts + pooled/per-cell ρ + 2000× date bootstrap). NO new Pangu
inference (R3(a) reused on-disk trajectories; pilot R3(b) not run), no GPU, no
pipeline/submission artifacts touched. Recon script in session scratch (read-only
scope; nothing written outside reports/ + scratch).

- 2026-07-20 / CC: d14 blend recon (reports/d14_blend_recon_20260720.md, commit
  5195f0d) and kill addition appended to MERGED_20260720. Compute: 28 existing DJF
  trajectories reused, no new inference. Dead end logged with reason (ρ = −0.36, CI
  below +0.28 bar).

<!-- ANCHOR: swap-runbook-20260721 -->
## 2026-07-21 — Swap re-run runbook + rehearsal + compliance guard

Deliverable `docs/swap_runbook_20260721.md`: mechanical swap-year re-run of the
frozen Task-1 pipeline (HEAD 5ab4be3, submission 855076). Full pipeline traced to
file:line (subagent) — Leg A (kit notebooks 1→2 + d14 clim + dir-residual + d7 bias/α
+ 360 fix → 835026) and Leg B (ERA5 fetch → Pangu 7-step rollout → splice → f=1.25 →
855076), with a frozen-vs-fresh table and per-step wall-clock (~4-4.5 h, rollout
dominates).

**Key reproducibility finding (R5):** the kit base downscaler `dn.train_downscaler`
has NO random_state (downscaling.py:43-44) and is retrained fresh each run → **d1
speed q05/q50/q95 jitter run-to-run and cannot be reproduced byte-for-byte** (the
original unseeded downscaler was never saved). d7 speed (pinned downscaler SHA
b68eb5fe), d14 speed (deterministic climatology), and all directions (seeded models)
ARE byte-reproducible. Rehearsal pass criterion: byte-identical → PASS; only d1 speed
within 3-dp floor → PASS-with-note; else FAIL. Optional determinism hardening
(random_state=42 on the kit downscaler) recommended for the swap but flagged as a
frozen-pipeline change needing Matteo's OK.

**R3 compliance guard** `scripts/tier2_swap_compliance_guard.py` (Clause 1: every
init ≤ context_end) — written, tested, **32/32 PASS** on the 2021 windows. **R5
rehearsal** `scripts/tier2_swap_rehearsal.py` — written; comparison engine self-tested
(855076 vs itself = byte-identical; 854984 vs 855076 = only d7 q05/q95 differ, q50
unchanged — cross-confirms f=1.25 scope). Rehearsal NOT run (multi-hour; Matteo
triggers).

**Cautions surfaced:** (1) Z: runtime tree (pinned downscaler pkl, extracts, ONNX
fallback) NOT mounted this session — must be restored + SHA-verified before swap or
the Pangu leg silently retrains unseeded; (2) ws_d7_set_chosen_alpha.py pins α=1.0,
must force 0.90 for the base; (3) ERA5 WB2 mirror covers 1959-2023 — if swap year ≥
2024, an alternative source is needed.

**Compute disclosure:** read-only pipeline trace (subagent) + two local sub-second/
~20 s checks (compliance guard on 2021 metadata; rehearsal compare-engine self-test
reading the two existing CSVs). NO Pangu inference, NO ERA5 fetch, NO submission
artifact touched (rehearsal written, not executed, per dispatch).

- 2026-07-21 / CC: swap runbook amendment (Matteo). (1) APPROVED random_state=42 on
  the kit downscaler (downscaling.py:43-51) for swap-day repeatability — deterministic
  going forward; does NOT reproduce 855076's original unseeded d1 (rehearsal expected
  PASS-WITH-NOTE on d1). (2) Fail-loud alpha=0.90 guards added: ws_d7_apply_bias_
  correction.py (asserts chosen_alpha==0.90, guarding the set_chosen_alpha 1.0 pin)
  and tier2_d7_build_submission.py (asserts splice ALPHA==0.90). (3) Runbook Step 0
  pre-flight added: Z: connect (human), downscaler SHA-verify, alpha check, CDS ERA5
  fallback for swap year >=2024 (set up account now). All three scripts py_compile OK.
  Rehearsal still gated on Matteo's go (not run). Compute: code edits + compile checks
  only, no inference, no submission artifacts touched.

- 2026-07-21 / CC: nested-repo correction. `phase_2/kit/` is a NESTED git clone of the
  organizer repo (<ORGANIZER_GH_ORG>/…, branch phase_2) — the main repo tracks ZERO kit
  files, so the approved downscaling.py seed was NOT in main commit 17ef3d8 (git
  silently skipped the nested path). Resolved: committed the seed in the kit repo
  locally (kit commit 0159ab9, NOT pushed to organizer) + backed it up as
  docs/kit_patches/downscaling_random_state_20260721.patch in main + runbook §0.4
  documents the nested repo and re-clone reapply step. First kit-code modification in
  the project (pattern to date: kit pristine, our code in scripts/). Flagged for Matteo:
  whether to de-nest/vendor the kit long-term. No inference; no submission artifacts.

<!-- ANCHOR: figb-pipeline-trace-20260721 -->
## 2026-07-21 — Fig B pipeline trace (Dim 3 flowchart): two provenance flags

Read-only trace of the d7-arm build (`scripts/tier2_d7_build_submission.py` producing
`submission_pangu_d7_allhours_20260719.csv`) and its imported machinery
(`scripts/tier2_d7_score_blocks.py`, `tier2_f2_d14_precheck.py`, `tier2_pangu_couple.py`).

**Flag 1 — d7 speed MOS provenance: the kit MOS models are BYPASSED, not reused-with-
Pangu-features and not retrained on Pangu.** The Pangu d7 speed path never touches a
MOS. In the build's per-block loop, the center is `spd_fine = S.downscaled((ctx["d7_u"],
ctx["d7_v"]))` (`tier2_d7_build_submission.py:131`) — the Pangu-coupled 125 m wind
downscaled directly — and the interval is `S.pg_interval(None, spd_fine, K_PANGU)`
(`:132`), a FIXED-ratio spread `q05=spd·(1−k·0.45)`, `q95=spd·(1+k·0.60)` with PG_LO/HI
0.55/1.60 (`tier2_d7_score_blocks.py:69-71`), then per-cell×season `S.build_bias`
(`:109`) and α-tighten (`:135-136`). No `mos`/`qmos`/`adj`/`coarse_fields`/`predict_mos`
appears anywhere in `tier2_d7_build_submission.py` (grep: none). The kit deterministic
u/v MOS + quantile-speed MOS survive in `tier2_d7_score_blocks.py` ONLY inside the A0
comparison arm (`a0_ctx`→`P.coarse_fields(mos,qmos,adj,…)` `:53-54`; `a0_interval`→
quantile-MOS `:60`), which is the SCORER's HRES baseline, not the shipped build. In the
final 855076 the frozen kit MOS (trained 2016-2020 on HRES) therefore drive **d1 speed
only** (base 835026); they are neither fed Pangu inputs nor retrained. → Flowchart:
d7-speed nodes = "Pangu 125 m wind → downscaler → fixed-ratio interval + bias + α",
NO MOS node; the MOS node is d1-only, "frozen (kit-trained on HRES)".

**Flag 2 — downscaler IS in the Pangu arm, but it is a DIFFERENT instance than the kit
baseline.** The Pangu 125 m wind lands on the coarse 45×57 = 2,565-cell grid
(`couple_global` returns (45,57), `tier2_pangu_couple.py:126-128`), then passes through
the coarse→fine downscaler: `S.downscaled` calls `dn.downscale(_DWN, coarse_uv[0],
coarse_uv[1])` (`tier2_d7_score_blocks.py:75-77`) → fine (479,433), subset `[ys,xs]` to
the 43,715 footprint (`tier2_d7_build_submission.py:133`). So it is NOT fine-grid-direct.
The instance is NOT the kit baseline's: `_DWN = get_downscaler()` (`tier2_d7_score_blocks.py:48`)
loads the PINNED cache `downscaler_blockexcl.pkl` (SHA `b68eb5fe…`, asserted
`tier2_d7_build_submission.py:55,100-101`), trained on `d2020_red` = 2020 target days
MINUS the eval-block±buffer exclusion set (`tier2_f2_d14_precheck.py:55-61`). The kit
baseline (835026) instead retrains a FRESH downscaler on the FULL 2020 sample
(`dn.train_downscaler(d2020,…)`, `2_downscale_to_target.ipynb` cell e5f5a269) each run,
uncached. Same architecture (`dn.train_downscaler` coarse→fine LightGBM), different
training subset → different weights. → Flowchart: d7 downscaler = "pinned block-excluded
instance (SHA b68eb5fe)", distinct from the base/d1 downscaler node.

**Compute disclosure:** read-only static trace (grep + file reads), ~2 min wall-clock,
no model loading, no inference, no GPU, no builds, no submission artifacts touched.

## 2026-07-23 — Bonus feasibility recon (read-only; quantile-bidding +15 pt)

**Scope:** read-only reconnaissance feeding Matteo's go/no-go on the
quantile-bidding bonus (decision his, before 2026-07-30). No builds, no
training, no inference, no data generation. Task 1 frozen at 855076
untouched. Only this log append written. Answers Q1–Q5 of the dispatch.

**Q1 — d1 coverage capability.**

(a) *HRES d1 coarse input on disk — CONTINUOUS 2016–2021* on the coarse
45×57 (=2,565-cell) grid, in two parquets read by `config.hres_parquets()`
(`forecast_hres.py::_load_hres` `:37-48`, keyed on `time` = issue date):
- `phase_2/phase2_dataset_ship/train/hres/north_sea_hres_2016_2018.parquet`
  — time 2016-01-01…2018-12-31, 1096 days, cols `fcst_{speed,dir}_d{1,7}_h{0,6,12,18}`.
- `phase_2/phase2_dataset_ship/train/hres/hres_north_sea.parquet` (the
  Phase-1 ship file, 296 MB) — time **2019-01-01…2021-12-31**, 1096 days,
  cols d1, d7 **and d10**. *This closes the "HRES 2019–2020 absent" gap in
  the 2026-07-02 inventory memory:* the Phase-1 file supplies 2019–2021 d1
  coarse forecasts. So d1 HRES input exists for every year 2016–2021.

(b) *AROME truth on disk (u125m/v125m, needed for realized production, NOT
for producing forecasts):* `train/arome/{2016..2020}` only — 366/365/364/
365/366 daily .nc (2018 missing 2018-08-07; **no 2021 dir**). Coarse twin
`train/arome_coarse125/{2016..2020}` same counts. Daily reanalysis
`train/reanalysis/{2016..2020}` complete (no 2021); `reanalysis_extra`
(PL/SL) 2016–2018 only. **2021 AROME truth is withheld** (it is the scored
eval year) → 2021 cannot be scored / cannot supply realized production.

(c) *Model readiness — frozen d1 artifacts produce output for arbitrary
dates WITHOUT retraining.* d1 speed is the plain kit path (MOS → conformal
→ downscaler); d1 direction is the residual model. Neither is retrained per
date. The engine core is date-agnostic: `forecast_pipeline.coarse_fields(
mos,qmos,adj,issue_date)` `:60-62` and `downscale_window` `:99-121` take any
issue date; d1 speed reads only the HRES parquet row for that date
(`forecast_hres.py:119,138-139`, `with_truth=False` → no AROME needed to
produce forecasts) + `arome_static.nc` terrain. **Hardcoded-window pins
(would need a new driver loop, NOT retraining):** `splits.py:22-26`
(TRAIN_YEARS 2016-2020, EVAL_PUBLIC_YEAR 2021), `splits.py:67-171`
(`_BASE_WINDOWS` = 8 hardcoded dicts; `eval_windows(year)` is
year-parameterizable but only re-dates the same 8 shapes, non-leap
assumed `:50-51`); notebook drivers loop `for wi in range(len(windows))`
(`1_predict_target.ipynb:183-187`, `2_downscale_to_target.ipynb:192-195`);
splice scripts pinned to 96 blocks / 8 windows (`apply_d14_climatology.py:
124` `for window in range(8)`, `:91` asserts `8*3*4`; `dir_residual_apply.py:
53,164,177`; `tier2_d7_build_submission.py:66-67,122-125`); datemaps
`artifacts/tier2_sub_datemap.json` (8 issue dates). None of these are
retraining; a full-year run needs a new date-loop over the frozen models.

(d) *Feasible years.* Both HRES d1 input AND AROME truth AND daily
reanalysis present → **2016, 2017, 2018, 2019, 2020** (2018 = 364/365,
99.7%; others 100%). **2021 has d1 input but no truth and no daily
reanalysis → infeasible for a scored / production-closing simulation.**
KEY CAVEAT (surfaced, not resolved): the frozen d1 model was
trained/calibrated on 2016–2020 — quantile-MOS on years <2020
(`forecast_pipeline.py:47`), conformal calibration on 2020 (`:46`), center
MOS + `spd_infl` pooled over 2016–2020, and the base/d1 downscaler retrains
on the full 2020 target each run (`2_downscale_to_target.ipynb` cell
e5f5a269, uncached, no seed → d1 speed jitters on rerun). **Therefore every
truth-bearing year is in-sample; there is NO year that is both
out-of-sample and has AROME truth on disk.** 2021 is the only genuine
out-of-sample year but lacks truth. Recommended primary candidate if an
in-sample run is acceptable: **2019** (100% coverage; not the 2020
conformal-calib year, though still in the qmos center-MOS training set).
Whether in-sample forecast quality is acceptable for the bidding sim's
economic question is Matteo's call.

**Q2 — cost (most feasible year, full 43,715-cell grid).**
Anchor: the 2026-07-08 d1 bias diagnostic ran **365 issue dates × 4 init
hours over the full grid in ~30 min wall-clock** (`swnd`, single machine,
no GPU) — LLM log 2026-07-08 "d1 speed bias diagnostic", ~identical shape
to a full-year d1 generation. So **wall-clock ≈ 30 min** (generation only;
truth-compare was included in that 30 min, so pure generation is ≤ that).
Disk: full-grid full-year d1 = 365×4×43,715 ≈ 63.8 M rows; in submission
CSV format (~80 B/row, from the 338 MB / 4.2 M-row shipped submission) ≈
**~5 GB CSV**, or ~1–2 GB as float32 parquet. C: has **only ~15 GB free**
→ a 5 GB CSV is a real dent; parquet or subset-only output (Q3) avoids
pressure. Compute fits easily alongside the swap rehearsal (rehearsal is
read-only, no inference) — no compute contention; disk is the only watch item.

**Q3 — simulation scope reduction (need only cell-63 farm ≈ 55 turbines).**
(a) The downscaler is **structurally all-or-nothing** without code edits:
`downscaling.downscale(models,cu,cv)` `:152-165` predicts over the full
static seamask (`_sea_flat()` `:121-122` = whole `arome_static.seamask`,
no subset arg) and returns the full (479,433) field; footprint (43,715) is
applied only at row emission (`build_forecast_submission.py:50-51`), also
whole-mask. No parameterizable point set. Subsetting the downscaler would
require editing the seamask/mask source (a code change to frozen kit
modules — out of scope for a freeze-safe run).
(b) Revised estimate: because full-grid d1/year is already only ~30 min,
subsetting the *compute* is unnecessary. The right move is to generate
full-grid in memory and **write only the cell-63 subset to disk** (~55 pts
× 365 × 4 ≈ 80 K rows ≈ a few MB) — same ~30 min compute, negligible disk,
no edits to frozen modules.

**Q4 — Netherlands / TenneT imbalance settlement scheme (web).**
NL/TenneT uses a **hybrid, predominantly single-price** scheme, NOT a
systematic two-price system: single imbalance price in regulation states
−1, 0, +1 (system long / balanced / short); **dual (two-price) settlement
only in regulation state 2** — when the TSO activates both up- and
down-regulation within the same 15-min ISP/PTU. This structure held across
2020 and 2021: ACM approved continuation of dual pricing, and TenneT
confirmed continuation at the BAS workshop 2021-05-19 (to be finalized by
2021-07-15). No scheme *change* found inside the candidate years 2016–2021;
the later material change (PICASSO/aFRR effect on imbalance prices) is
post-2022, and a scarcity component is slated for 2028 — both outside the
candidate window. **Ambiguity for the analytic bid formula (surfaced, not
resolved):** the two-price newsvendor result assumes systematic dual
pricing; NL 2020/2021 is single-price in the large majority of ISPs with
dual pricing only in R2. So the analytic two-price formula does **not**
apply directly to NL — the one-price variant is the more representative
baseline, with R2 dual-pricing as a correction. Which formula to adopt is
Matteo's call; a fully authoritative per-year statement should be read
from TenneT's own imbalance-settlement documentation. Sources: ACM
dual-pricing approval (acm.nl), TenneT settlement/balancing pages
(tennet.eu), ENTSO-E single-pricing standardization objective; regulation-
state mechanism corroborated across secondary summaries.

**Q5 — freeze-compliance statement (plan only; nothing created).**
A greenlit full-year d1 run would READ only frozen artifacts (the two HRES
parquets, `arome_static.nc`, the frozen kit MOS/quantile/conformal models,
the base/d1 downscaler as retrained-in-memory per kit design, and — for
scoring — AROME truth 2016–2020) and WRITE only to a **new top-level
directory `bidding_sim/` (proposed subdir `bidding_sim/d1_fullyear_2019/`),
untracked, outside every runbook path.** It touches none of:
`submission.csv` / `phase_2/kit/phase_2/part1_forecast/submission.{csv,zip}`,
`scripts/artifacts/` (frozen tables/downscaler cache/datemaps), `models/`,
or `phase_2/phase2_dataset_ship/inference/window_*/` — the swap
compliance guard only globs `window_*/metadata.json`
(`tier2_swap_compliance_guard.py:38`), which `bidding_sim/` does not
contain. It would require a NEW date-loop driver (per Q1c) over the frozen
models; no frozen kit module is edited.

**Compute disclosure:** read-only trace (grep + file reads + 1 read-only
parquet schema/date-range probe via a scratchpad script `inspect_hres.py`,
outside the repo) + 2 web searches. ~15 min wall-clock, no GPU, no model
loading, no inference, no builds, no submission artifacts touched. No
pipeline state modified; Task 1 remains frozen at 855076.

**Report sync:** recon feeds a decision, not the report; no
`phase2_report_material_*` entries touched. Mark `[REPORT: n/a]`.

## 2026-07-23 — Bonus build Stage 1 COMPLETE (d1 full-year 2019, cell-63 farm)

**Scope:** Stage 1 of the quantile-bidding bonus, authorised by Matteo
(decision "C", plan approved this date; Stages 2–5 PARKED until explicit
post-return trigger). New files under `bidding_sim/` only. Task 1 frozen at
855076 — untouched. Read-only on all frozen artifacts.

**What ran:** `bidding_sim/generate_d1_2019.py` (seeded, committed,
cold-runnable). Recipe verified against the kit, not memory:
- MOS center / quantile-MOS / conformal `adj` / `offs` **LOADED** from the
  shipped cache `phase_2/kit/phase_2/part1_forecast/cache/coarse_forecasts.pkl`
  (SHA-256 `3a4e4538…c80278`, 2026-07-06) — the exact shipped d1 model set
  (the d1 producer `2_downscale_to_target.ipynb` pickle.loads it and never
  refits the MOS). SHA pinned + asserted fail-loud in the script. No
  `fit_forecast`, no MOS re-seeding.
- Downscaler **REFIT** `dn.train_downscaler(2020[::5])` = 74 days, seeded
  `random_state=42` (downscaling._LGBM, 2026-07-21). Reproducible but NOT
  byte-identical to 835026's original unseeded downscaler — disclosed in the
  output MANIFEST and the [[phase2 recon]] entry above. Not persisted from the
  ship build, hence refit.
- `spd_infl`/`dir_off` **RECOMPUTED** via `P.calibrate_intervals`
  (deterministic; AROME truth 2016–2020): `spd_infl={1:1.067, 7:4.747,
  14:12.782}`; d1 widening k used = **1.067** (consistent with the
  well-centred d1 finding, ≈1.0).
- Date-loop: valid days 2019-01-01…2019-12-31, issue = valid−1, 4 init hours,
  **lead 1 only** downscaled; direction quantile *model* skipped (speed-only
  sim), free deterministic `dir_50` retained.

**Subset:** farm centre (52.5 N, 3.0 E), Task-2 layout (55 × IEA 22 MW, cell
63); footprint points within ±0.12° lat / ±0.18° lon = **385 points**
(deliberate superset of the ±7.5 km box; aligned to the kit's canonical
`np.where(mask)` footprint order, verified).

**Output** (`bidding_sim/d1_fullyear_2019/`):
- `d1_quantiles_cell63.parquet` — **562,100 rows** (385 × 365 × 4), cols
  point_id, lat, lon, valid_time, hour, spd_q05/q50/q95, dir_50. SHA-256
  **`ed84693a4d68938047565f8576e688f235c7de3432a2750cd7eb97b043558f7b`**.
  Archive `.parquet.zip` + `.sha256` written.
- `MANIFEST.md` — provenance + mandatory D4.1 in-sample disclosure verbatim.

**QA:** q05≤q50≤q95 on all rows (0 violations); 0 missing issue dates (both
HRES parquets concatenate cleanly across the 2018/2019 boundary); no NaNs;
speeds physically sane (q50 median 9.5 m/s, mean PI width 7.33 m/s); correct
North Sea seasonality (monthly mean q50: Jul 7.2 → Dec 12.9 m/s); dir_50
median ~220° (SW). 

**Mandatory disclosure (D4.1, carried in MANIFEST):** 2019 is in-sample to
the frozen model (trained/calibrated 2016–2020); no year is both
out-of-sample and truth-bearing in the provided data — the quantile-strategy
gain will be an upper-bound estimate.

**Compute disclosure:** one generation run, `swnd` env, single machine, no
GPU. **14.2 min wall-clock** (well under the 30-min recon anchor and the
60-min >2× hard-stop; faster because only lead 1 is downscaled). Plus ~1 min
read-only validation. No external API calls. No frozen artifact modified:
no touch of `submission.csv`, `part1_forecast/submission.{csv,zip}`,
`scripts/artifacts/`, `models/`, or `inference/window_*/`.

**Status:** Stages 2–5 (production conversion, ENTSO-E market data, bidding
strategies, outputs) remain PARKED pending Matteo's post-return trigger
(target 2026-08-18/19) and the ENTSO-E API token.

**Report sync:** Stage-1 build artifact, not report prose; no
`phase2_report_material_*` touched. `[REPORT: n/a]`.

## 2026-07-23 — Opus session (checkpoint relay from checkpoint_update_20260723.md)

- 2026-07-23 / Opus: session open per reading list (six files +
  report-sync grep). Bonus recon dispatched to CC and results
  processed (5 questions, all answered, two forks surfaced).
  Decisions: bonus GO (2019, Option B, Option C split). Stage 1
  pickled-artifact question raised → CC confirmed shipped models
  in coarse_forecasts.pkl → Stage 1 plan confirmed and executing
  (loaded MOS, refit downscaler seeded, full disclosure). Three
  further CC dispatches written (SHAP, compute accounting, ablation
  verification) — parked for Fri 24 relay. Validation-protocol fork
  closed: include leaky-split as removable box. Report assembly
  design committed: standalone sections, cold-reader pass. Fig C
  (Winkler stock-and-flow) promoted to report. EVPI/EVIU verified
  against primary attributions (Raiffa & Schlaifer 1961). Schedule
  hard-cut through the July hard-cut. My employer's work completed except
  Tue meeting.
- Compute disclosure: claude.ai session, text + tool calls (web
  search ×3, file reads, past-chat search ×0). CC bonus recon was
  read-only, logged by CC. CC Stage 1 executing (~45 min estimated,
  logged separately).

## 2026-07-24 — SHAP explainability (Dim 5 deliverable)

**Scope:** produce the Dim 5 (Responsible & Transparent AI) SHAP
explainability deliverable from the frozen Task-1 learned models. New files
under `explainability/` only (top-level, untracked, outside runbook globs —
swap guard globs only `window_*/metadata.json`, verified). READ-ONLY on all
models (loaded, never refit). Task 1 frozen at 855076 — untouched. No
submission/runbook artifact modified.

**Method:** TreeSHAP (`shap.TreeExplainer` 0.51.0, tree_path_dependent — exact,
no background set). Per-quantile on the quantile models. Sample = held-out
val split (`train_val_dates(seed=42)`), season-stratified, **N=20,000/model,
seed=42**. Feature engineering + feature lists copied verbatim from the
training/kit source (no improvised loaders; boosters loaded via
`lgb.Booster(model_file=...)`, kit MOS from the pinned
`coarse_forecasts.pkl`, SHA `3a4e4538…` asserted).

**Priority 1–2 — direction residual d1 & d7** (`explainability/
shap_direction_residual.py`), 6 boosters, 10 features. Key result: **the
interval-width quantiles use different features than the median.** Width
(q05/q95) is dominated by `hres_speed` (mean|SHAP| d1 16.9/18.2°, d7
17.4/17.3° — far above all others), i.e. low wind speed → larger directional
uncertainty → wider arc. The median (q50) is instead driven by the HRES
direction features (`hres_dir_sin/cos`) plus `lon` (d1) and seasonality
(`month_*`, stronger at d7). Full mean|SHAP| top-5 per model + cross-quantile
rank table in `explainability/SHAP_findings.md`.

**Priority 3–4 (stretch) — kit MOS d1** (`explainability/shap_kit_mos_d1.py`),
7 features, sample built via `fh.build_hres_table(150 seeded train-year issue
dates, lead 1, with_truth=False)`. Center u/v each key on their own forecast
component (`fcst_u`→u 5.17, `fcst_v`→v 5.41; sanity pass). Contrast with the
direction arm: the **speed quantile MOS uses the same dominant feature
(`fcst_speed` ≈3.4–3.6 m/s) across all three quantiles** — intervals scale
with forecast speed rather than switching feature basis. Facts in
`explainability/SHAP_findings_kit_mos.md`.

**Architecture finding (Dim 3/5, stated as finding not gap):** the shipped
architecture concentrated learned complexity in the direction arm; the d7/d14
speed arms are deliberately model-free (d7 = Pangu fixed-ratio shell +
scalars, d14 = climatology lookup), so there is no per-forecast learned speed
model to explain for those horizons.

**Outputs (`explainability/`):** 11 beeswarm PNGs (no baked-in titles — report
captions carry them), 11 `shap_values_*.parquet` (values + feature values +
base value) each with a `.sha256` sidecar, 2 archive zips
(`shap_values_direction.zip`, `shap_values_kit_mos.zip`), 2
`shap_importance_*.json`, `SHAP_findings.md` + `SHAP_findings_kit_mos.md`, 2
committed seeded scripts. Interpretation drafting left to Opus (facts +
numbers only in findings).

**Compute disclosure:** `swnd` env, single machine, no GPU. Direction run
(load 14.6M-row val split + 6× TreeSHAP on 20k) ≈ a few min wall-clock; kit
MOS stretch (build_hres_table 150 dates + 5× TreeSHAP) < 2 min. Well under
the 30-min/model budget (TreeSHAP on 300-tree depth-7 models is seconds). No
external API calls. No model retrained or modified.

**Report sync:** the width-vs-median feature split and the model-free-speed-
arm finding are report-worthy → `[REPORT: pending]` (Dim 5 explainability +
Dim 3 architecture). Interpretation prose is Opus-side.

## 2026-07-24 — Ablation table verification (Dim 3 deliverable)

**Scope:** READ-ONLY verification pass on the candidate ablation ladder
before Opus assembles the Dim 3 ablation table. LAW ZERO applied to our own
record: every candidate number checked as (a) real, (b) quoted correctly,
(c) computed on a comparable protocol. No re-scores run; every number verified
against an existing on-disk artifact. Task 1 frozen at 855076 — untouched. No
pipeline/submission artifact modified. Deliverable:
`docs/ablation_verification.md` (new, committed with this append).

**Five discrepancies found (the success condition of the pass):**
1. Rung 0 mislabels submission 835026 (ours, WS d7 27.14 — already carries the
   k=30 bias, d14 fix, dir residuals) as the "kit baseline." As-shipped kit WS
   d7 = 29.8 (organizer) / 29.53 (our clean kit run). Src `hunt_record_task1.md:22-24,82`.
2. "A0" overloaded: rung-0 kit (29.8/29.53, board) vs A0 = HRES-MOS four-block
   arm (32.70, dev, `tier2_d7_fourblock_20260719.md:27`) vs 835026 board 27.14 —
   three distinct numbers.
3. Rung 3 (+31.0%) already contains the per-cell bias + spd_infl recal
   (identical machinery both arms, `tier2_d7_fourblock_20260719.md:12-18`); it
   is the driver-swap effect, not raw bypass — so rung 4 double-counts it.
4. Rung 4's −28.1% belongs to the HRES-MOS `spd_infl` recal that was
   board-REJECTED (+0.51 worse, `LLM_AGENT_LOG.md:946-950`), not a shipped
   component; and its cited k=1.846 is the Pangu arm's multiplier, unrelated to
   the −28.1% (which lived at k=1.7/1.9). Mis-attributed.
5. Rung 5 coverage "0.84→0.91" is the dev four-block sweep, not the board
   recovery (board 855076: coverage_d7 74.6→85.5%, WS d7 25.32→24.7543 = −2.2%,
   `checkpoint_update_20260720b.md:9-11`). Dev-Winkler optimum was f=1.0, not
   f=1.25 (`tier2_d7_widthcal_sweep.json`); f=1.25 was a board-transfer hedge.

**Also flagged (open, not adjudicated per LAW ZERO — both records reported):**
+15.8% vs 15.9% center-RMSE full-precision reconciliation Opus already flagged
(`checkpoint_update_20260720.md:64-66`); F1 coupling floor quoted as ~1.1 m/s
(`tier2_d7_fourblock_20260719.md:55`) vs ~1.76 m/s residual
(`LLM_AGENT_LOG.md:1959-1960`); the "~6% quadrature" is a d14 error-variance
fraction (ARM-D14 kill), not the d7 F1 floor. Supporting claims verified:
d14 blend ρ=−0.358 CI[−0.544,−0.118] (DJF single block), Dir MAE 15.16/55.87
(dev validation, not board), center RMSE +15.8% (dev four-block).

**Recommendation:** two tables — Table A board per-dimension (kit→shipped),
Table B dev four-block driver isolation (WS d7 only), clearly labeled as
non-comparable protocols. Do not present the −28.1% recal as a shipped rung.

**Compute disclosure:** read-only. Grep + file reads only (repo markdown, JSON
artifacts, one script). No env activated, no model loaded, no scoring, no GPU,
no external API. One file written (`docs/ablation_verification.md`) + this
append. Handoff to Opus for table assembly / prose.

**Report sync:** verification feeds the Dim 3 table assembly (Opus-side); the
discrepancy list above is the actionable output. No `[REPORT: pending]` flag —
this is a QA pass on numbers already tracked, not a new result.

## 2026-07-24 — Dim 5 compute & carbon accounting (Parts A–D)

**Agent:** CC, builder/executor per contract §2. **Task scope:** read-only on
pipeline state (Task 1 FROZEN at 855076, untouched); new outputs to top-level
untracked `compute_accounting/`. Two small sample workloads (Part B) were the
only new compute; approved by Matteo 2026-07-24 with the two method
substitutions below.

**Two dispatch assumptions broke on this platform, escalated + approved:**
1. Part B specified Linux RAPL (`turbostat`/`/sys/class/powercap`). Host is
   **Windows 10**; RAPL MSRs need a signed kernel driver (no Power Gadget /
   LibreHardwareMonitor / HWiNFO; no admin in this session). → **CPU watts are a
   TDP-anchored ESTIMATE (PL1=45 W), not a measurement**, idle-corrected, with a
   bracketed range (37 W lower → 67.5 W = 1.5× whole-system upper). GPU watts
   *were* measured (`nvidia-smi`) but the GPU was unused in production.
2. Pangu `pangu_weather_24.onnx` not resident (was on `Z:`). → **not re-run**;
   per-step cost from the logged tier2 smoke (43–46 s/step, L1917).

**Part A (records aggregation).** Phase 2 = **≈ 42 CPU-h logged**, CPU-only
(RTX A4000 present but UNUSED, L1917/L1924). Categories: Pangu inference 27.1 h
(L2129/2149/2216/1999) · LightGBM training 6.3 h (L311/1556/1884) · downscaler
training 0.1 h (L2001) · scoring/diagnostics ~7.0 h (L446/544/664/2150) · data
processing ~1.5 h (L206/365). Split: **81 % inference/eval, 15 % training, 4 %
data-prep.** Pangu arithmetic reconciled: 182×7×43 s = 54,782 s vs 65,623 s
logged = **+19.8 %** (per-date ERA5 I/O; under the 20 % flag). Untracked
exploration bounded at **≤15 %** (42 sessions × ≤10 min interstitial) → total
≤ ~48 CPU-h; logged is a lower bound.

**Part B (measured, `compute_accounting/measure_workloads.py`, seeded).** Idle
CPU 19.4 %, load CPU **97.8 %** (near-saturation → justifies full-PL1 anchor);
GPU flat ~20 W throughout (idle, unused). LightGBM d1 exact-PARAMS sample:
1.5 M-row seeded sample of 29.2 M, 3 quantiles in 38.3 s. Sample model saved as
measurement artifact only; frozen production `.lgb` untouched.

**Part C.** Energy **≈ 1.6–2.8 kWh**; at NS grid **660 gCO₂e/kWh (2022, CER/
ECCC NIR 1990–2022)** → **≈ 1.0–1.9 kgCO₂e** (≤ ~2.2 with untracked). Canada-avg
100 gCO₂e/kWh sensitivity → 0.16–0.28 kgCO₂e (grid, not compute, dominates).
Method per Lacoste et al. 2019; CodeCarbon named but not used (frozen pipeline +
Windows-RAPL gap; retro-instrumentation would cost >27 CPU-h of re-runs).

**Part D (reconstruction, not measurement).** Phase 1 (2026-05-16→06-24, ~5.5 wk,
CPU-only boosting kits + hybrid calibration): **≤ ~40 CPU-h, ≤ ~1.8 kgCO₂e**,
±1 order of magnitude. **Program total ≤ ~90 CPU-h, ≤ ~4 kgCO₂e.**

**Compute disclosure (this accounting).** One local Part-B measurement run
(~155 s wall, ~42 s CPU-saturating) + hardware-probe shell commands (<60 s).
Total new compute **< 0.05 CPU-h → < 0.005 kWh → ~1–2 gCO₂e** — negligible vs
the ~42 CPU-h accounted, itemised for completeness. No GPU. No pipeline touched.

**Outputs:** `compute_accounting/{compute_table.md, measurement_notes.md,
report_paragraph.md, measure_workloads.py, measure_samples.csv,
measure_summary.json, sample_lgbm_d1_q50.txt}`.

**Report sync:** `[REPORT: pending]` — Dim 5 environmental-cost section;
`report_paragraph.md` is the verified-numbers feed, prose assembly Opus-side.

---

## [LOG: 2026-07-27, CC dispatch 2026-07-27 Part 2 — two flag-closure checks (READ-ONLY)]

**Task:** Close two flags carried open in the report prose: (1) the day+7 centre
RMSE 15.8% vs 15.9% full-precision reconciliation, open since
`checkpoint_update_20260720.md:64-66`; (2) the F1 coupling-floor magnitude,
quoted ~1.1 m/s in `reports/tier2_d7_fourblock_20260719.md:55` vs ~1.76 m/s in
`LLM_AGENT_LOG.md:1959-1960`.

**Mode: READ-ONLY. Artifacts touched: none.** No pipeline stage re-run, no
submission, model, or frozen artifact modified. Compute: arithmetic on stored
JSON values only (< 1 s CPU); no compute disclosure warranted. Sources read:
`scripts/artifacts/tier2_d7_fourblock.json` (a8872c5, sha256 046d7c23…c00af1),
`scripts/artifacts/tier2_f1_coupling.json` (2ae9a3d, sha256 b0f4eb2a…7b9a53fa3),
`scripts/tier2_d7_score_blocks.py`, `scripts/tier2_f1_coupling_check.py`,
`scripts/artifacts/f1_run.log`, `reports/tier2_d7_fourblock_20260719.md`,
`reports/tier2_night1_20260718.md`. Both JSONs verified identical to their
commits (`git diff` empty).

**Check 1 — centre RMSE, RESOLVED, published figure stands.** Full stored
precision: A0 pooled `center_rmse` = **5.5952**, Pangu-d7 = **4.7094**.
Improvement = (5.5952 − 4.7094)/5.5952 = **15.831%** → **+15.8%**. The JSON's own
stored field agrees (`pangu_vs_a0_center_rmse_pct` = 15.83). The 15.89% came from
recomputing on the report's 2-dp *display* values (5.60, 4.71) — a display-rounding
artifact, not a source discrepancy. Note: the scorer applies `round(…, 4)` at write
time (`tier2_d7_score_blocks.py:187,192-193`), so 4 dp *is* the full stored
precision and the raw float64 is not recoverable without a rerun; this does not
affect the answer — over the whole 4-dp rounding interval the improvement is
bounded to 15.8298–15.8331%, rounding to 15.8% throughout. **No figure moves;
`HANDOFF_20260719.md` §1, `ADDITION_report_material_20260719.md` Addition 1 and
the merged ablation tables need no change on this flag.**

**Check 2 — F1 floor, RESOLVED, one source document is mislabelled.** The two
figures are different quantities, not two estimates of one: **−1.1377** is the
pooled *speed bias* (mean signed error, correctable by the per-cell×season bias
table); **1.76** is the *de-biased residual RMSE*, derived by quadrature from the
two stored pooled values — `sqrt(2.0968² − 1.1377²) = 1.7613` — and not itself a
stored field. `round(|−1.1377|, 1) = 1.1`, so **−1.14 is confirmed as the origin
of the 1.1**; no other F1 quantity rounds to 1.1. The floor the +15.8% is net of
is **1.76 m/s**, since both four-block arms run the mean-only bias table and the
bias component is therefore already absorbed. `reports/tier2_night1_20260718.md:13-14,35`
and `LLM_AGENT_LOG.md:1959-1960` are both consistent with the artifact;
**`reports/tier2_d7_fourblock_20260719.md:55` is wrong** — it attaches the word
"floor" to the bias magnitude. A labelling error, not a numerical disagreement.

**Kept separate (not conflated):** the ~6% coupling figure is a **d14** quantity
from the ARM-D14 kill pre-check (7.49² − 1.76² = 53.0025 vs 2 × 5.30² = 56.1800,
ratio 0.9434) — the F1 floor as a fraction of *d14* error variance. Unrelated to
the d7 flag and unchanged by the above.

**Stop conditions:** neither triggered. Both figures recoverable from committed
artifacts; nothing reconstructed from memory or rerun.

**Deliverable:** `docs/flag_closure_20260727.md` (facts only, no interpretation,
no recommendation, no document edits made).

**Report sync:** `[REPORT: pending]` — Check 2 requires an Opus-side correction to
`reports/tier2_d7_fourblock_20260719.md:55` and to any report prose inheriting the
"~1.1 m/s floor" phrasing. Check 1 requires no report action. CC made no edits.

**Follow-up, same date (2026-07-27), authorized by Matteo after the read-only
pass:** the "Artifacts touched: none" scope above applies to the *checks*; it no
longer describes the session. On Matteo's instruction an errata note was appended
to `reports/tier2_d7_fourblock_20260719.md` (§errata-d7-f1floor-20260727), matching
the `phase2_report_material` errata convention (§errata-d7-datecount-20260720).
Additions-only: 43 lines added, 0 deleted (`git diff --numstat`); line 55 and all
existing prose verified unchanged; note closes with "Original prose unchanged
(additions-only)." The note states what line 55 says, distinguishes the pooled
speed bias (−1.1377, correctable, origin of the "~1.1") from the de-biased residual
RMSE (1.76 = sqrt(2.0968² − 1.1377²), the floor), records that the +15.8% center
gain is net of 1.76 because both arms run the same mean-only bias table so the bias
is absorbed identically in each, and cites `reports/tier2_night1_20260718.md:13-14,35`
and `LLM_AGENT_LOG.md:1959-1960` as labelling the pair correctly. No measured value
changed; labelling only. Still no pipeline stage run, no model, submission, or
frozen artifact touched. Committed with the flag-closure evidence.

---

## 2026-07-29 (Wednesday) - CC session: ENTSO-E API token smoke test (read-only)

Anchor: `entsoe-token-smoketest-20260729`.

**Scope as given:** read-only, five minutes, no repo changes. Confirm that the
credential held in the `ENTSOE_API_TOKEN` environment variable authenticates
against the ENTSO-E Transparency Platform. Explicit instruction: never print,
log, or write the token value. Explicit instruction: on failure, report the
exact error and stop, without retrying with the token inline.

**Environment change (the only state change made):** `entsoe-py` was absent
from the `swnd` conda env. Installed `entsoe-py==0.8.0` with
`conda run -n swnd python -m pip install entsoe-py`. Every dependency was
already satisfied (pandas, requests, beautifulsoup4, numpy 2.4.6); nothing in
the env was upgraded, downgraded, or removed. This install lives outside the
repo.

**Call executed.** Passed to the `swnd` interpreter over stdin rather than
saved as a script, to honour the no-file instruction:
`EntsoePandasClient.query_day_ahead_prices("NL", start=2019-01-15 00:00
Europe/Amsterdam, end=2019-01-16 00:00 Europe/Amsterdam)`.

**Result: the call succeeds.** Returned a pandas Series, shape (25,), dtype
float64, `Freq: h`.

- first index 2019-01-15 00:00:00+01:00
- last index 2019-01-16 00:00:00+01:00
- first three values 46.67, 42.00, 38.30 EUR/MWh

25 rows, not 24: the `end` bound is inclusive, so the window returns the 24
hours of 2019-01-15 plus the 00:00 stamp of 2019-01-16. Any later use must drop
the trailing stamp or pass `end` at 23:00, or hourly counts will be off by one.

**Credential handling.** The token was read from `os.environ` into a local
variable and passed to the client constructor only. It was never printed,
echoed, interpolated into a command line, written to any file, or recorded in
this entry. Presence was confirmed without displaying the value.

**Artifacts touched:** none, other than this log addition. No script, notebook,
data file, or submission was created, edited, or deleted; no commit was made;
no pipeline stage was run; no model or frozen artifact was touched.

**Compute disclosure.** Negligible. One pip install (1.6 MB wheel) and one
HTTPS GET against the ENTSO-E Transparency API. No training, no inference, no
GPU, well under one minute of CPU.

**Not decided. Flagged for Matteo and Opus.**

1. Scope. A working ENTSO-E credential is a capability, not yet a wired Phase 2
   data dependency. No part of Task 1 or Task 2 reads day-ahead prices. Clause
   added by Matteo on review, 2026-07-29: the bonus deliverable does read them,
   and it is worth 15 points, so the credential has a scored home even though
   the scored tasks do not touch it. The manifest requirement stands and Matteo
   has confirmed it will be honoured at Stage 2: a pinned entry carrying
   version and SHA-256 before any pipeline reads price data (contract A1). No
   price data has been read into the repo by this session.
2. Timing. This session runs the day before the July 30 gate, with the
   2026-07-31 to 2026-08-17 no-computing wall immediately after. New scope
   should not be opened against that gate.
3. Retention drift, pre-existing, untouched by this session:
   `checkpoint_update_20260720.md` and `checkpoint_update_20260720b.md` remain
   tracked alongside the current `checkpoint_update_20260727.md`, against the
   retention rule adopted 2026-07-16 (latest checkpoint only). Removing tracked
   files is not a CC decision; recorded here as an observation. Matteo's answer
   on review, 2026-07-29: leave it; the question resolves once the pending
   checkpoints are pushed. No untracking performed.

**No new `checkpoint_update_*.md` was written.** CC's first reason was that
rotation would spend `checkpoint_update_20260727.md`, whose "Next" section
(Dim 3 criterion 2, then criterion 5) read as the live plan. Corrected by
Matteo, 2026-07-29: that state is already spent and the working repo is two
days behind project knowledge. Criteria 2 and 5 were drafted 2026-07-28 and
merged into `phase2_report_material_MERGED_20260729.md` as
`§design-choices-and-physics-20260728` and `§innovation-20260728`; Dim 4 §§5-9
and the revenue estimate were migrated 2026-07-29. Matteo will push
`checkpoint_update_20260728.md`, `checkpoint_update_20260729.md`,
`dimension_gap_list_20260729.md` and
`phase2_report_material_MERGED_20260729.md`; after that push the latest
checkpoint describes actual state and the retention question resolves itself.
Standing instruction from Matteo: write no checkpoint file and rotate nothing
until then. Whether this smoke test also wants a checkpoint file rather than
this log entry alone is to be raised after the push. This session is therefore
checkpointed here only, by instruction, not by CC's own judgement of the
retention rule.

**Report sync:** no report-worthy result. This session marks nothing
`[REPORT: pending]`. The three pre-existing `[REPORT: pending]`
compute-disclosure hits in `checkpoint_update_20260720.md:59,79` and
`checkpoint_update_20260720b.md:75` were raised by CC and ruled Opus-side by
Matteo: they are Dim 5 criterion 1 report prose, already tracked in the gap
list. CC is not to fix, untrack, or resolve them, and has not.

<a id="cc-20260818-two-cell-check"></a>
## 2026-08-18 - S0 repo reconciliation, governance commit, and S2 two-cell check

**Agent:** CC (Claude Code), role = builder/executor per contract §2.
**Prompt / authority:** "CC DISPATCH, swap re-run on the new inference set +
read-only two-cell check" dated 2026-08-18, blocks S0, S2, and the S1 gate.
Matteo authorized the governance commit and push in-session after CC reported S0.

**S0, repo state reconciliation (read-only).** Settled the `[U]` housekeeping item
on whether the 2026-07-29 documents were pushed before departure. Finding: they
were not, and no document *file* dated 2026-07-29 exists in this repository, on
disk or in any commit on any branch. A 2026-07-29 CC session did occur and did
write to the record: the ENTSO-E API token smoke test entry, anchor
`entsoe-token-smoketest-20260729`, sits in `LLM_AGENT_LOG.md` as an uncommitted
188-line addition, discovered while appending this entry. The pushed record
stopped at 2026-07-24; the four 2026-07-27 documents and that 2026-07-29 log
addition were written to disk and never committed. `origin/main` and local `main` were
level at `37fc7f8` (fetch-verified 2026-08-18). Banked Task 1 artifact
`scripts/artifacts/submission_pangu_d7_f125_20260720.csv` verified present,
SHA-256 `169524f9ffa5c3e53f66a4d7f686299b1344c9f303040a68e610ed0816b33079`,
matching the expected value. `data/MANIFEST_zenodo_20335351.md` present, tracked
and clean.

**Governance commit `ac5ecaa`** (Matteo-authorized). Added the four 2026-07-27
documents plus `docs/session_handoff_20260810.md`,
`docs/session_handoff_20260814_mobile.md`, `docs/session_handoff_20260818.md` and
`docs/dimension_gap_list_20260818.md`. All eight passed the CLAUDE.md byte-level
mojibake scan before commit. Applied the checkpoint retention rule: removed from
tracking `checkpoint_update_20260720.md`, `checkpoint_update_20260720b.md`,
`docs/checkpoint_update_20260721b.md`, `docs/checkpoint_update_20260723.md`,
`phase_2/checkpoint_update_20260717.md`, all recoverable from history at
`37fc7f8`. **Flagged collision:** the preceding log entry records a standing
instruction that CC is not to untrack the two 2026-07-20 checkpoints on account of
their `[REPORT: pending]` markers. Today's dispatch explicitly ordered the
retention rule applied. CC executed today's explicit instruction and is recording
the collision here rather than resolving it silently; the `[REPORT: pending]`
items remain tracked in the gap list and were not edited.

**S1, HALTED at the gate, no compute spent.** The dispatch instructed CC to diff
`5ab4be3..HEAD` over the pipeline paths the swap runbook touches and to stop if
anything in it touched inference code. It does. Two inference-path files are
modified in that range, both at commit `17ef3d8`:
`scripts/tier2_d7_build_submission.py` (+3) and
`scripts/ws_d7_apply_bias_correction.py` (+6). Both changes are assert-only
fail-loud alpha guards; neither assigns a value, neither sits in a branch, and
output is bit-identical when `alpha == 0.90`. Two files the runbook requires,
`scripts/tier2_swap_compliance_guard.py` and `scripts/tier2_swap_rehearsal.py`, do
not exist at `5ab4be3` at all, so literal execution from that commit cannot
produce the compliance evidence the dispatch demands. Reported to Matteo with a
recommendation to run from HEAD; awaiting decision. No download, no pin, no
inference.

**S2, two-cell check (read-only).** New scripts
`scripts/task2_two_cell_check_20260818.py` and
`scripts/task2_two_cell_probe_20260818.py`; outputs
`reports/two_cell_check_20260818.md` and `reports/two_cell_check_20260818.json`.
Both deterministic (static raster reads, CSV reads, closed-form arithmetic); no
stochastic step, so no seed applies. No layout, submission or fitted artifact was
touched, and no AEP was recomputed.

Result, verdict (a): **cell 63 resource is better than the organizer baseline
centre**, by +4.90% in pooled mean v^3 (1,925.17 versus 1,835.23) and +0.77% in
mean 125 m wind speed over AROME 2016-2020, ranking 12 of 159 against 58 of 159,
at 94.56% of domain best against 90.14%. Better in four of five training years.
The 2026-07-13 record's "94.6% of domain-best, +4.9% over kit baseline" is
confirmed against the underlying data. Cell 63's 15x15 km box is fully within
50 m; the baseline centre's box is not (4 of 3,721 samples reach 50.71 m). Wind
roses share the same dominant 225-247 degree sector, angular offset 0 degrees.
The CF gap is therefore not a resource gap.

Decomposing `CF_net = CF_gross x (1 - wake)` on recorded wake fractions puts
**90% of the 5.99 pp CF gap in gross capacity factor** (5.39 pp), with only
0.60 pp attributable to the wake difference. LCOE decomposition on the unchanged
kit cost stack splits the 12.07 EUR/MWh gap into a 6.18 site component (of which
distance to shore 3.63 exceeds foundation depth 2.55) and a 9.97 energy component,
leaving a **-4.08 residual**: the kit cost model returns 79.02 EUR/MWh, not 83.1,
from the organizer's own stated site and AEP. Three record corrections are logged
in the report: the distance term is not a constant and our own chain does override
it (79.197 km, not the 60 km default); the box screen is 250 m, not 1 km; and
`checkpoint_update_20260713.md` is in no commit on any branch, so its claim was
verified against `data/cell_resource_ranking.csv` instead.

**Compute disclosure:** local CPU only, no GPU, no model training, no inference,
no submission I/O. Measured wall-clock 3 s for the check script and 3 s for the
probe script including interpreter startup, single-threaded, plus two earlier
development runs of the same scripts at comparable cost. Total well under one
CPU-minute. Git operations and read-only repository inspection are not separately
metered.

**Report sync:** the S2 findings are report-worthy and are marked
`[REPORT: pending]` in `reports/two_cell_check_20260818.md` by reference from the
session checkpoint. Specifically pending: the (a) verdict on resource, the 90%
gross-CF finding, the distance-over-depth LCOE split, and the non-reproducibility
of the organizer's 83.1 from the shipped cost model.

<a id="cc-20260819-s11-pin-gate"></a>
## 2026-08-19 - S1.1 pin gate and structure verification, Zenodo 20874645

**Agent:** CC (Claude Code), role = builder/executor per contract §2.
**Prompt / authority:** CC dispatch 2026-08-18, block S1.1. Matteo authorized
"GO from HEAD" on 2026-08-19 after CC reported the `5ab4be3..HEAD` inference-code
diff and recommended HEAD.

**Frozen-tree decision.** The dispatch pinned the pipeline at `5ab4be3`. The diff
`5ab4be3..HEAD` over the runbook's pipeline paths is not empty: two inference-path
files are modified at commit `17ef3d8`, `scripts/tier2_d7_build_submission.py`
(+3) and `scripts/ws_d7_apply_bias_correction.py` (+6), both assert-only fail-loud
alpha guards with no numeric effect when `alpha == 0.90`, and two files the
runbook requires (`tier2_swap_compliance_guard.py`, `tier2_swap_rehearsal.py`) do
not exist at `5ab4be3` at all. Reported to Matteo, who approved running from HEAD.

**Swap year is 2022.** Record 20874645 is v4 of concept record 19538993, created
2026-08-09T20:53:52Z. Its `metadata.version` field is null; "v4" is the displayed
badge. Its `publication_date` field, 2026-06-25, is stale and inherited from v3;
`created` is the field to trust.

**Pin gate (contract A1), `scripts/fetch_pin_inference_2022.py`.** Downloaded
`inference_2022.zip` only, by version-specific DOI 10.5281/zenodo.20874645, with
an in-script assertion that the concept record 19538993 is never resolved (the
swap-week rule in `MANIFEST_zenodo_20335351.md`). Size 21,742,784 B as listed;
MD5 `07bedad96f60de39134e01780be54a9d` matches the API listing; SHA-256 computed
locally on download is
`bb329f042b6c638f5ea70f26e32a044bf5c31c35f462b652ba697c3c87be59a7`. New manifest
written at `data/MANIFEST_zenodo_20874645.md`. `MANIFEST_zenodo_20335351.md` was
NOT edited, per the dispatch.

`phase2_dataset.zip` in v4 carries MD5 `96988634e1dbce27cb5369b4809de964`,
byte-identical to the v3 file already pinned, so the training archive is unchanged
and was deliberately not re-downloaded. `phase1_dataset.zip` and
`mini_challenge_dataset.zip` are out of scope and were not downloaded.

**Structure verification, `scripts/verify_inference_2022_structure.py`:
GATE PASS, no structural difference found.** Compared field by field against the
2021 set: window count and directory naming, per-window file set, `metadata.json`
key set, context span, predict span, implied horizon set, `context_end` to
`predict_start` gap, parquet column names, parquet dtypes, distinct point counts,
row counts, target grid, and implied submission shape. Eight windows
`window_1..8`; context 13 days, predict 13 days, horizons {1,7,14}, one-day gap;
`context_reanalysis_north_sea.parquet` 143,640 rows per window with
`time, latitude, longitude, u10, v10, u100, v100` all float32;
`context_hres_north_sea.parquet` 2,565 rows with the 24
`fcst_speed/dir_d{1,7,10}_h{0,6,12,18}` columns, still **d10 not d14**, unchanged
from 2021. Target grid 43,715 distinct points confirmed from
`phase_2/phase2_dataset_ship/static/footprint_points.parquet`, giving the
contracted 43,715 x 8 x 3 x 4 = 4,196,640 rows.

**Correction made during the gate run.** The first version of the verification
script compared the expected 43,715 against the raw `arome_static` seamask sum
(75,653) and reported a spurious FAIL. Those are different quantities: the seamask
counts every AROME sea pixel, while the target grid is defined by
`footprint_points.parquet`. The check was corrected to read the footprint file and
re-run clean. No data difference was involved. Logged rather than silently fixed.

**Compliance guard run early (read-only).**
`scripts/tier2_swap_compliance_guard.py --meta-dir phase_2/inference_2022/inference`
returns **32 checks, all PASS**, Clause 1 satisfied on all 8 windows x 4 init
hours. Full per-check output is in the session report to Matteo.

**ERA5 source question resolved.** Runbook §R2 flagged that a swap year of 2024 or
later would require the CDS fallback. The swap year is 2022, inside the
WeatherBench2 mirror's 1959-2023 coverage, so the built-in path applies and no CDS
account, `~/.cdsapirc`, or `fetch_global` adaptation is needed.

**Provenance defect found and flagged, not fixed.**
`MANIFEST_zenodo_20335351.md` records MD5 `4633a8232a69c4b723e991eb906c2259` for
`hres_north_sea.parquet` (296.5 MB). The Zenodo API reports that same MD5 for
`phase1_dataset.zip` (10.5 GB). One MD5 cannot belong to both, so that manifest's
MD5 field holds the containing archive's checksum rather than the parquet's. Its
SHA-256 for that file was computed locally on the parquet and remains sound, so
nothing downstream is compromised. That manifest declares itself immutable and
correcting it is outside S1.1 scope; raised for Matteo and Opus.

**Operational note.** The working drive is at 98% capacity, 11.2 GB free. The
runbook prerequisite is roughly 5 GB for extracts and CSVs plus 1.2 GB for the
ONNX. It fits without margin. Raised before the 3.2 h Pangu leg rather than during
it.

**S1.2 not started.** Its Step 0 has two human-only prerequisites outstanding: the
Z: drive is not mounted (`Test-Path <NETWORK_SHARE>/` returns False; the drive letter exists as
a stale mapping reporting no capacity), so the pinned `downscaler_blockexcl.pkl`
SHA `b68eb5fe...` cannot be verified, and the alpha check has not been run.

**Compute disclosure:** local CPU only, no GPU, no model training, no inference,
no submission I/O. One 21.7 MB HTTPS download from Zenodo plus one Zenodo API
metadata call. Hashing 21.7 MB twice, one zip extraction, and eight pairs of small
parquet reads for the structure gate. Total well under one CPU-minute of compute;
the download dominates wall-clock at a few seconds.

**Report sync:** S1.1 carries no report-worthy result of its own beyond the swap
year and the pin, which belong in the methods section. Marked `[REPORT: pending]`
for the pin SHA and the 2022 swap-year fact.

<a id="cc-20260819-s3-three-case"></a>
## 2026-08-19 - S3 three-case like-for-like scorer run, and the Z: blocker

**Agent:** CC (Claude Code), role = builder/executor per contract §2.
**Prompt / authority:** CC dispatch 2026-08-18 block S3, authorized by Matteo
2026-08-19 ("S3: AUTHORIZED, read-only, in parallel with S1"). Read-only. No
submission, no refit, no layout change, no parameter change.

**Artifacts:** `scripts/task2_three_case_scorer_20260818.py`,
`reports/three_case_scorer_20260818.md`, `reports/three_case_scorer_20260818.json`,
`data/task2_three_case_scorer_20260818.csv`. Deterministic, no stochastic step.

**Environment finding.** The Task 2 scorer does NOT run in `swnd`: that env has no
`py_wake`. It runs in the `pywake` env (py_wake 2.6.17). The first S3 attempt
failed on `swnd` with `ModuleNotFoundError: No module named 'py_wake'` and was
re-run in `pywake`. Recorded because the standing note for this repo is to use
`conda run -n swnd python`, which is wrong for the Task 2 siting chain.

**Result table** (real AROME 125 m, single nearest sea pixel, 2016-01-01 00:00 to
2020-12-31 21:00, years 2016-2020, 14,608 finite steps from 1,826 daily files;
shear 125 m to 170 m at alpha 0.11, factor 1.0344):

| case | CF gross | wake | CF net | AEP GWh | LCOE |
|---|---|---|---|---|---|
| 1. plain 7D grid at 53.5N 1.5E | 52.15% | 9.70% | 47.09% | 4,991.9 | 89.23 |
| 2. plain 7D grid at cell 63 | 51.46% | 9.60% | 46.52% | 4,931.3 | 96.31 |
| 3. submitted Stage 3 layout at cell 63 | 51.46% | 8.27% | 47.21% | 5,003.9 | 95.17 |

**Validity anchors.** Case 3 reproduces the banked winner figures exactly (CF
0.472086 vs recorded 0.4721; wake 0.082698 vs 0.0827; AEP 5,003.923 vs 5,003.92).
Case 1 reproduces the 2026-07-13 well-tie row `arome_baseline_53.5_1.5` to all
stored digits. Every case was run through BOTH the independent replica and the
kit's own `simulate_year()`; the two agree to **0.0 pp on CF and 0.0 pp on wake on
all three cases**. Cases 2 and 3 return identical gross CF to the last digit, as
they must on identical wind, which is a free correctness check.

**Power curve NOT substituted, per the dispatch.** The real IEA 22 MW CSV is absent
from this checkout (`_TURBINES_DIR` at `turbines_catalog.py:29` does not exist), so
`load_turbine("IEA_22MW")` falls through to `_generic_power_ct`
(`turbines_catalog.py:89-120`), a cubic ramp with `rated_ws = 12.0`. The replica
path builds the same curve from `data/iea22mw_power_ct.csv`.

**Finding: the site explanation is dead.** Our chain at the organizer's OWN centre,
with a plain 7D grid, returns CF net **47.09%** against their reported 53.2%, a
**6.11 pp** gap, essentially the same gap we carry at cell 63. AEP falls 643 GWh
short and LCOE lands at 89.23 against 83.1. The deficit does not move when the site
moves, so it is not siting and not our layout.

**Site effect** (case 2 minus case 1, byte-identical layout): CF net -0.57 pp, CF
gross -0.69 pp, AEP -60.6 GWh, LCOE +7.09 EUR/MWh. **Layout effect** (case 3 minus
case 2, byte-identical wind): wake -1.33 pp, CF net +0.69 pp, CF gross exactly
0.00 pp, AEP +72.6 GWh, LCOE -1.15 EUR/MWh. The Stage 3 layout earns its place
purely through geometry.

**Tension flagged, not smoothed.** Cell 63 has a HIGHER mean hub wind speed
(10.524 vs 10.465 m/s) but a LOWER gross CF at the single scorer pixel. S2's
+4.90% advantage was measured on mean v^3 pooled over every AROME sea pixel in the
0.25 degree cell (289 vs 265 pixels); the scorer uses the single nearest pixel.
Different spatial samples, so the two results are not in contradiction, but neither
substitutes for the other, and the report must not cite the +4.90% as if it
predicted the scorer outcome.

**S1.2 BLOCKED: the pinned downscaler is missing.** Z: is now mounted (18.0 TB,
8.6 TB free) and `<NETWORK_SHARE>\Matteo\large downloads\` exists, but
**`tier2_smoke\` is gone**. `downscaler_blockexcl.pkl` (SHA
`b68eb5fe57fab817364f3df2feb4f4bd77a6658cf91642301b9159e9d0fa8e0a`) was not found
under `<NETWORK_SHARE>\Matteo` to depth 4, under `<USER_HOME>` to
depth 6, or anywhere under `<LOCAL_DRIVE>/Pythonwork`. A full-depth Z: sweep was still running
when this entry was written; scope of the negative is exactly those three searches.
The repo's own `tier2_smoke/` holds only `pangu_README.md`, `pangu_smoke.py`,
`prep_input_era5.py` and `PANGU_SMOKE_RESULTS.md`.

Consequence, traced: `tier2_d7_build_submission.py:100-101` asserts the downscaler
SHA and would abort. `tier2_f2_d14_precheck.py:50-66` would, if reached with the
cache absent, RETRAIN the downscaler; runbook §0.1 records that the pinned pkl was
fitted under the OLD unseeded code and must be preserved rather than regenerated,
so a retrain produces a different object and cannot reproduce 855076's Leg B. Leg B
steps B4 and B5 therefore cannot run as frozen. Leg A is unaffected.

Present and verified: `pangu_weather_24.onnx` at `<USER_HOME>\tier2_model\`,
1,181,711,187 B, matching the runbook's expected size; and the pinned base
`835026_submission__2_.zip` under `scripts/artifacts/codabench_downloads_20260719/`.

**Compute disclosure:** local CPU only, no GPU, no model training, no inference, no
submission I/O. S3 total wall-clock 198 s, of which 182 s was AROME loading (two
sweeps of 1,826 daily files) and about 7 s PyWake across six simulations (three
cases x two code paths). One failed 3 s run in the wrong env. Filesystem searches
across Z: and C: are not separately metered.

**Report sync:** S3 is report-worthy and marked `[REPORT: pending]`. Specifically
pending: the case 1 result killing the site explanation, the 1.33 pp wake gain of
the Stage 3 layout at identical gross CF, the 0.0 pp agreement between the two
scorer paths, and the pooled-cell versus single-pixel sampling caveat.

<a id="cc-20260819-s3-addendum-search-scope"></a>
### Addendum to the 2026-08-19 S3 entry: downscaler search now complete

The preceding entry recorded the full-depth Z: sweep as still running and scoped
its negative accordingly. That sweep has since completed. Closing the scope:

`downscaler_blockexcl.pkl` was NOT found by any of the following, all run
2026-08-19:

- `<NETWORK_SHARE>/` recursive, depth 6, filter `downscaler_blockexcl.pkl`: no match.
- `<NETWORK_SHARE>\Matteo` recursive, depth 4, matching
  `downscaler|tier2|pangu|\.onnx$|arm_extract`: no match.
- `<USER_HOME>` recursive, depth 6, filter `downscaler_blockexcl.pkl`:
  no match. Same search for a `tier2_smoke` directory: no match.
- `<LOCAL_DRIVE>/Pythonwork` for `downscaler*.pkl` and `*blockexcl*`: no match. The only
  `tier2_smoke` directory on that drive is the repo's own, holding four docs and
  scripts and no pickle.
- `<LOCAL_DRIVE>/` recursive, depth 4, matching `tier2|downscaler_blockexcl`: one unrelated
  hit, [REDACTED: unrelated work file, not a Sea Winds artifact].

Also checked and empty: `<NETWORK_SHARE>\Matteo\large downloads\exports\`.
`<LOCAL_DRIVE>/` enumerated as empty or inaccessible.

The negative is therefore no longer provisional: within every location searched
above, the pinned block-excluded downscaler is absent. Not searched, and so not
covered by this negative: any external or removable media, any cloud storage, any
other machine, and `<LOCAL_DRIVE>/` beyond its root.

Nothing was created, moved, or deleted by these searches; all were read-only
enumerations.

<a id="cc-20260819-runbook-errata-legA-2022"></a>
## 2026-08-19 - RUNBOOK ERRATA and Leg A executed on the 2022 final-evaluation set

**Agent:** CC (Claude Code), role = builder/executor per contract §2.
**Authority:** CC dispatch 2026-08-18 (S1), Matteo's "GO from HEAD" 2026-08-19,
his approval of the three-blocker Step 0 remediation, his approval of the
blocker-4 handling with seven conditions, and his decision to adopt the organizer
kit update. Every parameter change below was approved before it was made.

### Runbook errata: four blockers, none of them in docs/swap_runbook_20260721.md

**Blocker 1, alpha was 1.0 not 0.90.** Runbook §0.2 warned about this and its own
Step 0 fixes it, so the defect is that the frozen artifact shipped at the wrong
value. Fixed via the runbook's own path: APPROVED_ALPHA 1.0 to 0.90 in
ws_d7_set_chosen_alpha.py, then ran the setter. Result: chosen_alpha 1.0 to 0.9,
coverage 85.7%, Winkler 32.485 (vs 35.287 pre-correction).
ws_d7_bias_correction_params.json SHA-256:

    before  94962715aed2ae3f949e39a7afd37747fa058c60ed46d4967262a50bb715bbb0
    after   df15b39ba2aca5780b60f41420a3644f9a92158cce707ba52b043b37b562fe1f

**The runbook frozen-artifact SHA table is a PRE-Step-0 snapshot.** It pins the
before value, the alpha=1.0 state. That table and the runbook's own alpha=0.90
requirement cannot both hold for this file. The after value is the correct one for
any run of the frozen pipeline, because ws_d7_apply_bias_correction.py asserts
chosen_alpha == 0.90. The table should be corrected or annotated.

**Blocker 2, A5 would have silently produced a 2021 submission.**
ws_d7_apply_bias_correction.py:124 reads its SOURCE from BACKUP_PATH, not from the
live submission.csv, and the BACKUP_PATH.exists() branch at :84-85 takes
"reading FROM this". The backup from 2026-07-08 was still on disk, so A5 would
have bias-corrected the 2021 submission and written it out as the 2022 result,
discarding everything A1 and A2 had just computed. The output would have been
4,196,640 well-formed rows of 2021 predictions and **would have passed the
validation gate**. Most dangerous defect found in this run. Remediation: the three
stale backups renamed to _2021.csv suffixes, not deleted. A5's run log then showed
"Checkpointed pre-fix submission.csv" to the backup path, confirming it
checkpointed the fresh 2022 build rather than reading a stale one.

**Blocker 3, every step pointed at 2021.** The notebooks call eval_windows() with
no argument, and A3/A4/A5 hardcoded EVAL_YEAR = 2021. **Resolved upstream** by
organizer commit 9edd92b, which makes eval_windows() auto-detect the year from the
installed window_1/metadata.json.

**Blocker 4, no 2022 HRES in the pipeline input path.** The kit does not read the
per-window context files; _load_hres() sources from config.hres_parquets(), whose
patterns were all under train/, and those parquets stop at 2021-12-31.
Empirically: build_hres_table returned 20,520 rows at a 2021 issue date and
**0 rows** at 2022-01-14. Leg A would have produced degenerate d1/d7 output.
**Resolved upstream** by the same 9edd92b, which adds the
inference/window_*/context_hres_north_sea.parquet pattern to hres_parquets().

CC's own proposed remediation for blocker 4 (concatenating the eight context files
into a train/ parquet) was **dropped** in favour of the organizers' path, on
Matteo's decision, after a git fetch he ordered surfaced the update. Recorded
because the fetch was his instruction and it changed the outcome materially: the
adopted path writes no organizer inference data into any training location.

### Kit adoption

Nested kit repo, local commits only, never pushed to the organizer:

    9edd92b  organizer, "update: eval final year"          (adopted base)
    503b747  organizer, "fix: keep directions in [0,360)"  (ancestor of 9edd92b)
    17baf59  ours, random_state=42, cherry-picked from 0159ab9

b15623e (CC's SEAWINDS_EVAL_YEAR splits patch) dropped as superseded by the
organizers' auto-detect. docs/kit_patches/downscaling_random_state_20260721.patch
re-captured against 9edd92b; splits_eval_year_env_20260819.patch removed. 503b747
also means A6 is now an idempotent safety net rather than a necessary step, which
the run confirmed: **A6 replaced 0 values**.

### Guards

**Guard A PASS.** hres_parquets() resolves to exactly 10 files (2 train + 8 window
context). All eight 2022 issue dates return exactly 20,520 rows with
fcst_u/v/speed 1.000 finite. At 2021-01-14 the feature table is content-identical
with and without the 2022 sources (SHA-256 7d89bbc0ce9869d94007b971e3e8a9dd over a
canonically sorted serialisation), proving the new sources did not perturb a
single 2021 feature value.

**Guard B PASS, with a scope correction stated rather than papered over.** The
condition was to compare fitted objects against the 2021 rehearsal. **No such
reference exists**: tier2_swap_rehearsal.py was written but never run (runbook
§R5). Rather than skip the condition or silently redefine it, the property was
established directly and more strongly: fit_forecast was run twice, once with the
2022 windows installed and once with the archived 2021 windows physically swapped
in, and mos/qmos/adj/offs are **byte-identical** across both. Instrumentation on
build_hres_table and build_climatology_forecast recorded 671 dates requested
during fitting, all in 2016-2020, no eval year. train_dates('6D') is a literal
2016-01-01 to 2020-12-29 range, 305 dates.

**build_climatology_forecast traced:** reads TRAIN_YEARS_CLIM coarse files only;
_coarse_grid is guarded by with_truth, which inference sets False. coarse_fields
never reads context_reanalysis_north_sea.parquet, so Leg A needs only the HRES
context.

**2022-issue-date assertion PASS** (the check the runbook did not have). All 8
windows have every date field and all three score_days in 2022, none in 2021;
shape 4,196,640 rows / 96 blocks of 43,715 / windows 0-7 / horizons 1,7,14 /
hours 0,6,12,18; and a content check against the archived 2021 build shows
**99.72% of q50 values differ**, so the build cannot be a stale reuse.

### Leg A execution, per-step wall-clock

| Step | Wall-clock | Note |
|---|---|---|
| Guard A | 40 s | PASS |
| Guard B | 208 s | two fits, 107 s + 101 s, PASS |
| A1 notebook 1 | 98 s | 8 windows cached, all 2022 issue dates |
| A2 notebook 2 | 253 s | downscaler on 74 days (2020 only); 4,196,640 rows |
| A3 d14 climatology | 47 s | 1,398,880 d14 rows across 32 blocks |
| A4 direction residual | 61 s | 2,565 HRES rows per window |
| A5 d7 bias correction | 46 s | alpha 0.90; fresh backup checkpointed |
| 2022 assertion | 57 s | PASS |
| A6 dir 360 wrap | 44 s | **0 values replaced** |
| archive + zip | 16 s | |
| validation gate | 8 s | PASS |

A1 to A6 total **549 s (9.2 min)**, against the runbook's ~20-50 min estimate.

### Output

    scripts/artifacts/submission_legA_base_2022_20260819.csv
      432,929,903 B
      SHA-256 4f632fef000d8526d64cce9ab5a01d14dd8aeefcf5e8e1fb1faee4ed5d3a0789
    scripts/artifacts/submission_legA_base_2022_20260819.zip
      80,503,779 B
      SHA-256 5c8c4701bf0620c354e74d4a66be7d4c081ea79734a6921f4d48ead67256c41b

Validation gate: all 12 structural checks PASS, 0 NaN, directions in [0,360),
q05 <= q50 <= q95, zip contains submission.csv. Calibration: d1 width 7.597 m/s,
dir half-width 30.317 deg, q50 median 9.224; d7 22.251, 117.655, 8.594; d14
14.652, 139.383, 9.215. Both median STOP-flag checks in range.

**This is Leg A only. It is NOT the 855076-equivalent**: Leg B (the Pangu d7
splice and f=1.25) is blocked on the missing downscaler_blockexcl.pkl, with IT
asked to restore tier2_smoke/ from a 20-29 July snapshot. CC has not submitted and
will not. Matteo uploads.

**Compute disclosure:** local CPU only, no GPU. Total measured wall-clock across
guards, A1-A6, archiving and validation: 878 s (14.6 min), single machine, swnd
env. Two notebook executions headless via nbconvert to executed-copy notebooks so
the source notebooks were not mutated. Three full LightGBM MOS fits (one in A1,
two in Guard B) and one downscaler fit. No inference, no GPU, no submission I/O.
Earlier this session: 10.26 GB of regenerable Phase-1 prediction CSVs deleted with
Matteo's approval, freeing C: from 12 GB to 22 GB.

**Report sync:** [REPORT: pending] on the four-blocker errata (a methods and
reproducibility exhibit), the Guard B double-fit design, and the fact that two of
the four blockers were resolved by an organizer update found only because Matteo
ordered a fetch before proceeding.

<a id="cc-20260819-board-2022-legA"></a>
## 2026-08-19 - 2022 final board result, submission 1 of 10 (Leg A)

**Agent:** CC (Claude Code), relaying a result reported by Matteo. CC did not
submit and does not select. Matteo uploaded.

**Artifact submitted:**

    scripts/artifacts/submission_legA_base_2022_20260819.csv
    SHA-256 4f632fef000d8526d64cce9ab5a01d14dd8aeefcf5e8e1fb1faee4ed5d3a0789
    scripts/artifacts/submission_legA_base_2022_20260819.zip
    SHA-256 5c8c4701bf0620c354e74d4a66be7d4c081ea79734a6921f4d48ead67256c41b

Build provenance: Leg A only (A1-A6), kit 9edd92b + 17baf59, alpha=0.90, d14
climatological replacement, direction residual models, no Pangu d7 splice and no
f=1.25 (Leg B blocked on the missing downscaler_blockexcl.pkl). Full build record
in the 2026-08-19 runbook-errata entry above.

**Board figures as reported by Matteo (submission 1 of 10):**

| Metric | d1 | d7 | d14 |
|---|---|---|---|
| WS (speed Winkler) | 9.7251 | 27.4386 | 16.6917 |
| Dir (circular Winkler) | 83.0877 | 313.6588 | 343.4354 |
| Coverage | 91.3% | 83.8% | 91.4% |

Match 100%, so every submitted row was accepted and joined; no structural or
alignment defect reached the scorer. That is the first independent confirmation
that the 2022 rebuild is well-formed end to end.

### Comparisons, each labelled by what it is

**Like-for-like, same pipeline stage, different year.** The 2021 board scored our
Leg A base (submission 835026, alpha=0.90) at **WS d7 = 27.14**
(`LLM_AGENT_LOG.md:679, :2767`). The 2022 Leg A base scores **27.4386**, +0.30
(+1.1%). Same construction, same frozen models, different evaluation year. The
close agreement is a useful sanity signal that nothing structural broke in the
year swap, though a year-to-year difference of this size is not interpretable as
skill change.

**Against the kit reference anchors** (`CLAUDE.md`, Phase_2.pdf pp.4,6; these are
kit references, NOT a 2022 baseline, so read them as orientation only):

| | kit ref | 2022 Leg A | direction |
|---|---|---|---|
| WS d1 | 9.2 | 9.7251 | slightly worse |
| WS d7 | 29.8 | 27.4386 | better |
| WS d14 | 40.1 | 16.6917 | far better |
| Dir d1 | 173 | 83.0877 | far better |
| Dir d7 | 312 | 313.6588 | flat |
| Dir d14 | 334 | 343.4354 | slightly worse |

The two large gains are the two components built for exactly those horizons: the
d14 climatological replacement (WS d14 40.1 -> 16.69) and the d1 direction
residual model (Dir d1 173 -> 83.09). Both behave on 2022 as designed on 2021.

**Coverage against the Winkler optimum.** The standing anchor is that the Winkler
optimum sits near coverage 0.88, not 0.90 (`CLAUDE.md`). Measured: d1 91.3% and
d14 91.4% are ABOVE that, i.e. intervals wider than optimal; d7 83.8% is BELOW,
i.e. narrower than optimal. Directionally this says d1 and d14 are paying a width
penalty and d7 a miss penalty. No action proposed: the pipeline is frozen, and
tuning on a 10-submission budget against a live board is precisely the probing
the calendar rules exist to prevent.

### What Leg B is worth, stated as an expectation and not a measurement

On the 2021 board, adding the Pangu d7 splice and f=1.25 moved WS d7 from
**27.14 to 24.7543**, a gain of 2.39 Winkler (-8.8%), with coverage_d7 rising
74.6% -> 85.5% (`checkpoint_update_20260720b.md:9-11`, relayed at
`LLM_AGENT_LOG.md:2781`). If that transfer held on 2022, Leg B would take WS d7
from 27.4386 to roughly **25.0**, and would raise d7 coverage from the current
83.8% toward the 0.88 optimum rather than away from it.

That is an extrapolation from one year to another, not a prediction, and f=1.25
was itself a pre-registered transfer bet fitted on dev-year 00 UTC rather than
measured on 2021 (`tier2_d7_apply_f125.py:19-26`). It is recorded here only to
size what the missing `downscaler_blockexcl.pkl` is actually costing, since that
is the live decision: d7 is our weakest horizon against the kit reference on the
speed axis, and Leg B is the component built to fix it.

**Leg B remains held** at Matteo's instruction pending the IT restore of
`<NETWORK_SHARE>\Matteo\large downloads\tier2_smoke\`. Nine submissions
remain.

**Compute disclosure:** none. This entry is a relay of a board result plus
arithmetic on figures already in the record. No pipeline step run, no model
touched, no data read beyond the log itself.

**Report sync:** [REPORT: pending] on the 2022 board result, the 835026-to-2022
like-for-like d7 agreement (27.14 vs 27.4386), and the coverage-versus-optimum
reading.

<a id="cc-20260820-bonus-stage3"></a>
## 2026-08-20 - Bonus build Stage 3: ENTSO-E 2019 NL market data (pull OK, Stage 4 held)

**Agent:** CC (Claude Code), role = builder/executor per contract §2.
**Authority:** `docs/cc_prompt_bonus_build_PARKED.md`, triggered by Matteo
2026-08-20; stages 2-5 plan proposed by CC and confirmed the same day with four
explicit decisions (1,460-hour framing accepted with no scaled column; full PyWake
for Stage 2; Danglars construction as proposed; Stage 3 first).

**Rider R1, answered honestly: NO.** `cc_prompt_bonus_build_PARKED.md` was NOT in
the repo, on disk or in any commit on any branch. The stages 2-5 plan was read
from Matteo's paste in session, not from a repo copy. Committed at `abd50e6` to
close the gap. Third instance this week of a document authored in project
knowledge and never committed (see also MERGED_20260729 and the 2026-07-27 set).

**Rider R2, errata E7:** spread-grid validation is commit
`c16b1a2984e4f8fbf6ebf553522aaa0e5a73561c`, dated 2026-07-13, "Task 2 QA
continued: spread-grid full-series check + spacing fix".

**Stage 3 outcome: data pull SUCCEEDED.** New script
`bidding_sim/fetch_market_data_2019.py`; outputs and provenance under
`bidding_sim/market_data_2019/` with `MANIFEST.md` (contract A1) written before
anything downstream reads them.

- Day-ahead, hourly: **8,760 rows**, 0 NaN, SHA-256 `b900a7f5db0e17b9...`
- Imbalance, 15 min: **35,040 rows**, 0 NaN, SHA-256 `518ed090ef2451d8...`

Both exactly the expected counts for a non-leap year. Elapsed 193 s on the
re-pull, well inside the 30 min estimate and nowhere near the 2x stop.

**CC defect found and fixed mid-stage.** The first pull closed each monthly window
at 23:00 local. The end bound is inclusive (2026-07-29 smoke test), which is right
for an hourly series but truncates the final hour of a 15-minute one, silently
dropping 4 quarters per month boundary, 44 across January to November. Fixed by
closing at 23:45 and re-pulling. The day-ahead SHA is byte-identical across both
pulls, confirming the fix touched only the 15-minute series. The gap would not
have affected the simulation (delivery hours are 00/06/12/18 UTC; every missing
quarter fell at 22:00-22:45 UTC), but a cached artifact with an undocumented hole
is a defect regardless. Logged, not quietly corrected.

**Option A does not arise.** The imbalance response carries only `Long` and
`Short`; the scan for a regulation-state column returned nothing. Not trivially
available, so per the dispatch it is not implemented.

**STAGE 4 HELD: the `Long`/`Short` semantics do not yield a valid two-price
scheme.** Over the 8,748 hours joinable to day-ahead (DA mean 41.20, `Long` mean
36.25 with max 590.91, `Short` mean 8.98 with min -194.97):

- Reading A (`pi_plus`=Long, `pi_minus`=Short): `psi_minus` negative in **99.7%**
  of hours, i.e. being short is rewarded. Critical ratio in [0,1] in **31.5%**.
- Reading B (swapped): `psi_plus` fine, but `psi_minus` negative in **68.6%**.
  Critical ratio in [0,1] in the same **31.5%**.

`Long >= Short` in 70.7% of hours, equal in 1.2%, so the pair is genuinely
two-sided; the question is what the sides mean. The distributions look like
system-side regulation prices (upward with scarcity spikes, downward going deeply
negative) rather than resolved BRP settlement prices.

Assigning `pi_plus`/`pi_minus` is a settlement-model decision, not a build step,
and reversing it would invert the entire result. Escalated to Matteo per contract
§2 and the CLAUDE.md escalation rule. **No Stage 2 compute spent**, which is
exactly what the Stage-3-first ordering was for.

**Compute disclosure:** local CPU plus network only. Two ENTSO-E pull runs (217 s
and 193 s wall-clock, 24 monthly requests each), a handful of pandas diagnostics.
No GPU, no training, no inference, no frozen artifact read or written. Writes
confined to `bidding_sim/`.

**Report sync:** [REPORT: pending] on the Stage 3 data provenance and, if the
semantics question resolves, the settlement-model choice, which is a Dim 4
methods point rather than a result.

<a id="dim5-failure-r1-dispatch-from-project-knowledge-20260820"></a>
## 2026-08-20 - Dim 5 criterion 3 (failures): dispatching from project-knowledge state as if it were repo state

**Logged at Matteo's explicit instruction** (amended Stage 4 unblock dispatch,
2026-08-20, section 5: "Log it against me in LLM_AGENT_LOG.md as a Dim 5 failure
entry"). Recorded as observable behaviour only, per the narration rules.

**The behaviour.** Rider R1 of the 2026-08-20 bonus trigger asked CC to "Confirm
cc_prompt_bonus_build_PARKED.md is committed in docs/ and this plan was read from
the repo copy, not project knowledge." The rider was phrased as a confirmation,
that is, it presumed the answer was yes.

The file was not in the repository: not on disk under any casing, and never added
in any commit on any branch. CC had proposed the stages 2-5 build plan by reading
the dispatch from Matteo's paste in the session. The verification the rider asked
for could not be performed, and the claim embedded in the rider was wrong.

**Scope: this is the third instance in one week.**

1. `phase2_report_material_MERGED_20260729.md`, named as the comparison baseline
   for the 2026-08-20 MERGED batch. Not in the repo; supplied by paste after CC
   reported the gap. The verification could not run until it was supplied.
2. The 2026-07-27 governance set (checkpoint, gap list, handoff, MERGED). Written
   to disk on 27 July and never committed; discovered by the S0 reconciliation of
   2026-08-18, which was itself commissioned to settle whether the 29 July
   documents had been pushed. They had not, and neither had the 27 July ones.
3. `cc_prompt_bonus_build_PARKED.md`, this entry.

**What distinguishes instance 3.** Instances 1 and 2 were omissions: a document
existed in project knowledge and was not committed. Instance 3 attached a
*verification request* to the omission, asking CC to confirm a state that did not
hold. Had CC answered the rider from expectation rather than from the filesystem,
the log would now carry a false provenance claim: that a build plan worth 15
scored points was derived from a committed repo artifact when it was in fact
derived from a chat paste.

**Consequence, observed.** No incorrect work was produced. CC checked the
filesystem and git history before answering, reported the negative, and committed
the pasted text at `abd50e6` so the repo carries the instruction it is executing
against. The cost was one round trip and the loss of an independent check: the
plan and the dispatch it was built from now share a single origin, the paste, with
no repo copy that could have corroborated either at the time the plan was written.

**Standing mitigation already adopted.** Contract amendment v2.1 (2026-08-19)
addresses the artifact half of this pattern through 5a, which requires pinned
artifacts to be listed in `data/PINNED_ARTIFACTS.md` and named to Matteo. The
document half is not yet covered by any rule: nothing currently requires that a
dispatch be committed before CC executes against it, and nothing requires a
verification rider to be checked against the repo before it is written.

**Not adjudicated here.** Whether that gap should become a v2.2 clause is Matteo's
and Opus's call, not CC's. CC's role in all three instances was to check and
report, which it did in instances 1 and 3 and which the S0 dispatch commissioned in
instance 2.

**Compute disclosure:** none. This entry records behaviour already logged in the
2026-08-20 Stage 3 entry and adds no new measurement.

<a id="dim5-failure-r1-instance-4-openapi-spec-20260820"></a>
### Addition, 2026-08-20: instance 4, and the first instance to cost compute

**Logged at Matteo's explicit instruction**, appended to the entry above rather
than filed separately, because it is the same pattern and not a new one.

**The behaviour.** The amended Stage 4 unblock dispatch stated: "OpenAPI spec for
Settlement prices (v1) is at bidding_sim/ (filename as downloaded from the
portal)." The file was not there, and was not in the Downloads directory either.
As in instance 3, an intended file state was asserted as a current one without
verification. Matteo has accepted the correction; the dispatch line was stale.

**What distinguishes instance 4: it was not free.** Instances 1 to 3 cost a round
trip each. This one cost the work itself, because the undelivered document was
the one that carried the operating limits.

CC proceeded without it, deriving the API contract from the endpoint's own error
responses. That recovered the endpoint path, the authentication header, the
parameter names, the date format, the half-open range convention and the local
time base, all correctly. It could not recover the two facts that were not
observable from a successful response:

1. **Maximum request range is 1 month.** Not inferable; a range request either
   succeeds or is rate-limited, and CC never saw a successful multi-day range.
2. **Production rate limit is 25 requests per day** (with 1/s and 5/min), and
   the spec warns that exceeding it "may result in temporary blocking of your
   API keys."

CC inferred from the 429 pattern that the limit was a rolling quota and
responded by retrying with backoff. Against a per-day budget that is the exactly
wrong response: a rejected request still consumes budget, so each retry deepened
the deficit. Two fetch designs were built and both failed on this. Version 1
attempted 365 daily requests, which exceeds the daily budget by more than 14x.

**Consequence, observed.** Between roughly 170 and 195 requests were issued
against a 25-per-day allowance on 2026-08-20. The exact figure is not
recoverable, because the failing run checkpointed its progress only every 30
days; the bounds come from the probe scripts (30 + 38 + 2 + 1 + 3 requests, all
counted) plus a daily-pull run that passed its day-60 checkpoint and failed
before day 90, with 29 logged rate-limited responses. No data was corrupted and
no deliverable was affected. The costs are: two discarded fetch designs, the
2019 pull not completed on the day it was attempted, and an unquantified risk of
temporary key blocking that was taken without knowing it existed.

**Not the strategist's failure alone.** The dispatch's stale line is the
originating fact and is what this entry records. But CC issued the requests, and
CC had a cheaper option it did not take: on discovering the spec was absent, it
could have said so and asked before spending three rounds of probing and two
fetch runs against an unknown quota. It reported the absence in the same message
as the successful probe result, by which point most of the budget was already
spent. The reporting was accurate and prompt; the sequencing was not.

**Standing mitigation.** Unchanged from the parent entry: contract v2.1 section
5a covers the artifact half of this pattern, and the document half remains
uncovered. This instance sharpens what a rule would need to say. It is not
enough that a dispatch's referenced document be committed before execution; the
gap that bit here is that CC treated "the document is missing" as an obstacle to
work around rather than as a stop condition, because nothing said which of those
it was. Whether that becomes a v2.2 clause is Matteo's and Opus's call, not CC's.

**Remediation applied.** Fetch version 3 is built to the spec rather than to
inference: monthly windows only (the 1-month maximum makes a single-request year
impossible, and makes weekly and daily plans exceed the daily budget), 15-second
spacing to sit at 4 requests per minute against the 5-per-minute limit, and no
retry on 429 at all. Every successful window is cached, so a quota stall costs
progress but never repeats a request. The fallback ladder that versions 1 and 2
implemented has no meaning under these limits: exactly one plan is legal.

**Compute disclosure:** network only, no modelling compute. Between roughly 170
and 195 HTTPS GET requests to api.tennet.eu, total payload on the order of 3 MB,
across five probe scripts and two failed fetch runs. No GPU. No model was
trained, loaded or evaluated.

<a id="bonus-bidding-sim-stage-4-5-closed-20260821"></a>
## 2026-08-21 - Bonus bidding simulation: Stage 4/5 shipped, build CLOSED

**Rung 1: real 2019 Netherlands settlement prices** (TenneT Aether
settlement-prices API), not the synthetic fallback pair. Matteo closed the bonus
build on receipt.

### Result [REPORT: pending]

Four day-ahead bidding strategies settled hour by hour over **1,460 delivery
hours** of 2019 (four per day at 00, 06, 12 and 18 UTC). Not a full year, and no
figure anywhere is scaled to one.

| Strategy | Bid | Revenue (EUR) |
|---|---|---|
| naive | q50 point forecast | 34,886,667 |
| quantile | `F^-1(alpha*)`, alpha* = 0.6280 | 35,069,102 |
| danglars | overconfident, biased triple | 34,830,164 |
| perfect | realized production, zero imbalance | 35,041,624 |

EVIU +182,435 EUR (+0.523% of naive). EVPI -27,478 EUR (-0.078% of quantile).

Modest, and predicted to be modest in advance by Opus before the numbers existed.
Reported at face value, neither softened nor inflated.

### Three findings that qualify the headline [REPORT: pending]

**1. The expected-penalty gate FAILED on the specified population.** The dispatch
specified the delivery-instant join; on those 5,840 settlement periods
`mean(Shortage - DA)` is **-0.0211 EUR/MWh**, so the critical ratio is negative
and the newsvendor bid is undefined. Over the full 35,040 periods the pair is
+0.8047 and +1.3583, giving alpha* = 0.6280. A seeded bootstrap (20,000
resamples) puts the delivery-hour figure's 95% interval at -1.303 to +1.265 with
48.6% of resamples above zero: it is a mean sitting on zero, not evidence that
under-delivery is rewarded. Cause is hour-of-day sampling, not data. The
population switch to the annual aggregate was **forced, not chosen**, and is
disclosed as such with the full diagnostic reproduced in the summary body.

**2. EVPI is negative, and the parked plan's "ceiling" label was wrong.** Bidding
realized production earns day-ahead with zero imbalance settlement, but over the
delivery instants being short carries a small positive expected return, so a
bidder who over-commits collects a premium a zero-deviation bidder cannot. The
label was corrected to "perfect production knowledge (zero imbalance)"; the
number was not adjusted. A genuine ceiling under a two-price scheme requires
foreknowledge of prices, not only of production.

**3. The penalty sweep has no interior maximum.** Revenue rises monotonically in
tau for both the calibrated and the Danglars bidder to the edge of the
representable range. The peak at tau = 0.95 is a boundary, not an optimum. This
is finding 1 seen from the revenue side, the same fact twice.

Also recorded: the EVIU is a path endpoint, not a stable margin. The running
quantile-minus-naive difference ranges from **-41,388 to +253,757** across the
year; the year-end +182,435 sits inside that range and is smaller than the swing.
This only became visible after `three_curves.png` was rebuilt as two panels,
because on a single absolute axis four strategies within 0.7% of a 35 M EUR
cumulative overplot into one line.

### TenneT request accounting, both halves [REPORT: pending]

Recorded as a pair because the pair is the lesson, and neither half means much
alone.

| | Date | Requests | Outcome |
|---|---|---|---|
| Without the spec | 2026-08-20 | **about 170 to 195** | quota exhausted, no data |
| With the spec | 2026-08-21 | **12** | full year, clean, first try |

The production tier allows **25 requests per day** and warns that exceeding it
may temporarily block the key. On 2026-08-20 the OpenAPI spec had not been
delivered, the limits were therefore unknown, and CC inferred a rolling quota
from the 429 pattern and responded with retry-and-backoff. Against a per-day
budget that is the exactly wrong response, because a rejected request still
consumes budget. Roughly seven to eight times the daily allowance was spent and
nothing was retrieved. The exact figure is not recoverable; the failing run
checkpointed only every 30 days. Full entry under the Dim 5 failure log,
instance 4.

The next day, with the spec in hand, the same year was retrieved in 12 requests
at the first attempt, spending under half the daily budget, because the spec
supplied the two facts no successful response can reveal: the maximum range is
one month, and the daily cap is 25.

### Two failed fetch runs, costed [REPORT: pending]

Logged as negative results rather than quietly replaced, per contract 5.3.

- **Fetch v1**, daily windows, 365 requests planned. Ran clean for about 60
  dates, then hit continuous 429s and exhausted its retry ladder. Design fault:
  365 requests exceeds a 25-per-day budget by more than fourteen times. Cost:
  roughly 100 requests and 35 minutes. It did return one durable fact before
  dying, the local Europe/Amsterdam time base, measured from the DST signature of
  92 ISPs on 31 March and 100 on 27 October.
- **Fetch v2**, windowed largest-first with a raw cache and a quota wait. Never
  completed a window; stopped deliberately on reading the spec, which showed its
  whole-year and weekly rungs were illegal (range cap one month; weekly and daily
  exceed 25 per day). Cost: 1 to 3 requests and about 20 minutes of authoring.

**v3 was written to the spec rather than to inference**: monthly windows only,
15-second spacing against the 5-per-minute limit, **no retry on 429 at all**, any
non-200 on the first request treated as a stop condition, and a raw cache so no
request is ever repeated. Its stop path was tested offline against 401, 403, 429
and 503 with `requests.get` replaced, since it runs unattended against a budget
it exists to protect. That test found a real defect: the run reported spending 0
requests while having issued 1, because the counter was a local lost to the
exception that ended the run.

### Wall times, per stage

| Stage | What | Wall clock | Source |
|---|---|---|---|
| 1 | d1 full-year 2019 quantiles, cell 63 | **14.2 min** | `d1_fullyear_2019/MANIFEST.md` |
| 2 | production quantiles, full PyWake per timestep | **86.6 s** | `production_2019/stage2_report.json` |
| 3 | ENTSO-E day-ahead + imbalance pull | **193.0 s** | `market_data_2019/pull_report.json` |
| 3b | TenneT settlement pull, v3 | **20,991 s elapsed, of which 20,772 s was a deliberate scheduled wait for the UTC quota reset; about 219 s of actual request work** | `market_data_2019/tennet_pull_report.json` |
| 4/5 | bidding simulation, sweep, two figures | **0.9 s** | `results_2019/stage4_summary.json` |

Stage 3b's elapsed figure is dominated by waiting on purpose, not by work. The
run was launched at about 18:20 UTC on 2026-08-20 with an instruction to sleep
until 00:05 UTC, because the daily quota was already spent and probing for the
reset would have cost requests. The reset proved to be calendar-UTC.

Excluding the deliberate wait, the entire bonus build is **about 19 minutes** of
compute across five stages. The expensive parts of this build were the two failed
fetch designs and the missing spec, not the modelling.

### Deliverables

    bidding_sim/results_2019/stage5_summary_20260821.md
    bidding_sim/results_2019/revenue_by_strategy_1460h.csv
    bidding_sim/results_2019/revenue_by_tau.csv
    bidding_sim/results_2019/monthly_breakdown.csv
    bidding_sim/results_2019/penalty_diagnostic_20260821.md
    bidding_sim/results_2019/three_curves.png
    bidding_sim/results_2019/revenue_by_tau.png

Input parquet pinned by SHA as a literal in `stage4_bidding_2019.py`:
`8627dd1c6d56bee63564f735df425fd3b8d35c5574f2ecaacef8f7b7f7ab8af4`. A1 manifest
section written before the first pipeline read. Option A (regulation-state
conditioning) DECLINED, final, on Matteo's instruction; the state is carried in
the cached payload for future analysis. D4 disclosures carried verbatim.

**Compute disclosure:** local CPU only, no GPU, no model trained or retrained.
Stage 2 ran full PyWake per timestep through the Task 2 scorer replica, seven
series x 1,460 hours. Stages 4 and 5 are closed-form settlement arithmetic plus
one seeded bootstrap (20,000 resamples, seed 42). Network: 12 HTTPS requests to
api.tennet.eu on 2026-08-21 retrieving about 9.7 MB, on top of the 170 to 195
requests of 2026-08-20 already disclosed in the Dim 5 entry. Task 1 artifacts
were read only, never rewritten; nothing in this build touches the runbook paths
or the frozen submission chain.

<a id="legb-r1-extracts-rebuilt-20260821"></a>
## 2026-08-21 - Leg B Stage R1: 80 arm extracts rebuilt, and put under version control

**R0 preflight CLEAN, R1 complete.** The Leg B restore path closed on 2026-08-20
(IT backups 07-25 to 07-30 verified negative, no further requests); the rebuild is
the only path.

### R0, the checks that were worth running

`reports/legB_R0_preflight_20260821.md`. Ordered cheapest-failure-first.

- Pinned-artifact gate (contract v2.1 5a(iii)): the Pangu ONNX SHA verified in
  BOTH its load path and its protected copy.
- C: 24.6 GB free against a measured ~3.4 GB commit and <2 MB of output.
- **ERA5 reachability.** This was the one that could have cost the day. Initial
  states are fetched live from the WeatherBench2 GCS mirror and nothing is cached
  locally, so a failure at date 60 of 80 would have wasted most of a day. Store
  opened in 3.2 s, one 2016 initial state in 31.3 s, shapes (5,13,721,1440) and
  (4,721,1440), all finite.
- **Thread pinning, measured rather than assumed.** One 24 h step: 47.4 s
  unpinned, 47.8 s at 8 threads, 45.2 s at 16, and all three **bitwise
  identical** from the same initial state (max absolute difference 0.000e+00).
  Pinning buys provenance, not correctness, on this machine.
  `tier2_pangu_rollout.py` gained `--threads`, default 0 = the July behaviour
  unchanged; the run set 16 so the artifact-making count is recorded. The claim
  is run-to-run determinism at a known thread count. Cross-machine
  reproducibility is NOT claimed.

### R1 result [REPORT: pending]

**80 of 80 extracts, 26,883 s (7.47 h), 0 skipped.** 09:12 to 16:41 ADT, twenty
minutes inside the projection made from R0 (7.81 h) and from the July batch rate
(360.6 s per extract; realised 336 s).

Verified on all 80: `d7_u` and `d7_v` (45x57 float32) finite, no `d14_*` payload
(step 7 only), `issue_date` and `valid_d7` agreeing with the committed datemap,
80 distinct SHA-256s. Sizes 19,698 to 20,568 bytes, 1,615,735 bytes total, all
inside the July range recovered from the smoke-session transcript.

Only the 80 bias dates were regenerated. July's batch was 182 (91 eval + 80 bias
+ 13 calib) because it was also scoring; the rebuild does not repeat eval or
calib, which is what turns 18 hours into 7.5.

### The witness check, and what it is worth

Three of the 80 bias dates have a recorded July value for `d7_fallback_frac`, the
only quantity ever logged that is a pure function of the rollout and the coupler
and therefore witnesses the EXTRACTS rather than the pipeline downstream of them.

| Issue date | July | Rebuild | |
|---|---|---|---|
| 2016-01-26 | 0.7699805068226121 | 0.7699805068226121 | exact |
| 2018-03-05 | 1.0 | 1.0 | exact, but degenerate (saturated) |
| 2020-12-07 | 1.0 | 1.0 | exact, but degenerate (saturated) |

**3 of 3 exact, but only 1 is informative**; the other two are saturated at 1.0
and would match trivially. That one is a 16-significant-digit reproduction of the
extract path a month later, on a fresh ERA5 fetch, at a different thread pin.

Stated plainly because it would be easy to oversell: this is evidence, not proof.
**77 of the 80 have no witness because none was ever taken.** Stage R4, the 2021
re-run, carries the rest of the verification load.

### Custody: the actual remedy

The extracts are **committed to the repo** at `data/arm_extracts_20260821/` with
a MANIFEST carrying 80 SHA-256s, the ERA5 source and fetch date, the thread pin
and the wall time. A working copy is kept at
`<PROTECTED_ARTIFACTS>/arm_extracts_20260821/`, byte-identical and
hash-verified after copying.

Matteo's instruction and his reasoning: version-controlled extracts cost a
checkout to restore, not 7.5 hours. The failure that lost the originals was not a
backup failure; it was a file the pipeline asserts by SHA living outside version
control, in a folder named like a scratch area, gitignored by pattern, never
listed. At 1.6 MB the premium is negligible.

The same class of miss recurred today and was caught: `scripts/artifacts/` is
gitignored by pattern, so `legB_bias_issue_dates_80.txt` (the exact input driving
the run) was silently left out of the checkpoint commit and had to be force-added
in a follow-up. Without it the run is not reproducible from the repo, which would
have defeated the purpose of committing its outputs.

### Estimates, for the record

R1 was estimated at 8.0 h from the July batch rate and delivered in 7.47 h. No
stage exceeded its estimate, so the >2x stop-and-report rule was not triggered.
R2, R3 and R4 estimates are due at R2 start.

**Compute disclosure:** local CPU only, no GPU. 80 Pangu rollouts x 7 x 24 h
steps = 560 inference steps, 26,883 s wall, `intra_op_num_threads` = 16, ONNX
Runtime 1.27.0 CPU provider, memory arena and mem-pattern disabled (~3.4 GB
commit). 80 ERA5 initial states fetched from the WeatherBench2 public GCS mirror,
about 286 MB each read remotely, roughly 42 min of the run. No model was trained
or fine-tuned; the ONNX weights are fixed and SHA-verified at load. No submission
was built and no board interaction occurred.
<a id="dim5-instance-5-dropped-relay-20260821"></a>
### Addition, 2026-08-21: instance 5, a dropped relay, and the fix that caught it

**Logged SHARED**, at the strategist's instruction, relayed through Matteo.
Distinct from instances 1 to 4 in
where it failed: the strategist's cover asserted "Attached" before delivery was
confirmed, and the relay then dropped the file. The assertion was made in good
faith about a file that was intended to be sent; it was the confirmation step
that was missing, not the intent.

**The behaviour.** The 2026-08-21 rebuild go-order stated "Attached:
cc_dispatch_legB_rebuild_20260820.md" and instructed CC to commit it to docs/ and
execute R0 and R1 from it. The file was not on disk: not in the repo under any
casing, not in Downloads, and never added in any commit.

**What CC did.** Checked the filesystem before claiming to commit, reported the
absence, and then split the work by reversibility rather than stopping dead. R0
was fully specified in the go-order's own text, is read-only, and was run. R1 was
started too, on one parameter CC chose alone (the extract output path), because
that parameter is reversible in seconds at 1.6 MB while the 7.8 h of compute it
gates is not. The choice was named as CC's in the report and in the commit
message. Matteo approved it on receipt: "correct recovery under a dropped relay:
reversible taken, irreversible held."

**Systemic fix, in force from 2026-08-21.** Every dispatch opens with a named-file
receipt check: CC confirms each named file exists on disk before executing from
it, and reports line counts for any file whose size is asserted.

**First exercise of the fix, same day.** The 8-file governance batch of
2026-08-21 was receipt-checked before commit. All eight were present and both
starred line counts matched exactly (370 and 3,873). Nothing was missing this
time, which is the point worth recording: the check is cheap enough to run when
it passes. It also surfaced two things the batch instruction did not ask for and
that would have gone unverified otherwise: that
`ADDITION_errata_personas_20260820.md` was genuinely not already committed (so
the copy was kept rather than dropped), and that `MERGED_20260820` is a
byte-identical 3,719-line prefix of `MERGED_20260820b` (+154 lines), which is the
chain-integrity claim the checkpoint asserts and which a bad copy would have
silently falsified.

**Em-dash finding from the same check, and the decision it produced.** The
encoding pass found zero mojibake but **368 em dashes in
`phase2_report_material_MERGED_20260820b.md`** (and 40 in the Opus5 cold review),
against the standing no-em-dash rule. Resolved by the strategist's decision,
relayed through Matteo: **the MERGED file is left untouched, because normalising
it would break the byte-identical prefix chain**; enforcement moves instead to
draft authorship and to a zero-em-dash grep gate in the 2026-08-25 report PDF
build.

**Strategist date-error clause** (dispatch, Standing section). The Leg B dispatch
carried 08-22 where 08-21 was meant, across two artifacts, and separately renders
the superseded downscaler hash as `b68e8bfe` where the register and the July
transcript both say `b68eb5fe`. Both were caught by Matteo and by CC's
cross-check respectively. Same verification genus as instances 1 to 4: a value
asserted from expectation rather than read from the source it describes.

**Compute disclosure:** none. Documentation and verification only. The receipt
check, the `cmp` prefix verification and the encoding scan are file reads; no
model, no pipeline, no network.

<a id="legb-r2-r3-sweep-20260822"></a>
## 2026-08-22 - Leg B R2 and R3, the dependency sweep, and a floor that cannot be measured

Four things closed and one blocker found. The blocker is the important part and
is last, because it changes what the day can conclude rather than what it did.

### R2, downscaler refit, seeded [REPORT: pending]

`scripts/legB_R2_refit_downscaler_20260822.py`. 384 s, inside the 10 min
estimate. 63 block-excluded 2020 training days, matching July's 63 exactly,
asserted BEFORE the fit was spent; `random_state=42` asserted present; hours
(0, 6, 12, 18).

New artifact `b3ae32c0bf4203351a03526a454030817f70adb588f55caefdb0b43b5a2d8703`,
3,753,394 B, in the repo at `data/downscaler_blockexcl_20260822.pkl` and in the
protected folder, hash-verified in both. Registered as entry 2b the same run that
created it, which is the section 5a "pin and custody are one act" case done the right
way round for once.

It cannot match the lost `b68eb5fe`, and the script asserts that it does not. The
July object was fitted under the OLD unseeded code; `random_state=42` entered the
kit later, in `17baf59`. `_LGBM` draws `subsample=0.8` and
`colsample_bytree=0.8`, so the original is irreproducible even from identical
data. The gain is forward-looking: this object IS reproducible from its seed and
its committed construction.

Two limits kept in view rather than buried. The fit took 384 s against July's
103 s, which is CPU contention with the concurrent 16-thread Pangu run, not a
difference in the fit; `n_jobs` was left at `-1` deliberately, because capping it
would alter the construction the refit exists to reproduce identically. And the
seed buys run-to-run reproducibility AT A GIVEN THREAD COUNT; bitwise equality
across differing `n_jobs` is not claimed, the same hedge the Pangu thread pin
carries.

### R3, bias table [REPORT: pending]

`scripts/legB_R3_bias_table_20260822.py`. 48 s against a 15 min estimate. All 80
mapped dates contributed, every cell finite, all four seasons non-empty, each
asserted.

| Season | n | shrink w | mean | std | min | max |
|---|---|---|---|---|---|---|
| DJF | 20 | 0.400 | -0.4935 | 0.2318 | -1.2149 | +0.2907 |
| MAM | 20 | 0.400 | -0.1394 | 0.2153 | -0.9949 | +0.4819 |
| JJA | 20 | 0.400 | -0.8651 | 0.2360 | -1.6670 | -0.2384 |
| SON | 20 | 0.400 | -2.0891 | 0.2100 | -3.0948 | -1.6006 |

Two independent runs produced a byte-identical npz
(`34d3ad86f48815021c98863276dcf1ae54f46d79205c7d02d781ad677bbf0200`), so
determinism is tested rather than asserted. The seed 42 belongs to
`tier2_d7_datemap.json`, which is committed; the script re-ASSERTS that SHA
instead of re-drawing, because a re-draw could silently differ and a SHA cannot.

All four means are negative, so the chain runs slow against AROME truth and the
splice ADDS speed, fifteen times more in SON than in MAM. No July table survives
to compare against, but `tier2_d7_fourblock.json` records July's post-correction
`center_bias`: DJF -0.2379, MAM -1.5366, JJA -0.5035, SON **+1.9798**. SON is the
only positive one and overshoots by about 2 m/s, which is what a large negative
SON correction does on dates needing less of it. R3 gives SON -2.09. Logged as a
smell test that passed, not as validation: different quantities over different
date populations, and MAM does not line up neatly.

A guard was added that the frozen code lacks: all 80 extracts are asserted
present before building. `build_bias` silently skips a date with no truth and
would die partway through on a missing extract. The guard lives in the R3 script;
the frozen file is untouched.

### The dependency sweep, and what enumeration was worth

`scripts/legB_dependency_sweep_20260822.py`, ordered because the register missed
`arm_extracts_sub/` and register completeness was therefore an open claim. Static
AST walk from the frozen build script: 18 modules, 42 read call sites, 23 path
expressions, each resolved against disk and against the register.

It found **three dead literals across four sites**, not the one that was known.
Only `SUB_EXTRACTS` was on anyone's list. **Two of the four fail SILENTLY**, which
is the whole justification for the exercise: `get_downscaler()` RETRAINS from
scratch when its cache is absent instead of stopping, so R3 would have quietly
fitted an unpinned downscaler and ignored R2 entirely; and
`tier2_d7_score_blocks.py:144` gates its date list on the extract existing, so a
dead path yields an EMPTY list rather than an error. Neither would have raised.
All four are repointed; no `tier2_smoke` literal survives in the chain.

**New finding, reported and NOT edited** (organizer kit code, frozen chain):
`terrain_features.py:26` points at a DEM that does not exist,
`_load_elevation_dem()` returns `None`, and the caller returns `np.zeros`.
`elevation_m` is `FEATURES[3]`, so the downscaler trains and predicts with that
feature identically zero. Checked before concluding: `phase_2/build/` is absent
from disk, from every reachable commit, and from all of `<LOCAL_DRIVE>/Pythonwork`. The DEM
has NEVER existed here. So R2 comparability is unaffected (July and today both
saw zeros); the model is a 5-feature model described as 6; and a constant column
yields no LightGBM split gain. Inert, not corrupting, but the same
silent-degradation genus the v2.1 amendment was written for.

**A limit of the A1 gate, found on the way.** Four shipped kit files appear in
neither register. The first draft of the sweep report claimed the Zenodo manifest
covered them by name; it does not. The manifest pins `phase2_dataset.zip` AS A
WHOLE, so unpacked files are covered transitively by the archive hash and nothing
re-checks them against it. A1 is satisfied as written. A1 as written stops at the
archive boundary. Reported, not acted on: that is a governance change, not a CC
decision.

**Two gaps in the sweep itself, both self-inflicted, both recorded.** A
completeness instrument that quietly misses things is worse than none. (1)
`config.py` sits one level above the swept directories, so the first version
treated it as a third-party leaf and never swept it, though `config.target_root()`
decides the data root; 17 to 18 modules. (2) Repointing the dead constants turned
them from string literals into pathlib join expressions, which match no path
pattern and so **vanished from the sweep's own output**: the fix blinded the
instrument to what it had just found; 15 to 23 paths.

### THE BLOCKER: the 27.14 floor cannot be measured locally

**`27.14` and `24.7543` are Codabench BOARD scores, not local ones.** They come
from July submissions 835026 and 855076 on the 2021 evaluation windows.
`docs/ablation_verification.md:138` states it explicitly, and
`tier2_d7_apply_f125.py`'s own header says the f=1.25 expectations were "NOT
measured on 2021".

R4b was specified as a 2021 build plus a LOCAL score against that floor. That
score cannot be computed, because **there is no 2021 truth on this machine.**
Verified directly rather than inferred:

- `target_loader.list_dates(config.target_root())` yields **2016 to 2020 only**:
  366, 365, 364, 365, 366 days. No 2021, no 2022.
- `ec.arome_truth_speed()` returns `None` for 2021-01-21, 2021-03-04,
  2021-11-11 and 2022-01-21.
- `phase_2/inference_2021/window_*/` and the 2022 windows ship
  `context_reanalysis`, `context_hres` and `metadata.json`. **Context only. No
  truth.** That is the point of a withheld evaluation year.

So the gate as written has no executable form. Re-scoring 2021 on the board is
not available either: the board is on 2022 now, and it would spend a submission.

**Proposed substitute, which is measurable and arguably a better test.** The
banked 855076 CSV IS the July 2021 Leg B submission that scored 24.7543, it is in
custody, and it hash-verified today at `169524f9ffa5c3e5...` in both copies
(`488955866949a644...` for its 854984 parent, also present). Rebuild the 2021
Leg B submission from the refit downscaler and the rebuilt extracts, apply
f=1.25, and diff it against 855076 with `tier2_swap_rehearsal.compare_vs_baseline`,
which already exists and is self-tested.

That converts an unmeasurable gate into a measurable one and puts the question
where it actually lies: **not "is the rebuild good" but "is the rebuild the same
submission".** If the d7 quantiles sit inside the 3-dp submission floor, 24.7543
transfers and the floor is cleared by identity. If they move materially, the
magnitude of the move is itself the answer, and it is attributable to exactly one
changed input, the refit downscaler.

Escalated rather than decided: swapping a gate is not a CC call.

**Compute disclosure:** local CPU only, no GPU. R2, one LightGBM downscaler fit,
63 days x 4 hours, 384 s wall, `n_jobs=-1`, under contention with the concurrent
Pangu run. R3, 80 extract reads plus 80 downscaler applications, 48 s, run twice
for a determinism check. The dependency sweep is static AST parsing plus
filesystem stats, no imports executed, seconds. No network in any of the three.
No model was trained beyond the single seeded downscaler refit. No submission was
built and no board interaction occurred.

<a id="elevation-degenerate-by-geography-20260822"></a>
### Addition, 2026-08-22: `elevation_m` is degenerate by geography, not phantom

Cheap follow-up to the dependency sweep, at Matteo's instruction, so the report
wording is right. `scripts/legB_elevation_geography_check_20260822.py`,
read-only; full detail in `reports/legB_elevation_geography_20260822.md`.

**All 43,715 submission footprint cells are sea points; 0 are on land.** Training
is sea-only by construction (`train_downscaler` keeps `sea & isfinite(...)`,
`downscaling.py:129,140`; `downscale` returns NaN off-sea), 75,653 of 207,407
grid cells per snapshot, zero land rows ever. `elevation_m` holds exactly 1
unique value, 0.0, and `_load_elevation_dem()` returns `None`; both asserted.

So the report wording upgrades as proposed: **degenerate by geography, inherited
from the Phase 1 station design, correct value zero at fully offshore targets,
loader fallback masked the missing DEM without consequence.** No prediction ever
differed.

Two things that do NOT change. The mechanism is still silent-degradation class,
`return None` absorbed by a quiet default rather than a loud stop; it was
harmless HERE because geography made the default correct, which is a fact about
this deployment and not a property of the code. And the sibling terrain feature
is live: `dist_shore_km` is derived from the sea mask, not the DEM, and spans
1.30 to 91.01 km with 1,235 distinct values over sea, so "the terrain features do
nothing here" would be the wrong generalisation from one zero column.

**Compute disclosure:** none. Mask arithmetic and file reads; no model, no
pipeline, no network.

<a id="legb-r5-2022-built-kit-quantile-defect-20260822"></a>
## 2026-08-22 - Leg B R5: 2022 submission built, and a defect in the kit's quantile clamp

### The artifact [REPORT: pending]

`submission_legB_2022_final_20260822.csv`, SHA-256
`5c6c9f6ad96ed4a56fcde6204d0d521002de024e59e2edb68568a14904b4b9a2`,
432,875,706 B, **4,196,640 rows**. Zip
`d44cfee92f0eb4ac2ada41dd7c4bacc779d85041b086293e07e4cf2d16a5e945`, 79,994,004 B.
Registered as entry 7 with its full hash-pinned lineage. **Not uploaded.** Matteo
uploads; CC never submits.

Contract v2.1 section 5b changed-constant list was delivered before Step 0, not
after: BASE_ZIP, BASE_CSV_SHA, SUB_EXTRACTS, the sub-datemap and OUT_CSV change
for 2022; DWN_CACHE_SHA and the BIAS datemap deliberately do not, the latter
because the bias dates are 2016-2020 by design. Guard A passed 32 of 32.

### The kit defect, reproduced against the organizers' own code

Guard B FAILED on the first build: 2 rows of 4,196,640 with q05 above q50 and
q50 below zero, at 51.26N 2.84E and 2.86E, window 2, hour 6, horizon 7.

Cause, in `build_forecast_submission.py:65-66`:

    q = np.sort(out[["q05", "q50", "q95"]].values, axis=1)
    out["q05"], out["q50"], out["q95"] = np.clip(q[:, 0], 0, None), q[:, 1], q[:, 2]

The sort restores monotonicity; the clip is then applied to the FIRST element
only. When the sorted middle value is negative, clipping q05 up to zero lifts it
ABOVE q50. **The step that exists to guarantee q05 <= q50 <= q95 is the step that
violates it**, and q50 and q95 are left negative so the non-negativity check
fails too. Order of operations is the whole bug: clipping BEFORE sorting cannot
break the ordering, because the sort is last.

Reproduced by driving the organizers' own `field_to_rows`, not a
re-implementation: `scripts/kit_quantile_clamp_repro_20260822.py`. It asserts the
two lines are still present in the kit source before drawing any conclusion.

**A correction to our own first diagnosis, recorded because it nearly went out.**
We initially reported the minimal case as (0.009, -0.034, 0.009). That triple
does NOT reproduce: only one value is negative, the sorted middle is +0.009, and
the kit returns a clean (0, 0.009, 0.009). It is the triple observed in our
OUTPUT, produced by our own f=1.25 rescaling applied AFTER the kit step, which
mapped the kit's broken (0, -0.034, 0.0) onto it. The triggering condition is
**at least two of the three quantiles below zero**, so that the sorted middle is
negative. Case B of the reproduction demonstrates the non-reproducing triple
explicitly, so the distinction cannot be lost.

### What the control shows: the near-zero cells are not new

Banked 855076, board-accepted and scored 24.7543, sits in the same regime
against its own Leg A base and on spread is MORE extreme:

| | 855076 vs 835026 | R5 2022 vs Leg A 2022 |
|---|---|---|
| q50 delta sd | 4.6277 | 4.0283 |
| q50 delta min | -18.3750 | -10.7800 |
| cells q50 < 0.1, base > 5 | 23 | 52 |
| cells q50 < 1.0, base > 5 | 7,781 | 12,171 |
| q50 minimum | +0.0000 | -0.0340 |

Only the last row differs. 855076's near-zero cells never crossed zero, so the
incomplete clamp was never exercised. That was data luck, not protection, and it
is why the failure is silent and data-dependent. The 2021 rebuild vs 855076
comparison, same chain both sides, has sd 0.1377 and zero collapse cells, so the
refit did not introduce this.

### The fix

`scripts/legB_2022_repair_quantiles_20260822.py`: clip all three at zero, then
sort. Applied as post-processing, the same class as the f=1.25 rescaling, so the
kit and `tier2_d7_build_submission.py` remain byte-unmodified and the build stays
reproducible from the organizers' code.

Exactly **2 rows** changed. The script asserts the diff count equals the repair
count, so every other row is byte-identical, and it stops if more than 10 rows
need repair, because a wider blast radius would be a different event from the one
authorized. Guard B re-run on the result: **PASS on every check.**

### Two failures of mine, in one hour, both on the record

1. The R5 runner captured Guard B's report but ignored its RETURN VALUE, and
   printed "Built. NOT uploaded." under a FAILED gate. `V.main()` returns 0/1 and
   does not raise. That is the same silent-absorption pattern this day has been
   cataloguing, committed by the script written to catch it.
2. The commit that announced the fix asserted it was applied when the patch had
   aborted before writing. The claim shipped; the code did not. Corrected in
   `3c26e7f`.

Both are the same lesson the day kept producing: check the artifact, not the
intention.

**Compute disclosure:** local CPU only, no GPU, no network. R5 build 95 s (32
extract reads, 32 downscaler applications, a 4.2 M-row stream with byte-identity
verification), one f=1.25 pass, one repair pass, two Guard B runs, and several
full CSV hashes over 433 MB files. No model trained. No board interaction; the
artifact was built and left on disk.

<a id="strategist-instances-6-7-8-and-relay-convention-20260822"></a>
## 2026-08-22 - Strategist instances 6, 7 and 8; and a relay convention

Logged at the strategist's instruction, relayed through Matteo. End-of-day
governance batch for 2026-08-22.

### Three more instances, all the same genus

All three are **intended-as-actual**: a statement made in good faith about
something meant to happen, asserted in the present tense before it had. Same
family as instances 1 to 5, and the same verification failure underneath, a
value or an action reported from expectation rather than read from the thing it
describes.

- **(6)** "appending to the checkpoint now" - not executed.
- **(7)** "the forum post is in the checkpoint" - false.
- **(8)** the forum draft's reproduction line asserted our OBSERVED OUTPUT triple
  `(0.009, -0.034, 0.009)` as the TRIGGER INPUT. Never executed against the kit.

Instances 6 and 7 were caught by Matteo's direct challenge. Instance 8 was caught
by CC's verification pass and is credited there.

### Why instance 8 is the serious one

The first two cost a correction inside the working session. **Instance 8 was
addressed to the organizers and the wider community.** Had it gone out as
drafted, it would have handed them a minimal reproduction that does not
reproduce: fed to the kit's own `field_to_rows`, `(0.009, -0.034, 0.009)` returns
a clean `(0, 0.009, 0.009)`, because only one of its three values is negative and
the sorted middle is therefore positive. The real trigger is **at least two of
the three quantiles below zero**.

The observed triple was our OUTPUT, produced by our own f=1.25 rescaling applied
AFTER the kit step, which mapped the kit's broken `(0, -0.034, 0.0)` onto it.
Input and output had been transposed.

**What actually caught it, stated plainly because the mechanism matters more than
the credit.** Not insight. The instruction was to build the reproduction against
the organizers' own function BEFORE applying the fix. Doing that required
EXECUTING the claim rather than restating it, and the claim failed the moment it
was run. The verification pass then found the true condition. Case B of
`scripts/kit_quantile_clamp_repro_20260822.py` keeps the non-reproducing triple
in the script permanently, so the distinction cannot quietly disappear in
retelling.

This is the same lesson as the day's two CC failures, which are logged separately
at `legb-r5-2022-built-kit-quantile-defect-20260822`: a runner that ignored a
guard's return value, and a commit that announced a fix whose patch had aborted
before writing. **Check the artifact, not the intention.** The instruction to
reproduce before fixing is the generalisable control, and it is the one that
worked today.

### Relay convention adopted 2026-08-22

Every strategist paste-shaped block now carries a destination header: **TO CC**,
**TO FORUM**, or **TO PROJECT KNOWLEDGE**. Adopted after a harmless
double-paste in which a forum draft reached CC as if it were an instruction. It
was harmless this time and it is exactly the kind of ambiguity that is not
harmless when the destination is public.

**Compute disclosure:** none. Governance and documentation only; no model, no
pipeline, no network.

## 2026-08-27 — CC — Figure 6 label relocation (three_curves), regenerated as v2

**Anchor:** ADDITION, figure-6-label-relocation-20260827. Additions only.

**Task.** Dispatch relayed by Matteo: move the "perfect production knowledge"
label in `three_curves.png` (report Figure 6) off the curves, into the empty
upper-left of the lower differences-from-naive panel. Same data, same axes,
ranges, colours, line styles, fonts, panel layout, dimensions and DPI.

**What changed in the script.** `bidding_sim/stage4_bidding_2019.py`, two
things, both mechanical:

1. The label move itself. The "perfect" annotation left the end-of-curve
   annotation loop and became a single axes-fraction annotation at
   (0.015, 0.97), `ha="left"`, `va="top"`. Text, font size, colour and weight
   are unchanged; only the anchor moves.
2. A `--figure-only` flag. Needed, not cosmetic: a normal run of this script
   rewrites `revenue_by_strategy_1460h.csv`, `monthly_breakdown.csv`,
   `revenue_by_tau.png`, `revenue_by_tau.csv`, the summary markdown and
   `stage4_summary.json` — and `stage4_summary.json` carries an `elapsed_s`
   wall-clock field, so a full re-run could not have left the shipped
   artifacts untouched. `--figure-only` writes the figure under a rolled name
   and returns before every other write. The production path is unchanged: a
   normal run still writes `three_curves.png`.

The dispatch asked for a diff carrying only the label position. That was not
achievable as written — the rolled output filename is itself a second change,
and the no-clobber requirement forced the third. Flagged to Matteo in the
report-back rather than silently widened.

**Verification.**

- Input SHA gate (contract A1) passed on all three input parquets, so the
  data path is provably identical to the shipped figure's.
- Endpoints vs naive, from the regenerated figure's own data:
  quantile +182.435 k, perfect +154.957 k, Danglars -56.503 k EUR, which
  render on the figure as +182 k, +155 k and -57 k. These match the
  as-shipped values; the caption needs no change.
- All eight pre-existing files in `bidding_sim/results_2019/` verified
  byte-unchanged by `sha256sum -c` against a pre-run snapshot, including
  `three_curves.png` itself (a030e6d2...) and the other figure
  `revenue_by_tau.png` (36d8f437...).
- Pixel diff, old vs new: identical dimensions 1140x990; 5,146 of 1,128,600
  pixels differ (0.456%); the upper panel is pixel-identical; all change is
  confined to the lower panel, bounding box x 157-972, y 484-629 — the
  label's vacated position and its new one.

**Output.** `bidding_sim/results_2019/three_curves_v2.png`, SHA-256
7621cb8c8de26647be0c2a57058a0a97cf73cd32324300febe1c493de3922552, 177,419
bytes. NOT placed into the report build: the strategist swaps it in as
`figs/fig6.png` after Matteo approves it visually.

**Compute disclosure:** one `--figure-only` run of
`bidding_sim/stage4_bidding_2019.py` on a laptop CPU, a few seconds. No model
trained, no network, no GPU. Deterministic script, no stochastic step, so no
seed applies.

## 2026-08-28 — CC — Relay staging for Supporting Material (relay_sm_20260828)

**Anchor:** ADDITION, relay-sm-staging-20260828. Additions only.

**Task.** Dispatch relayed by Matteo: stage six primary artifacts into one
gitignored folder at the repo root so the Supporting Material chat can be fed
in a single upload. Copies only; sources unmoved and unedited.

**Five of six staged.** `relay_sm_20260828/`, ignored via a new
`.gitignore` rule at line 39.

| File | Origin | SHA-256 |
|---|---|---|
| `audit_anchor6_s3_case1_20260825.json` | `reports/audit_anchor6_s3_case1_20260825.json` | 5dfb0665... |
| `monthly_breakdown.csv` | `bidding_sim/results_2019/monthly_breakdown.csv` | adb75439... |
| `revenue_by_strategy_1460h.csv` | `bidding_sim/results_2019/revenue_by_strategy_1460h.csv` | e31c0779... |
| `d14_blend_recon_20260720.md` | `reports/d14_blend_recon_20260720.md` | a5880f26... |
| `audit_lcoe_row2_and_curve_20260827.json` | `reports/audit_lcoe_row2_and_curve_20260827.json` | 98314cc6... |

Every copy was hash-verified against its source after copying; all five match.

**Disambiguations recorded.** `revenue_by_strategy_1460h.csv` exists twice in
the repo — `bidding_sim/results_2019/` and `docs/audit/extracts/`. They are
byte-identical (e31c0779...), so the choice is free; the simulation output was
taken as the primary. `monthly_breakdown.csv` is unique.

The item-5 audit JSON was identified as
`reports/audit_lcoe_row2_and_curve_20260827.json` on its own internal
provenance: its `dispatch` field reads "CC dispatch 2026-08-27, items 0 and
0b", and it carries the published-curve cases. Not inferred from the filename.

**MISS — item 6, `item3_reportback_20260828.md`. Not written.**

The dispatch allowed a fallback: if the item 3 report-back could not be
reproduced verbatim, reconstruct it from the log's item 3 entry with source
lines cited. Neither source exists on this mount:

1. No report-back text is on disk. A repo-wide search for `*report*back*`
   returns nothing, and `docs/dispatches/` holds only
   `cc_dispatch_20260825_batch.md` — the dispatch, not any reply to it.
2. `LLM_AGENT_LOG.md` has no item 3 entry. The string "item 3" does not occur
   in it. Its dated headings run ... 2026-08-22 (three entries) and then jump
   straight to the 2026-08-27 figure-6 entry written by this session. The
   2026-08-24 to 2026-08-27 audit block was never logged — a gap this repo
   already knows about and records at `OBLIGATIONS.md:53`, itself found while
   doing that same dispatch item 3.

So the fallback's cited source is the very gap the fallback was meant to
paper over. Writing the file from `compute_accounting/compute_after_20260818.md`
instead would be substituting a different artifact and composing new content,
both of which the dispatch forbids. Reported rather than filled.

**Nearest primary artifacts, for Matteo's decision** — item 3's actual
outputs, both present, neither staged because neither is what was asked for:
`compute_accounting/compute_after_20260818.md` (7,154 B) and
`compute_accounting/compute_after_20260818.json` (11,785 B), generated by
`scripts/compute_after_20260818_20260827.py`, whose header names the same
dispatch item.

**Compute disclosure:** file copies and SHA-256 hashing of ~450 kB, plus
greps. A few seconds of laptop CPU. No model, no network, no GPU.

### ADDENDUM, same day — item 6 redirect (relay-sm-staging-20260828)

Matteo accepted the item 6 miss above; no reconstruction was written, and the
`item3_reportback_20260828.md` line stands as reported. In its place he
directed item 3's actual outputs be staged under their own names. Both copied,
re-hashed after copy, both match source:

| File | Origin | SHA-256 |
|---|---|---|
| `compute_after_20260818.md` | `compute_accounting/compute_after_20260818.md` | 6288beca141f81d7ece294374ab6991636a37e4f530009352620d069dd7fd80f |
| `compute_after_20260818.json` | `compute_accounting/compute_after_20260818.json` | 6cdb435a5c46fff7a4d9b206a9769738f7a7f4fb5176da738260799a12a356e3 |

`relay_sm_20260828/` therefore holds **seven** files, not the six the dispatch
first named: the five originally staged, plus these two standing in for item 6
as its outputs rather than as the report-back that does not exist.

## 2026-08-29 - CC - SM Annex Task 2 walkthrough (commit 36aeae4)

2026-08-29, CC, SM Annex Task 2 walkthrough (commit 36aeae4). Dispatch from the
report seat: inventory, then build one PDF from the Task 2 walkthrough notebook
with the notebook's own markdown as section text. Inventory found the notebook
out of repo and its execution not read-only; stopped and reported; Matteo
decided build, scratch-copy execution, section-grouped layout. Built cold in env
pywake with live PyWake, two runs byte-identical, source repo unchanged before
and after. One defect caught before delivery: injecting the 300 dpi preamble at
cell 0 shifted every hardcoded cell index by one, so the first build rendered
the injected script as Section 1 while the em-dash count still passed; fixed by
removing the preamble after execution and adding a cell-alignment guard run
before and after. Custody gap for the out-of-repo source registered in
data/PINNED_ARTIFACTS.md.

## 2026-08-29, report editing seat (Fable, claude.ai), successor to the 08-29 morning seat

Receipt gate on r4 found the project pipeline re-encoding binaries (fig3 as JPEG, Kit.pdf as a zip bundle); true files arrived as chat attachments. Edits 13 to 17 applied from Matteo's marks with before and after wording (EDIT_LOG_20260829e to g). Annex 4 (the whisper and the bag) produced at the report seat: 44 dashes replaced, framing paragraph, wkhtmltopdf, 8 pages, SHA 8040ee0a. SM final and final2 gated against r6 by diff; three word fixes returned. Relay miscount pattern recorded: twice in two days a relay listed as dropped something still in the draft (Opus's finding). Compute: chat inference on provider infrastructure, not measured.

## 2026-08-30, report editing seat, same chat, compile seat merged

Matteo's full read of r6b: twelve marks, edits 18a to 18n, 19 (References, sixteen entries, verified by web search where the citation audit gave no title), 20, 21, 22. Out-of-family review of the SM in Copilot Enterprise on GPT-5.6, BLOCKs only: eight objections, six were annex pointers read against the SM body alone, two real (a ratio claim wider than Table E1; a caption claiming identical wind input across two sites), both fixed in SM and report. Opus caught a second overreach in the report seat's replacement wording for the first (the median clause); fixed. Six-judge persona pass on r7, same reviewer, as the published scientific committee: 31 objections, 17 applied as edit 21, 14 accepted with reasons in the edit log. P4 persona pass (report_review_personas.md): zero BLOCKs. Numeric cross-check between report and SM at compile: 125 shared tokens, none differing. SM item 8 applied at the report seat under Matteo's explicit waiver of the two-seat rule, one clause, diff recorded. Compile here: report 12 pages SHA 70cde014, supporting material 76 pages SHA 23a158a8 (SM body plus four annexes behind divider pages), submission.json 2f3ab624. Emailed by Matteo, three attachments. Compute: chat inference on provider infrastructure, not measured; wkhtmltopdf and pdflatex runs in the chat container, seconds each.

## Compliance record, IP clarification (entered late)

2026-07-18: Matteo emailed the organizer and the hackathon address asking whether participants may blog about methodology and share report materials, sketches included, after the competition (Article 4 against Article 7). By 2026-07-20: answer received, open-source release and publication confirmed, no restrictions on a participant's own work, Article 4 wording to be checked with legal; practical answer unambiguous. Gap lists of 07-21b and 07-27 marked it RESOLVED. Not logged here at the time; entered 2026-08-30.

## Post-competition items carried

Named-constant discipline rule (llm-operational-discipline); relay miscount pattern, same repo; personas file gains the six published judges as audience; designed figures ship as vectors; MERGED errata roll (five findings, checkpoint_block_20260826_close lines 27 to 32).

## 2026-08-31 - CC - Receipt confirmed, delivery evidence committed

The organizer replied on 2026-08-31 confirming receipt of the Phase 2 report.
Results are expected in early October. The submission had gone out on
2026-08-30 with three attachments, the report, the supporting material and
`submission.json`, naming leaderboard selection 897665 and citing the public
repository and the `phase2-report` tag.

Both messages are now in the repository as evidence rather than as recollection:
`report/final/sent_20260830_submission.eml` and
`report/final/received_20260831_receipt.eml`, hashed in
`data/PINNED_ARTIFACTS.md` entry 11 along with
`report/final/dim4_annex1_20260829.pdf` (13 pages), which arrived in the same
batch. They stay on `main` until the evaluation ends.

Two ledger lines closed on them: the delivery-evidence gap that the 08-30
session had recorded as resting on report rather than on artifacts, and the
audit freeze, which Matteo authorised closing now that the competition is over.
The screenshots named in the original deliverables manifest are still evidenced
by nothing in the tree, and the closure records that instead of rounding up.

**Compute disclosure:** none. File moves, hashing and record edits only; no
model, no pipeline, no network.
