# Item 6 post-statement: what was actually done

CC, 2026-08-25. Pairs with `anchor6_prestatement_20260825.md`, which was
written before the run. Authorised after cell 4 was dropped (08-24 decision),
in-repo, no quarantine.

## Result

**ANCHOR 6: PASS.** Not merely within tolerance: **bit-identical to the last
double** on every reported quantity.

| Quantity | Recomputed | S3 record (08-19) | Delta |
|---|---|---|---|
| net capacity factor | 0.47094829333315191 | 0.47094829333315191 | identical |
| wake loss fraction | 0.096992407141249015 | 0.096992407141249015 | identical |
| AEP GWh | 4991.8635300140768 | 4991.8635300140768 | identical |
| mean hub wind speed | 10.46489499321267 | 10.46489499321267 | identical |

Tolerance asked for was 0.1 pp on net CF. Delta is 0.000000 pp.

Machine-readable: `reports/audit_anchor6_s3_case1_20260825.json`.
Script: `scripts/audit_anchor6_s3_case1_recompute_20260825.py`.

## Environment, and the PyWake pin

Fresh conda env `s3anchor20260825`, created for this run, python 3.11.14,
installed from `phase_2/kit/requirements.txt` in full (not a subset), no
cached artifacts, no reuse of `swnd` and no reuse of the existing `pywake`
env.

- **pip freeze SHA-256: `f6fbf1d754e376e06c4ce941d744b0d72e38d45bf0cb1100f6002ee49fa0b315`**
- Full freeze: `reports/audit_anchor6_pipfreeze_20260825.txt`
- **PyWake resolved and now pinned: `py_wake==2.6.20`**

The kit declares `py_wake>=2.6`, a floor rather than a pin, which the
pre-statement named as the single largest threat to a 0.1 pp tolerance. The
floor resolved to 2.6.20 on 2026-08-25 and that is the version this result
belongs to. The kit's `requirements.txt` is not edited here: it is a kit file
and changing it is not CC's call. The pin lives in this document, in the JSON,
and in the freeze file.

**Why the bit-identity is worth more than it looks.** The environment is
materially different from the one nearest the original run:

| Package | `pywake` env (nearest candidate) | fresh `s3anchor20260825` |
|---|---|---|
| py_wake | 2.6.17 | 2.6.20 |
| numpy | 2.4.0 | 2.4.6 |
| pandas | 2.3.3 | **3.0.5** (major version change) |
| xarray | 2025.12.0 | 2026.7.0 |
| scipy | 1.16.3 | 1.17.1 |

A pandas major-version change and a PyWake patch bump produced no difference
at all, to the last representable bit. On this code path the result is
genuinely library-version-insensitive, which is a stronger statement than
"reproduces within 0.1 pp".

**FINDING, and the reason the table above says "nearest candidate" rather than
"the original".** `reports/three_case_scorer_20260818.json` records
`generated_by`, the three cases and a wall clock, and **no environment at
all**: no python version, no library versions, no freeze. So the environment
the 2026-08-19 run used cannot be determined from the record, and the
comparison above is against an inferred baseline, not a recorded one. The
identity of the numbers makes that inference safe in this instance. It would
not have been safe if they had differed: a mismatch would have been
un-diagnosable, because there would be nothing to diff against. This is
exactly the gap anchor 6 exists to expose, and it is why the pin above is
recorded rather than left implicit.

## Wall time

| Stage | Seconds |
|---|---|
| AROME load, 1,826 daily files, single pixel | 92.5 |
| Wake model, 14,608 steps, 55 turbines | 2.1 |
| Total | 100.5 |

Environment build (conda create plus the full requirements install) ran
separately and is not in the 100.5 s.

## Input gate, before the physics

Asserted before any wake model ran, so that a matching CF could not be a
coincidence and a mismatching CF could not be misdiagnosed as a physics
difference:

- nearest AROME pixel to (53.5, 1.5) is (53.4982, 1.4928), `sea=True`
- 14,608 finite steps, matching the S3 record exactly
- mean hub wind speed 10.464895 m/s, matching the S3 record to below 1e-6

The dataset root had to be repointed: `phase_2/build/phase2_dataset` no longer
exists and the train tree now lives under
`phase_2/inference_2022/phase2_dataset_ship`. Set via `PHASE2_DATA_ROOT`, the
same resolution the anchor-5 export used on 08-24. This is the same stale-root
condition that makes `scripts/tier2_d7_build_submission.py` unrunnable
(guard (d), item 4 of this batch); here it is handled by an environment
variable the loader already supports.

## The seven unstated defaults

Values taken from `scripts/task2_scorer_replica.py`, the module the 08-19 run
used, as the dispatch permits. Each is flagged for whether `Phase_2.pdf`
determines it.

| # | Default | Value used | Source | Spec determines it? |
|---|---|---|---|---|
| 1 | PyWake version | `2.6.20` | resolved from the kit floor | **NO.** Kit declares `>=2.6`. Now pinned here. |
| 2 | Superposition model | `LinearSum` | `task2_scorer_replica.py:190` | **NO.** Spec names the deficit model (Bastankhah-Gaussian) and is silent on how deficits combine. |
| 3 | Turbulence model | `None` (also no blockage model, no rotor-average model) | `task2_scorer_replica.py:188-191` | **NO.** Spec is silent. |
| 4 | Grid orientation and origin | 8 rows x 7 columns, row-major fill, first 55 kept, mean-centred, 0 degrees; 7D = 1988 m; extent 11,928 x 13,916 m | `task2_scorer_replica.py:128-141` | **NO.** Spec fixes 55 units and a 5D minimum; it does not determine the arrangement. Largest free parameter. |
| 5 | Air density / density correction | none applied; `PowerCtTabular` used as tabulated | `task2_scorer_replica.py:122-123` | **NO.** Spec is silent. |
| 6 | Wind resource time base | real AROME 2016-2020, native 3-hourly, single nearest pixel, 14,608 finite steps | `task2_scorer_replica.py:244-276` | **NO.** Spec is silent on years, cadence and spatial sampling. |
| 7 | Shear reference height | `REF_HEIGHT_M = 125.0`, factor `(170/125)^0.11 = 1.0344` | `task2_scorer_replica.py:98` | **PARTLY.** Spec states hub 170 m and shear 0.11 but not the reference height of the source series. |

All seven are free under the spec, to some degree, and six are entirely free.

Default 7 carries the sharpest consequence and it is already documented in the
replica's own docstring: the 2026-07-13 prompt said to shear "10 m to 170 m for
reanalysis ws10", but the site data is AROME at 125 m native. Applying the 10 m
factor would multiply speed by `(170/10)^0.11 = 1.366` instead of the correct
`(170/125)^0.11 = 1.034`, a 32-percentage-point error rather than a rounding
difference. The replica takes the 125 m reference. That choice is correct and
it is not derivable from `Phase_2.pdf` alone.

Defaults 4 and 6 were named in the pre-statement as the two most likely to move
the answer past 0.1 pp on their own. Both are confirmed free.

## Diff: pre-statement against execution

For the strategist. Every commitment in the pre-statement was kept, with one
correction and one addition.

| Pre-statement said | What happened |
|---|---|
| Fresh env from the kit's pinned requirements, no reuse of `swnd` or `pywake` | Done. `s3anchor20260825`, full requirements file installed. |
| Report environment identity as the SHA-256 of `pip freeze` | Done, `f6fbf1d7...`, freeze file committed. |
| PyWake floor-not-pin is the largest threat to the tolerance | Held as a risk; resolved to 2.6.20 and pinned. In the event it did not bite: bit-identical across a patch bump. |
| Name the defaults in advance | Done, seven named before the run; all seven confirmed free, none discovered late. |
| Grid orientation (4) and time base (6) most likely to break tolerance | Both confirmed free under the spec. Neither broke anything here, because both were read from repo code as the dispatch permits. |
| Not stated in advance | The dataset root had moved and had to be repointed via `PHASE2_DATA_ROOT`. Not anticipated in the pre-statement; recorded here rather than passed over. |
| Not stated in advance | The 08-19 run recorded no environment, so the comparison baseline had to be inferred. Reported as a finding above. |

## What this does and does not establish

**Establishes.** The S3 case1 number is reproducible from committed code and
committed data, in an environment built from scratch, with no cached
artifacts, across a pandas major-version boundary, to the last bit. The
pipeline is re-runnable cold on this path.

**Does not establish.** That the configuration is right. Every one of the seven
defaults was read from the repo, as the dispatch permits, so this run cannot
disagree with the repo about them. A from-spec-only reconstruction was the job
of anchor 5 cell 4, which was dropped on 08-24; rung 2's independence now rests
on the kit `simulate_year` cross-check of 2026-07-13, not on this run.

Those are different claims and a PASS here should not be read as the stronger
one.

## Status

DONE. PASS, bit-identical. Two findings recorded above: the 08-19 run has no
environment provenance, and the kit declares a PyWake floor rather than a pin
(now resolved and pinned at 2.6.20).
