# CLAUDE.md — Sea Winds Phase 2 repo

Standing constraints for Claude Code in this repository. This file is a
distillation, not the source of truth. Authority chain:
`ORCHESTRATION_CONTRACT_v2_0.md` (active 2026-07-03) > this file. On any
conflict, the contract wins. v1.1 and its amendments remain on disk as
archive; "contract §/A" pointers below refer to the archived v1.1 numbering,
kept for traceability. The ✗-list is delegated to this file by v2.0 §5.

## Role (contract §2)

You are CC: builder and executor, code-side only.

- You MAY: write code, run pipelines, create files, propose parameters for
  approval, checkpoint to disk.
- You MAY NOT: make design decisions autonomously; proceed past parameter
  proposal without Matteo's explicit approval; modify Opus-stewarded documents
  (contract, handoffs, checkpoints, LLM_AGENT_LOG.md, report prose) without
  Opus review.
- "OK" from Matteo is approval only when he has stated what he checked.
  Ambiguous acknowledgment → ask, don't build.

## Build discipline (contract §4, non-negotiable)

1. Save every model on creation. No model exists only in memory or as an
   unnamed temp file.
2. Every artifact comes from a named, seeded script. The script lives in the
   repo, is committed, and uses explicit random seeds wherever applicable.
   If a stochastic step cannot be seeded, document why in the script.
3. No `_scratch` or throwaway names. Every file has generation provenance.
4. Pipeline re-runnable cold, at all times: raw inputs → submission with no
   manual intervention, no missing files, no implicit state.
5. Checkpoint to disk BEFORE risk, not at the brink. Before any risky
   operation or when approaching context limits, write state to disk first.
6. Parameter proposal → Matteo approves → then build. Autonomy is in the
   building, not the deciding.
7. Pin data versions. Record exact Zenodo DOI, version, and SHA-256 on
   download, in `data/MANIFEST_zenodo_20335351.md`. Never let a pipeline
   read data that is not in the manifest (contract A1).

## Self-check before declaring anything done (the ✗ list, contract §4)

Your output fails audit if it contains any of:

- ✗ an unsaved model
- ✗ an unseeded stochastic step without documented justification
- ✗ a `_scratch` or unnamed temp file in the deliverable chain
- ✗ a pipeline step needing manual intervention or implicit state
- ✗ a numeric value in a deliverable without a traceable source
- ✗ a file created or modified without a named, committed script
- ✗ a checkpoint written only at session end, not before risky operations
- ✗ parameters acted on without evidence of Matteo's approval
- ✗ data used without a pinned version/checksum

Opus audits against this list after every task. Run it yourself first.

## Task prompts (contract §6)

Every task prompt should declare: scope (touch / don't touch), file
references (authoritative vs superseded), constraints, success criteria.
If a prompt is missing one of these, ask before starting — do not infer scope.

## Escalation (contract §2)

Stay in CC for: boilerplate, import/syntax fixes, plot formatting, small
iterations, standard pandas/numpy ops.

Stop and escalate to Matteo/Opus when: results are >10% off expected with
unclear cause; an error can't be read from the traceback; an architectural
decision is needed; a module is finished and wants validation; unsure whether
a discrepancy is a bug or a real difference. Rule of thumb: >15 min stuck →
escalate.

## Data provenance (pinned — verify checksums before use)

- Phase 2 data: Zenodo record 20335351 — hashes in
  `data/MANIFEST_zenodo_20335351.md`. Nothing is unzipped, moved, or read by
  any pipeline unless its manifest entry exists (contract A1).
- Phase 1 HRES: `hres_north_sea.parquet` (296.5 MB), Zenodo record 19538994
  v1 (2026-04-12). SHA-256
  `7b00c61df2d56f2f69445ec8677ba50de734aa782a335447dad8d353e1587be3`
  (checkpoint 2026-07-02).

## Phase 2 hard constants (Phase_2.pdf — do NOT confuse with the employer's
## IEA 15 MW work; different turbine, different project)

| Parameter | Value | Source |
|---|---|---|
| Turbine | IEA 22 MW, exactly 55 units (1.21 GW) | Phase_2.pdf p.5 |
| Rotor diameter D | 284 m | Phase_2.pdf p.5 |
| Hub height | 170 m (forecast target height: 125 m) | Phase_2.pdf pp.4–5 |
| Min spacing | ≥ 5 D = 1420 m | Phase_2.pdf p.5 |
| Layout box | 15 × 15 km; centre depth ≤ 50 m | Phase_2.pdf p.5 |
| Scorer physics | PyWake, Bastankhah-Gaussian, shear α = 0.11 | Phase_2.pdf p.6 |
| Submission size | 4,196,640 rows = 43,715 pts × 8 windows × 3 horizons × 4 hours | Phase_2.pdf |
| Deadline | 2026-08-30, 23:59 CET (18:59 ADT); terminal actions 2026-08-29 18:00 ADT | organizer final-window email, docs/organizer/organizer_emails_verbatim_20260825.md |

Sanity anchors (kit references, Phase_2.pdf pp.4,6):
- Task 1 speed Winkler d1/d7/d14: 9.2 / 29.8 / 40.1
- Task 1 direction circular Winkler d1/d7/d14: 173 / 312 / 334
- Winkler optimum sits near coverage ≈ 0.88, not 0.90
- Task 2 baseline: CF ≈ 53.2%, AEP ≈ 5,635 GWh, wake ≈ 7.1%, LCOE ≈ 83.1
  EUR/MWh (organizer correction on the Codabench board, the organizer, 2026-08-10)

Domain: 159 allowed cells = Phase_2.pdf p.2 map figure (title + legend;
square-count verified 2026-06-25 and 2026-07-06) — the coarse
AROME ∩ reanalysis ∩ sea domain for BOTH tasks ("the only domain we predict
& site on"). Not shipped as a file in the kit; membership criterion is on
the figure. Open check: distinct 0.25° reanalysis cells over
footprint_points.parquet == 159? Separate fact: `SITING_YEARS=()` in
splits.py — the siting evaluation YEARS are withheld (unrelated to the 159).

## Logging (scored — Dim 5)

All LLM-agent work is appended to `LLM_AGENT_LOG.md` (additions only, dated).
Compute disclosure entries accompany every substantial run. Dead ends are
logged with the reason, never silently abandoned (contract §5.3).

Report sync (contract v2.0 §6): report-worthy results in checkpoints carry
`[REPORT: pending]`, flipped to `[REPORT: §X, YYYYMMDD]` once written into
`phase2_report_material.md`; report additions cite their source checkpoint
date. At session close, run:
`grep -n "REPORT: pending" checkpoint_update_*.md`
Any hit older than the current session = sync failure; fix before closing.

Checkpoint retention (adopted 2026-07-16): only the latest
`checkpoint_update_*.md` is kept in the repo; superseded checkpoints are
removed from tracking and live in project knowledge only.

Multi-chat document rule (adopted 2026-07-04, after a caught divergence):
project-knowledge copies are per-conversation snapshots; parallel chats
each see their own. Therefore, when a shared document may have been touched
elsewhere: (1) never emit a full replacement file — emit only the addition
block with a named anchor; (2) before writing any addition, state the last
line of the version you can see, so divergence surfaces at write time;
(3) full-file merges happen in one designated session only, diff-verified
against every contributing version. Applies to this file too: the repo
CLAUDE.md is authoritative; chat copies are drafts.

## Calendar walls (contract §7, §1)

- July 30 gate: Tasks 1 and 2 computationally complete; report strong-draft;
  swap-week re-run wired.
- July 31 – Aug 17: Matteo away, NO computing. Do not plan work into this
  window. Swap-path ladder (contract A2) resolves on organizer window dates —
  do not build below rung 2 until dates land.
- Crunch evenings: mechanical work only on pre-approved parameters; no
  parameter approvals, architecture decisions, or submissions in evening
  sessions (contract A5).

## Values anchor (contract §1)

Cut scope, not hours. If a task would trade against the July 30 gate or the
vacation wall, stop and escalate rather than compress quality.

## File and context integrity (added 2026-07-17)

**Encoding check before delivering files.** Before presenting or
committing any prose file, grep at byte level for double-encoded
UTF-8 signatures (the `â€` pattern and similar mojibake). Files
crossing project-knowledge / outputs / repo / paste boundaries are
vulnerable to silent double-encoding of em dashes, quotes, and
accented characters. Fix before delivery; do not ship corrupted
files.

**Post-compaction integrity.** After any context compaction, the
compacted summary is lossy compression, not a verified source. Treat
all specific values carried in it (numbers, scores, hashes, dates,
paths, quotes) as unverified until cross-checked against a primary
source (repo files, LLM_AGENT_LOG.md, checkpoints, tool output).
Do not write unverified compacted details into any deliverable or
commit.

## Contract amendment v2.1 (adopted 2026-08-19, mirrored here 2026-08-20)

Additions to `ORCHESTRATION_CONTRACT_v2_0.md` §5. Source:
`docs/CONTRACT_AMENDMENT_v2_1_20260819.md`. Additions-only; v2.0 text
unchanged. Mirrored here because §5a and §5b bind CC directly.

**Why.** The pinned block-excluded downscaler (`downscaler_blockexcl.pkl`,
SHA b68eb5fe) and its 80 bias-calibration extracts lived in a folder named
like a scratch area, outside the repo, gitignored by pattern, never listed,
and never named to Matteo. The folder was deleted to make room. The pin was
in code; the artifact was not in custody. Separately, three
silent-corruption paths (alpha left at 1.0; a correction step reading a
2021 backup as its source; five hardcoded 2021 year constants) all passed
the July rehearsal because that rehearsal ran on 2021 data.

### 5a. Pinned artifacts

Any file a frozen pipeline asserts by SHA is a **pinned artifact**. A
pinned artifact:

1. lives in the repo, or, if too large or gitignored, in a single named
   protected folder that is not a scratch or smoke-test location, and is
   listed in `data/PINNED_ARTIFACTS.md` with path, SHA, size, the
   `script:line` that asserts it, and the date pinned;
2. is named to Matteo once, in the session it is pinned, with the sentence
   "this file must not be deleted";
3. is SHA-verified at session open by CC whenever a frozen run is on the
   calendar.

The pin in code and the custody of the file are one act, not two.
Promotion from scratch to pinned is a logged event.

### 5b. Rehearsal must vary what the production run varies

A rehearsal run on the same inputs as the original build tests the
mechanics, not the swap. Therefore:

- Before any production run on new inputs, CC greps the pipeline for every
  constant the new inputs would change (year, window set, data root, backup
  paths, cache files) and reports them **as a list before Step 0**.
- Any step that reads from a backup or cache rather than from the live
  pipeline output is named explicitly in the runbook, with the condition
  under which the backup is stale.
- A year-of-output assertion, or the equivalent for whatever the swap
  varies, is part of the validation gate, not an optional check.

### Consequences for the ✗ list

Two additions to the self-check above:

- ✗ a SHA-asserted file that is not listed in `data/PINNED_ARTIFACTS.md`
- ✗ a production run on new inputs with no pre-Step-0 changed-constant list
