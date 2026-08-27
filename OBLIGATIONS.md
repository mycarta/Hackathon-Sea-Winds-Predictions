# OBLIGATIONS.md

Founding instrument. Created 2026-08-25 by CC under the dispatch of the same
date (`docs/dispatches/cc_dispatch_20260825_batch.md`, item 3).

## The rule

**A line leaves this table only through an explicit closure entry: a date plus
the artifact or the decision that closed it. Never by deletion.**

A line that turns out to have been wrong is closed by a decision saying so, and
the closure records that. A line that is overtaken by events is closed by the
event. A line nobody remembers opening stays open until somebody closes it on
the record. Editing the `item` text of an open line is allowed only to correct
a transcription error, and the correction is noted in `closed-by` as an
amendment with its date; rewriting an obligation into a different obligation is
opening a new line, not editing an old one.

Deleting a line, blanking it, or letting it fall off the bottom of the file is
the failure mode this instrument exists to prevent. The value of the ledger is
that its open set is complete, and a silently dropped line destroys that
property for every other line as well.

`closed-by` is empty for every open line. That emptiness is the point: the open
set is whatever has an empty `closed-by`, readable at a glance, with no
cross-referencing required.

## Ledger

| opened | item | closed-by |
|---|---|---|
| 2026-08-24 | deliverables manifest: full owed set checked as a set on 08-29 (report PDF, submission.json clean copy, selection flag 897665, email sent confirmation, screenshots to register) | |
| 2026-08-24 | clean Task 2 submission copy | |
| 2026-08-24 | consolidated checkpoint uploaded under rolled name, fragments deleted (Matteo, after the chain concatenation) | |
| 2026-08-24 | organizer email of 08-09 transcribed verbatim into the record (Matteo provides) | 2026-08-27, by decision: the 2026-08-27 dispatch states the transcription is done. **Closed on the decision, not on a verified artifact** — see note C1 |
| 2026-08-24 | obligation ledger full design, post-competition, llm-operational-discipline; founding cases 08-24 (Task 2 save) and 08-25 (ratified text not persisted) | |
| 2026-08-22 | named-constant discipline rule, post-competition | |
| 2026-08-24 | close's two written practices into llm-operational-discipline, post-competition | |
| 2026-08-24 | register rules distillation into the rules repo, post-competition | |
| 2026-08-25 | meta-rule: a ratification is not closed until the ratified text itself is persisted where successors read | |
| 2026-08-25 | RATIFIED_OPENINGS_20260824.md: opening 2 "on a 2021 dry run" to read "on the 2021 board" at next upload (Matteo) | |
| 2026-08-25 | merge ADDITION_dim1_leg_b_2022_20260825.md into MERGED_20260825 (strategist folds, Matteo uploads) | 2026-08-27, by decision: the 2026-08-27 dispatch states the merge is done. **Closed on the decision, not on a verified artifact** — see note C1 |
| 2026-08-25 | Fig B review: confirm feature engineering is visible on the flowchart (deck criterion: feature engineering, model architecture, uncertainty quantification) | |
| 2026-08-25 | bonus section: one sentence stating the sorted-triple interpolation rule for the 21 cut-out hours | |
| 2026-08-25 | LLM_AGENT_LOG additions: (a) Tier B extract export (CC, anchor-5 addition format), (b) Tier A F1 wording-versus-source catch (strategist drafts) | |
| 2026-08-25 | terminal-tier script | 2026-08-25, `scripts/terminal_check.py`, commit `66a2af6`, dry-run clean in 0.4 min. **Artifact verified present 2026-08-27** |
| 2026-08-26 | CLAUDE.md deadline and Task 2 baseline constants corrected to organizer values | 2026-08-26, commit e3b2eae |
| 2026-08-27 | Opus draft 2 folded into the skeleton | 2026-08-27, by decision: the 2026-08-27 dispatch states the fold is done. Opened and closed in one entry because no prior line existed to close — see note C2 |
| 2026-08-27 | model seat list placeholder | 2026-08-27, closed by Matteo's supplied text in the 2026-08-27 draft (dispatch item 7b). Opened and closed in one entry — see note C2 |
| 2026-08-27 | CLAUDE.md Task 2 sanity anchors stale, superseded by the organizers' 2026-08-10 correction (dispatch item 7c) | 2026-08-27, **already closed before this line was opened**: commit `e3b2eae` (2026-08-26) carries CF 53.2 %, wake 7.1 %, AEP 5,635 GWh, LCOE 83.1 EUR/MWh. Verified in the repo copy 2026-08-27 — see note C3 |
| 2026-08-27 | CLAUDE.md deadline line stale, reads 2026-08-15 20:55 ADT (dispatch item 7c) | 2026-08-27, **already closed before this line was opened**: commit `e3b2eae` (2026-08-26) carries 2026-08-30 23:59 CET. Verified in the repo copy 2026-08-27 — see note C3 |
| 2026-08-27 | build recipe for the report PDF not recorded anywhere: the v4 invocation, template and margins are unrecoverable, and neither pandoc nor pdflatex is installed on this machine, so CC cannot measure the page count (dispatch item 4d) | |
| 2026-08-27 | `LLM_AGENT_LOG.md` ends at 2026-08-22; the 08-24 to 08-27 audit recomputations are absent from it entirely (found doing dispatch item 3) | |
| 2026-08-27 | two substantial Pangu inference blocks carry no compute disclosure: the 2026-08-22 R4a submission-window extracts for 2021 and 2022, 448 inference steps, about 6.1 CPU-h. Contract §5.3 requires one per substantial run (found doing dispatch item 3) | |
| 2026-08-27 | nothing refreshes `data/PINNED_ARTIFACTS.md` when a pinned *producer script* is committed: entry 1's recorded SHA for `tier2_pangu_rollout.py` decayed silently between 2026-08-21 and 2026-08-27 (found doing dispatch item 8) | |
| 2026-08-27 | `bidding_sim/generate_d1_2019.py:62` reads `coarse_forecasts.pkl`, which the 2026-08-19 Leg A run replaced with the 2022 vintage; the pinned bytes now live at `coarse_forecasts_2021.pkl`, so the bonus Stage 1 chain will not re-run cold. Repoint deferred: `bidding_sim/` is outside the 2026-08-27 dispatch's touch list (dispatch item 8) | |
| 2026-08-27 | the daily change-control attestation did not run for 2026-08-26; 08-25 and 08-27 exist (dispatch item 8) | |
| 2026-08-27 | `CLAUDE.md` cites `docs/organizer/organizer_emails_verbatim_20260825.md` as the source of the governing deadline, and that file is not in the repository (found doing dispatch item 7) | |
| 2026-08-27 | `checkpoint_block_20260826_close.md`, named authoritative by the 2026-08-27 dispatch, is not in the repository; nor are the 08-25 and 08-26 close blocks, so their sixteen ledger items could not be transcribed (dispatch item 7a) | |
| 2026-08-27 | **no audit was ever scoped to publication readiness.** Every pass to date, internal and external, checked the report's claims, numbers and citations; none asked what happens when the working repository becomes public. A distinct failure class from the ones already named, and it surfaced only because a confidentiality grep was attached to a push item. Post-competition, to `llm-operational-discipline` (dispatch addendum 2026-08-27) | |

## Notes on seeding

Every seeded line above was supplied by the 2026-08-25 dispatch and is
transcribed as given, with two mechanical changes: opened dates are written
absolute (2026-08-DD) rather than as bare month-day, and the two lines that
referred to "item 2", "item 5" and "item 7" of that dispatch name the work
instead of the item number, so the line still reads correctly once the
dispatch is no longer the document in hand.

No line here was invented by CC, and no line supplied by the dispatch was
dropped. Seeded count: 15 open, 0 closed.

Lines opened after the seeding are appended below the seeded block, in the
same table. 2026-08-26: one line opened and closed in the same session
(CLAUDE.md constants, commit e3b2eae). Count after 2026-08-26: 15 open, 1 closed.

## 2026-08-27 session (CC dispatch 2026-08-27, item 7)

**Count: 21 open, 8 closed.** Thirteen lines opened, four of the fifteen seeded
lines closed, and four of the thirteen new lines closed in the same entry. The
thirteenth (publication readiness) was opened by the 2026-08-27 dispatch
addendum, after the batch report-back.

### C1 — two closures rest on a decision, not on a verified artifact

The dispatch instructs CC to close "organizer email transcribed verbatim" and
"`ADDITION_dim1_leg_b_2022` merged". Both are closed above, and both are marked,
because **neither artifact is in this repository**:

- `docs/organizer/` does not exist. `CLAUDE.md` cites
  `docs/organizer/organizer_emails_verbatim_20260825.md` as the source of the
  governing deadline constant, so that citation currently points at nothing.
  A separate open line records this.
- `ADDITION_dim1_leg_b_2022_20260825.md` is not present, and neither is any
  `MERGED_20260825`.

Both may well exist in project knowledge, which CC cannot see. The ledger rule
allows closure by decision as well as by artifact, so the closures stand — but
recorded as decision-closures, so that a later reader looking for the artifact
knows in advance that the repository does not hold it.

### C2 — two lines opened and closed in the same entry

"Opus draft 2 folded into the skeleton" and "the model seat list placeholder"
were named for closure by the dispatch, but neither had an open line to close:
they were never in the seeded set. Closing a line that was never opened would
leave no record of the obligation at all, so each is opened and closed in one
entry, following the precedent set on 2026-08-26.

### C3 — dispatch item 7c was already done

Item 7c asked CC to fix two stale `CLAUDE.md` lines. Both were already correct
in the repository copy when this session opened: commit `e3b2eae` (2026-08-26)
carries the organizers' corrected Task 2 anchors and the 2026-08-30 23:59 CET
deadline, and commit `0acbf0a` already recorded that in this ledger. The
dispatch was written against a staler view of the file than the repository
holds. The two lines are entered anyway, so the instruction leaves a trace, but
they are entered already-closed rather than as new open obligations.

### The sixteen items that could not be transcribed

Item 7a asks for "the ten items from the 2026-08-25 close block" and "the six
from the 2026-08-26 close block". **Neither block is on this mount.** The
2026-08-25 dispatch (`docs/dispatches/cc_dispatch_20260825_batch.md`) is
present, and its item 3 is what seeded the original fifteen lines, but that is
the dispatch, not the close block. `checkpoint_block_20260826_close.md` — which
the 2026-08-27 dispatch names as the authoritative checkpoint superseding all
others — is in neither the working tree nor any commit on any branch.

Those sixteen items are therefore **not** in the ledger, and this note is the
record of that gap rather than a silent omission. An open line above tracks it.

