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
| 2026-08-24 | deliverables manifest: full owed set checked as a set on 08-29 (report PDF, submission.json clean copy, selection flag 897665, email sent confirmation, screenshots to register) |  2026-08-30, closed by delivery. Report `70cde014` (12 pp), supporting material `23a158a8` (76 pp) and `submission.json` `2f3ab624` emailed by Matteo, three attachments (`LLM_AGENT_LOG.md`, 2026-08-30 entry). The eight shipped files are in `report/final/` and registered as entry 10 of `data/PINNED_ARTIFACTS.md`. **Closed on the send as reported, not on a verified artifact**: the email is not in this repository, and neither the selection flag 897665 nor the screenshots are evidenced by anything in the tree -- see note D1 |
| 2026-08-24 | clean Task 2 submission copy |  2026-08-30, `data/submission/submission.json`, SHA `2f3ab624`, built by `scripts/task2_build_submission_json_20260827.py` (commit `5c67014`) and verified unchanged on 2026-08-29 and 2026-08-30. Clean copy relayed as `relay_final_20260830/submission.json`, same SHA, byte-identical by `cmp`. Sent as one of the three attachments. Artifact verified; the send itself rests on the report, as in the line above |
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
| 2026-08-27 | build recipe for the report PDF not recorded anywhere: the v4 invocation, template and margins are unrecoverable, and neither pandoc nor pdflatex is installed on this machine, so CC cannot measure the page count (dispatch item 4d) | 2026-08-27, `docs/report_build_recipe_20260827.md` at commit `1727041`: the command was recovered and VERIFIED against the v4 artifact in the strategist's container, 11 pages both, pages 3-11 identical. The page count is reproducible again |
| 2026-08-27 | `LLM_AGENT_LOG.md` ends at 2026-08-22; the 08-24 to 08-27 audit recomputations are absent from it entirely (found doing dispatch item 3) | |
| 2026-08-27 | two substantial Pangu inference blocks carry no compute disclosure: the 2026-08-22 R4a submission-window extracts for 2021 and 2022, 448 inference steps, about 6.1 CPU-h. Contract §5.3 requires one per substantial run (found doing dispatch item 3) | |
| 2026-08-27 | nothing refreshes `data/PINNED_ARTIFACTS.md` when a pinned *producer script* is committed: entry 1's recorded SHA for `tier2_pangu_rollout.py` decayed silently between 2026-08-21 and 2026-08-27 (found doing dispatch item 8) | |
| 2026-08-27 | `bidding_sim/generate_d1_2019.py` `CACHE_PKL` points at `coarse_forecasts.pkl`, whose contents the 2026-08-19 Leg A 2022 run replaced; the pinned bytes (`3a4e4538`) survive as `coarse_forecasts_2021.pkl`. A one-line path fix, SHA untouched, restores cold re-runnability of the shipped bonus chain. **Deferred to post-competition by decision (Matteo, 2026-08-27):** the file is a cache, it is excluded from the public tag, so no judge can encounter the guard, and no report claim depends on it. Not applied. *(item text amended 2026-08-27, prompt B; see note B2)* | |
| 2026-08-27 | the daily change-control attestation did not run for 2026-08-26; 08-25 and 08-27 exist (dispatch item 8) | |
| 2026-08-27 | `CLAUDE.md` cites `docs/organizer/organizer_emails_verbatim_20260825.md` as the source of the governing deadline, and that file is not in the repository (found doing dispatch item 7) | |
| 2026-08-27 | `checkpoint_block_20260826_close.md`, named authoritative by the 2026-08-27 dispatch, is not in the repository; nor are the 08-25 and 08-26 close blocks, so their sixteen ledger items could not be transcribed (dispatch item 7a) | |
| 2026-08-27 | **no audit was ever scoped to publication readiness.** Every pass to date, internal and external, checked the report's claims, numbers and citations; none asked what happens when the working repository becomes public. A distinct failure class from the ones already named, and it surfaced only because a confidentiality grep was attached to a push item. Post-competition, to `llm-operational-discipline` (dispatch addendum 2026-08-27) | |
| 2026-08-27 | uncommitted modifications to `phase_1/2d_starting_kit_heavy.ipynb` and `phase_1/make_hybrid.py` destroyed during publication-branch pruning (`rm -f` over the drop list, then the branch switch restored HEAD). Never staged, so no blob exists; an object-store similarity scan over all 7,345 blobs returned nothing above 0.80 and the best candidate was a different tracked file. Unrecoverable. Cause: working-tree deletion instead of `git rm --cached` or a separate worktree | 2026-08-27, closed by acknowledgement (Matteo, prompt B). No competition deliverable referenced uncommitted `phase_1/` state. Redo only if Matteo identifies lost content that matters. Opened and closed in one entry, per the 2026-08-26 precedent |
| 2026-08-27 | **audit freeze in force.** No further external or internal audit passes before delivery, except one P4 persona pass against the assembled PDF, BLOCKs only. Findings arising after this date go to this ledger as post-competition and are closed by decision (Matteo, 2026-08-27) |  2026-08-31, closed by authorisation (Matteo, dispatch A2r): the competition is closed, delivery is complete and receipt is confirmed, so the condition the freeze protected has passed. Evidence: `report/final/sent_20260830_submission.eml` and `report/final/received_20260831_receipt.eml`. This is the line note D2 of the 08-30 session left open as Matteo's judgement rather than CC's |
| 2026-08-27 | **the publication scan had no register category.** It asked what a string revealed about a *machine* -- paths, usernames, hostnames, credentials, private remotes -- and what it revealed about third parties. It never asked what a *document* revealed about the principal: private working voice, personal state, employer context. Three superseded working papers in `docs/audit/` carried an informal message quoted verbatim and two cold reviews discussing his working condition; two employer references survived elsewhere that the scan's own corporate-labels category should have caught. Found by Matteo clicking at random, then swept systematically by the strategist, read-only over the tag tarball, all 112 published files; every register hit was confined to those three files | 2026-08-27, tag `phase2-report` moved to `caabfce0`: the three files removed, employer and personal-schedule wording replaced in place, `README.md` and `SANITISATION.md` updated, and the register category added to `SANITISATION.md` so the sweep is repeatable before any future tag. Verified on an anonymous clone: 109 files, `docs/audit/` holds only the anchor 6 statements and the extracts index, and all six marker greps return zero across every branch and all history. Produced by `scripts/publication_register_sweep_20260827.py` and `scripts/publication_selfdescribe_20260827.py`. Post-competition: fold the register category into the publication-readiness rule in `llm-operational-discipline` |
| 2026-08-27 | **GitHub still serves the pre-removal commit by full SHA.** Rewriting `publication` to a single root commit removed the working papers from the published history -- a normal clone has one commit and no trace -- but unreachable objects survive on the remote until GitHub garbage-collects, and fork networks retain them longer. `raw.githubusercontent.com` returned 200 for the pre-removal blob addressed by its full 40-character SHA, verified 2026-08-27. That SHA appears nowhere in the public tree, its refs, or its public events feed, and the report cites the tag and not a SHA, so discovery requires having fetched the repository while it was live. Narrow, not nil. Only Matteo can decide whether to ask GitHub Support to purge unreachable objects, or to accept the residual | |
| 2026-08-30 | delivery evidence absent from the repository: the 08-30 closures of the deliverables manifest and the clean Task 2 submission copy rested on the send as reported, with no artifact in the tree for the email, the selection flag 897665 or the screenshots (note D1) | 2026-08-31, `report/final/sent_20260830_submission.eml` (`c031c261`) and `report/final/received_20260831_receipt.eml` (`c1da2803`), moved into the tree by dispatch A2r and registered as entry 11 of `data/PINNED_ARTIFACTS.md`. The sent message carries the three attachments and states selection flag 897665; the reply confirms receipt and gives early October for results. **Screenshots remain unevidenced** and were not part of what these two files show. Opened and closed in one entry, per the 2026-08-26 precedent and note C2 |

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

**Count after the register sweep: 22 open, 11 closed.** Earlier on 2026-08-27
the count stood at 21 open, 8 closed, then 21 open, 10 closed after prompt B.
The register sweep then opened and closed the missing-register-category line in
one entry, and opened one new line for the residual: GitHub still serves the
pre-removal commit when it is addressed by full SHA.

Earlier the same day: the batch dispatch opened twelve lines and closed four of
the fifteen seeded ones, and the dispatch addendum opened the
publication-readiness line after that report-back.

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

### Prompt B, later on 2026-08-27

**B1 — the build-recipe line is closed.** It was opened this morning because the
v4 invocation could not be recovered here. It has since been recovered and
verified against the v4 PDF in the strategist's container, page by page. The
closure cites the document and its commit, not the fact of someone saying so.

**B2 — the `coarse_forecasts.pkl` line was AMENDED, not duplicated.** Prompt B
asks to open a line for it. A line for the same obligation was already opened
this morning under dispatch item 8, so opening a second would have put one
obligation in the ledger twice and made the open count a lie. The existing line
was amended instead, to carry Matteo's deferral decision and its reasoning, and
`closed-by` was left empty because the obligation is deferred, not discharged.

The instrument's own rule is that editing an `item` text is allowed only to
correct a transcription error, with the amendment noted in `closed-by`. That
rule and the rule that `closed-by` is empty for every open line cannot both be
followed here: writing the amendment into `closed-by` would move the line out
of the open set, which is exactly wrong for a deferred obligation. The open set
is the property worth protecting, so the amendment is noted here and marked
inline in the row instead. Flagged as a genuine gap in the instrument, for the
post-competition ledger design.

**B3 — the `phase_1` loss is opened and closed in one entry.** It is CC's
mistake, recorded in CC's own words in the row, and closed by Matteo's
acknowledgement rather than by any artifact, because there is no artifact: the
content is gone. Recording it as an obligation that briefly existed is the only
honest way to keep it on the record at all.

### The register sweep, later on 2026-08-27

**C1 - a removal commit does not remove.** The dispatch specified one commit on
`publication`, then re-point the tag. Committing the removals as a child of
`1ac98a0b` would have left every removed file fully recoverable from the parent
commit in a public repository, and the commit message named the three files, so
the published history would have been a *map to* the content rather than its
removal. The branch was rebuilt instead as a single root commit carrying the
same verified tree (`caabfce0`: one commit, no ancestors) and force-pushed with
`--force-with-lease`. That is the same reason the branch was made an orphan in
the first place: git ships commits, not trees. The pre-removal commit is
preserved locally as branch `pub-preremoval-1ac98a0b`, so the change is
reversible on Matteo's word.

**C2 - the residual is a second line, deliberately.** The dispatch asked for one
entry, opened and closed. The closure is real: the content is gone from the
published history. The residual is also real, and it is nobody's judgement to
close silently, so it is opened as its own line rather than buried in the
closure text of the first.

**C3 - one deviation from the specified wording, flagged.** The dispatch gave
the employer name -> `my employer` for the log entry. Applied literally the
sentence read "my employer work completed", which is ungrammatical, and it began
a sentence in lower case where a proper noun had stood. Written as **"My
employer's work completed"**: the same possessive adjustment the dispatch itself
authorised one line above for `CLAUDE.md`, where it asked that the article be
adjusted so the sentence reads naturally. One apostrophe-s and one capital
beyond the literal instruction; trivially revertible if Matteo wants the literal
form.

**C4 - a redundancy left in place, not silently smoothed.** `departure Jul 31`
-> `the July hard-cut` makes that sentence read "Schedule hard-cut through the
July hard-cut." It is clumsy but not wrong, and removing the redundancy would
mean editing words the dispatch said to keep intact. Left as specified, and
reported rather than repaired.

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


## 2026-08-30 session (CC dispatch 2026-08-30, post-delivery)

Two lines closed, both report-delivery: the 2026-08-24 deliverables manifest
and the 2026-08-24 clean Task 2 submission copy. Nothing else in the ledger was
touched.

### D1 -- the email is evidence this repository does not hold

The dispatch names "the email" as the evidence closing the delivery lines. The
email is not an artifact here, and this session could not read it: the Gmail
connector is unauthorised in a non-interactive session. What is verifiable in
the tree is the shipped set itself, by SHA, and that is what the closures cite.
The send, the selection flag 897665 and the screenshots rest on Matteo's
report, exactly the situation note C1 describes for the 08-27 closures. Recorded
so the distinction survives: the artifacts are verified, the send is attested.

### D2 -- delivery-adjacent lines deliberately left open

The 2026-08-27 line **audit freeze in force** is worded "before delivery", so
delivery arguably overtakes it. It was NOT closed here: the dispatch scoped this
session to the report-delivery obligations, and reading a freeze as
self-cancelling is a judgement for Matteo, not a record-keeping act. It stays
open. The same applies to the 2026-08-24 consolidated-checkpoint line, which is
Matteo's to close.

## 2026-08-31 session (CC dispatch A2r)

Two closures, both delivery. The **audit freeze** line closed on Matteo's
authorisation now that the competition is closed and receipt is confirmed; note
D2 of the 08-30 session had left it open precisely because that call was his.
The **delivery-evidence gap** of note D1 closed on artifacts: the sent message
and the organizer's reply are now in `report/final/` and hashed in the register.

D1 is superseded, not deleted. What it recorded stays true of the 08-30 commit:
at that date the evidence was not in the tree. It is now. One element of the
original manifest, the screenshots, is still evidenced by nothing here, and the
closure says so rather than rounding up.

`report/final/` stays on `main`. Dispatch A1b publishes the code, the standing
constraints, the log, the audit files and the ledger; it publishes none of the
shipped documents or the delivery evidence, and this session added nothing to
what A1b publishes beyond the log entry and these ledger lines.
