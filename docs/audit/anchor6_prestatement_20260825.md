# Item 6 pre-statement, and an escalation before running

CC, 2026-08-25. Written before any run, per the dispatch. Nothing has been
executed for item 6.

## Why this is a pre-statement and not a result

The dispatch asks CC to recompute one S3 cell from the spec in a clean
environment: the plain 7D grid at the organizer baseline centre (53.5N 1.5E),
IEA 22 MW, 55 units, 5D spacing, Bastankhah-Gaussian, shear 0.11, TI
per-sector from repo code as recorded 08-24.

That instruction collides with the standing instrument of anchor 5, recorded
in `ADDITION_llm_agent_log_20260824_audit_anchor5.md` and committed at
`8c681b3`:

> CC wrote no simulation configuration anywhere, not in code, not in
> comments, not in the manifest, and ran no part of cell 4's physics in any
> form. **That absence is the instrument.** If a later session "completes"
> cell 4, the anchor is destroyed, not finished.

Item 6 is a different case from cell 4. Cell 4 is `case3_submitted_cell63`,
the winner layout at 52.50N 3.00E. Item 6 is a plain 7D grid at 53.5N 1.5E.
The two are not the same recomputation, and item 6 explicitly authorises
reading repo code for the TI values, which cell 4 forbids. So the dispatch is
plainly aware of anchor 5 and is asking for something else.

The collision is narrower than "same case" and it is still real: to run item 6
CC must write a working PyWake configuration for the IEA 22 MW farm and commit
it. Once that exists in the repo, the principal writing cell 4 can read it. The
thing cell 4 tests is whether the S3 number can be reproduced from
`Phase_2.pdf` alone by someone who has not seen the repo's configuration. A
sibling configuration in the repo does not answer cell 4, but it does remove
the condition that made cell 4 evidence.

This is an architectural call about what the audit is measuring, not a build
step, so CC is not making it (CLAUDE.md, Role). **Held for Matteo.**

## The three ways forward, so the decision is one word

1. **Run it, accept the cost.** Item 6 proceeds as written. Anchor 5's cell 4
   is downgraded from "written with no repo code available" to "written with a
   sibling configuration present in the repo", and that downgrade is disclosed
   in the report rather than discovered later.
2. **Run it quarantined.** Item 6 proceeds, but its configuration and
   environment live outside the repo, in a location the principal agrees not
   to open until cell 4 is written and committed. The result number comes back;
   the configuration does not land until cell 4 does.
3. **Defer until cell 4 is written.** Cell 4 first, by the principal, from the
   spec. Item 6 immediately afterwards, with the contamination question moot.

CC's recommendation is 3 if cell 4 is close, and 2 otherwise. 2 preserves both
instruments at the cost of one piece of bookkeeping.

## What CC would do, stated at the level the dispatch asks for

Deliberately stopping short of the PyWake object graph, the wind resource
construction and the parameter set, because writing those down here is the
contaminating act itself. This is the procedure, not the configuration.

**Environment.** A fresh conda env created from the kit's pinned requirements
(`phase_2/kit/requirements.txt`), no reuse of `swnd` and no reuse of the
existing `pywake` env, no cached artifacts. Environment identity reported as
the SHA-256 of `pip freeze` output. Note already established on 08-24: the kit
declares a PyWake **floor**, not a pin (`requirements.txt:33`), so "the pinned
requirements" resolves to whatever PyWake satisfies that floor on the day. That
is a default the spec does not state, and it is the single largest threat to a
0.1 pp CF tolerance.

**Inputs.** `Phase_2.pdf` pp. 5-6 for turbine, count, spacing, wake model and
shear. Wind resource at 53.5N 1.5E. TI per-sector from repo code as the
dispatch permits, cited by `file:line` in the after-statement.

**Steps.** Build env; record pip freeze SHA; construct the 7D grid layout;
construct the wind resource at the baseline centre; run the wake model; report
net CF and wake fraction; compare against the expected 47.09 percent net CF
and 9.70 wake at 0.1 pp CF tolerance; write the after-statement.

**Defaults the spec does not state, which CC would have to choose and would
report.** Listed now rather than after the fact, because the strategist's diff
of instruction against execution is only meaningful if the gaps were named in
advance:

1. PyWake version, per the floor-not-pin point above.
2. Superposition model. The scorer replica uses LinearSum; `Phase_2.pdf` names
   the deficit model (Bastankhah-Gaussian) but not the superposition rule.
3. Turbulence model, and whether any is applied at all.
4. Grid orientation and origin of the "plain 7D grid": 7D spacing and 55 units
   do not determine a unique arrangement, and CF depends on it.
5. Air density, and whether the power curve is density-corrected.
6. Time base of the wind resource: which years, which hours, and whether the
   series is the same 14,608-step series the banked case used.
7. Hub-height adjustment: 125 m to 170 m at shear 0.11 is stated, but the
   reference height of the source series is not.

Items 4 and 6 are the ones most likely to move the answer by more than the
0.1 pp tolerance. If the expected 47.09 / 9.70 was produced under a particular
choice for either, a mismatch will read as a reproduction failure when it is
actually an under-specification. That is worth knowing before the run, not
after.

## Status

NOT RUN. No environment created, no configuration written, no physics
executed, no compute spent.
