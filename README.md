# Sea Winds Predictions — Phase 2

Code and working record for a Phase 2 entry to the Sea Winds Predictions
competition: probabilistic wind forecasting over the North Sea (Task 1) and
offshore wind-farm siting and layout (Task 2).

Entrant: **Matteo Niccoli**.

This repository is the tagged companion to the Phase 2 report. It is a curated,
**path-sanitised** copy of a private working repository — see
[SANITISATION.md](SANITISATION.md) for exactly what was changed and why.

---

## What is unusual about this repository

Most competition repositories publish the code and keep the process private.
This one publishes the process, because on this entry the process *was* much of
the method: the work was carried out by LLM agents under a written contract,
and the record of how that was supervised is part of what the entry is claiming.

Three files carry that record, and they are worth more than the scripts:

- **[`LLM_AGENT_LOG.md`](LLM_AGENT_LOG.md)** — an additions-only log of every
  substantial agent session, with a compute disclosure attached to each. Dead
  ends are logged with the reason rather than quietly abandoned, and several
  entries are corrections of earlier entries in the same file. It is long, and
  it is not tidied.
- **[`OBLIGATIONS.md`](OBLIGATIONS.md)** — a ledger in which a line leaves only
  through an explicit closure entry: a date plus the artifact or decision that
  closed it, never by deletion. It exists because obligations had previously
  been lost by being quietly dropped.
- **[`CLAUDE.md`](CLAUDE.md)** — the standing constraints the agents worked
  under: what may be decided autonomously and what may not, the build
  discipline, and the self-check list an output has to survive before it can be
  called done.

If you only read one thing here, read the ledger and then the log entries it
points at.

## Layout

```
scripts/           the pipeline: forecasting, siting, guards, audits, tooling
docs/audit/        pre- and post-statements for the reproduction runs
LLM_AGENT_LOG.md   the agent working record, additions-only
OBLIGATIONS.md     the obligation ledger
CLAUDE.md          standing constraints for the agents
SANITISATION.md    what was replaced for publication, by category
phase_2/           dataset pin (provenance only)
```

109 files in all. It is small because no data is shipped; see
[What is not here](#what-is-not-here).

### Reading `scripts/`

Every artifact in this project comes from a named, committed, seeded script;
there are no notebook-only results and no throwaway names. The naming is
chronological and thematic rather than alphabetical:

| Prefix | What it is |
|---|---|
| `task2_*` | siting and layout: bathymetry, projection, layout search, an independent PyWake scorer replica, QA cross-checks |
| `tier2_*` | the Pangu-Weather d+7 forecast arm: ERA5 fetch, rollout, coupling, downscaling, calibration |
| `dir_residual_*` | the direction arm: unwrapped circular residual models |
| `ws_d7_*`, `*_d14_*` | speed-arm experiments and the climatological d+14 replacement |
| `legB_*` | the swap-year rebuild onto the withheld evaluation year |
| `guard_*`, `*_check*`, `audit_*` | verification: guards, gates, mutation testing, recomputations |
| `change_control_attest.py` | the daily change-control attestation |

Scripts are documented in their own docstrings, which state what they read,
what they write, why the approach is what it is, and whether the step is
deterministic. Several docstrings record a **deviation from an instruction**
and the reason — those are deliberate and are the interesting ones.

## What is not here

No data of any kind: no submissions, no intermediate stacks, no caches, no
model weights, no bathymetry, and not the organizers' starting kit. Everything
excluded is pinned by SHA-256 in the record rather than shipped. The report
prose lives with the report.

The scripts therefore **document** a pipeline you cannot re-run end to end from
this tree alone. That is the intended trade: the record is complete, the bulk
is not redistributed.

## Running anything

Paths that were sanitised are read from environment variables through
`scripts/_publication_paths.py`. An unset variable raises rather than resolving
to nothing. See [SANITISATION.md](SANITISATION.md).

```bash
SW_PROTECTED_ARTIFACTS=/data/pinned python scripts/legB_R1_finalise_20260821.py
```

Dependencies are in `requirements.txt`. The Task 2 scorer replica additionally
needs [PyWake](https://github.com/DTUWindEnergy/PyWake).

## A note on the pinned-artifact register

The record quotes a pinned-artifact register that describes the **private**
working tree. Where this public tree holds a sanitised copy of a script the
register pins by SHA-256, that recorded hash refers to the private original,
not to the copy here — the sanitisation changed those bytes. It did not change
any of the artifacts the pins are actually about.

## Citation

```
Niccoli, M. (2026). Sea Winds Predictions Phase 2 — code and working record.
https://github.com/mycarta/Hackathon-Sea-Winds-Predictions/tree/phase2-report
```

Tag: **`phase2-report`**.

## Licence

[MIT](LICENSE). Third-party components keep their own licences: the
Pangu-Weather weights (CC BY-NC-SA 4.0, not redistributed here) and the
IEA-22-280-RWT reference turbine data (Apache-2.0, IEA Wind TCP Task 55).
