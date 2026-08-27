# Sanitisation of this published tree

This repository is a **curated, path-sanitised copy** of a private working
repository. It was prepared for publication on 2026-08-27 and tagged
`phase2-report`. It holds **109 files**.

Nothing here is a summary or a rewrite. The working record — the agent log, the
obligation ledger, the standing constraints, the scripts — is published as it
was written. What changed is a small, mechanical, category-by-category
substitution of things that identify a machine or a person rather than a
result.

## The principle

**An absolute filesystem path is evidence only to someone who can mount the
drive. The SHA-256 beside it is evidence to anyone.**

Every artifact in this record is pinned by hash. The path is working-record
residue: it tells a reader where a file sat on one laptop, which is not a fact
they can check, act on, or learn from. Replacing it with a placeholder costs the
record nothing and costs a reader nothing, and it removes the only part of the
record that was never anybody else's business.

Three things were **removed rather than anonymised**, because anonymising
them would not have worked: their content, not their location, was the
problem.

## What was replaced, by category

| Category | Treatment |
|---|---|
| Windows user home | `<USER_HOME>` |
| Network share root and its drive label | `<NETWORK_SHARE>` |
| Protected pinned-artifact folder | `<PROTECTED_ARTIFACTS>` |
| Other local drive roots | `<LOCAL_DRIVE>` |
| Repository root, where it was hardcoded | derived from `__file__` instead |
| Claude session transcript directory and session UUIDs | `<CLAUDE_SESSION_PATH>`, `<SESSION_ID>` |
| Account name, where it survived the path rules | `<USER>` |
| Real names other than the entrant's | the role, not the name |
| Register: private working voice, personal state, employer context | removed with the file, or replaced by the role |
| Private repository name | `<PRIVATE_REPO>` |

Counts for the pass: 33 absolute-path replacements, 11 network-share, 9
session-path, 4 real-name, 18 code rewrites, 2 line redactions, across 17 files.

A **second pass the same day** added the register category, which the first
pass did not have: 3 files removed and 3 word replacements across 2 files.
The first pass asked what a string revealed about a *machine*; it never asked
what a document revealed about a *person*.

## What was removed, and why anonymising was not enough

**1. One line of the agent log** referenced a file on a local drive that has
nothing to do with this project and whose *filename itself* named an unrelated
piece of work. A placeholder would have kept the name. The line now reads
`[REDACTED: unrelated work file, not a Sea Winds artifact]`, and the surrounding
log entry is untouched.

**2. One verbatim quotation** of organizer-supplied text from the competition
brief. The citation survives with its page and clause number; the quoted words
do not. Organizer correspondence and organizer prose are not published here,
wherever they sat in the tree.

**3. Three superseded working papers** from `docs/audit/`: an early version of
the audit design and its two cold reviews. They were superseded by the frozen
design that the reproduction runs actually followed, so they are not the
instrument. They also quoted the entrant's own informal messages verbatim and
discussed his working condition — private working voice, not method. A
placeholder cannot help a document whose register is the problem.

`docs/audit/` keeps what is evidence: the pre- and post-statements for the
anchor 6 reproduction run, and the index to the extracts they cover.

The first two are **marked, not deleted**. The agent log is an additions-only
record, and a silently shortened line would break that property for every
other line in it.

## What this means if you want to run the code

Where a placeholder landed in prose it simply reads as a placeholder. Where one
landed in **executable code**, the code now reads the real location from an
environment variable through `scripts/_publication_paths.py`:

| Placeholder | Environment variable |
|---|---|
| `<USER_HOME>` | `SW_USER_HOME` |
| `<NETWORK_SHARE>` | `SW_NETWORK_SHARE` |
| `<PROTECTED_ARTIFACTS>` | `SW_PROTECTED_ARTIFACTS` |
| `<LOCAL_DRIVE>` | `SW_LOCAL_DRIVE` |
| `<CLAUDE_SESSION_PATH>` | `SW_CLAUDE_SESSION_PATH` |

```bash
SW_PROTECTED_ARTIFACTS=/data/pinned python scripts/legB_R1_finalise_20260821.py
```

**There is no silent fallback, deliberately.** An unset variable raises. A
variable pointing at a directory that does not exist raises. That second check
matters more than it looks: this project has already been bitten once by a
missing data root that produced an *empty list* instead of an error, and a
pipeline that finds nothing is far more dangerous than one that stops. The
resolver is unit-tested against exactly that case.

`must_exist=False` appears only where the original code legitimately tolerated
absence — a fallback candidate it probes with `.exists()`, or an output path it
is about to create.

## What is not here

- **All data.** No submissions, no `.npz` stacks, no parquet, no caches, no
  bathymetry, no model weights. The competition dataset is the organizers' to
  distribute, and the derived artifacts are large and pinned by hash in the
  record rather than shipped.
- **The Pangu-Weather ONNX weights** (~1.18 GB), which are third-party and
  pinned by SHA-256 in the record.
- **The organizer's starting kit**, which is theirs to publish.
- **Report prose drafts**, which belong with the report.
- **The sanitisation tool itself.** Its rule table contains the literal strings
  this pass removes, so publishing it would republish them. It stays in the
  private repository. This document is the account of what it did.

## One consequence worth stating plainly

`data/PINNED_ARTIFACTS.md` — the pinned-artifact register quoted in the record —
describes the **private working tree**. Where this public tree contains a
sanitised copy of a script that the register pins by SHA-256, that recorded SHA
refers to the **private original**, not to the copy you are reading. The
sanitisation changed those bytes; it did not change any artifact the pins are
actually about.

## Verification

The publication scan run over this tree before tagging covers absolute paths of
every shape including UNC, usernames, real names, email addresses, hostnames, IP
addresses, port bindings, credential patterns and key blobs, corporate drive
labels and internal project names, private git remotes, conda and virtualenv
prefixes, and third-party correspondence. It returned **zero unresolved hits**,
and every Python file in the tree was asserted to parse after the rewrite.
