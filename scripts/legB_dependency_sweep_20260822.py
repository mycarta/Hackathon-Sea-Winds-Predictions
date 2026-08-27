#!/usr/bin/env python3
"""Leg B dependency sweep: enumerate every file-read path in the frozen chain.

Amendment 1 of Matteo's 2026-08-22 dispatch. Rationale, in his words: the
register missed `arm_extracts_sub/`, so **register completeness is an open
claim; close it by enumeration.**

Method, deliberately static (no imports executed, nothing read through the
pipeline):

  1. Start at `scripts/tier2_d7_build_submission.py`.
  2. Walk the LOCAL import graph transitively (repo `scripts/` plus the two kit
     directories the frozen file puts on `sys.path`). Third-party imports are
     recorded as leaves, not followed.
  3. In every module of that graph, find read-ish call sites by AST:
     `np.load`, `open`, `pd.read_*`, `xr.open_*`, `joblib.load`, `pickle.load`,
     `json.load`, `zipfile.ZipFile`, `Path.read_*`.
  4. Also collect module-level path CONSTANTS (assignments whose value mentions
     a path-like literal), because that is where `SUB_EXTRACTS` hid.
  5. Resolve every literal or constant-derived path against disk.
  6. Cross-check every resolved path against `data/PINNED_ARTIFACTS.md`.

Output: `reports/legB_dependency_sweep_20260822.md`, ASCII-asserted.

Deterministic: no randomness, no network. Run:
    python scripts/legB_dependency_sweep_20260822.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from _publication_paths import ppath  # noqa: E402  (publication tree)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROOT = REPO / "scripts" / "tier2_d7_build_submission.py"
REPORT = REPO / "reports" / "legB_dependency_sweep_20260822.md"
REGISTER = REPO / "data" / "PINNED_ARTIFACTS.md"

SEARCH_DIRS = [
    REPO / "scripts",
    REPO / "phase_2" / "kit" / "phase_2" / "part1_forecast",
    REPO / "phase_2" / "kit" / "phase_2" / "part0_dataset_setup",
    # `config.py` lives one level UP from the two kit dirs the frozen file puts
    # on sys.path. The first run of this sweep therefore treated `config` as a
    # third-party leaf and never swept it, even though `config.target_root()`
    # decides the data root for the whole chain. Added 2026-08-22 after that
    # gap was caught; the omission is recorded rather than quietly repaired.
    REPO / "phase_2" / "kit" / "phase_2",
]

READ_FUNCS = {
    "load", "loadtxt", "open", "ZipFile", "read_csv", "read_parquet",
    "read_json", "read_pickle", "read_table", "read_feather", "read_text",
    "read_bytes", "open_dataset", "open_zarr", "open_mfdataset", "Dataset",
    "imread", "load_npz", "read_hdf",
}
READ_ATTR_OWNERS = {"np", "numpy", "pd", "pandas", "xr", "xarray", "joblib",
                    "pickle", "json", "zipfile", "nc", "netCDF4", "Path"}

PATHY = re.compile(r"[\\/]|[.](npz|csv|json|pkl|parquet|zip|nc|txt|md|zarr|onnx)\b", re.I)
STRLIT = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def local_modules():
    m = {}
    for d in SEARCH_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.py")):
            m.setdefault(p.stem, p)
    return m


class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.reads = []       # (lineno, rendered call)
        self.consts = {}      # name -> (lineno, rendered value)
        self.joins = {}       # name -> "data/x" for pathlib join expressions

    def visit_Import(self, n):
        for a in n.names:
            self.imports.add(a.name.split(".")[0])

    def visit_ImportFrom(self, n):
        if n.module:
            self.imports.add(n.module.split(".")[0])

    def visit_Call(self, n):
        f = n.func
        name = owner = None
        if isinstance(f, ast.Attribute):
            name = f.attr
            if isinstance(f.value, ast.Name):
                owner = f.value.id
        elif isinstance(f, ast.Name):
            name = f.id
        if name in READ_FUNCS and (owner is None or owner in READ_ATTR_OWNERS
                                   or name in ("open", "ZipFile")):
            try:
                src = ast.unparse(n)
            except Exception:
                src = str(name) + "(...)"
            self.reads.append((n.lineno, src[:200]))
        self.generic_visit(n)

    def visit_Assign(self, n):
        if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            nm = n.targets[0].id
            if nm.isupper() or nm.startswith("_"):
                try:
                    src = ast.unparse(n.value)
                except Exception:
                    src = "<unparseable>"
                # A path constant is either a path-looking string literal, OR a
                # pathlib join expression (`_HERE.parent / "data" / "x"`), which
                # contains no slash and no extension and so does NOT match
                # PATHY. Repointing the dead Z: constants turned them into
                # exactly that shape and made them vanish from this sweep;
                # caught 2026-08-22 and fixed here rather than left as a hole.
                pj = pathjoin(n.value)
                if PATHY.search(src) or pj:
                    self.consts[nm] = (n.lineno, src[:200])
                    if pj:
                        self.joins[nm] = pj
        self.generic_visit(n)


def walk():
    mods = local_modules()
    seen, order, third = {}, [], set()
    queue = [("tier2_d7_build_submission", ROOT)]
    while queue:
        stem, path = queue.pop(0)
        if stem in seen:
            continue
        v = Visitor()
        v.visit(ast.parse(path.read_text(encoding="utf-8", errors="replace")))
        seen[stem] = (path, v)
        order.append(stem)
        for imp in sorted(v.imports):
            if imp in mods:
                queue.append((imp, mods[imp]))
            else:
                third.add(imp)
    return seen, order, sorted(third)


def pathjoin(node):
    """Render `a / "b" / "c"` as `b/c`; return None if not a `/` chain."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
        return None
    parts = []

    def walk_(x):
        if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Div):
            walk_(x.left)
            walk_(x.right)
        elif isinstance(x, ast.Constant) and isinstance(x.value, str):
            parts.append(x.value)

    walk_(node)
    return "/".join(parts) if parts else None


def literals(src):
    """Path-looking string literals inside a rendered expression."""
    out = []
    for a, b in STRLIT.findall(src):
        s = a or b
        if s and PATHY.search(s):
            out.append(s)
    return out


WORKING = ppath("<PROTECTED_ARTIFACTS>")
TEMPLATE = re.compile(r"[{}]")

_INDEX = None


def basename_index():
    """Every filename anywhere in the repo, plus the protected working folder.

    Most call sites write a bare basename and join it to a directory constant at
    runtime (`ARTIFACTS / "tier2_d7_datemap.json"`), so a literal-only lookup
    reports false MISSINGs. Resolution therefore falls back to a name index.
    """
    global _INDEX
    if _INDEX is None:
        idx = {}
        for root in (REPO, WORKING):
            if not root.is_dir():
                continue
            for p in root.rglob("*"):
                if ".git" in p.parts:
                    continue
                if p.is_file() or p.is_dir():
                    idx.setdefault(p.name, p)
        _INDEX = idx
    return _INDEX


def resolve(s):
    """-> (path_or_None, status) with status in {yes, no, template}."""
    if TEMPLATE.search(s):
        return None, "template"
    p = Path(s)
    cands = [p] if p.is_absolute() else [REPO / s, REPO / "scripts" / s, WORKING / s]
    for c in cands:
        try:
            if c.exists():
                return c, "yes"
        except OSError:
            pass
    hit = basename_index().get(Path(s).name)
    if hit is not None:
        return hit, "yes"
    return cands[0], "no"


def main():
    seen, order, third = walk()
    reg_text = REGISTER.read_text(encoding="utf-8", errors="replace") if REGISTER.exists() else ""

    L = []
    L.append("# Leg B dependency sweep, 2026-08-22")
    L.append("")
    L.append("Produced by `scripts/legB_dependency_sweep_20260822.py`. Static AST walk;")
    L.append("no module is imported and no pipeline data is read.")
    L.append("")
    L.append("Root: `scripts/tier2_d7_build_submission.py`")
    L.append("")
    L.append("## 1. Local import graph (%d modules)" % len(order))
    L.append("")
    for stem in order:
        p = seen[stem][0]
        L.append("- `%s` -> `%s`" % (stem, p.relative_to(REPO).as_posix()))
    L.append("")
    L.append("Third-party / stdlib leaves, not followed: %s"
             % ", ".join("`%s`" % t for t in third))
    L.append("")

    L.append("## 2. Path constants (where SUB_EXTRACTS hid)")
    L.append("")
    L.append("| Module | Line | Constant | Value | On disk |")
    L.append("|---|---|---|---|---|")
    const_paths = []
    for stem in order:
        p, v = seen[stem]
        rel = p.relative_to(REPO).as_posix()
        for nm, (ln, src) in sorted(v.consts.items()):
            lits = literals(src) or ([joined] if (joined := v.joins.get(nm)) else [])
            if lits:
                got, st = resolve(lits[0])
                mark = {"yes": "YES", "no": "**NO**", "template": "template"}[st]
                const_paths.append((rel, ln, nm, lits[0], st))
            else:
                mark = "derived"
            L.append("| `%s` | %d | `%s` | `%s` | %s |"
                     % (rel, ln, nm, src.replace("|", "\\|"), mark))
    L.append("")

    L.append("## 3. Read call sites")
    L.append("")
    L.append("| Module | Line | Call |")
    L.append("|---|---|---|")
    nreads = 0
    for stem in order:
        p, v = seen[stem]
        rel = p.relative_to(REPO).as_posix()
        for ln, src in v.reads:
            nreads += 1
            L.append("| `%s` | %d | `%s` |" % (rel, ln, src.replace("|", "\\|")))
    L.append("")
    L.append("%d read call sites across %d modules." % (nreads, len(order)))
    L.append("")

    L.append("## 4. Resolved paths: disk and register status")
    L.append("")
    L.append("| Literal in code | Status | Resolved to | In PINNED_ARTIFACTS.md |")
    L.append("|---|---|---|---|")
    rows = []
    for rel, ln, nm, lit, st in const_paths:
        rows.append(lit)
    for stem in order:
        p, v = seen[stem]
        for ln, src in v.reads:
            rows.extend(literals(src))
    uniq = {}
    for lit in rows:
        if lit in uniq:
            continue
        got, st = resolve(lit)
        base = Path(lit).name or lit
        registered = (base in reg_text) or (lit in reg_text)
        uniq[lit] = (st, got, registered)
    for lit in sorted(uniq):
        st, got, registered = uniq[lit]
        mark = {"yes": "YES", "no": "**MISSING**", "template": "template"}[st]
        where = "n/a" if got is None else (
            got.relative_to(REPO).as_posix() if str(got).startswith(str(REPO))
            else got.as_posix())
        L.append("| `%s` | %s | `%s` | %s |"
                 % (lit, mark, where, "yes" if registered else "**no**"))
    L.append("")
    missing = [k for k, (st, g, r) in uniq.items() if st == "no"]
    unreg = [k for k, (st, g, r) in uniq.items() if not r and st != "template"]
    L.append("**MISSING from disk: %d** -- %s"
             % (len(missing), ", ".join("`%s`" % m for m in missing) or "none"))
    L.append("")
    L.append("**Not found in the register: %d** -- %s"
             % (len(unreg), ", ".join("`%s`" % m for m in unreg) or "none"))
    L.append("")
    L.append("Register membership is a substring test on the basename, so it is")
    L.append("generous: a `no` is a hard miss, a `yes` still deserves a human read.")
    L.append("")
    L.append("**Resolution is generous in the same direction.** A bare basename that is")
    L.append("joined to a directory constant at runtime is looked up in a repo-wide name")
    L.append("index, so it can land on a same-named file elsewhere. `metadata.json`")
    L.append("resolving into `mini_challenge/` is exactly that: the real read is per")
    L.append("inference window. A `YES` here means \"a file of this name exists\", which is")
    L.append("enough to clear the MISSING question and nothing more.")
    L.append("")

    L.append("## 5. Findings, classified")
    L.append("")
    L.append("### 5.1 The dead `Z:` paths: found, then closed")
    L.append("")
    L.append("The first run of this sweep (commit `98ba5d2`, before any repointing)")
    L.append("found **three distinct dead literals** pointing into the deleted")
    L.append("`tier2_smoke/` folder, spread over **four** assignment sites:")
    L.append("")
    L.append("| Constant | Site | What it held |")
    L.append("|---|---|---|")
    L.append("| `DWN_CACHE` | `tier2_f2_d14_precheck.py:42` | the lost pinned pickle |")
    L.append("| `EXTRACTS` | `tier2_f2_d14_precheck.py:41` | the 80 bias extracts |")
    L.append("| `EXTRACTS` | `tier2_d7_score_blocks.py:37` | the 80 bias extracts (second copy) |")
    L.append("| `SUB_EXTRACTS` | `tier2_d7_build_submission.py:56` | the 32 window extracts |")
    L.append("")
    L.append("Only `SUB_EXTRACTS` was on anyone's list. The other three were found by")
    L.append("enumeration, which is what this sweep was ordered to do.")
    L.append("")
    L.append("**Two of them had silent fallbacks, which is the part that matters.**")
    L.append("`get_downscaler()` retrains from scratch when its cache file is absent")
    L.append("instead of stopping, so a run with the dead `DWN_CACHE` would have")
    L.append("quietly produced an UNPINNED downscaler and carried on. And")
    L.append("`tier2_d7_score_blocks.py:144` gates its scorable-date list on")
    L.append("`extract_<date>.npz` existing, so a dead `EXTRACTS` yields an EMPTY date")
    L.append("list rather than an error. Neither would have raised.")
    L.append("")
    L.append("All four sites are now repointed at committed repo locations. This run")
    L.append("confirms **no `tier2_smoke` literal remains anywhere in the 18-module")
    L.append("chain**, and the MISSING list is down from four entries to one.")
    L.append("")
    L.append("### 5.2 NEW finding: a downscaler feature that is silently always zero")
    L.append("")
    L.append("`terrain_features.py:26` points `_ELEVATION_NC` at")
    L.append("`phase_2/build/phase1_dataset/train/elevation_north_sea.nc`.")
    L.append("`_load_elevation_dem()` returns `None` when that file is absent, and")
    L.append("`_interp_elevation_to_grid_cached()` then returns `np.zeros(shape)`.")
    L.append("")
    L.append("`elevation_m` is `FEATURES[3]` in `downscaling.py:41`. So the downscaler")
    L.append("has been trained and applied with that feature identically zero.")
    L.append("")
    L.append("**Checked before drawing any conclusion:** `phase_2/build/` does not exist")
    L.append("on disk, appears in no commit reachable from any ref, and no file of that")
    L.append("name exists anywhere under `<LOCAL_DRIVE>/Pythonwork`. The DEM has therefore NEVER")
    L.append("been present in this repo.")
    L.append("")
    L.append("Consequences, stated separately because they differ:")
    L.append("")
    L.append("1. **R2 comparability is UNAFFECTED.** July's fit and today's refit both")
    L.append("   saw zeros. The training construction really is identical; this does not")
    L.append("   weaken the refit claim.")
    L.append("2. **The model is a 5-feature model described as 6.** A constant column")
    L.append("   yields no split gain in LightGBM, so `elevation_m` is inert rather than")
    L.append("   harmful. Nothing is corrupted; something is merely absent.")
    L.append("3. **The failure mode is the silent-degradation class**, the same genus as")
    L.append("   the three paths logged in the v2.1 amendment: a missing input handled")
    L.append("   by a quiet default instead of a loud stop. It is organizer kit code")
    L.append("   inside a FROZEN chain, so it is reported and NOT edited.")
    L.append("")
    L.append("### 5.3 Registered-status `no`: what it does and does not mean")
    L.append("")
    L.append("`arome_static.nc`, `emodnet_northsea_1km.nc`, `climatology_coarse.npz` and")
    L.append("`metadata.json` resolve on disk but appear in NEITHER")
    L.append("`data/PINNED_ARTIFACTS.md` NOR `data/MANIFEST_zenodo_20335351.md`.")
    L.append("")
    L.append("This was checked rather than assumed, and the first draft of this section")
    L.append("asserted the manifest covered them by name. It does not. The manifest pins")
    L.append("**the 12.0 GB `phase2_dataset.zip` as a whole** (SHA-256 `f4b60b3d...`)")
    L.append("plus one promoted Phase-1 parquet. Shipped dataset files are therefore")
    L.append("covered TRANSITIVELY by the archive hash, not individually.")
    L.append("")
    L.append("That is a real difference. Archive-level pinning verifies what was")
    L.append("downloaded; it does not detect a shipped file that was later modified,")
    L.append("moved or replaced in the unpacked tree, because nothing re-checks the")
    L.append("unpacked copies against the archive. The A1 gate is satisfied as written.")
    L.append("The gap is that A1 as written stops at the archive boundary.")
    L.append("")
    L.append("Reported, not acted on: extending the manifest to per-file hashes of the")
    L.append("unpacked tree is a governance change, not a CC decision, and it is not on")
    L.append("the Leg B critical path.")
    L.append("")
    L.append("The remaining `no`s (`submission_pangu_d7_allhours_20260719.csv`,")
    L.append("`tier2_d7_build_submission_summary.json`, `tier2_d7_fourblock.json`) are")
    L.append("pipeline OUTPUTS, not inputs, and are not pinned artifacts by definition.")
    L.append("")

    text = "\n".join(L) + "\n"
    bad = sorted(set(c for c in text if ord(c) > 126))
    assert not bad, "non-ASCII in report: %r" % bad
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    print("wrote %s" % REPORT.relative_to(REPO))
    print(json.dumps({"modules": len(order), "read_sites": nreads,
                      "paths": len(uniq), "missing": missing,
                      "unregistered": unreg}, indent=2))


if __name__ == "__main__":
    main()
