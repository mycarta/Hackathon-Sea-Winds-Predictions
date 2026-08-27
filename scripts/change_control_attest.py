"""
Daily change-control attestation (audit design Tier 3, section 4.1).

Re-runnable, read-only, offline. Writes one dated attestation into
reports/attestations/ and prints the same content to stdout.

Five sections, in the order the audit design states them:

  1. MANIFEST re-hash      every (path, SHA-256) pair found in every
                           MANIFEST*.md, re-hashed against the file on disk.
  2. PINNED_ARTIFACTS      every register entry, its declared status, and its
                           (path, SHA) pairs re-hashed. Entry count asserted.
  3. Frozen-file diff      frozen artifacts: working-tree cleanliness, and the
                           commits that have touched them, so unauthorized
                           edits are visible rather than assumed absent.
  4. Push state            unpushed commits against the tracking branch.
  5. Clean tree            every modified, staged or untracked tracked-path.

DESIGN NOTE, and the reason this script is longer than it looks like it should
be. The failure mode an attestation has to avoid is its own silence. A parser
that skips a line it cannot read, or a path it cannot resolve, produces a
clean-looking report that attests to nothing. So every (path, SHA) pair found
is placed in exactly one bucket: MATCH, MISMATCH, MISSING, or UNRESOLVED, and
UNRESOLVED is printed in full and counted as a failure of the attestation, not
as a skip. The attestation PASSES only when there are zero MISMATCH, zero
MISSING outside declared-LOST register entries, and zero UNRESOLVED.

Line endings. core.autocrlf is true in this checkout, so a text file's raw
working-copy hash is not its committed blob content. Where a recorded SHA was
taken LF-normalised (register entry 7), the raw hash will not match and the
LF-normalised one will. Both are computed; a pair matching either is a MATCH,
and the report names which hash matched. A pair matching neither is a MISMATCH.

Usage:  conda run -n swnd python scripts/change_control_attest.py
        conda run -n swnd python scripts/change_control_attest.py --date 2026-08-25
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
from collections import OrderedDict
from _publication_paths import ppath  # noqa: E402  (publication tree)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "reports", "attestations")

SHA_RE = re.compile(r"\b[0-9a-f]{64}\b")
# A path-ish token: has a dot-extension, or is a directory path with a slash.
PATH_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,8}\b")

# Protected folders outside the repo that pinned artifacts legitimately live
# in (contract v2.1 5a allows one named protected folder for artifacts too
# large or gitignored). Searched after the repo, never before it.
EXTERNAL_ROOTS = [
    str(ppath("<PROTECTED_ARTIFACTS>")),
    os.path.join(os.path.expanduser("~"), "tier2_model"),
]

REGISTER = os.path.join(REPO, "data", "PINNED_ARTIFACTS.md")
EXPECTED_REGISTER_ENTRIES = 8

# Frozen artifacts. Sources: data/PINNED_ARTIFACTS.md, the "Frozen artifacts
# NOT SHA-asserted in code" table and the submission entries 4, 5 and 7.
FROZEN = [
    "phase_2/kit/phase_2/part1_forecast/submission.csv",
    "phase_2/kit/phase_2/part1_forecast/submission.zip",
    "scripts/artifacts/d14_climatology_season.parquet",
    "scripts/artifacts/ws_d7_bias_shrunk_table.parquet",
    "scripts/artifacts/tier2_d7_datemap.json",
    "scripts/artifacts/tier2_sub_datemap.json",
    "data/task2_layout_winner.json",
    "data/iea22mw_power_ct.csv",
    "reports/three_case_scorer_20260818.json",
]


# Pins the MANIFEST parser cannot resolve on its own, checked explicitly.
# Two reasons a pin lands here: the file lives off-repo (a Zenodo download the
# repo never holds), or its basename exists at more than one path so automatic
# resolution would have to guess. Guessing is what `resolve` refuses to do, so
# the guess is made once, here, in the open.
EXPLICIT_PINS = [
    ("Phase-1 HRES (Zenodo 19538994 v1), repo copy",
     "phase_1/hres_north_sea.parquet",
     "7b00c61df2d56f2f69445ec8677ba50de734aa782a335447dad8d353e1587be3"),
    ("Phase-1 HRES, second copy under the 2022 inference tree",
     "phase_2/inference_2022/phase2_dataset_ship/train/hres/hres_north_sea.parquet",
     "7b00c61df2d56f2f69445ec8677ba50de734aa782a335447dad8d353e1587be3"),
    ("Pangu ONNX (register entry 1, protected copy)",
     str(ppath("<PROTECTED_ARTIFACTS>/pangu_weather_24.onnx",
               must_exist=False)),
     "613a5c140a1399abcaffb4dbce32af373a1f5f56c515704f5be61925bb9fdcfd"),
    ("Downscaler 2b (register entry 2b, protected copy)",
     str(ppath("<PROTECTED_ARTIFACTS>/downscaler_blockexcl_20260822.pkl",
               must_exist=False)),
     "b3ae32c0bf4203351a03526a454030817f70adb588f55caefdb0b43b5a2d8703"),
]


def git(*args):
    out = subprocess.run(["git"] + list(args), cwd=REPO,
                         capture_output=True, text=True)
    return out.stdout.rstrip("\n"), out.returncode


def sha_raw(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_lf(path):
    """SHA of the content with CRLF collapsed to LF: the committed blob content
    for a text file under core.autocrlf=true."""
    with open(path, "rb") as fh:
        data = fh.read()
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def find_manifests():
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "node_modules", "__pycache__")]
        for f in files:
            if f.startswith("MANIFEST") and f.endswith(".md"):
                hits.append(os.path.join(root, f))
    return sorted(hits)


def resolve(token, near_dir):
    """Resolve a path token from a document to a real file. Returns the
    absolute path, or None. Never guesses between two candidates."""
    token = token.strip().strip("`*|").replace("\\", "/")
    if not token:
        return None
    cands = [os.path.join(near_dir, token),
             os.path.join(REPO, token),
             os.path.join(near_dir, os.path.basename(token))]
    for r in EXTERNAL_ROOTS:
        cands.append(os.path.join(r, token))
        cands.append(os.path.join(r, os.path.basename(token)))
    for c in cands:
        if os.path.isfile(c):
            return c
    # Last resort: a unique basename match anywhere in the repo. Ambiguous
    # matches are deliberately NOT resolved; they are reported UNRESOLVED.
    base = os.path.basename(token)
    found = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "node_modules", "__pycache__")]
        if base in files:
            found.append(os.path.join(root, base))
            if len(found) > 1:
                return None
    return found[0] if len(found) == 1 else None


def pairs_in(doc):
    """Every (sha, path_token, line_no, raw_line) in a document.

    A SHA is paired with a path token on its own line where one exists; failing
    that, with the nearest path token on a preceding line within a short
    window. Anything left unpaired is yielded with a None token so the caller
    reports it as UNRESOLVED rather than dropping it."""
    with open(doc, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    out = []
    for i, line in enumerate(lines):
        for sha in SHA_RE.findall(line):
            # All path tokens on the line, then those on the preceding few
            # lines, in that order. Every one is a candidate: a table row can
            # name several paths and only one of them is the hashed file, so
            # the parser must not commit to the first token it sees.
            toks = [t for t in PATH_RE.findall(line) if not SHA_RE.match(t)]
            for back in range(1, 6):
                if i - back < 0:
                    break
                toks += [t for t in PATH_RE.findall(lines[i - back])
                         if not SHA_RE.match(t)]
            seen, ordered = set(), []
            for t in toks:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
            out.append((sha, ordered, i + 1, line.strip()))
    return out


_DIR_INDEX = {}
DIR_INDEX_FILE_CAP = 512 * 1024 * 1024     # skip any single file above this
DIR_INDEX_DIR_CAP = 8 * 1024 * 1024 * 1024  # skip the directory above this


def dir_index(dirpath):
    """sha -> path for the files sitting directly in one directory.

    This exists because several MANIFESTs key their per-file rows on something
    that is not a path (the arm-extract tables key on issue date), so a path
    token simply is not present to parse. Hashing the manifest's own directory
    resolves those rows exactly, and does it without hashing 31 GB of repo."""
    if dirpath in _DIR_INDEX:
        return _DIR_INDEX[dirpath]
    idx = {}
    try:
        names = [n for n in os.listdir(dirpath)
                 if os.path.isfile(os.path.join(dirpath, n))]
    except OSError:
        _DIR_INDEX[dirpath] = idx
        return idx
    total = 0
    for n in names:
        p = os.path.join(dirpath, n)
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    if total > DIR_INDEX_DIR_CAP:
        _DIR_INDEX[dirpath] = idx
        return idx
    for n in names:
        p = os.path.join(dirpath, n)
        try:
            if os.path.getsize(p) > DIR_INDEX_FILE_CAP:
                continue
            idx.setdefault(sha_raw(p), p)
            idx.setdefault(sha_lf(p), p)
        except OSError:
            continue
    _DIR_INDEX[dirpath] = idx
    return idx


def zip_member_match(path, sha):
    """A recorded SHA is sometimes the hash of a file INSIDE an archive rather
    than of the archive (register entry 5 says so explicitly). Checked rather
    than assumed, so that case reports MATCH with the member named instead of
    a MISMATCH a reader has to know to discount."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                h = hashlib.sha256()
                with z.open(info) as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() == sha:
                    return info.filename
    except Exception:
        return None
    return None


def classify(sha, toks, near_dir):
    """Bucket one recorded hash. `toks` is the ordered candidate path list."""
    toks = toks or []
    resolved = []
    for t in toks:
        p = resolve(t, near_dir)
        if p and p not in resolved:
            resolved.append(p)

    # 1. Any candidate path whose content hashes to the recorded value.
    for p in resolved:
        if sha_raw(p) == sha:
            return "MATCH", p, "raw"
        if sha_lf(p) == sha:
            return "MATCH", p, "lf"
    # 2. A recorded hash of a member inside a candidate archive.
    for p in resolved:
        if p.lower().endswith(".zip"):
            member = zip_member_match(p, sha)
            if member:
                return "MATCH", "%s -> member %s" % (p, member), "zip-member"
    # 3. Any file sitting in the document's own directory.
    hit = dir_index(near_dir).get(sha)
    if hit:
        return "MATCH", hit, "dir-scan"

    # Nothing matched. Report MISMATCH only against a candidate whose BASENAME
    # is the one the document named. Without this guard a hash whose real file
    # lives off-repo gets pinned on whatever unrelated path happened to share
    # the line, and the attestation invents a change-control failure. A
    # resolution that does not agree on the filename is not a resolution.
    named = {os.path.basename(t.strip().strip("`*|").replace("\\", "/")).lower()
             for t in toks}
    for p in resolved:
        if os.path.basename(p).lower() in named:
            return "MISMATCH", p, None
    if resolved:
        return "UNRESOLVED", ("hash names %r; nearest resolvable path %s has a "
                              "different filename, so no claim is made"
                              % (sorted(named)[:2],
                                 os.path.relpath(resolved[0], REPO))), None
    if toks:
        return "UNRESOLVED", "no candidate of %r resolves to a file" % (
            toks[:3],), None
    return "UNRESOLVED", "no path token near the hash", None


def register_entries():
    """Top-level register entries with their declared status."""
    with open(REGISTER, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    ents = OrderedDict()
    for i, line in enumerate(lines):
        m = re.match(r"^### (\d+[a-z]?)\.\s+(.*)$", line)
        if m:
            ents[m.group(1)] = {"title": m.group(2), "line": i + 1,
                                "lost": "(LOST" in m.group(2),
                                "in_progress": "IN PROGRESS" in m.group(2)}
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="attestation date, YYYY-MM-DD. Required: this script "
                         "does not read the clock, so the date in the filename "
                         "is always a stated input rather than an assumption.")
    args = ap.parse_args()
    if not args.date:
        print("ERROR: --date YYYY-MM-DD is required.", file=sys.stderr)
        return 2
    date = args.date

    L = []
    def say(s=""):
        L.append(s)
        print(s)

    fails = {"mismatch": 0, "missing": 0, "unresolved": 0}

    say("# Change-control attestation, %s" % date)
    say()
    say("Produced by `scripts/change_control_attest.py`. Read-only, offline.")
    head, _ = git("rev-parse", "HEAD")
    say("HEAD `%s`." % head[:12])
    say()

    # --- 1. MANIFEST re-hash ----------------------------------------------
    say("## 1. MANIFEST re-hash")
    say()
    mans = find_manifests()
    say("%d MANIFEST files found." % len(mans))
    say()
    say("| MANIFEST | pairs | match | mismatch | missing | unresolved |")
    say("|---|---:|---:|---:|---:|---:|")
    detail = []
    for m in mans:
        near = os.path.dirname(m)
        c = {"MATCH": 0, "MISMATCH": 0, "MISSING": 0, "UNRESOLVED": 0}
        for sha, toks, ln, raw in pairs_in(m):
            kind, info, how = classify(sha, toks, near)
            c[kind] += 1
            if kind != "MATCH":
                detail.append((os.path.relpath(m, REPO).replace("\\", "/"),
                               ln, kind, sha, info, raw[:110]))
        fails["mismatch"] += c["MISMATCH"]
        fails["missing"] += c["MISSING"]
        fails["unresolved"] += c["UNRESOLVED"]
        say("| `%s` | %d | %d | %d | %d | %d |"
            % (os.path.relpath(m, REPO).replace("\\", "/"), sum(c.values()),
               c["MATCH"], c["MISMATCH"], c["MISSING"], c["UNRESOLVED"]))
    say()
    if detail:
        say("Every non-MATCH, in full:")
        say()
        for d in detail:
            say("- `%s:%d` **%s** `%s...`  %s" % (d[0], d[1], d[2], d[3][:16], d[4]))
            say("  > %s" % d[5])
        say()
    else:
        say("No non-MATCH pairs.")
        say()

    say("### Explicitly checked pins")
    say()
    say("Pins the parser cannot resolve on its own, checked by stated path. "
        "The Phase-1 HRES appears at two paths, so automatic resolution is "
        "refused by design; both copies are hashed here instead.")
    say()
    say("| Pin | path exists | result |")
    say("|---|---|---|")
    for label, rel, sha in EXPLICIT_PINS:
        ap_ = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        if not os.path.isfile(ap_):
            say("| %s | no | **ABSENT** |" % label)
            fails["missing"] += 1
            continue
        got = sha_raw(ap_)
        if got == sha:
            say("| %s | yes | MATCH |" % label)
        else:
            say("| %s | yes | **MISMATCH** got `%s...` |" % (label, got[:16]))
            fails["mismatch"] += 1
    say()

    # --- 2. PINNED_ARTIFACTS ----------------------------------------------
    say("## 2. PINNED_ARTIFACTS register")
    say()
    ents = register_entries()
    top = [k for k in ents if k.isdigit()]
    say("Entries found: %d total headings (%s), of which %d top-level numbered."
        % (len(ents), ", ".join(ents.keys()), len(top)))
    if len(top) == EXPECTED_REGISTER_ENTRIES:
        say("Top-level entry count %d matches the expected %d. PASS"
            % (len(top), EXPECTED_REGISTER_ENTRIES))
    else:
        say("Top-level entry count %d does NOT match the expected %d. **FAIL**"
            % (len(top), EXPECTED_REGISTER_ENTRIES))
        fails["mismatch"] += 1
    say()
    lost = [k for k, v in ents.items() if v["lost"]]
    prog = [k for k, v in ents.items() if v["in_progress"]]
    say("Declared LOST: %s. Declared IN PROGRESS: %s."
        % (", ".join(lost) if lost else "none",
           ", ".join(prog) if prog else "none"))
    say()
    near = os.path.dirname(REGISTER)
    rc = {"MATCH": 0, "MISMATCH": 0, "MISSING": 0, "UNRESOLVED": 0,
          "EXPECTED-ABSENT": 0}
    rdetail = []
    # Which entry each hash belongs to, so a hash inside an entry the register
    # itself declares LOST is scored as EXPECTED-ABSENT rather than as a
    # change-control failure. Those entries are retained on purpose and their
    # files are gone on purpose; counting them as failures would train the
    # reader to ignore the failure column, which is how a real one gets missed.
    bounds = sorted([(v["line"], k) for k, v in ents.items()])
    def entry_of(ln):
        cur = None
        for start, key in bounds:
            if start <= ln:
                cur = key
            else:
                break
        return cur
    for sha, toks, ln, raw in pairs_in(REGISTER):
        kind, info, how = classify(sha, toks, near)
        ek = entry_of(ln)
        if kind != "MATCH" and ek and ents[ek]["lost"]:
            kind, info = "EXPECTED-ABSENT", "entry %s is declared LOST" % ek
        rc[kind] += 1
        if kind != "MATCH":
            rdetail.append((ln, kind, sha, info, raw[:110]))
    say("| pairs | match | mismatch | missing | unresolved | expected-absent |")
    say("|---:|---:|---:|---:|---:|---:|")
    say("| %d | %d | %d | %d | %d | %d |"
        % (sum(rc.values()), rc["MATCH"], rc["MISMATCH"], rc["MISSING"],
           rc["UNRESOLVED"], rc["EXPECTED-ABSENT"]))
    say()
    say("Hashes inside entries 2, 3 and 6 are scored EXPECTED-ABSENT: the "
        "register declares those files LOST, the entries are retained on "
        "purpose, and their hashes cannot match anything. They are still "
        "listed below, because an attestation that hides them cannot be told "
        "apart from one that missed them.")
    say()
    for d in rdetail:
        say("- `PINNED_ARTIFACTS.md:%d` **%s** `%s...`  %s"
            % (d[0], d[1], d[2][:16], d[3]))
        say("  > %s" % d[4])
    if not rdetail:
        say("No non-MATCH pairs.")
    say()
    fails["mismatch"] += rc["MISMATCH"]
    fails["unresolved"] += rc["UNRESOLVED"]

    # --- 3. Frozen-file diff audit ----------------------------------------
    say("## 3. Frozen-file diff audit")
    say()
    say("| Frozen path | on disk | tree state | commits touching it |")
    say("|---|---|---|---:|")
    for f in FROZEN:
        ap_ = os.path.join(REPO, f)
        exists = "yes" if os.path.isfile(ap_) else "**ABSENT**"
        st, _ = git("status", "--porcelain", "--", f)
        state = "clean" if not st.strip() else "**%s**" % st.strip()[:2]
        log, _ = git("log", "--oneline", "--", f)
        n = len([x for x in log.splitlines() if x.strip()])
        say("| `%s` | %s | %s | %d |" % (f, exists, state, n))
    say()
    say("Commit list per frozen path, most recent first, so an unauthorized "
        "edit is visible as a commit nobody can name:")
    say()
    for f in FROZEN:
        log, _ = git("log", "--oneline", "-5", "--", f)
        say("- `%s`" % f)
        for line in (log.splitlines() or ["  (no commits)"]):
            say("  - %s" % line.strip())
    say()

    # --- 4. Push state ----------------------------------------------------
    say("## 4. Push state")
    say()
    sb, _ = git("status", "-sb")
    branchline = sb.splitlines()[0] if sb else "(unknown)"
    say("Branch line: `%s`" % branchline)
    unpushed, rc_ = git("log", "--oneline", "@{u}..HEAD")
    ups = [x for x in unpushed.splitlines() if x.strip()] if rc_ == 0 else []
    if rc_ != 0:
        say("No upstream tracking branch resolved. **FAIL**")
        fails["mismatch"] += 1
    elif ups:
        say()
        say("**%d unpushed commit(s):**" % len(ups))
        say()
        for u in ups:
            say("- `%s`" % u)
        say()
        say("Push state is NOT clean. The attestation records this; it does "
            "not push, because pushing is not a read-only act.")
    else:
        say("No unpushed commits. PASS")
    say()

    # --- 5. Clean tree ----------------------------------------------------
    say("## 5. Clean tree")
    say()
    porc, _ = git("status", "--porcelain")
    mod = [x for x in porc.splitlines() if x and not x.startswith("??")]
    unt = [x for x in porc.splitlines() if x.startswith("??")]
    if mod:
        say("**%d tracked path(s) modified or staged:**" % len(mod))
        say()
        for m in mod:
            say("- `%s`" % m)
    else:
        say("No tracked path modified or staged. PASS")
    say()
    say("Untracked paths: %d. Untracked files do not break the attestation, "
        "but a frozen or pinned artifact must never be among them; section 3 "
        "checks that separately." % len(unt))
    say()

    # --- verdict ----------------------------------------------------------
    say("## Verdict")
    say()
    total = fails["mismatch"] + fails["missing"] + fails["unresolved"]
    say("| Failure class | count |")
    say("|---|---:|")
    say("| MISMATCH | %d |" % fails["mismatch"])
    say("| MISSING | %d |" % fails["missing"])
    say("| UNRESOLVED | %d |" % fails["unresolved"])
    say()
    tree_ok = not mod
    push_ok = (rc_ == 0 and not ups)
    say("Hash checks: %s. Clean tree: %s. Push state: %s."
        % ("PASS" if total == 0 else "FAIL",
           "PASS" if tree_ok else "FAIL",
           "PASS" if push_ok else "FAIL"))
    say()
    verdict = "PASS" if (total == 0 and tree_ok and push_ok) else "FAIL"
    say("**ATTESTATION %s**" % verdict)

    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    out = os.path.join(OUTDIR, "attestation_%s.md" % date.replace("-", ""))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print()
    print("written: %s" % os.path.relpath(out, REPO).replace("\\", "/"))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
