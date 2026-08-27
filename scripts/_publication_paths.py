"""Placeholder resolution for the published Sea Winds Phase 2 tree.

This tree is a path-sanitised copy of a private working repository. Absolute
filesystem paths were replaced with placeholders, because an absolute path is
evidence only to someone who can mount the drive, while the SHA-256 beside it
is evidence to anyone. See SANITISATION.md.

Where a placeholder sat in *executable* code rather than in prose, the code now
reads it from an environment variable through `ppath()` below. There is no
silent fallback: an unset variable raises, and by default a resolved path that
does not exist raises too. A script that cannot find its inputs must say so,
not quietly find nothing.

    SW_USER_HOME             the machine's user home directory
    SW_NETWORK_SHARE         root of the network share holding the smoke-test tree
    SW_PROTECTED_ARTIFACTS   the protected pinned-artifact folder
    SW_LOCAL_DRIVE           a local drive root
    SW_CLAUDE_SESSION_PATH   directory holding Claude session .jsonl transcripts

Example:

    SW_PROTECTED_ARTIFACTS=/data/pinned python scripts/legB_R1_finalise_20260821.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

PLACEHOLDER_ENV = {
    "<USER_HOME>": "SW_USER_HOME",
    "<NETWORK_SHARE>": "SW_NETWORK_SHARE",
    "<PROTECTED_ARTIFACTS>": "SW_PROTECTED_ARTIFACTS",
    "<LOCAL_DRIVE>": "SW_LOCAL_DRIVE",
    "<CLAUDE_SESSION_PATH>": "SW_CLAUDE_SESSION_PATH",
}

_TOKEN = re.compile(r"<[A-Z_]+>")


class UnresolvedPlaceholder(RuntimeError):
    """A placeholder had no environment variable set for it."""


def ppath(template: str, must_exist: bool = True) -> Path:
    """Expand <PLACEHOLDER> tokens from the environment and return a Path.

    Raises UnresolvedPlaceholder if a token has no environment variable set, or
    if the variable's own value is not an existing directory -- that second
    check is the one that stops a mistyped root from silently resolving to
    nothing and producing an empty result set instead of an error.

    `must_exist=False` is used only where the original code legitimately
    tolerated absence: a fallback candidate it probes with .exists(), or an
    output path it is about to create.
    """
    out = template
    for token in _TOKEN.findall(template):
        env = PLACEHOLDER_ENV.get(token)
        if env is None:
            raise UnresolvedPlaceholder(
                f"{token} in {template!r} is not a known placeholder; "
                f"known: {sorted(PLACEHOLDER_ENV)}"
            )
        value = os.environ.get(env)
        if not value:
            raise UnresolvedPlaceholder(
                f"{token} is unset. This tree is path-sanitised (see "
                f"SANITISATION.md); set {env} to the real location before "
                f"running this script. Wanted: {template!r}"
            )
        root = Path(value)
        if not root.is_dir():
            raise UnresolvedPlaceholder(
                f"{env}={value!r} is not an existing directory. Refusing to "
                f"continue: a root that does not exist yields an empty result "
                f"rather than an error, which is the failure this check exists "
                f"to prevent."
            )
        out = out.replace(token, str(root))

    resolved = Path(out)
    if must_exist and not resolved.exists():
        raise UnresolvedPlaceholder(
            f"resolved path does not exist: {resolved}  (from {template!r})"
        )
    return resolved
