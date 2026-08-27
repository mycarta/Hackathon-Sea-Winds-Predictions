#!/usr/bin/env bash
# Leg B R4a: the 32 submission-window extracts for 2021.
# 8 window issue dates x 4 init hours (00/06/12/18 UTC), 7-step, threads pinned to 16.
# Named with the init hour so the four inits never collide (--name-hour).
set -u
D="scripts/artifacts/legB_sub_issue_dates_2021_8.txt"
# Path sanitised for publication (see SANITISATION.md). SW_PROTECTED_ARTIFACTS
# must name the protected pinned-artifact folder. ":?" makes an unset variable
# a fatal error, and the -d test makes a wrong one fatal too: a root that does
# not exist would otherwise yield an empty extract set instead of an error.
: "${SW_PROTECTED_ARTIFACTS:?set SW_PROTECTED_ARTIFACTS to the protected artifact folder}"
[ -d "$SW_PROTECTED_ARTIFACTS" ] || { echo "SW_PROTECTED_ARTIFACTS=$SW_PROTECTED_ARTIFACTS is not a directory" >&2; exit 1; }
OUT="$SW_PROTECTED_ARTIFACTS/arm_extracts_sub_2021_20260822"
for H in 0 6 12 18; do
  echo "=== init hour ${H} UTC ==="
  conda run -n swnd python scripts/tier2_pangu_rollout.py \
    --dates-file "$D" --harvest 7 --threads 16 \
    --init-hour "$H" --name-hour --outdir "$OUT" || exit 1
done
echo "ALL_INIT_HOURS_DONE"
ls "$OUT"/*.npz | wc -l
