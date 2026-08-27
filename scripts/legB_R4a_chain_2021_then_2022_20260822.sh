#!/usr/bin/env bash
# Leg B R4a: finish the 2021 window extracts, then run the 2022 set.
#
# WHY THIS EXISTS. The original launcher (legB_R4a_sub_extracts_2021.sh) loops
# init hours 0, 6, 12, 18. Its bash parent was killed when the harness task was
# stopped at 2026-08-22 ~10:0x, but the `--init-hour 0` child survived as an
# orphan and kept writing extracts. So the run is NOT stopped, and it is also
# NOT going to finish: when the orphan completes its 8 dates at hour 0, nothing
# will start hours 6, 12 and 18. It would halt at 8 of 32 with no error.
#
# This script waits for that orphan to exit, then re-runs the full 2021 loop
# (resumable: existing extract_<date>_h<HH>.npz are skipped, each is written to
# .tmp then renamed, so an interruption costs at most one extract), and then
# runs the 2022 loop.
#
# The 2022 leg is amendment 2 of the 2026-08-22 dispatch: it launches
# "immediately and unconditionally" when the 2021 set finishes, before or
# alongside R4b. Floor fails, they cost electricity. Floor passes, R5's long
# pole is already done. It is chained here rather than launched by hand so it
# does not depend on someone being awake at the handover.
#
# Safe to re-run from cold at any point. Idempotent by construction.

set -u
cd "$(dirname "$0")/.."

# Path sanitised for publication (see SANITISATION.md). SW_PROTECTED_ARTIFACTS
# must name the protected pinned-artifact folder. ":?" makes an unset variable
# a fatal error, and the -d test makes a wrong one fatal too: a root that does
# not exist would otherwise yield an empty extract set instead of an error.
: "${SW_PROTECTED_ARTIFACTS:?set SW_PROTECTED_ARTIFACTS to the protected artifact folder}"
[ -d "$SW_PROTECTED_ARTIFACTS" ] || { echo "SW_PROTECTED_ARTIFACTS=$SW_PROTECTED_ARTIFACTS is not a directory" >&2; exit 1; }
OUT21="$SW_PROTECTED_ARTIFACTS/arm_extracts_sub_2021_20260822"
OUT22="$SW_PROTECTED_ARTIFACTS/arm_extracts_sub_2022_20260822"
D21="scripts/artifacts/legB_sub_issue_dates_2021_8.txt"
D22="scripts/artifacts/legB_sub_issue_dates_2022_8.txt"
HOURS="0 6 12 18"

count() { ls "$1"/*.npz 2>/dev/null | wc -l | tr -d ' '; }

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# --- 1. wait for any in-flight rollout to exit -------------------------------
# Matching on the script name, not on "python", so an unrelated interpreter is
# never mistaken for ours.
#
# HARDENED 2026-08-22 against a race I put in the first version. The original
# loop hands off between init hours in milliseconds, but `conda run` takes a
# second or two before its python.exe appears, so a poll watching ONLY
# python.exe could land in that gap, declare the run finished, and start a
# SECOND concurrent 2021 loop against the same output directory. Over three
# remaining hour transitions at a 60 s poll that was roughly a 15% chance, not
# a theoretical one. Two fixes: watch conda.exe as well, and require three
# consecutive clear polls before believing it.
rollout_running() {
  wmic process where "name='python.exe' or name='conda.exe'" get CommandLine 2>/dev/null     | grep -q "tier2_pangu_rollout.py"
}

echo "[$(stamp)] waiting for any in-flight tier2_pangu_rollout to exit ..."
clear_polls=0
while [ "$clear_polls" -lt 3 ]; do
  if rollout_running; then
    clear_polls=0
    sleep 60
  else
    clear_polls=$((clear_polls + 1))
    echo "[$(stamp)] no rollout seen ($clear_polls of 3 consecutive)"
    sleep 20
  fi
done
echo "[$(stamp)] no rollout running. 2021 extracts on disk: $(count "$OUT21") of 32"

# --- 2. finish 2021 ----------------------------------------------------------
for H in $HOURS; do
  echo "[$(stamp)] 2021 init hour $H (have $(count "$OUT21") of 32)"
  conda run -n swnd python scripts/tier2_pangu_rollout.py \
    --dates-file "$D21" --harvest 7 --threads 16 \
    --init-hour "$H" --name-hour --outdir "$OUT21" || {
      echo "[$(stamp)] FAILED on 2021 hour $H"; exit 1; }
done

N21=$(count "$OUT21")
echo "[$(stamp)] 2021 complete: $N21 of 32"
if [ "$N21" -ne 32 ]; then
  echo "[$(stamp)] STOP: expected 32 extracts, found $N21. Not launching 2022."
  exit 2
fi

# --- 3. 2022, unconditionally (amendment 2) ----------------------------------
echo "[$(stamp)] launching the 2022 set, unconditionally per amendment 2"
for H in $HOURS; do
  echo "[$(stamp)] 2022 init hour $H (have $(count "$OUT22") of 32)"
  conda run -n swnd python scripts/tier2_pangu_rollout.py \
    --dates-file "$D22" --harvest 7 --threads 16 \
    --init-hour "$H" --name-hour --outdir "$OUT22" || {
      echo "[$(stamp)] FAILED on 2022 hour $H"; exit 1; }
done

echo "[$(stamp)] 2022 complete: $(count "$OUT22") of 32"
echo "[$(stamp)] DONE. 2021 $(count "$OUT21")/32, 2022 $(count "$OUT22")/32"
