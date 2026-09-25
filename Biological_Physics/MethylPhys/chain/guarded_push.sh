#!/bin/sh
# Push only if propagate.py passes. Usage: guarded_push.sh "<commit message>"
#
# Why this exists (2026-09-25). propagate.py exits non-zero on drift, and the instruction is to run it before
# every push - but it was being run as `python3 propagate.py | tail -4`, and a pipe hands the shell TAIL's
# exit status, not the gate's. Under `set -e` the script therefore sailed past a printed
# "1 FAILURE(S) - fix these before pushing" and pushed anyway; main briefly carried documents with broken
# relative references. A gate whose result can be discarded by a pipe is not a gate.
#
# This wrapper never pipes the gate. It runs it, keeps the full output in a file, and returns its real status.
set -e
[ -n "$1" ] || { echo "guarded_push.sh: a commit message is required"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
LOG="$ROOT/.propagate_last_run.txt"

echo "== propagate =="
if ! (cd "$HERE" && python3 propagate.py > "$LOG" 2>&1); then
    tail -20 "$LOG"
    echo
    echo "REFUSED: propagate.py failed. Nothing was committed and nothing was pushed."
    echo "Full output: $LOG"
    exit 1
fi
tail -3 "$LOG"

cd "$ROOT"
git add -A
if git diff --cached --quiet; then
    echo "nothing staged - no commit made"
    exit 0
fi
git -c user.name="IAMPerformance" -c user.email="iamperformance@users.noreply.github.com" commit -q -F - <<MSG
$1
MSG
git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@github.com/hmahaffeyges/IAM-Validation.git"
git push -q origin main
git remote set-url origin https://github.com/hmahaffeyges/IAM-Validation.git
echo "pushed $(git rev-parse --short HEAD)"
