#!/bin/sh
# Push only if the canon check, build_all.py and release_check_v3.py pass. Usage: guarded_push.sh "<commit message>"
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
LOG="$ROOT/.release_check_last_run.txt"

echo "== build_all (regenerate every document that reports the chain) =="
# Canon gate (2026-10-01): every LIVE file must agree with CANON/iam_canon.json (constants and names). Exit status read directly, no pipe.
CANON_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
if ! python3 "$CANON_ROOT/CANON/canon_check.py" > "$CANON_ROOT/.canon_check_last_run.txt" 2>&1; then
    echo "guarded_push.sh: CANON CHECK FAILED - a LIVE file uses a retired name or a wrong constant. See .canon_check_last_run.txt / CANON/canon_report.json"
    cat "$CANON_ROOT/.canon_check_last_run.txt" | head -20
    exit 1
fi
if ! (cd "$HERE" && "${PYTHON:-python3}" build_all.py > "$ROOT/.build_all_last_run.txt" 2>&1); then
    tail -25 "$ROOT/.build_all_last_run.txt"
    echo
    echo "REFUSED: build_all.py failed (a generator or a gate). Nothing was committed and nothing was pushed."
    exit 1
fi
tail -2 "$ROOT/.build_all_last_run.txt"

echo "== release check (chain v3 end to end, frozen-input hashes) =="
if ! (cd "$HERE" && "${PYTHON:-python3}" release_check_v3.py > "$LOG" 2>&1); then
    tail -20 "$LOG"
    echo
    echo "REFUSED: release_check_v3.py failed. Nothing was committed and nothing was pushed."
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
