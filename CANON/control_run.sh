#!/bin/bash
# Run a control that commits and resets (LESSONS C6, C11): refuse if anything is uncommitted, so a reset cannot wipe unsaved work.
# Usage: bash CANON/control_run.sh "<shell to run>"
set -uo pipefail; cd "$(git rev-parse --show-toplevel)"
git checkout -q -- CANON/canon_report.json 2>/dev/null || true
if [ -n "$(git status --porcelain)" ]; then echo "control_run: REFUSED - commit your work first (a control resets the tree):"; git status --short | head; exit 1; fi
H0=$(git rev-parse HEAD); eval "$1"; rc=$?
git reset -q --hard "$H0"; git checkout -q -- . 2>/dev/null || true
[ "$(git rev-parse HEAD)" = "$H0" ] && [ -z "$(git status --porcelain)" ] && echo "control_run: tree restored to $H0" || { echo "control_run: TREE NOT RESTORED"; exit 1; }
exit $rc
