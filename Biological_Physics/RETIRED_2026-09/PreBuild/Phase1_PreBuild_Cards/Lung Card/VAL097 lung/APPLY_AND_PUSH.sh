#!/bin/bash
# VAL-097 GitHub push — apply on Heath's local clone of IAM-Validation
# Two paths:
#   PATH A (cleanest): git am the .patch file
#   PATH B (manual):   copy raw_files/* into your repo, commit, push
#
# Usage: from inside your local IAM-Validation clone, run:
#   bash /path/to/APPLY_AND_PUSH.sh

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PATCH_FILE=$(ls "$SCRIPT_DIR"/*.patch 2>/dev/null | head -1)

echo "=== VAL-097 push application ==="
echo "Repo working dir: $(pwd)"
git status --short | head -5

# PATH A — git am the patch
if [ -n "$PATCH_FILE" ]; then
  echo
  echo "Applying patch: $PATCH_FILE"
  git am "$PATCH_FILE" || {
    echo "git am failed. Falling back to PATH B (manual copy)..."
    cp -r "$SCRIPT_DIR/raw_files/Biological_Physics/validation_runs/VAL-097" Biological_Physics/validation_runs/
    cp "$SCRIPT_DIR/raw_files/Biological_Physics/README.md" Biological_Physics/README.md
    git add Biological_Physics/validation_runs/VAL-097/ Biological_Physics/README.md
    git commit -m "VAL-097: never-smoker LUAD GSE256092 — O5_BASELINE_DOMINATED" \
               -m "See VAL-097/outcome.md for full details. LL-CROSS-COHORT-CALIBRATION + LL-PRELOCK-DEGENERATE-COMPARATOR lessons logged."
  }
fi

echo
echo "Pushing to origin/main..."
git push origin main
echo
echo "=== Done ==="
