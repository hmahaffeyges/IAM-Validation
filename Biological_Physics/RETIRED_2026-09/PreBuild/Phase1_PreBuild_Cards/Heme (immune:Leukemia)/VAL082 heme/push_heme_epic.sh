#!/bin/bash
# heme-epic v0.1 + cervical-epic v0.1 GitHub push helper
# Run from the IAM-Validation repo root on Heath's machine.
#
# This applies the combined commit (39 files):
#   cervical-epic v0.1: VAL-071/072/073/074/076/077/081 (preregs, seals,
#                       outcomes, results, scripts, manifests)
#   heme-epic v0.1:     VAL-082 (prereg, seal, outcome, results, script)
#   Biological_Physics/README.md update (cervical + heme rows + summaries)
#
# Per memory #14: cookbook README, card JSON, master LESSONS_LEARNED,
# TESTING_CHECKLIST stay Heath-only and are NOT in this commit.

set -e

PATCH_FILE="${1:-heme_epic_v0.1_GITHUB_PUSH.patch}"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: patch file not found: $PATCH_FILE"
    echo "Usage: $0 [path/to/heme_epic_v0.1_GITHUB_PUSH.patch]"
    exit 1
fi

if [ ! -d ".git" ]; then
    echo "ERROR: not in a git repository. Run from IAM-Validation repo root."
    exit 1
fi

echo "Applying cervical-epic + heme-epic v0.1 patch..."
git am "$PATCH_FILE"

echo ""
echo "Patch applied. Verifying..."
git log --oneline -1
echo ""
git show --stat HEAD | tail -8

echo ""
echo "Ready to push. Run:"
echo "  git push origin main"
