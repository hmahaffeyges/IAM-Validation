#!/bin/bash
# cervical-epic v0.1 GitHub push helper
# Run from the IAM-Validation repo root on Heath's machine.
#
# This applies the cervical-epic v0.1 commit (34 files, 2018 insertions)
# that was prepared in the Walther session on 2026-04-25.
#
# Files included:
#   - VAL-071 landscape survey
#   - VAL-072/073/074/076/077/081: prereg.md, PREREG_SEAL.txt, outcome.md, results.json
#   - val072_cervical_epic_tcga_cesc.py
#   - val_073/074/076/077/081_cervical_epic.py
#   - CESC_matched_manifest.json, GSE99511_manifest.json
#   - Biological_Physics/README.md (cervical-epic VAL rows + build summary)
#
# Per memory #14: cookbook README, card JSON, master LESSONS_LEARNED, TESTING_CHECKLIST
# stay Heath-only and are NOT in this commit.

set -e

PATCH_FILE="${1:-cervical_epic_v0.1_GITHUB_PUSH.patch}"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: patch file not found: $PATCH_FILE"
    echo "Usage: $0 [path/to/cervical_epic_v0.1_GITHUB_PUSH.patch]"
    exit 1
fi

if [ ! -d ".git" ]; then
    echo "ERROR: not in a git repository. Run from IAM-Validation repo root."
    exit 1
fi

echo "Applying cervical-epic v0.1 patch..."
git am "$PATCH_FILE"

echo ""
echo "Patch applied. Verifying..."
git log --oneline -1
echo ""
git show --stat HEAD | tail -8

echo ""
echo "Ready to push. Run:"
echo "  git push origin main"
