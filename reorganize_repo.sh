#!/bin/bash

# IAM-Validation GitHub Repository Setup Script
# This script reorganizes your repository into the clean structure
# Run this from your IAM-Validation root directory

set -e  # Exit on error

echo "======================================================================"
echo "IAM-Validation Repository Reorganization Script"
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Create new directory structure"
echo "  2. Move test files to development/archive/"
echo "  3. Move PDFs to docs/"
echo "  4. Update README files"
echo "  5. Stage changes for git commit"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# ======================================================================
# STEP 1: Create New Directory Structure
# ======================================================================
echo ""
echo "[1/6] Creating new directory structure..."

mkdir -p development/archive/tests_01-03
mkdir -p development/archive/tests_04-10
mkdir -p development/archive/tests_11-20
mkdir -p development/archive/tests_21-26
mkdir -p development/archive/tests_27-29
mkdir -p development/archive/test_30
mkdir -p development/deprecated
mkdir -p development/results
mkdir -p docs
mkdir -p figures
mkdir -p data

echo "  ✓ Directories created"

# ======================================================================
# STEP 2: Move Test Files to Archive
# ======================================================================
echo ""
echo "[2/6] Moving test files to development/archive/..."

# Tests 1-3
if [ -f "tests/test_01_h0_prediction.py" ]; then
    mv tests/test_01*.py development/archive/tests_01-03/ 2>/dev/null || true
fi
if [ -f "tests/test_02_growth_solver.py" ]; then
    mv tests/test_02*.py development/archive/tests_01-03/ 2>/dev/null || true
fi
if [ -f "tests/test_03_final.py" ]; then
    mv tests/test_03*.py development/archive/tests_01-03/ 2>/dev/null || true
fi

# Tests 4-10
for i in {04..10}; do
    if [ -f "tests/test_${i}*.py" ]; then
        mv tests/test_${i}*.py development/archive/tests_04-10/ 2>/dev/null || true
    fi
done

# Tests 11-20
for i in {11..20}; do
    if [ -f "tests/test_${i}*.py" ]; then
        mv tests/test_${i}*.py development/archive/tests_11-20/ 2>/dev/null || true
    fi
done

# Tests 21-26
for i in {21..26}; do
    if [ -f "tests/test_${i}*.py" ]; then
        mv tests/test_${i}*.py development/archive/tests_21-26/ 2>/dev/null || true
    fi
done

# Tests 27-29 (the breakthrough!)
if [ -f "tests/test_27_cmb_lensing_FIXED.py" ]; then
    mv tests/test_27*.py development/archive/tests_27-29/ 2>/dev/null || true
fi
if [ -f "tests/test_28_dual_sector.py" ]; then
    mv tests/test_28*.py development/archive/tests_27-29/ 2>/dev/null || true
fi
if [ -f "tests/test_29_beta_gamma_constraint.py" ]; then
    mv tests/test_29*.py development/archive/tests_27-29/ 2>/dev/null || true
fi

# Test 30
if [ -f "tests/test_30_final_beta_only.py" ]; then
    mv tests/test_30*.py development/archive/test_30/ 2>/dev/null || true
fi

# Move any result files from tests/
if [ -d "tests/results" ]; then
    mv tests/results/* development/results/ 2>/dev/null || true
fi

echo "  ✓ Test files moved to archive"

# ======================================================================
# STEP 3: Move PDFs to docs/
# ======================================================================
echo ""
echo "[3/6] Moving PDF files to docs/..."

# Look for PDFs in common locations
find . -maxdepth 2 -name "*.pdf" -type f ! -path "./docs/*" ! -path "./development/*" -exec mv {} docs/ \; 2>/dev/null || true

# If you have specific PDF names, move them explicitly:
# mv IAM_Manuscript.pdf docs/ 2>/dev/null || true
# mv IAM_Test_Validation_Compendium.pdf docs/ 2>/dev/null || true
# mv Supplementary_Methods_Reproducibility_Guide.pdf docs/ 2>/dev/null || true

echo "  ✓ PDFs moved to docs/"

# ======================================================================
# STEP 4: Move Figure PDFs (if separate from general PDFs)
# ======================================================================
echo ""
echo "[4/6] Moving figure PDFs to figures/..."

# Move figure PDFs if they exist
mv figure*.pdf figures/ 2>/dev/null || true
mv results/figure*.pdf figures/ 2>/dev/null || true

echo "  ✓ Figure PDFs moved to figures/"

# ======================================================================
# STEP 5: Copy README files
# ======================================================================
echo ""
echo "[5/6] Setting up README files..."

# Note: You'll need to manually copy the content of README.md and 
# README_development.md from the outputs you received

echo ""
echo "  MANUAL STEP REQUIRED:"
echo "  ---------------------"
echo "  1. Copy README.md content to repository root"
echo "  2. Copy README_development.md to development/"
echo ""
echo "  Use these commands:"
echo "    cp /path/to/downloaded/README.md ."
echo "    cp /path/to/downloaded/README_development.md development/"
echo ""
read -p "Press Enter once you've copied the README files..."

# ======================================================================
# STEP 6: Git Operations
# ======================================================================
echo ""
echo "[6/6] Git operations..."

# Check git status
echo ""
echo "Current git status:"
git status

echo ""
echo "Do you want to:"
echo "  1. Stage all changes (git add .)"
echo "  2. Review changes manually"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "Staging all changes..."
    git add .
    
    echo ""
    echo "Changes staged. Ready to commit."
    echo ""
    echo "Suggested commit message:"
    echo "  git commit -m \"Reorganize repository: clean main structure + development archive\""
    echo ""
    echo "  git commit -m \"Major reorganization:"
    echo "    - Move tests to development/archive/"
    echo "    - Update README with clean presentation"
    echo "    - Add development history documentation"
    echo "    - Consolidate validation into single script\""
    echo ""
    
    read -p "Do you want to commit now? (y/n): " commit_choice
    
    if [ "$commit_choice" = "y" ]; then
        git commit -m "Reorganize repository: clean main structure + development archive

- Move all test files (1-30) to development/archive/
- Update main README with focused presentation
- Add development/README_development.md for transparency
- Move PDFs to docs/
- Move figures to figures/
- Prepare for final validation script consolidation"
        
        echo ""
        echo "✓ Changes committed!"
        echo ""
        echo "Next steps:"
        echo "  1. Review commit: git log -1"
        echo "  2. Push to GitHub: git push origin main"
    fi
else
    echo ""
    echo "Skipping automatic staging. Review changes manually with:"
    echo "  git status"
    echo "  git diff"
    echo ""
fi

# ======================================================================
# FINAL SUMMARY
# ======================================================================
echo ""
echo "======================================================================"
echo "Repository Reorganization Complete!"
echo "======================================================================"
echo ""
echo "New Structure:"
echo "  /"
echo "  ├── README.md                 (clean, focused)"
echo "  ├── iam_validation.py         (TODO: add consolidated script)"
echo "  ├── data/                     (observational data)"
echo "  ├── figures/                  (8 publication PDFs)"
echo "  ├── docs/                     (3 main documents)"
echo "  └── development/"
echo "      ├── README_development.md (transparency)"
echo "      └── archive/              (tests 1-30)"
echo ""
echo "Next Steps:"
echo "  1. Add iam_validation.py (consolidated final script)"
echo "  2. Verify all PDFs are in docs/"
echo "  3. Verify all figures are in figures/"
echo "  4. Test: python iam_validation.py"
echo "  5. Git push to GitHub"
echo ""
echo "======================================================================"
