#!/bin/bash
# ============================================================
# IAM Level 2b + 2c + 3: Chained runs
# ============================================================
source ~/iam-env/bin/activate
cd ~/CAMB_IAM_L2

# --- LEVEL 2b: Modified background, Planck only ---
echo "=== L2b Run A (IAM background + dual-sector, Planck) starting at $(date) ==="
cobaya-run iam_level2_runA.yaml -p ~/IAM-Validation/packages -f -o chains/iam_l2b_runA
echo "=== L2b Run A finished at $(date) ==="

echo "=== L2b Run D (IAM background + dual-sector, Planck+RSD) starting at $(date) ==="
cobaya-run iam_level2_runD.yaml -p ~/IAM-Validation/packages -f -o chains/iam_l2b_runD
echo "=== L2b Run D finished at $(date) ==="

# Flip to LCDM for baseline
sed -i 's/iam_dual_sector = .true./iam_dual_sector = .false./' fortran/equations.f90
# Also need to remove the dtauda modification for true LCDM baseline
# We'll use a flag approach - comment out the background mod
sed -i 's/if (a > 1.0d-6) then/if (.false. .and. a > 1.0d-6) then/' fortran/equations.f90
pip install -e . 2>&1 | tail -3

echo "=== L2b Run C (LCDM baseline) starting at $(date) ==="
cobaya-run iam_level2_runC_lcdm.yaml -p ~/IAM-Validation/packages -f -o chains/iam_l2b_runC_lcdm
echo "=== L2b Run C finished at $(date) ==="

# Restore everything
sed -i 's/iam_dual_sector = .false./iam_dual_sector = .true./' fortran/equations.f90
sed -i 's/if (.false. .and. a > 1.0d-6) then/if (a > 1.0d-6) then/' fortran/equations.f90
pip install -e . 2>&1 | tail -3

echo "=== ALL LEVEL 2b RUNS COMPLETE at $(date) ==="
