#!/bin/bash
# ============================================================
# IAM Level 2b: Add background modification to dtauda
# Run this AFTER Level 2a chains are complete
# ============================================================
source ~/iam-env/bin/activate
cd ~/CAMB_IAM_L2

echo "=== Preparing Level 2b: Background modification ==="

# 1. Ensure IAM dual-sector flag is ON
sed -i 's/iam_dual_sector = .false./iam_dual_sector = .true./' fortran/equations.f90

# 2. Add IAM background term to dtauda
# Insert after: grhoa2 = this%grho_no_de(a) + grhov_t * a**2
# Insert before: if (grhoa2 <= 0) then
sed -i '/grhoa2 = this%grho_no_de(a) +  grhov_t \* a\*\*2/a\
\
    ! === IAM BACKGROUND MODIFICATION (Level 2b) ===\
    ! H² = H²_LCDM + beta_m * E(a) * H0²\
    ! E(a) = exp(1 - 1/a), beta_m = Omega_m/2 = 0.15765\
    ! In CAMB units: grhoa2 = 8*pi*G*rho*a^4, so IAM term = beta_m * E(a) * a^2 * grho_today\
    if (a > 1.0d-6) then\
        block\
            real(dl) :: iam_Ea_bg, iam_grho0_bg, iam_grho_bg\
            iam_Ea_bg = exp(1.0_dl - 1.0_dl / a)\
            iam_grho0_bg = this%grhob + this%grhoc + this%grhornomass + this%grhog + this%grhov\
            iam_grho_bg = 0.15765_dl * iam_Ea_bg * a * a * iam_grho0_bg\
            grhoa2 = grhoa2 + iam_grho_bg\
        end block\
    end if\
    ! === END IAM BACKGROUND MODIFICATION ===' fortran/equations.f90

# 3. Verify the patch applied
echo ""
echo "=== Verifying patch ==="
grep -A 20 "grhoa2 = this%grho_no_de" fortran/equations.f90 | head -25

# 4. Rebuild CAMB
echo ""
echo "=== Rebuilding CAMB ==="
pip install -e . 2>&1 | tail -5

echo ""
echo "=== Level 2b preparation complete ==="
echo "Run: bash run_level2b_chain.sh"
