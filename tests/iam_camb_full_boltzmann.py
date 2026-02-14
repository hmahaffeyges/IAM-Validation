#!/usr/bin/env python3
"""
IAM-CAMB Full Boltzmann Solver: Fortran Modification Guide
===========================================================

This script:
1. Clones CAMB from source
2. Creates the IAM dark energy module (DarkEnergyIAM.f90)
3. Applies patches to equations.f90 for mu(a) modification
4. Builds CAMB from source
5. Runs the full Boltzmann comparison

PREREQUISITES (Mac with Anaconda):
    brew install gcc           # Fortran compiler (gfortran)
    pip install numpy scipy matplotlib

USAGE:
    cd ~/IAM-Validation
    python iam_camb_full_boltzmann.py

The script will handle everything. If the build fails, check that
gfortran is installed: gfortran --version

Heath Mahaffey, February 2026
"""

import os
import sys
import subprocess
import shutil
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
CAMB_DIR = os.path.expanduser("~/CAMB_IAM")
BETA_M = 0.157
H0 = 67.4

# ============================================================================
# STEP 1: Clone CAMB source
# ============================================================================
def step1_clone():
    print("\n" + "="*70)
    print("STEP 1: Cloning CAMB from source...")
    print("="*70)
    
    if os.path.exists(CAMB_DIR):
        print(f"  {CAMB_DIR} already exists.")
        resp = input("  Delete and re-clone? [y/N]: ").strip().lower()
        if resp == 'y':
            shutil.rmtree(CAMB_DIR)
        else:
            print("  Using existing clone.")
            return
    
    subprocess.run(["git", "clone", "https://github.com/cmbant/CAMB.git", CAMB_DIR], check=True)
    print(f"  Cloned to {CAMB_DIR}")

# ============================================================================
# STEP 2: Create IAM Dark Energy Module
# ============================================================================
IAM_MODULE = r'''
! DarkEnergyIAM.f90
! IAM (Informational Actualization Model) dark energy module
! Implements beta_m * E(a) as an additional energy component
! where E(a) = exp(1 - 1/a)
!
! This module handles the BACKGROUND modification only.
! The mu(a) perturbation modification is in equations.f90
!
! Heath Mahaffey, February 2026

module DarkEnergyIAM
    use precision
    use classes
    use DarkEnergyInterface
    implicit none

    type, extends(TDarkEnergyEqnOfState) :: TDarkEnergyIAMModel
        real(dl) :: beta_m = 0.157_dl  ! Matter-sector coupling
    contains
        procedure :: Init => IAM_Init
        procedure :: BackgroundDensityAndPressure => IAM_BackgroundDensityAndPressure
        procedure :: w_de => IAM_w_de
        procedure :: grho_de => IAM_grho_de
        procedure :: PrintFeedback => IAM_PrintFeedback
    end type TDarkEnergyIAMModel

    public TDarkEnergyIAMModel

contains

    subroutine IAM_Init(this, State)
        class(TDarkEnergyIAMModel), intent(inout) :: this
        class(TCAMBdata), intent(in), target :: State
        
        this%is_cosmological_constant = .false.
        this%num_perturb_equations = 0  ! No extra perturbation variables
        ! The w/wa values for halofit approximation
        this%w_lam = -1.0_dl - 1.0_dl/3.0_dl  ! w at a=1
        this%wa = 0._dl
        
    end subroutine IAM_Init
    
    function IAM_activation(a) result(E_a)
        real(dl), intent(in) :: a
        real(dl) :: E_a
        
        if (a > 1e-10_dl) then
            E_a = exp(1.0_dl - 1.0_dl/a)
        else
            E_a = 0.0_dl
        end if
    end function IAM_activation

    subroutine IAM_BackgroundDensityAndPressure(this, grhov, a, grhov_t, w)
        ! Returns the dark energy density and pressure at scale factor a
        ! grhov is 8piG*rho_de(today)*a0^2 (the base density parameter)
        ! grhov_t is the total dark energy contribution to grho at time a
        class(TDarkEnergyIAMModel), intent(inout) :: this
        real(dl), intent(in) :: grhov, a
        real(dl), intent(out) :: grhov_t
        real(dl), optional, intent(out) :: w
        real(dl) :: E_a, grho_lambda, grho_iam

        ! Standard cosmological constant part
        grho_lambda = grhov * a**2
        
        ! IAM extra component: beta_m * E(a) * H0^2 * a^2
        ! In CAMB units, grhov already contains the H0^2 factor
        ! We add beta_m * E(a) as fraction of the CC
        E_a = IAM_activation(a)
        
        ! Total DE = Lambda + beta_m * E(a)
        ! Note: grhov is 8piG*rho_Lambda*a0^2, so grhov/H0^2 = Omega_Lambda
        ! We want to add beta_m * E(a) * H0^2 * a^2
        ! = beta_m/Omega_Lambda * E(a) * grhov * a^2
        ! But simpler: rescale grhov to include both
        grho_iam = this%beta_m / 0.685_dl * grhov * E_a * a**2
        
        grhov_t = grho_lambda + grho_iam
        
        if (present(w)) then
            ! Effective equation of state
            if (abs(grhov_t) > 1e-30_dl) then
                ! p = w * rho for effective total
                ! Lambda: w=-1, p_Lambda = -rho_Lambda
                ! IAM: p_IAM = w_IAM * rho_IAM
                ! w_IAM from: d(rho*a^3)/da = -3*p*a^2
                ! rho_IAM ~ E(a)/a^3, so dlnrho/dlna = -3 + dlnE/dlna / ???
                ! Simpler: compute effective w from total
                ! For now, approximate
                w = -1.0_dl  ! Dominant term is Lambda
            end if
        end if
        
    end subroutine IAM_BackgroundDensityAndPressure

    function IAM_w_de(this, a) result(w)
        class(TDarkEnergyIAMModel), intent(inout) :: this
        real(dl), intent(in) :: a
        real(dl) :: w
        w = -1.0_dl  ! Effective w for halofit etc.
    end function IAM_w_de

    function IAM_grho_de(this, a) result(grho)
        ! Returns 8*pi*G*rho_de*a^2 as function of scale factor
        class(TDarkEnergyIAMModel), intent(inout) :: this
        real(dl), intent(in) :: a
        real(dl) :: grho
        real(dl) :: E_a
        
        E_a = IAM_activation(a)
        ! Cosmological constant + IAM term
        ! In normalized units where sum of Omega = 1
        grho = 1.0_dl + this%beta_m * E_a  ! Relative to Lambda
    end function IAM_grho_de

    subroutine IAM_PrintFeedback(this, FeedbackLevel)
        class(TDarkEnergyIAMModel), intent(inout) :: this
        integer, intent(in) :: FeedbackLevel
        
        if (FeedbackLevel > 0) then
            write(*,'(a,f8.4)') '  IAM beta_m = ', this%beta_m
            write(*,'(a)') '  IAM activation: E(a) = exp(1 - 1/a)'
            write(*,'(a)') '  Sigma = 1 (photon geodesics unmodified)'
        end if
    end subroutine IAM_PrintFeedback

end module DarkEnergyIAM
'''

def step2_create_iam_module():
    print("\n" + "="*70)
    print("STEP 2: Creating IAM dark energy module...")
    print("="*70)
    
    path = os.path.join(CAMB_DIR, "fortran", "DarkEnergyIAM.f90")
    with open(path, 'w') as f:
        f.write(IAM_MODULE)
    print(f"  Created: {path}")

# ============================================================================
# STEP 3: Patch equations.f90 for mu(a) modification
# ============================================================================
def step3_patch_equations():
    print("\n" + "="*70)
    print("STEP 3: Patching equations.f90 for mu(a)...")
    print("="*70)
    
    eqn_path = os.path.join(CAMB_DIR, "fortran", "equations.f90")
    
    with open(eqn_path, 'r') as f:
        content = f.read()
    
    # Backup
    with open(eqn_path + '.backup', 'w') as f:
        f.write(content)
    print(f"  Backed up original to {eqn_path}.backup")
    
    # ---- PATCH 1: Add IAM mu(a) function ----
    # Insert after the module declaration and before the first function
    
    mu_function = '''
    ! ============================================================
    ! IAM: Modified gravity parameter mu(a)
    ! mu(a) = H^2_LCDM(a) / [H^2_LCDM(a) + beta_m * E(a)]
    ! This suppresses the gravitational source term for matter
    ! perturbations while leaving photon geodesics unchanged (Sigma=1)
    ! ============================================================
    function iam_mu_of_a(a, beta_m) result(mu)
        real(dl), intent(in) :: a, beta_m
        real(dl) :: mu, E_a, H2_lcdm
        real(dl), parameter :: Om = 0.315_dl, OL = 0.685_dl, Or = 9.24e-5_dl
        
        if (a > 1e-10_dl) then
            E_a = exp(1.0_dl - 1.0_dl / a)
        else
            E_a = 0.0_dl
        end if
        
        H2_lcdm = Om * a**(-3) + Or * a**(-4) + OL
        mu = H2_lcdm / (H2_lcdm + beta_m * E_a)
        
    end function iam_mu_of_a
'''
    
    # Find insertion point: after "contains" in the module
    # The first "function dtauda" appears early - insert mu function right before it
    insert_marker = "    function dtauda(this,a)"
    if insert_marker in content:
        content = content.replace(insert_marker, mu_function + "\n" + insert_marker)
        print("  PATCH 1: Added iam_mu_of_a function")
    else:
        print("  WARNING: Could not find insertion point for mu function!")
        print("  You may need to manually add it.")
    
    # ---- PATCH 2: Modify dgrho_matter to include mu(a) ----
    # The key line: dgrho = dgrho_matter
    # We want: dgrho = mu(a) * dgrho_matter
    # BUT we need to be careful - dgrho_matter includes neutrinos
    # We only want to modify CDM + baryons
    
    # Actually, the cleanest approach is to modify the Poisson equation directly
    # Line: phi = -((dgrho +3*dgq*adotoa/k)/EV%Kf(1) + dgpi)/(2*k2)
    # Modify to: multiply matter part of dgrho by mu(a)
    
    # Safer approach: modify dgrho_matter before it's used
    old_dgrho = "    dgrho = dgrho_matter\n"
    new_dgrho = """    ! IAM MODIFICATION: Apply mu(a) to matter perturbations
    ! mu(a) < 1 suppresses gravitational growth (matter sector)
    ! Photon perturbations are NOT modified (Sigma = 1)
    ! To disable IAM: set iam_beta_m = 0.0
    block
        real(dl) :: iam_beta_m, iam_mu
        iam_beta_m = 0.157_dl  ! IAM matter-sector coupling
        iam_mu = iam_mu_of_a(a, iam_beta_m)
        ! Apply mu to CDM + baryon density perturbation only
        ! dgrho_matter = grhoc_t*clxc + grhob_t*clxb + neutrino terms
        ! We scale the CDM+baryon part by mu, leave neutrinos unchanged
        dgrho = iam_mu * (grhob_t*clxb + grhoc_t*clxc) + (dgrho_matter - grhob_t*clxb - grhoc_t*clxc)
    end block
"""
    
    if old_dgrho in content:
        content = content.replace(old_dgrho, new_dgrho, 1)  # Only first occurrence
        print("  PATCH 2: Modified dgrho to include mu(a)")
    else:
        print("  WARNING: Could not find 'dgrho = dgrho_matter' line!")
        print("  You may need to manually modify equations.f90")
    
    # Write patched file
    with open(eqn_path, 'w') as f:
        f.write(content)
    print(f"  Patched: {eqn_path}")

# ============================================================================
# STEP 4: Build CAMB from source
# ============================================================================
def step4_build():
    print("\n" + "="*70)
    print("STEP 4: Building CAMB from source...")
    print("="*70)
    
    # Check for gfortran
    try:
        result = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
        print(f"  Found gfortran: {result.stdout.splitlines()[0]}")
    except FileNotFoundError:
        print("  ERROR: gfortran not found!")
        print("  Install with: brew install gcc")
        print("  Then ensure gfortran is in your PATH")
        return False
    
    # Build CAMB
    os.chdir(CAMB_DIR)
    print("  Running: pip install -e .")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Build failed! Error:\n{result.stderr[-500:]}")
        print("\n  Try manually:")
        print(f"    cd {CAMB_DIR}")
        print(f"    python -m pip install -e .")
        return False
    
    print("  Build successful!")
    return True

# ============================================================================
# STEP 5: Run comparison
# ============================================================================
def step5_run_comparison():
    print("\n" + "="*70)
    print("STEP 5: Running IAM-CAMB full Boltzmann comparison...")
    print("="*70)
    
    import importlib
    import camb as camb_mod
    importlib.reload(camb_mod)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # --- LCDM baseline ---
    print("  Running LCDM baseline...")
    pars_lcdm = camb_mod.CAMBparams()
    pars_lcdm.set_cosmology(H0=67.4, ombh2=0.02242, omch2=0.11933, 
                            mnu=0.06, omk=0, tau=0.0561)
    pars_lcdm.InitPower.set_params(As=2.1e-9, ns=0.9649)
    pars_lcdm.set_for_lmax(2500, lens_potential_accuracy=1)
    pars_lcdm.Want_CMB_lensing = True
    pars_lcdm.set_matter_power(redshifts=[0.0], kmax=2.0)
    pars_lcdm.WantTransfer = True
    
    results_lcdm = camb_mod.get_results(pars_lcdm)
    powers_lcdm = results_lcdm.get_cmb_power_spectra(pars_lcdm, CMB_unit='muK')
    cl_lcdm = powers_lcdm['total']
    lens_lcdm = results_lcdm.get_lens_potential_cls(lmax=2500)
    sigma8_lcdm = results_lcdm.get_sigma8_0()
    
    # --- IAM (with mu(a) modification compiled in) ---
    print("  Running IAM (mu(a) modified)...")
    pars_iam = camb_mod.CAMBparams()
    pars_iam.set_cosmology(H0=67.4, ombh2=0.02242, omch2=0.11933,
                           mnu=0.06, omk=0, tau=0.0561)
    pars_iam.InitPower.set_params(As=2.1e-9, ns=0.9649)
    pars_iam.set_for_lmax(2500, lens_potential_accuracy=1)
    pars_iam.Want_CMB_lensing = True
    pars_iam.set_matter_power(redshifts=[0.0], kmax=2.0)
    pars_iam.WantTransfer = True
    
    results_iam = camb_mod.get_results(pars_iam)
    powers_iam = results_iam.get_cmb_power_spectra(pars_iam, CMB_unit='muK')
    cl_iam = powers_iam['total']
    lens_iam = results_iam.get_lens_potential_cls(lmax=2500)
    sigma8_iam = results_iam.get_sigma8_0()
    
    ell = np.arange(cl_lcdm.shape[0])
    
    # --- Results ---
    print(f"\n  sigma_8 (LCDM): {sigma8_lcdm:.4f}")
    print(f"  sigma_8 (IAM):  {sigma8_iam:.4f}")
    print(f"  Ratio: {sigma8_iam/sigma8_lcdm:.4f}")
    
    derived_lcdm = results_lcdm.get_derived_params()
    derived_iam = results_iam.get_derived_params()
    print(f"\n  theta_s (LCDM): {derived_lcdm['thetastar']:.6f}")
    print(f"  theta_s (IAM):  {derived_iam['thetastar']:.6f}")
    
    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # TT spectrum comparison
    ax = axes[0,0]
    ell_p = ell[2:2501]
    Dl_lcdm = ell_p*(ell_p+1)/(2*np.pi) * cl_lcdm[2:2501, 0]
    Dl_iam = ell_p*(ell_p+1)/(2*np.pi) * cl_iam[2:2501, 0]
    ax.plot(ell_p, Dl_lcdm, 'b-', lw=1.5, label=r'$\Lambda$CDM')
    ax.plot(ell_p, Dl_iam, 'r--', lw=1.5, label=r'IAM $\mu(a)<1$')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell^{TT}$ [$\mu$K$^2$]')
    ax.set_title('(a) CMB TT Power Spectrum')
    ax.legend()
    
    # TT fractional difference
    ax = axes[0,1]
    diff = (Dl_iam - Dl_lcdm) / Dl_lcdm * 100
    ax.plot(ell_p, diff, 'r-', lw=1)
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\Delta D_\ell / D_\ell$ [%]')
    ax.set_title(r'(b) TT Fractional Difference (IAM$-\Lambda$CDM)')
    ax.set_xlim(2, 2500)
    
    # Lensing
    ax = axes[1,0]
    mask = (ell >= 2) & (ell <= 2000)
    pp_lcdm = ell[mask]**2 * (ell[mask]+1)**2 * lens_lcdm[mask, 0] / (2*np.pi)
    pp_iam = ell[mask]**2 * (ell[mask]+1)**2 * lens_iam[mask, 0] / (2*np.pi)
    ax.plot(ell[mask], pp_lcdm, 'b-', lw=1.5, label=r'$\Lambda$CDM')
    ax.plot(ell[mask], pp_iam, 'r--', lw=1.5, label=r'IAM')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$[L(L+1)]^2 C_L^{\phi\phi}/2\pi$')
    ax.set_title(r'(c) CMB Lensing Power')
    ax.legend()
    
    # Matter power spectrum
    ax = axes[1,1]
    kh_lcdm, z_pk, pk_lcdm = results_lcdm.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)
    kh_iam, z_pk2, pk_iam = results_iam.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)
    ax.loglog(kh_lcdm, pk_lcdm[0], 'b-', lw=1.5, label=r'$\Lambda$CDM')
    ax.loglog(kh_iam, pk_iam[0], 'r--', lw=1.5, label=r'IAM')
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_ylabel(r'$P(k)$ [Mpc/$h$]$^3$')
    ax.set_title('(d) Matter Power Spectrum (z=0)')
    ax.legend()
    
    fig.suptitle(f'IAM Full Boltzmann: $\\mu(a)<1$, $\\Sigma=1$\n'
                 f'$\\sigma_8$: {sigma8_lcdm:.4f} (LCDM) vs {sigma8_iam:.4f} (IAM)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    outpath = os.path.join(os.path.expanduser("~/IAM-Validation"), "iam_camb_full_boltzmann.pdf")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    
    # Save numerical results
    results_path = os.path.join(os.path.expanduser("~/IAM-Validation"), "iam_camb_full_results.txt")
    with open(results_path, 'w') as f:
        f.write("IAM FULL BOLTZMANN RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"sigma_8 (LCDM): {sigma8_lcdm:.6f}\n")
        f.write(f"sigma_8 (IAM):  {sigma8_iam:.6f}\n")
        f.write(f"Suppression:    {(1-sigma8_iam/sigma8_lcdm)*100:.2f}%\n\n")
        f.write(f"theta_s (LCDM): {derived_lcdm['thetastar']:.7f}\n")
        f.write(f"theta_s (IAM):  {derived_iam['thetastar']:.7f}\n")
        f.write(f"Shift:          {(derived_iam['thetastar']-derived_lcdm['thetastar'])/derived_lcdm['thetastar']*100:.5f}%\n\n")
        f.write(f"Max TT deviation: {np.max(np.abs(diff)):.3f}%\n")
    print(f"  Saved: {results_path}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("IAM-CAMB FULL BOLTZMANN SOLVER")
    print("Applying mu(a) < 1, Sigma = 1 modification to CAMB")
    print("="*70)
    
    step1_clone()
    step2_create_iam_module()
    step3_patch_equations()
    
    print("\n" + "="*70)
    print("MANUAL BUILD INSTRUCTIONS")
    print("="*70)
    print(f"""
The patches have been applied to: {CAMB_DIR}

To build and run:

  cd {CAMB_DIR}
  pip install -e .

If that works, then run step 5 manually:

  cd ~/IAM-Validation  
  python -c "
import sys
sys.path.insert(0, '{CAMB_DIR}')
exec(open('iam_camb_full_boltzmann.py').read().split('def step5_run_comparison')[1].split('if __name__')[0])
step5_run_comparison()
"

Or simply run this script again after building -- it will skip the clone
step and go straight to the comparison.

IMPORTANT: The mu(a) modification is hardcoded with beta_m = {BETA_M}.
To change it, edit line in {CAMB_DIR}/fortran/equations.f90:
    iam_beta_m = 0.157_dl

To DISABLE IAM and recover LCDM, set:
    iam_beta_m = 0.0_dl
""")
    
    # Try auto-build
    try:
        if step4_build():
            step5_run_comparison()
    except Exception as e:
        print(f"\n  Auto-build failed: {e}")
        print("  Follow the manual instructions above.")
