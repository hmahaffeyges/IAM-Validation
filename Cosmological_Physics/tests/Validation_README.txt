═══════════════════════════════════════════════════════════════════════════════
  INFORMATIONAL ACTUALIZATION MODEL (IAM)
  Complete Validation Suite - Instructions
═══════════════════════════════════════════════════════════════════════════════

Thank you for your interest in the Informational Actualization Model!

This script reproduces all validation tests and generates publication-quality
figures demonstrating that IAM resolves the Hubble tension through dual-sector
coupling.

═══════════════════════════════════════════════════════════════════════════════
REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

1. Python 3.8 or newer
2. Three Python packages: numpy, scipy, matplotlib

═══════════════════════════════════════════════════════════════════════════════
INSTALLATION
═══════════════════════════════════════════════════════════════════════════════

OPTION 1: If you already have Python installed
───────────────────────────────────────────────

Open Terminal (Mac/Linux) or Command Prompt (Windows) and run:

  pip install numpy scipy matplotlib

Or if you have Anaconda:

  conda install numpy scipy matplotlib


OPTION 2: If you don't have Python yet (EASIEST)
─────────────────────────────────────────────────

1. Download Anaconda (free):
   https://www.anaconda.com/download

2. Install Anaconda (it includes Python + all required packages!)

3. You're done! All packages are already installed.


OPTION 3: Using Python from python.org
───────────────────────────────────────

1. Download Python 3.11 or newer from:
   https://www.python.org/downloads/

2. During installation, check "Add Python to PATH"

3. After installation, open Terminal/Command Prompt and run:
   pip install numpy scipy matplotlib

═══════════════════════════════════════════════════════════════════════════════
RUNNING THE VALIDATION
═══════════════════════════════════════════════════════════════════════════════

1. Save iam_validation.py to your Downloads folder

2. Open Terminal (Mac/Linux) or Command Prompt (Windows)

3. Navigate to Downloads:
   cd ~/Downloads              (Mac/Linux)
   cd %USERPROFILE%\Downloads  (Windows)

4. Run the script:
   python3 iam_validation.py   (Mac/Linux)
   python iam_validation.py    (Windows)

5. Wait 3-4 minutes while it runs all tests

6. Find your 8 PDF figures in the Downloads folder!

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS SCRIPT DOES
═══════════════════════════════════════════════════════════════════════════════

TEST 1: ΛCDM Baseline
  Shows standard cosmology fails (5σ Hubble tension)

TEST 2: IAM Discovery
  Demonstrates 5.6σ improvement with β-only model

TEST 3: Matter-Sector Validation
  Isolates matter-coupled observables (DESI + local H₀)

TEST 4: Photon-Sector Validation
  Shows CMB requires β_γ ≈ 0 (empirical sector separation)

TEST 5: CMB Lensing Consistency
  Proves growth suppression creates natural lensing compensation

TEST 6: Profile Likelihood
  Rigorous statistical constraints: β_m = 0.157 ± 0.029

FIGURE GENERATION:
  Creates 8 publication-quality PDF figures in your Downloads folder

═══════════════════════════════════════════════════════════════════════════════
EXPECTED RESULTS
═══════════════════════════════════════════════════════════════════════════════

β_m = 0.157 ± 0.029 (68% CL)
β_γ < 0.004 (95% CL)
β_γ/β_m < 0.022 (empirical sector separation)

H₀(photon/CMB)  = 67.4 km/s/Mpc
H₀(matter/local) = 72.5 ± 0.9 km/s/Mpc

χ²(ΛCDM) = 41.63
χ²(IAM)  = 10.38
Δχ² = 31.25 (5.6σ improvement)

Growth suppression = 1.36%
σ₈(IAM) = 0.800

═══════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

ERROR: "python: command not found"
  → Try "python3" instead of "python"
  → Or install Python from python.org or anaconda.com

ERROR: "No module named 'numpy'" (or scipy, matplotlib)
  → Run: pip install numpy scipy matplotlib
  → Or: conda install numpy scipy matplotlib

ERROR: "Permission denied"
  → Try: python3 iam_validation.py (without sudo)
  → Or move the script to a folder you own

Script runs but no figures appear:
  → Check your Downloads folder
  → Or look for message showing where files were saved

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FILES
═══════════════════════════════════════════════════════════════════════════════

After running, you'll find these 8 PDFs in your Downloads folder:

  figure1_h0_comparison.pdf           - Hubble tension resolution
  figure2_growth_suppression.pdf      - Growth factor evolution
  figure3_desi_growth.pdf             - DESI growth rate fit
  figure4_beta_gamma_constraint.pdf   - Photon-sector constraint
  figure5_beta_m_profile.pdf          - Matter-sector likelihood
  figure6_h0_ladder_complete.pdf      - Complete H₀ compilation
  figure7_chi2_breakdown.pdf          - Statistical analysis
  figure8_summary_panel.pdf           - Physical quantities summary

═══════════════════════════════════════════════════════════════════════════════
CONTACT & QUESTIONS
═══════════════════════════════════════════════════════════════════════════════

For questions or issues:
  Email: hmahaffeyges@gmail.com

For more information about IAM:
  https://github.com/hmahaffeyges/IAM-Validation  
  https://doi.org/10.17605/OSF.IO/KCZD9 (OSF Preprints)

═══════════════════════════════════════════════════════════════════════════════

Happy validating! 🚀

- Heath W. Mahaffey
  February 2026
