"""
IAM μ(a) and Σ(a) Table Generator for MGCAMB
==============================================
Computes the exact IAM predictions for μ(a) at arbitrary redshift nodes
for use with MGCAMB's cubic-spline reconstruction mode (MG_flag = 3).

Usage:
    python iam_mu_sigma.py [--nnode 11] [--zmax 5.0] [--beta 0.1575]

Output:
    - Prints MGCAMB parameter lines (copy into params_MG.ini)
    - Saves iam_mu_sigma_table.dat (for plotting/verification)
"""

import numpy as np
import argparse

# =============================================================================
# COSMOLOGICAL PARAMETERS (Planck 2018)
# =============================================================================
Om = 0.315
Or = 9.24e-5
OL = 0.685
BETA_M = Om / 2.0  # = 0.1575 (virial theorem prediction)

def activation(a):
    """IAM activation function E(a) = exp(1 - 1/a)"""
    return np.exp(1.0 - 1.0 / a)

def H2_LCDM(a):
    """H²/H₀² for standard ΛCDM"""
    return Om * a**(-3) + Or * a**(-4) + OL

def mu_IAM(a, beta_m=BETA_M):
    """
    IAM gravitational coupling parameter.
    
    μ(a) = H²_ΛCDM / (H²_ΛCDM + β_m × E(a))
    
    This gives μ < 1 at late times (weaker effective gravity for clustering)
    and μ → 1 at early times (E(a) → 0 exponentially).
    """
    H2L = H2_LCDM(a)
    Ea = activation(a)
    return H2L / (H2L + beta_m * Ea)

def sigma_IAM(a):
    """
    IAM lensing parameter.
    
    Σ(a) = 1 exactly, by construction.
    
    Photon geodesics are unmodified because IAM's informational
    pressure acts only on the matter sector (Hubble friction),
    not on the Weyl potential that governs lensing.
    """
    return 1.0

def generate_table(nnode=11, zmax=5.0, beta_m=BETA_M):
    """Generate μ and Σ at equally-spaced redshift nodes."""
    z_nodes = np.linspace(0, zmax, nnode)
    a_nodes = 1.0 / (1.0 + z_nodes)
    
    mu_nodes = np.array([mu_IAM(a, beta_m) for a in a_nodes])
    sigma_nodes = np.array([sigma_IAM(a) for a in a_nodes])
    
    return z_nodes, a_nodes, mu_nodes, sigma_nodes

def print_mgcamb_params(z_nodes, mu_nodes, sigma_nodes):
    """Print lines ready to paste into params_MG.ini"""
    nnode = len(z_nodes)
    
    print(f"\n# IAM cubic-spline nodes for MGCAMB (MG_flag = 3)")
    print(f"# beta_m = {BETA_M}, Om = {Om}, OL = {OL}")
    print(f"# Redshift nodes: z = {', '.join(f'{z:.1f}' for z in z_nodes)}")
    print(f"Nnode = {nnode}")
    print()
    
    for i, mu in enumerate(mu_nodes):
        print(f"MGCAMB_Mu_idx({i+1:2d}) = {mu:.4f}")
    print()
    
    for i, sig in enumerate(sigma_nodes):
        print(f"MGCAMB_Sigma_idx({i+1:2d}) = {sig:.4f}")

def save_table(z_nodes, a_nodes, mu_nodes, sigma_nodes, filename="iam_mu_sigma_table.dat"):
    """Save full table for plotting."""
    header = (
        "# IAM mu(a) and Sigma(a) for MGCAMB\n"
        f"# beta_m = {BETA_M}, Om = {Om}, OL = {OL}\n"
        "# Columns: z, a, mu(a), Sigma(a), E(a)\n"
    )
    
    Ea = np.array([activation(a) for a in a_nodes])
    data = np.column_stack([z_nodes, a_nodes, mu_nodes, sigma_nodes, Ea])
    
    np.savetxt(filename, data, header=header, 
               fmt='%8.4f %8.6f %8.6f %8.6f %12.8f',
               comments='')
    print(f"\nTable saved to {filename}")

def print_verification():
    """Print key verification values."""
    print("\n" + "=" * 60)
    print("IAM KEY VALUES (VERIFICATION)")
    print("=" * 60)
    
    test_z = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0, 1090.0]
    
    print(f"{'z':>8s} {'a':>8s} {'mu(a)':>10s} {'E(a)':>12s} {'1-mu':>10s}")
    print("-" * 52)
    
    for z in test_z:
        a = 1.0 / (1.0 + z)
        mu = mu_IAM(a)
        Ea = activation(a)
        print(f"{z:8.1f} {a:8.6f} {mu:10.6f} {Ea:12.8f} {1-mu:10.6f}")
    
    print()
    print(f"mu(z=0) = {mu_IAM(1.0):.6f}  (should be 0.8644)")
    print(f"mu(z=1090) = {mu_IAM(1.0/1091.0):.10f}  (should be ~1.000)")
    print(f"Sigma(all z) = 1.0000  (exact by construction)")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IAM mu/Sigma tables for MGCAMB")
    parser.add_argument("--nnode", type=int, default=11, help="Number of spline nodes")
    parser.add_argument("--zmax", type=float, default=5.0, help="Maximum redshift")
    parser.add_argument("--beta", type=float, default=BETA_M, help="beta_m value")
    args = parser.parse_args()
    
    BETA_M = args.beta
    
    print("=" * 60)
    print("IAM mu(a), Sigma(a) TABLE GENERATOR FOR MGCAMB")
    print("=" * 60)
    print(f"Nodes: {args.nnode}, z_max: {args.zmax}, beta_m: {BETA_M}")
    
    z, a, mu, sigma = generate_table(args.nnode, args.zmax, BETA_M)
    
    print_mgcamb_params(z, mu, sigma)
    save_table(z, a, mu, sigma)
    print_verification()
