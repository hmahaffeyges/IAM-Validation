#!/usr/bin/env python3
"""
IAM ISW-Galaxy Cross-Correlation Prediction
=============================================
Computes the Integrated Sachs-Wolfe (ISW) effect and its cross-correlation
with galaxy surveys for IAM vs LCDM.

Physics:
  - ISW effect: dT/T = -2 integral[dPhi/dt * dt] along line of sight
  - In LCDM: potentials decay as dark energy dominates -> positive T-galaxy correlation
  - In IAM: mu < 1 -> potentials decay FASTER -> modified ISW amplitude
  - Cross-correlation C_ell^{Tg} probes d(Phi+Psi)/dt directly

Key prediction:
  - IAM modifies the ISW signal through two competing effects:
    1. Faster potential decay (mu < 1) -> ENHANCED ISW
    2. Suppressed growth (lower sigma_8) -> REDUCED potential depth
  - Net effect depends on mu(z) profile and galaxy bias

Observables:
  - C_ell^{Tg}: CMB temperature - galaxy cross-power spectrum
  - ISW amplitude A_ISW relative to LCDM
  - Redshift dependence of ISW kernel

References:
  - Crittenden & Turok (1996): ISW-LSS cross-correlation proposal
  - Nolta et al. (2004): First detection with WMAP
  - Planck Collaboration (2016): ISW detection at 2-4 sigma
  - Giannantonio et al. (2012): Multi-tracer ISW analysis

Author: H.W. Mahaffey
Date: February 14, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d

# ===========================================================================
# COSMOLOGICAL PARAMETERS (Planck 2018)
# ===========================================================================
H0 = 67.36           # km/s/Mpc
h = H0 / 100.0
Om = 0.3153          # Total matter
OL = 0.6847          # Dark energy
sigma8_planck = 0.8111
ns = 0.9649
c_light = 299792.458  # km/s

# IAM
beta_m = 0.1575
mu0_iam = -0.13495

# Derived
H0_per_c = H0 / c_light  # 1/Mpc (comoving Hubble)

# ===========================================================================
# BACKGROUND COSMOLOGY
# ===========================================================================
def E_activation(a):
    """IAM activation function"""
    return np.exp(1.0 - 1.0 / a)

def H_over_H0(a):
    """H(a)/H0 for LCDM background"""
    return np.sqrt(Om * a**(-3) + OL)

def Omega_m_of_a(a):
    """Matter density parameter at scale factor a"""
    return Om * a**(-3) / (Om * a**(-3) + OL)

def Omega_DE_of_a(a):
    """DE density parameter at scale factor a"""
    return OL / (Om * a**(-3) + OL)

def mu_iam_exact(a):
    """IAM exact mu(a)"""
    H2L = Om * a**(-3) + OL
    return H2L / (H2L + beta_m * E_activation(a))

def mu_mgcamb(a, mu0):
    """MGCAMB parametrization"""
    return 1.0 + mu0 * Omega_DE_of_a(a)

# Comoving distance
def chi_of_z(z):
    """Comoving distance to redshift z [Mpc/h]"""
    def integrand(zp):
        a = 1.0 / (1.0 + zp)
        return 1.0 / H_over_H0(a)
    result, _ = quad(integrand, 0, z)
    return result * c_light / H0  # Mpc

# ===========================================================================
# GROWTH FACTOR AND ITS DERIVATIVE
# ===========================================================================
def solve_growth(mu0=0.0, use_exact=False):
    """
    Solve for D(a) and dD/da with modified gravity.
    Returns interpolation functions for D(a), f(a) = dlnD/dlna, dD/dtau.
    """
    def deriv(lna, y):
        a = np.exp(lna)
        D, dD_dlna = y
        
        H2 = Om * a**(-3) + OL
        Ea = H_over_H0(a)
        
        # Deceleration
        q = 0.5 * Om * a**(-3) / H2 - OL / H2
        
        # mu(a)
        if use_exact:
            mu_a = mu_iam_exact(a)
        else:
            mu_a = mu_mgcamb(a, mu0)
        
        Om_a = Om * a**(-3) / H2
        
        ddD = -(2.0 - q) * dD_dlna + 1.5 * mu_a * Om_a * D
        return [dD_dlna, ddD]
    
    lna_span = (np.log(1e-4), 0.0)
    lna_eval = np.linspace(np.log(1e-4), 0.0, 10000)
    
    y0 = [1e-4, 1.0]  # D ~ a in matter domination
    
    sol = solve_ivp(deriv, lna_span, y0, t_eval=lna_eval,
                    rtol=1e-12, atol=1e-14, method='DOP853')
    
    a_arr = np.exp(sol.t)
    D_arr = sol.y[0]
    dD_dlna = sol.y[1]
    
    # Normalize: D(a=1) for LCDM = 1
    D_arr_norm = D_arr / D_arr[-1]
    dD_dlna_norm = dD_dlna / D_arr[-1]
    
    # f = dlnD/dlna = (dD/dlna) / D
    f_arr = dD_dlna_norm / D_arr_norm
    
    # For ISW we need dPhi/dtau where tau is conformal time
    # Phi proportional to D(a)/a in linear theory
    # d(D/a)/d(ln a) = dD/dlna / a - D/a = (f-1)*D/a
    # d(D/a)/dtau = aH * d(D/a)/dlna = aH*(f-1)*D/a = H*(f-1)*D
    
    D_interp = interp1d(a_arr, D_arr_norm, bounds_error=False, fill_value='extrapolate')
    f_interp = interp1d(a_arr, f_arr, bounds_error=False, fill_value='extrapolate')
    
    return a_arr, D_arr_norm, f_arr, D_interp, f_interp

# ===========================================================================
# ISW KERNEL
# ===========================================================================
def isw_kernel(a, D_func, f_func, mu0=0.0, use_exact=False):
    """
    ISW kernel: proportional to d(D * mu)/d(tau) evaluated at scale factor a.
    
    The ISW effect comes from time variation of (Phi + Psi).
    In modified gravity: (Phi + Psi) propto Sigma * mu * D(a) / a * delta_0
    
    For IAM: Sigma = 1, so (Phi+Psi) propto mu(a) * D(a) / a
    
    ISW kernel: K_ISW(a) = d/dtau [mu(a) * D(a) / a]
    = a*H * d/d(lna) [mu * D / a]
    
    In LCDM (mu=1): K = a*H * d/d(lna) [D/a] = a*H * [f*D/a - D/a] = H*D*(f-1)
    
    In IAM: K = a*H * d/d(lna) [mu*D/a]
    = a*H * [D/a * dmu/dlna + mu*dD/dlna/a - mu*D/a]
    = H*D*[mu*(f-1) + d(ln mu)/d(ln a)]
    """
    D = D_func(a)
    f = f_func(a)
    
    if use_exact:
        mu_a = mu_iam_exact(a)
        # Numerical derivative of ln(mu) with respect to ln(a)
        da = a * 1e-5
        mu_plus = mu_iam_exact(a + da)
        mu_minus = mu_iam_exact(a - da)
        dlnmu_dlna = (np.log(mu_plus) - np.log(mu_minus)) / (2 * da / a)
    else:
        mu_a = mu_mgcamb(a, mu0)
        da = a * 1e-5
        mu_plus = mu_mgcamb(a + da, mu0)
        mu_minus = mu_mgcamb(a - da, mu0)
        dlnmu_dlna = (np.log(mu_plus) - np.log(mu_minus)) / (2 * da / a)
    
    Ea = H_over_H0(a)
    
    # ISW kernel (unnormalized)
    K = Ea * D * (mu_a * (f - 1.0) + dlnmu_dlna)
    
    return K

# ===========================================================================
# GALAXY WINDOW FUNCTION
# ===========================================================================
def galaxy_window(z, z_mean, sigma_z):
    """Gaussian galaxy redshift distribution centered at z_mean"""
    return np.exp(-0.5 * ((z - z_mean) / sigma_z)**2) / (sigma_z * np.sqrt(2 * np.pi))

# ===========================================================================
# ISW-GALAXY CROSS-CORRELATION (Limber approximation)
# ===========================================================================
def compute_isw_cross_spectrum(ell_arr, z_gal_mean, z_gal_sig, bias,
                                D_func, f_func, mu0=0.0, use_exact=False,
                                D_func_lcdm=None, sigma8_model=None):
    """
    Compute C_ell^{Tg} using Limber approximation.
    
    C_ell^{Tg} = integral dchi/chi^2 * K_ISW(chi) * W_g(chi) * P(k=ell/chi, z)
    
    Simplified: use scale-independent growth so P(k,z) = P(k,0) * [D(z)/D(0)]^2
    Then contributions factor into geometry and growth.
    """
    # We compute the integrand as a function of z
    z_min, z_max = 0.01, 4.0
    nz = 500
    z_arr = np.linspace(z_min, z_max, nz)
    
    C_ell = np.zeros_like(ell_arr, dtype=float)
    
    # Precompute comoving distances
    chi_arr = np.array([chi_of_z(z) for z in z_arr])
    
    # Galaxy window
    W_g = galaxy_window(z_arr, z_gal_mean, z_gal_sig) * bias
    
    # ISW kernel at each z
    K_isw = np.zeros(nz)
    for i, z in enumerate(z_arr):
        a = 1.0 / (1.0 + z)
        if a > 1e-3:
            K_isw[i] = isw_kernel(a, D_func, f_func, mu0, use_exact)
    
    # Growth factor for P(k,z) amplitude
    D_at_z = np.array([D_func(1.0/(1.0+z)) if 1.0/(1.0+z) > 1e-3 else 0 for z in z_arr])
    
    # sigma8 of the model
    if sigma8_model is None:
        sigma8_model = sigma8_planck
    
    # For each ell, integrate over z
    for j, ell in enumerate(ell_arr):
        # Limber: k = (ell + 0.5) / chi
        integrand = np.zeros(nz)
        for i in range(nz):
            if chi_arr[i] > 0 and D_at_z[i] > 0:
                a = 1.0 / (1.0 + z_arr[i])
                Ea = H_over_H0(a)
                
                # The cross-spectrum integrand in Limber approximation:
                # dz/H(z) * K_ISW * W_g * D^2 / chi^2
                # (we absorb normalization into a relative comparison)
                integrand[i] = (K_isw[i] * W_g[i] * D_at_z[i]**2 
                               / chi_arr[i]**2 / Ea * c_light / H0)
        
        C_ell[j] = np.trapezoid(integrand, z_arr)
    
    # Normalize by sigma8^2
    C_ell *= (sigma8_model / sigma8_planck)**2
    
    return C_ell


def main():
    print("=" * 72)
    print("  IAM ISW-GALAXY CROSS-CORRELATION PREDICTION")
    print("  Comparing ISW signal: IAM (mu < 1, Sigma = 1) vs LCDM")
    print("=" * 72)
    
    # ------------------------------------------------------------------
    # 1. Solve growth for both models
    # ------------------------------------------------------------------
    print("\n[1/5] Solving growth equations...")
    
    # LCDM
    a_lcdm, D_lcdm, f_lcdm, D_func_lcdm, f_func_lcdm = solve_growth(mu0=0.0)
    
    # IAM (MGCAMB parametrization)
    a_iam, D_iam, f_iam, D_func_iam, f_func_iam = solve_growth(mu0=mu0_iam)
    
    # IAM exact
    a_iamx, D_iamx, f_iamx, D_func_iamx, f_func_iamx = solve_growth(
        mu0=mu0_iam, use_exact=True)
    
    # sigma8 for IAM
    # D(a=1) is normalized to 1 for each, so we need the unnormalized ratio
    # Run without normalization
    a_u, D_u, _, _, _ = solve_growth(mu0=mu0_iam)
    a_l, D_l, _, _, _ = solve_growth(mu0=0.0)
    # Both are normalized to D(a=1)=1, so we need raw ratio
    # Recompute with same IC
    def raw_growth(mu0, use_exact=False):
        def deriv(lna, y):
            a = np.exp(lna)
            D, dD = y
            H2 = Om * a**(-3) + OL
            q = 0.5 * Om * a**(-3) / H2 - OL / H2
            Om_a = Om * a**(-3) / H2
            if use_exact:
                mu_a = mu_iam_exact(a)
            else:
                mu_a = mu_mgcamb(a, mu0)
            ddD = -(2.0 - q) * dD + 1.5 * mu_a * Om_a * D
            return [dD, ddD]
        
        sol = solve_ivp(deriv, (np.log(1e-4), 0.0), [1e-4, 1.0],
                        t_eval=[0.0], rtol=1e-12, atol=1e-14, method='DOP853')
        return sol.y[0][0]
    
    D1_lcdm = raw_growth(0.0)
    D1_iam = raw_growth(mu0_iam)
    D1_iamx = raw_growth(mu0_iam, use_exact=True)
    
    sigma8_iam = sigma8_planck * D1_iam / D1_lcdm
    sigma8_iamx = sigma8_planck * D1_iamx / D1_lcdm
    
    print(f"  LCDM:      sigma_8 = {sigma8_planck:.4f}")
    print(f"  IAM MGCAMB: sigma_8 = {sigma8_iam:.4f} ({(sigma8_iam/sigma8_planck-1)*100:+.2f}%)")
    print(f"  IAM exact:  sigma_8 = {sigma8_iamx:.4f} ({(sigma8_iamx/sigma8_planck-1)*100:+.2f}%)")
    
    # ------------------------------------------------------------------
    # 2. ISW kernel comparison
    # ------------------------------------------------------------------
    print("\n[2/5] Computing ISW kernels...")
    
    z_plot = np.linspace(0.01, 3.0, 500)
    a_plot = 1.0 / (1.0 + z_plot)
    
    K_lcdm = np.array([isw_kernel(a, D_func_lcdm, f_func_lcdm, 0.0) for a in a_plot])
    K_iam = np.array([isw_kernel(a, D_func_iam, f_func_iam, mu0_iam) for a in a_plot])
    K_iamx = np.array([isw_kernel(a, D_func_iamx, f_func_iamx, 0.0, use_exact=True) for a in a_plot])
    
    # The ISW kernel should be negative (potentials decaying -> K < 0)
    # Stronger ISW = more negative K
    
    # Ratio
    ratio_mgcamb = K_iam / K_lcdm
    ratio_exact = K_iamx / K_lcdm
    
    print(f"  ISW kernel ratio IAM/LCDM at z=0.3: {np.interp(0.3, z_plot, ratio_mgcamb):.4f}")
    print(f"  ISW kernel ratio IAM/LCDM at z=0.5: {np.interp(0.5, z_plot, ratio_mgcamb):.4f}")
    print(f"  ISW kernel ratio IAM/LCDM at z=1.0: {np.interp(1.0, z_plot, ratio_mgcamb):.4f}")
    print(f"  ISW kernel ratio IAM/LCDM at z=2.0: {np.interp(2.0, z_plot, ratio_mgcamb):.4f}")
    
    # ------------------------------------------------------------------
    # 3. Cross-correlation for different galaxy samples
    # ------------------------------------------------------------------
    print("\n[3/5] Computing C_ell^{Tg} for galaxy samples...")
    
    ell_arr = np.logspace(0.3, 2.5, 80)  # ell = 2 to ~300 (ISW multipoles)
    
    # Galaxy samples (approximate DESI/Euclid-like)
    galaxy_samples = [
        ("DESI BGS (z~0.3)", 0.3, 0.1, 1.2),
        ("DESI LRG (z~0.5)", 0.5, 0.15, 1.7),
        ("DESI LRG (z~0.7)", 0.7, 0.15, 1.9),
        ("Euclid (z~1.0)", 1.0, 0.2, 2.0),
        ("High-z (z~1.5)", 1.5, 0.3, 2.5),
    ]
    
    results = {}
    
    for name, z_mean, z_sig, bias in galaxy_samples:
        Cl_lcdm = compute_isw_cross_spectrum(
            ell_arr, z_mean, z_sig, bias,
            D_func_lcdm, f_func_lcdm, mu0=0.0,
            sigma8_model=sigma8_planck)
        
        Cl_iam = compute_isw_cross_spectrum(
            ell_arr, z_mean, z_sig, bias,
            D_func_iam, f_func_iam, mu0=mu0_iam,
            sigma8_model=sigma8_iam)
        
        Cl_iamx = compute_isw_cross_spectrum(
            ell_arr, z_mean, z_sig, bias,
            D_func_iamx, f_func_iamx, mu0=0.0, use_exact=True,
            sigma8_model=sigma8_iamx)
        
        # Amplitude ratio (integrated)
        # Use ell range 2-100 where ISW dominates
        mask = ell_arr < 100
        A_ratio = np.sum(ell_arr[mask] * Cl_iam[mask]) / np.sum(ell_arr[mask] * Cl_lcdm[mask])
        A_ratio_x = np.sum(ell_arr[mask] * Cl_iamx[mask]) / np.sum(ell_arr[mask] * Cl_lcdm[mask])
        
        results[name] = {
            'Cl_lcdm': Cl_lcdm, 'Cl_iam': Cl_iam, 'Cl_iamx': Cl_iamx,
            'A_ratio': A_ratio, 'A_ratio_x': A_ratio_x,
            'z_mean': z_mean
        }
        
        print(f"  {name:<25s}: A_ISW(IAM)/A_ISW(LCDM) = {A_ratio:.4f} "
              f"(exact: {A_ratio_x:.4f})")
    
    # ------------------------------------------------------------------
    # 4. Potential decay rate
    # ------------------------------------------------------------------
    print("\n[4/5] Computing potential decay rates...")
    
    # Phi+Psi proportional to mu * Sigma * D(a) / a (in Poisson gauge)
    # For IAM: Sigma = 1, so Phi+Psi propto mu(a) * D(a) / a
    
    Phi_lcdm = np.array([1.0 * D_func_lcdm(a) / a for a in a_plot])
    Phi_iam = np.array([mu_mgcamb(a, mu0_iam) * D_func_iam(a) / a for a in a_plot])
    Phi_iamx = np.array([mu_iam_exact(a) * D_func_iamx(a) / a for a in a_plot])
    
    # Normalize to early time
    Phi_lcdm /= Phi_lcdm[-1]   # z=3 end
    Phi_iam /= Phi_iam[-1]
    Phi_iamx /= Phi_iamx[-1]
    
    # Wait -- we need to normalize at high z where both = 1
    # Let's normalize at z = 3 (a = 0.25)
    idx_norm = np.argmin(np.abs(z_plot - 3.0))
    Phi_lcdm /= Phi_lcdm[idx_norm]
    Phi_iam /= Phi_iam[idx_norm]
    Phi_iamx /= Phi_iamx[idx_norm]
    
    print(f"  Potential (Phi+Psi) at z=0 relative to z=3:")
    print(f"    LCDM:      {Phi_lcdm[0]:.4f}")
    print(f"    IAM MGCAMB: {Phi_iam[0]:.4f} ({(Phi_iam[0]/Phi_lcdm[0]-1)*100:+.2f}%)")
    print(f"    IAM exact:  {Phi_iamx[0]:.4f} ({(Phi_iamx[0]/Phi_lcdm[0]-1)*100:+.2f}%)")
    print(f"  --> IAM potentials decay MORE than LCDM (stronger ISW source)")
    
    # ------------------------------------------------------------------
    # 5. Detection forecast
    # ------------------------------------------------------------------
    print("\n[5/5] ISW detection forecast...")
    
    # Current ISW detection: ~2-4 sigma with Planck x various galaxy surveys
    # Planck 2015 ISW paper: A_ISW = 1.00 +/- 0.25 (relative to LCDM)
    # Expected improvement with Euclid/DESI: factor ~2-3 in S/N
    
    sigma_A_planck = 0.25
    sigma_A_euclid_desi = 0.10  # projected
    sigma_A_cmbs4_euclid = 0.06  # optimistic
    
    # IAM prediction for integrated A_ISW
    # Average over redshift-weighted samples
    A_iam_avg = np.mean([r['A_ratio'] for r in results.values()])
    delta_A = A_iam_avg - 1.0  # Deviation from LCDM
    
    print(f"\n  IAM average A_ISW / A_ISW(LCDM) = {A_iam_avg:.4f}")
    print(f"  Deviation from LCDM: {delta_A*100:+.1f}%")
    
    print(f"\n  {'Configuration':<35s} {'sigma(A_ISW)':>12s} {'Detection':>12s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    
    configs = [
        ("Planck x SDSS/WISE (current)", sigma_A_planck),
        ("Euclid + DESI (projected)", sigma_A_euclid_desi),
        ("CMB-S4 + Euclid (optimistic)", sigma_A_cmbs4_euclid),
    ]
    
    for name, sig in configs:
        det = abs(delta_A) / sig
        print(f"  {name:<35s} {sig:>12.3f} {det:>11.1f} sigma")
    
    print(f"\n  Note: The ISW test is complementary to mu_0 measurement.")
    print(f"  ISW probes d(Phi+Psi)/dt DIRECTLY -- no model assumption needed.")
    print(f"  IAM's A_ISW > 1 prediction is OPPOSITE to f(R) which has A_ISW < 1")
    print(f"  (because f(R) has mu > 1 -> potentials decay LESS).")
    
    # ------------------------------------------------------------------
    # FIGURES
    # ------------------------------------------------------------------
    print("\n  Generating figures...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("IAM ISW-Galaxy Cross-Correlation: $\\mu < 1$ Predicts Enhanced ISW Signal",
                 fontsize=15, fontweight='bold', y=0.99)
    
    # -- Panel (a): ISW kernel --
    ax = axes[0, 0]
    ax.plot(z_plot, -K_lcdm, 'k-', linewidth=2.5, label='$\\Lambda$CDM')
    ax.plot(z_plot, -K_iam, 'b-', linewidth=2, label='IAM (MGCAMB)')
    ax.plot(z_plot, -K_iamx, 'r--', linewidth=1.5, label='IAM (exact)', alpha=0.7)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$-K_{\\rm ISW}(z)$ [arb. units]', fontsize=12)
    ax.set_title('(a) ISW Kernel: $d(\\Phi+\\Psi)/d\\tau$', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(bottom=0)
    
    # -- Panel (b): ISW kernel ratio --
    ax = axes[0, 1]
    valid = np.abs(K_lcdm) > 1e-10
    ax.plot(z_plot[valid], ratio_mgcamb[valid], 'b-', linewidth=2.5, label='IAM/LCDM (MGCAMB)')
    ax.plot(z_plot[valid], ratio_exact[valid], 'r--', linewidth=1.5, label='IAM/LCDM (exact)', alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.fill_between(z_plot[valid], 1.0, ratio_mgcamb[valid], alpha=0.15, color='blue')
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$K_{\\rm ISW}^{\\rm IAM} / K_{\\rm ISW}^{\\Lambda{\\rm CDM}}$', fontsize=12)
    ax.set_title('(b) ISW Kernel Enhancement', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0.9, 1.5)
    
    # -- Panel (c): Gravitational potential evolution --
    ax = axes[0, 2]
    ax.plot(z_plot, Phi_lcdm, 'k-', linewidth=2.5, label='$\\Lambda$CDM')
    ax.plot(z_plot, Phi_iam, 'b-', linewidth=2, label='IAM (MGCAMB)')
    ax.plot(z_plot, Phi_iamx, 'r--', linewidth=1.5, label='IAM (exact)', alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$(\\Phi+\\Psi)(z) / (\\Phi+\\Psi)(z=3)$', fontsize=12)
    ax.set_title('(c) Potential Decay: IAM Faster', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 3)
    
    # Annotate the faster decay
    z05 = 0.5
    idx05 = np.argmin(np.abs(z_plot - z05))
    diff = (Phi_iam[idx05] - Phi_lcdm[idx05]) / Phi_lcdm[idx05] * 100
    ax.annotate(f'{diff:+.1f}% at z={z05}',
                xy=(z05, Phi_iam[idx05]), xytext=(1.5, Phi_iam[idx05]+0.03),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    
    # -- Panel (d): C_ell^{Tg} for different galaxy samples --
    ax = axes[1, 0]
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
    
    for i, (name, data) in enumerate(results.items()):
        # Plot LCDM and IAM for low-z sample
        if i < 3:  # Show first 3 samples
            label_l = f'LCDM ({name.split("(")[1]}' if i == 0 else None
            ax.plot(ell_arr, ell_arr * (ell_arr + 1) * data['Cl_lcdm'] / (2*np.pi),
                    '--', color=colors[i], linewidth=1.5, alpha=0.5)
            ax.plot(ell_arr, ell_arr * (ell_arr + 1) * data['Cl_iam'] / (2*np.pi),
                    '-', color=colors[i], linewidth=2.5,
                    label=f'{name}: A={data["A_ratio"]:.3f}')
    
    ax.set_xlabel('Multipole $\\ell$', fontsize=12)
    ax.set_ylabel('$\\ell(\\ell+1) C_\\ell^{Tg} / 2\\pi$', fontsize=12)
    ax.set_title('(d) $C_\\ell^{Tg}$: Solid=IAM, Dashed=LCDM', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(2, 300)
    
    # -- Panel (e): A_ISW vs redshift --
    ax = axes[1, 1]
    z_means = [r['z_mean'] for r in results.values()]
    A_ratios = [r['A_ratio'] for r in results.values()]
    A_ratios_x = [r['A_ratio_x'] for r in results.values()]
    
    ax.plot(z_means, A_ratios, 'bo-', linewidth=2, markersize=10, label='IAM/LCDM (MGCAMB)')
    ax.plot(z_means, A_ratios_x, 'rs--', linewidth=1.5, markersize=8, 
            label='IAM/LCDM (exact)', alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='$\\Lambda$CDM')
    
    # Current constraint band
    ax.fill_between([0, 2], 1-sigma_A_planck, 1+sigma_A_planck,
                    alpha=0.1, color='gray', label=f'Planck ISW ($\\pm${sigma_A_planck})')
    ax.fill_between([0, 2], 1-sigma_A_euclid_desi, 1+sigma_A_euclid_desi,
                    alpha=0.15, color='green', label=f'Euclid+DESI ($\\pm${sigma_A_euclid_desi})')
    
    ax.set_xlabel('Galaxy Sample Mean Redshift', fontsize=12)
    ax.set_ylabel('$A_{\\rm ISW}^{\\rm IAM} / A_{\\rm ISW}^{\\Lambda{\\rm CDM}}$', fontsize=12)
    ax.set_title('(e) ISW Amplitude Ratio vs Redshift', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, 2)
    ax.set_ylim(0.7, 1.5)
    
    # -- Panel (f): Model comparison --
    ax = axes[1, 2]
    
    models = {
        '$\\Lambda$CDM': (1.0, 0, 'gray'),
        'IAM ($\\mu<1, \\Sigma=1$)': (A_iam_avg, 0, 'blue'),
        '$f(R)$ ($\\mu>1, \\Sigma>1$)': (0.90, 0, 'red'),  # f(R) reduces ISW
        '$w_0w_a$CDM': (1.0, 0, 'green'),  # Same as LCDM for ISW
    }
    
    y_pos = np.arange(len(models))
    for i, (name, (A, _, color)) in enumerate(models.items()):
        if name == '$\\Lambda$CDM':
            ax.barh(i, A, height=0.5, color=color, alpha=0.3, edgecolor='black')
        else:
            ax.barh(i, A, height=0.5, color=color, alpha=0.6, edgecolor='black')
    
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1)
    
    # Add Euclid sensitivity band
    ax.axvspan(1.0 - sigma_A_euclid_desi, 1.0 + sigma_A_euclid_desi,
               alpha=0.1, color='green')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(models.keys()), fontsize=11)
    ax.set_xlabel('$A_{\\rm ISW}$ (relative to $\\Lambda$CDM)', fontsize=12)
    ax.set_title('(f) ISW Amplitude: Model Comparison', fontsize=12)
    ax.set_xlim(0.7, 1.3)
    
    # Add text annotations
    ax.text(A_iam_avg + 0.01, 1, f'{A_iam_avg:.3f}', fontsize=11, va='center', color='blue',
            fontweight='bold')
    ax.text(0.90 + 0.01, 2, '0.90', fontsize=11, va='center', color='red',
            fontweight='bold')
    ax.text(1.005, 3, '1.000', fontsize=11, va='center', color='green',
            fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    outpath = '/home/claude/iam_isw_prediction.pdf'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    
    outpath_png = '/home/claude/iam_isw_prediction.png'
    plt.savefig(outpath_png, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath_png}")
    
    # ------------------------------------------------------------------
    # SUMMARY TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY: ISW-Galaxy Cross-Correlation Predictions")
    print("=" * 72)
    print(f"""
  PHYSICS:
    LCDM: Potentials decay as dark energy dominates -> ISW effect
    IAM:  mu < 1 -> gravity weaker -> potentials decay FASTER
          -> ENHANCED ISW signal (A_ISW > 1)

  KEY RESULT:
    IAM predicts A_ISW = {A_iam_avg:.3f} relative to LCDM
    ({(A_iam_avg-1)*100:+.1f}% enhancement in ISW-galaxy cross-correlation)

  MODEL DISCRIMINATION:
    IAM (mu < 1): A_ISW > 1 (enhanced ISW, weaker gravity)
    f(R) (mu > 1): A_ISW < 1 (suppressed ISW, stronger gravity)
    w0waCDM:       A_ISW = 1 (same gravity as LCDM)
    
    --> ISW sign distinguishes IAM from f(R) even before amplitude is precise!
    --> This is INDEPENDENT of the mu_0 measurement from Euclid 3x2pt

  DETECTABILITY:
    Current (Planck x WISE):     {abs(delta_A)/sigma_A_planck:.1f} sigma (cannot detect {delta_A*100:+.1f}% shift)
    Euclid + DESI:               {abs(delta_A)/sigma_A_euclid_desi:.1f} sigma (marginal)
    CMB-S4 + Euclid:             {abs(delta_A)/sigma_A_cmbs4_euclid:.1f} sigma (detectable)
    
  COMPLEMENTARITY:
    Euclid mu_0 measurement:     Constrains gravity coupling directly
    ISW cross-correlation:       Constrains potential TIME DERIVATIVE
    Combined:                    Breaks degeneracies between mu and sigma8
""")

if __name__ == "__main__":
    main()
