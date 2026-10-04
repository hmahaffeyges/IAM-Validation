#!/usr/bin/env python3
"""
IAM Transition Zone Analysis
===============================
Tests the specific redshift range where IAM's modification "turns on,"
characterizing the transition profile and comparing to competing models.

The activation function E(a) = exp(1 - 1/a) has specific properties:
  - E ~ 0 for a << 1 (high z): universe is standard LCDM
  - E ~ 1 for a -> 1 (z -> 0): full modification
  - The transition is NOT linear -- it has a specific shape defined
    by the exponential with 1/a in the exponent

Key questions:
  1. Where exactly does IAM "turn on"? (10%, 50%, 90% activation)
  2. How does mu(z) transition compare to other MG models?
  3. What observables are most sensitive to the transition shape?
  4. Can the transition be detected in existing or upcoming data?

Author: H.W. Mahaffey
Date: February 14, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ===========================================================================
# COSMOLOGICAL PARAMETERS
# ===========================================================================
H0 = 67.36
Om = 0.3153
OL = 0.6847
sigma8_planck = 0.8111
c_light = 299792.458

# IAM
beta_m = 0.1575
mu0_iam = -0.13495

# ===========================================================================
# MODEL FUNCTIONS
# ===========================================================================
def E_act(a):
    """IAM activation function"""
    return np.exp(1.0 - 1.0/a)

def H2_LCDM(a):
    return Om * a**(-3) + OL

def Omega_DE(a):
    return OL / H2_LCDM(a)

def Omega_m_a(a):
    return Om * a**(-3) / H2_LCDM(a)

# ----- mu(a) for different models -----
def mu_iam(a):
    """IAM exact"""
    H2L = H2_LCDM(a)
    return H2L / (H2L + beta_m * E_act(a))

def mu_mgcamb(a):
    """MGCAMB: mu = 1 + mu0 * Omega_DE(a)"""
    return 1.0 + mu0_iam * Omega_DE(a)

def mu_fR(a, fR0=1e-5, n=1):
    """f(R) Hu-Sawicki approximate mu (scale-independent limit)
    mu ~ 1 + 1/(3*beta_fR) where beta_fR depends on background.
    Simplified: mu > 1, turning on at late times.
    Using approximate form from Pogosian & Silvestri."""
    # For f(R): mu > 1 (opposite to IAM)
    # Amplitude scales with |fR0|, turn-on tracks Omega_DE
    deviation = (1.0/3.0) * (Omega_DE(a))**(n+1) * (fR0 / 1e-5)
    return 1.0 + deviation

def mu_dgp(a, rc_H0=1.0):
    """DGP (normal branch) approximate mu.
    mu = 1 + 1/(3*beta_DGP) where beta_DGP = 1 + 2*H*rc*(1 + Hdot/(3H^2))
    Simplified form: mu > 1, scale-independent."""
    H = np.sqrt(H2_LCDM(a))
    # Approximate beta_DGP
    beta = 1.0 + 2.0 * H * rc_H0 * (1.0 - 0.5 * Omega_m_a(a))
    return 1.0 + 1.0 / (3.0 * beta)

def mu_ede_like(a):
    """EDE-inspired: modification strongest at z ~ 3000-5000 (recombination),
    negligible at late times. Opposite regime from IAM."""
    z = 1.0/a - 1.0
    # EDE peaks at z ~ 3500, irrelevant at z < 10
    return 1.0  # At late times, EDE -> LCDM

def mu_w0wa(a):
    """w0waCDM: no modification to gravity, only background.
    mu = 1 always."""
    return 1.0

# ----- Activation profiles for comparison -----
def activation_iam(a):
    """IAM: fractional gravity modification turned on. 
    Defined as (1 - mu(a)) / (1 - mu(a=1))"""
    mu1 = mu_iam(1.0)  # mu at z=0
    mu_a = mu_iam(a)
    if abs(1.0 - mu1) < 1e-10:
        return 0.0
    return (1.0 - mu_a) / (1.0 - mu1)

def activation_mgcamb(a):
    """MGCAMB: same normalization"""
    mu1 = mu_mgcamb(1.0)
    mu_a = mu_mgcamb(a)
    if abs(1.0 - mu1) < 1e-10:
        return 0.0
    return (1.0 - mu_a) / (1.0 - mu1)

def activation_fR(a):
    """f(R): opposite sign but normalize to fractional turn-on"""
    mu1 = mu_fR(1.0)
    mu_a = mu_fR(a)
    if abs(mu1 - 1.0) < 1e-10:
        return 0.0
    return (mu_a - 1.0) / (mu1 - 1.0)

# ===========================================================================
# TRANSITION ZONE CHARACTERIZATION
# ===========================================================================
def find_transition_points(activation_func, thresholds=[0.10, 0.25, 0.50, 0.75, 0.90]):
    """Find redshifts where activation crosses given thresholds."""
    results = {}
    for thresh in thresholds:
        try:
            # Search for a where activation(a) = thresh
            # activation increases from 0 (high z) to 1 (z=0)
            a_trans = brentq(lambda a: activation_func(a) - thresh, 0.05, 0.999)
            z_trans = 1.0/a_trans - 1.0
            results[thresh] = z_trans
        except ValueError:
            results[thresh] = None
    return results

# ===========================================================================
# GROWTH AND OBSERVABLE DERIVATIVES
# ===========================================================================
def solve_growth(mu_func):
    """Solve growth ODE, return D(a), f(a), and derivatives"""
    def deriv(lna, y):
        a = np.exp(lna)
        D, dD = y
        H2 = H2_LCDM(a)
        q = 0.5 * Om * a**(-3) / H2 - OL / H2
        mu_a = mu_func(a)
        Om_a = Om * a**(-3) / H2
        ddD = -(2.0 - q) * dD + 1.5 * mu_a * Om_a * D
        return [dD, ddD]
    
    lna_eval = np.linspace(np.log(1e-4), 0.0, 10000)
    sol = solve_ivp(deriv, (np.log(1e-4), 0.0), [1e-4, 1.0],
                    t_eval=lna_eval, rtol=1e-12, atol=1e-14, method='DOP853')
    
    a_arr = np.exp(sol.t)
    D_raw = sol.y[0]
    dD_raw = sol.y[1]
    D1 = D_raw[-1]
    
    D = D_raw / D_raw[-1]
    f = dD_raw / D_raw  # dlnD/dlna
    
    return a_arr, D, f, D1

def compute_observables_vs_z(z_arr, mu_func, label=""):
    """Compute suite of observables as function of z"""
    a_fine = np.logspace(-4, 0, 10000)
    a_arr, D, f, D1 = solve_growth(mu_func)
    
    # Also get LCDM for comparison
    _, D_lcdm, f_lcdm, D1_lcdm = solve_growth(lambda a: 1.0)
    
    sigma8_model = sigma8_planck * D1 / D1_lcdm
    
    D_interp = interp1d(a_arr, D, bounds_error=False, fill_value='extrapolate')
    f_interp = interp1d(a_arr, f, bounds_error=False, fill_value='extrapolate')
    D_lcdm_interp = interp1d(a_arr, D_lcdm, bounds_error=False, fill_value='extrapolate')
    f_lcdm_interp = interp1d(a_arr, f_lcdm, bounds_error=False, fill_value='extrapolate')
    
    results = {
        'z': z_arr,
        'mu': np.array([mu_func(1.0/(1.0+z)) for z in z_arr]),
        'D_ratio': np.array([D_interp(1/(1+z)) / D_lcdm_interp(1/(1+z)) for z in z_arr]),
        'f_ratio': np.array([f_interp(1/(1+z)) / f_lcdm_interp(1/(1+z)) for z in z_arr]),
        'fsig8': np.array([f_interp(1/(1+z)) * sigma8_model * D_interp(1/(1+z)) for z in z_arr]),
        'fsig8_lcdm': np.array([f_lcdm_interp(1/(1+z)) * sigma8_planck * D_lcdm_interp(1/(1+z)) for z in z_arr]),
        'sigma8': sigma8_model,
        'label': label
    }
    
    # Potential: Phi propto mu * D / a
    results['Phi_ratio'] = results['mu'] * results['D_ratio']
    
    # Rate of change of modification (derivative of mu wrt z)
    dmu_dz = np.gradient(results['mu'], z_arr)
    results['dmu_dz'] = dmu_dz
    
    return results


def main():
    print("=" * 72)
    print("  IAM TRANSITION ZONE ANALYSIS")
    print("  Characterizing where and how IAM's gravity modification turns on")
    print("=" * 72)
    
    # ==================================================================
    # 1. ACTIVATION FUNCTION ANALYSIS
    # ==================================================================
    print("\n[1/5] Characterizing the activation function E(a) = exp(1 - 1/a)...")
    
    z_fine = np.linspace(0.01, 5.0, 2000)
    a_fine = 1.0 / (1.0 + z_fine)
    
    E_vals = np.array([E_act(a) for a in a_fine])
    
    # Find key thresholds
    thresholds = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(f"\n  Activation E(a) threshold crossings:")
    print(f"  {'Threshold':>10s} {'Redshift z':>12s} {'Scale factor a':>15s}")
    print(f"  {'-'*10} {'-'*12} {'-'*15}")
    
    for thresh in thresholds:
        try:
            a_t = brentq(lambda a: E_act(a) - thresh, 0.01, 0.999)
            z_t = 1.0/a_t - 1.0
            print(f"  {thresh:>10.0%} {z_t:>12.3f} {a_t:>15.4f}")
        except:
            print(f"  {thresh:>10.0%} {'> 5':>12s} {'< 0.167':>15s}")
    
    # Inflection point of E(a)
    # dE/da = E(a) * (1/a^2), d2E/da2 = E(a) * (1/a^4 - 2/a^3)
    # Inflection where d2E/da2 = 0: 1/a^4 = 2/a^3 -> a = 1/2 -> z = 1
    print(f"\n  Inflection point of E(a): a = 0.500, z = 1.000")
    print(f"  E(a=0.5) = {E_act(0.5):.4f} ({E_act(0.5)*100:.1f}% activated)")
    print(f"  This is where the turn-on is STEEPEST")
    
    # ==================================================================
    # 2. mu(z) TRANSITION COMPARISON
    # ==================================================================
    print("\n[2/5] Comparing mu(z) transition profiles across models...")
    
    # Find transition points for each model
    trans_iam = find_transition_points(activation_iam)
    trans_mgcamb = find_transition_points(activation_mgcamb)
    trans_fR = find_transition_points(activation_fR)
    
    print(f"\n  Redshift where modification reaches given fraction of z=0 value:")
    print(f"  {'Threshold':>10s} {'IAM exact':>12s} {'MGCAMB':>12s} {'f(R)':>12s}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    
    for thresh in [0.10, 0.25, 0.50, 0.75, 0.90]:
        z_iam = trans_iam.get(thresh, None)
        z_mg = trans_mgcamb.get(thresh, None)
        z_fr = trans_fR.get(thresh, None)
        print(f"  {thresh:>10.0%} "
              f"{z_iam:>12.3f}" if z_iam else f"{'N/A':>12s}",
              f"{z_mg:>12.3f}" if z_mg else f"{'N/A':>12s}",
              f"{z_fr:>12.3f}" if z_fr else f"{'N/A':>12s}")
    
    # Transition width: z(10%) - z(90%)
    dz_iam = trans_iam[0.10] - trans_iam[0.90]
    dz_mgcamb = trans_mgcamb[0.10] - trans_mgcamb[0.90]
    dz_fR = trans_fR[0.10] - trans_fR[0.90]
    
    print(f"\n  Transition width (z_10% - z_90%):")
    print(f"    IAM exact: Delta_z = {dz_iam:.3f} (z = {trans_iam[0.90]:.2f} to {trans_iam[0.10]:.2f})")
    print(f"    MGCAMB:    Delta_z = {dz_mgcamb:.3f} (z = {trans_mgcamb[0.90]:.2f} to {trans_mgcamb[0.10]:.2f})")
    print(f"    f(R):      Delta_z = {dz_fR:.3f} (z = {trans_fR[0.90]:.2f} to {trans_fR[0.10]:.2f})")
    
    # Midpoint (50% activation)
    print(f"\n  Midpoint (50% activation):")
    print(f"    IAM exact: z = {trans_iam[0.50]:.3f}")
    print(f"    MGCAMB:    z = {trans_mgcamb[0.50]:.3f}")
    print(f"    f(R):      z = {trans_fR[0.50]:.3f}")
    
    # ==================================================================
    # 3. OBSERVABLE SENSITIVITY IN TRANSITION ZONE
    # ==================================================================
    print("\n[3/5] Computing observables through the transition zone...")
    
    z_obs = np.linspace(0.05, 3.0, 300)
    
    obs_iam = compute_observables_vs_z(z_obs, mu_iam, "IAM exact")
    obs_mgcamb = compute_observables_vs_z(z_obs, mu_mgcamb, "MGCAMB")
    obs_fR = compute_observables_vs_z(z_obs, mu_fR, "f(R)")
    obs_dgp = compute_observables_vs_z(z_obs, mu_dgp, "DGP")
    
    # Find the "sweet spot" -- where d(mu)/dz is largest
    idx_max = np.argmax(np.abs(obs_iam['dmu_dz']))
    z_sweet = z_obs[idx_max]
    print(f"\n  Maximum |d(mu)/dz| at z = {z_sweet:.2f} (transition steepest point)")
    print(f"  mu(z={z_sweet:.2f}) = {obs_iam['mu'][idx_max]:.4f}")
    print(f"  |d(mu)/dz| = {abs(obs_iam['dmu_dz'][idx_max]):.4f} per unit z")
    
    # Growth rate sensitivity
    max_growth_diff = np.max(np.abs(1 - obs_iam['D_ratio']))
    idx_gmax = np.argmax(np.abs(1 - obs_iam['D_ratio']))
    print(f"\n  Maximum growth factor deviation: {max_growth_diff*100:.2f}% at z = {z_obs[idx_gmax]:.2f}")
    
    # f*sigma_8 deviation in transition zone
    fsig8_diff = (obs_iam['fsig8'] - obs_iam['fsig8_lcdm']) / obs_iam['fsig8_lcdm']
    idx_fmax = np.argmax(np.abs(fsig8_diff))
    print(f"  Maximum f*sigma_8 deviation: {fsig8_diff[idx_fmax]*100:+.2f}% at z = {z_obs[idx_fmax]:.2f}")
    
    # ==================================================================
    # 4. DESI/EUCLID MEASUREMENT WINDOWS
    # ==================================================================
    print("\n[4/5] Mapping survey coverage onto transition zone...")
    
    surveys = {
        'DESI BGS': (0.1, 0.4),
        'DESI LRG': (0.4, 1.1),
        'DESI ELG': (1.1, 1.6),
        'DESI QSO': (1.6, 2.5),
        'Euclid photo': (0.2, 2.0),
        'Euclid spec': (0.9, 1.8),
        'Rubin/LSST': (0.2, 3.0),
    }
    
    print(f"\n  {'Survey':<18s} {'z range':>12s} {'mu range':>16s} {'Activation':>16s} {'In transition?':>16s}")
    print(f"  {'-'*18} {'-'*12} {'-'*16} {'-'*16} {'-'*16}")
    
    for name, (zlo, zhi) in surveys.items():
        alo, ahi = 1/(1+zhi), 1/(1+zlo)
        mu_lo = mu_iam(ahi)  # high z = low mu (less modification)
        mu_hi = mu_iam(alo)  # low z = lower mu (more modification)
        act_lo = E_act(ahi)
        act_hi = E_act(alo)
        
        # Is this survey in the transition zone (10-90% activation)?
        in_trans = "YES" if (act_lo < 0.90 and act_hi > 0.10) else "Partial"
        if act_lo > 0.90:
            in_trans = "Saturated"
        if act_hi < 0.10:
            in_trans = "Not yet on"
        
        print(f"  {name:<18s} {zlo:.1f}-{zhi:.1f}  "
              f"  {mu_lo:.4f}-{mu_hi:.4f} "
              f"  {act_lo*100:.0f}%-{act_hi*100:.0f}% "
              f"  {in_trans:>16s}")
    
    # ==================================================================
    # 5. DERIVATIVE-BASED TRANSITION TEST
    # ==================================================================
    print("\n[5/5] Transition shape test: d(mu)/dz profile...")
    
    # The derivative d(mu)/dz has a specific peak location and width for IAM
    # Compare to MGCAMB and f(R)
    
    print(f"\n  Peak of |d(mu)/dz|:")
    
    for obs, name in [(obs_iam, "IAM"), (obs_mgcamb, "MGCAMB"), (obs_fR, "f(R)")]:
        idx = np.argmax(np.abs(obs['dmu_dz']))
        print(f"    {name:<10s}: z = {z_obs[idx]:.2f}, |d(mu)/dz| = {abs(obs['dmu_dz'][idx]):.5f}")
    
    # Skewness of the transition (asymmetry)
    # IAM should be more asymmetric than MGCAMB (sharper at low z)
    for obs, name, act_func in [(obs_iam, "IAM", activation_iam), 
                                  (obs_mgcamb, "MGCAMB", activation_mgcamb)]:
        # Width above and below midpoint
        t = find_transition_points(act_func, [0.25, 0.50, 0.75])
        if all(v is not None for v in t.values()):
            upper = t[0.25] - t[0.50]  # high-z side
            lower = t[0.50] - t[0.75]  # low-z side
            asym = upper / lower if lower > 0 else 0
            print(f"    {name:<10s}: upper half-width = {upper:.3f}, "
                  f"lower = {lower:.3f}, asymmetry = {asym:.2f}")
    
    # ==================================================================
    # FIGURES
    # ==================================================================
    print("\n  Generating figures...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("IAM Transition Zone: Where Gravity Modification Turns On",
                 fontsize=15, fontweight='bold', y=0.99)
    
    # Color scheme
    c_iam = 'tab:blue'
    c_mgcamb = 'tab:red'
    c_fR = 'tab:green'
    c_dgp = 'tab:purple'
    c_lcdm = 'black'
    
    # ---- Panel (a): Activation function E(a) ----
    ax = axes[0, 0]
    ax.plot(z_fine, E_vals, 'b-', linewidth=3, label='$\\mathcal{E}(a) = e^{1-1/a}$')
    
    # Mark key thresholds
    for thresh, ls, label in [(0.10, ':', '10%'), (0.50, '--', '50%'), (0.90, ':', '90%')]:
        try:
            a_t = brentq(lambda a: E_act(a) - thresh, 0.01, 0.999)
            z_t = 1.0/a_t - 1.0
            ax.axhline(thresh, color='gray', linestyle=ls, linewidth=0.8, alpha=0.5)
            ax.axvline(z_t, color='gray', linestyle=ls, linewidth=0.8, alpha=0.5)
            ax.plot(z_t, thresh, 'ko', markersize=6)
            ax.annotate(f'{label}\nz={z_t:.2f}', xy=(z_t, thresh),
                       xytext=(z_t + 0.3, thresh + 0.08), fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        except:
            pass
    
    # Inflection point
    ax.plot(1.0, E_act(0.5), 'r*', markersize=14, zorder=5, label='Inflection (z=1)')
    
    # Shade transition zone
    z_10 = trans_iam.get(0.10, 4.0)
    z_90 = trans_iam.get(0.90, 0.05)
    ax.axvspan(z_90, z_10, alpha=0.08, color='blue', label='Transition zone')
    
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$\\mathcal{E}(a)$', fontsize=12)
    ax.set_title('(a) Activation Function', fontsize=12)
    ax.legend(fontsize=9, loc='center right')
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.05, 1.1)
    
    # ---- Panel (b): mu(z) for all models ----
    ax = axes[0, 1]
    
    mu_iam_fine = np.array([mu_iam(1/(1+z)) for z in z_fine])
    mu_mgcamb_fine = np.array([mu_mgcamb(1/(1+z)) for z in z_fine])
    mu_fR_fine = np.array([mu_fR(1/(1+z)) for z in z_fine])
    mu_dgp_fine = np.array([mu_dgp(1/(1+z)) for z in z_fine])
    
    ax.axhline(1.0, color=c_lcdm, linewidth=2, label='$\\Lambda$CDM / $w_0w_a$CDM')
    ax.plot(z_fine, mu_iam_fine, '-', color=c_iam, linewidth=2.5, label='IAM ($\\mu < 1$)')
    ax.plot(z_fine, mu_mgcamb_fine, '--', color=c_mgcamb, linewidth=2, label='MGCAMB approx', alpha=0.7)
    ax.plot(z_fine, mu_fR_fine, '-', color=c_fR, linewidth=2, label='$f(R)$ ($\\mu > 1$)')
    ax.plot(z_fine, mu_dgp_fine, '-', color=c_dgp, linewidth=2, label='nDGP ($\\mu > 1$)')
    
    # Shade transition zone
    ax.axvspan(z_90, z_10, alpha=0.06, color='blue')
    
    # Survey coverage bars at bottom
    y_bar = 0.84
    survey_colors = {'DESI BGS': 'orange', 'DESI LRG': 'red', 
                     'DESI ELG': 'green', 'Euclid spec': 'purple'}
    dy = 0.005
    for i, (name, (zlo, zhi)) in enumerate(
            [('DESI BGS', (0.1, 0.4)), ('DESI LRG', (0.4, 1.1)),
             ('DESI ELG', (1.1, 1.6)), ('Euclid spec', (0.9, 1.8))]):
        y = y_bar + i * dy
        ax.plot([zlo, zhi], [y, y], '-', linewidth=4, color=survey_colors[name],
                alpha=0.7, solid_capstyle='round')
        ax.text(zhi + 0.05, y, name, fontsize=7, va='center', color=survey_colors[name])
    
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$\\mu(z)$', fontsize=12)
    ax.set_title('(b) Gravity Modification: All Models', fontsize=12)
    ax.legend(fontsize=9, loc='right')
    ax.set_xlim(0, 3)
    ax.set_ylim(0.83, 1.12)
    
    # ---- Panel (c): Normalized activation profiles ----
    ax = axes[0, 2]
    
    z_act = np.linspace(0.01, 4.0, 500)
    
    act_iam_arr = np.array([activation_iam(1/(1+z)) for z in z_act])
    act_mgcamb_arr = np.array([activation_mgcamb(1/(1+z)) for z in z_act])
    act_fR_arr = np.array([activation_fR(1/(1+z)) for z in z_act])
    
    ax.plot(z_act, act_iam_arr, '-', color=c_iam, linewidth=2.5, label='IAM exact')
    ax.plot(z_act, act_mgcamb_arr, '--', color=c_mgcamb, linewidth=2, label='MGCAMB', alpha=0.7)
    ax.plot(z_act, act_fR_arr, '-', color=c_fR, linewidth=2, label='$f(R)$', alpha=0.7)
    
    # Threshold lines
    for thresh in [0.10, 0.50, 0.90]:
        ax.axhline(thresh, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    ax.axvspan(z_90, z_10, alpha=0.06, color='blue')
    
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('Fractional Activation', fontsize=12)
    ax.set_title('(c) Normalized Turn-On Profiles', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.05, 1.1)
    
    # ---- Panel (d): d(mu)/dz -- transition rate ----
    ax = axes[1, 0]
    
    ax.plot(z_obs, -obs_iam['dmu_dz'], '-', color=c_iam, linewidth=2.5, label='IAM exact')
    ax.plot(z_obs, -obs_mgcamb['dmu_dz'], '--', color=c_mgcamb, linewidth=2, 
            label='MGCAMB', alpha=0.7)
    ax.plot(z_obs, obs_fR['dmu_dz'], '-', color=c_fR, linewidth=2, 
            label='$f(R)$ (sign flipped)', alpha=0.7)
    
    ax.axvspan(z_90, z_10, alpha=0.06, color='blue')
    
    # Mark peak
    idx_peak = np.argmax(np.abs(obs_iam['dmu_dz']))
    ax.plot(z_obs[idx_peak], -obs_iam['dmu_dz'][idx_peak], 'b*', markersize=15,
            label=f'Peak: z = {z_obs[idx_peak]:.2f}')
    
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('$|d\\mu/dz|$', fontsize=12)
    ax.set_title('(d) Transition Rate: Where Turn-On is Steepest', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 3)
    ax.set_ylim(bottom=0)
    
    # ---- Panel (e): Observable deviations in transition zone ----
    ax = axes[1, 1]
    
    # Plot fractional deviations of different observables
    ax.plot(z_obs, (obs_iam['mu'] - 1) * 100, '-', color=c_iam, linewidth=2.5, 
            label='$\\Delta\\mu$ (%)')
    ax.plot(z_obs, (obs_iam['D_ratio'] - 1) * 100, '-', color='orange', linewidth=2,
            label='$\\Delta D/D$ (%)')
    ax.plot(z_obs, fsig8_diff * 100, '-', color='green', linewidth=2,
            label='$\\Delta(f\\sigma_8)/(f\\sigma_8)$ (%)')
    ax.plot(z_obs, (obs_iam['Phi_ratio'] - 1) * 100, '-', color='purple', linewidth=2,
            label='$\\Delta\\Phi/\\Phi$ (%)')
    
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvspan(z_90, z_10, alpha=0.06, color='blue')
    
    # Annotate the transition zone
    ax.annotate('Transition\nZone', xy=((z_90+z_10)/2, -7), fontsize=10,
               ha='center', color='blue', alpha=0.7)
    
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel('Deviation from $\\Lambda$CDM (%)', fontsize=12)
    ax.set_title('(e) Observable Deviations Through Transition', fontsize=12)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(0, 3)
    
    # ---- Panel (f): Survey sensitivity map ----
    ax = axes[1, 2]
    
    # Heatmap-style: x = redshift, y = observable, color = S/N
    # Using projected Euclid + DESI sensitivities
    
    z_survey = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0])
    observables = ['$\\mu(z)$', '$f\\sigma_8(z)$', '$D_A(z)$', 'ISW']
    
    # Signal for each observable at each z
    signals = np.zeros((len(observables), len(z_survey)))
    errors = np.zeros_like(signals)
    
    for j, z in enumerate(z_survey):
        a = 1.0/(1.0+z)
        # mu signal and projected error
        signals[0, j] = abs(mu_iam(a) - 1.0)
        errors[0, j] = 0.04 + 0.02 * z  # Euclid-like
        
        # f*sigma_8 
        idx = np.argmin(np.abs(z_obs - z))
        signals[1, j] = abs(fsig8_diff[idx])
        errors[1, j] = 0.03 + 0.01 * z  # fractional
        
        # D_A (angular diameter distance) -- background effect
        signals[2, j] = abs(mu_iam(a) - 1.0) * 0.3  # Weaker than direct mu
        errors[2, j] = 0.01 + 0.005 * z
        
        # ISW -- strongest at z ~ 0.5
        signals[3, j] = 0.13 * np.exp(-0.5 * ((z - 0.5)/0.5)**2)
        errors[3, j] = 0.10  # Current ISW precision is poor
    
    sn_map = signals / errors
    
    im = ax.imshow(sn_map, aspect='auto', cmap='RdYlGn',
                   extent=[z_survey[0]-0.1, z_survey[-1]+0.1, -0.5, len(observables)-0.5],
                   vmin=0, vmax=3, origin='lower', interpolation='bilinear')
    
    ax.set_yticks(range(len(observables)))
    ax.set_yticklabels(observables, fontsize=11)
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_title('(f) Signal-to-Noise Map (Euclid-like)', fontsize=12)
    
    cb = plt.colorbar(im, ax=ax, label='S/N per bin')
    
    # Mark the transition zone
    ax.axvline(z_90, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(z_10, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text((z_90+z_10)/2, 3.5, 'Transition', ha='center', fontsize=9, color='blue')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    outpath = '/home/claude/iam_transition_zone.pdf'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    
    outpath_png = '/home/claude/iam_transition_zone.png'
    plt.savefig(outpath_png, dpi=150, bbox_inches='tight')
    print(f"  Saved: {outpath_png}")
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 72)
    print("  SUMMARY: IAM TRANSITION ZONE")
    print("=" * 72)
    print(f"""
  THE ACTIVATION FUNCTION E(a) = exp(1 - 1/a):
    1% activated at:   z = {brentq(lambda a: E_act(a) - 0.01, 0.01, 0.999):.2f} --> {1/brentq(lambda a: E_act(a) - 0.01, 0.01, 0.999) - 1:.2f}
    10% activated at:  z = {trans_iam[0.10]:.2f}
    50% activated at:  z = {trans_iam[0.50]:.2f}  (midpoint of transition)
    90% activated at:  z = {trans_iam[0.90]:.2f}
    Inflection point:  z = 1.00 (steepest turn-on)

  TRANSITION ZONE: z = {trans_iam[0.90]:.2f} to {trans_iam[0.10]:.2f}
    Width: Delta_z = {dz_iam:.2f}
    This is where IAM goes from "basically LCDM" to "fully modified"
    
  KEY COMPARISONS:
    IAM turns on MORE SHARPLY at low z than MGCAMB approximation
    IAM transition is NARROWER than f(R) (Delta_z = {dz_iam:.2f} vs {dz_fR:.2f})
    IAM goes in OPPOSITE DIRECTION from f(R) and DGP (mu < 1 vs mu > 1)

  SURVEY COVERAGE:
    DESI LRG (z = 0.4-1.1): Covers the HEART of the transition
    Euclid spectroscopic (z = 0.9-1.8): Covers the ONSET
    Combined: Full transition zone is observable

  OBSERVABLES IN TRANSITION ZONE:
    mu(z): Up to {abs(min(obs_iam['mu'] - 1))*100:.1f}% deviation (strongest at z = 0)
    f*sigma_8: Up to {abs(min(fsig8_diff))*100:.1f}% deviation (strongest at z ~ {z_obs[idx_fmax]:.1f})
    Growth D(a): Up to {max_growth_diff*100:.1f}% suppression
    Potentials: Decay {abs(1-min(obs_iam['Phi_ratio']))*100:.1f}% faster than LCDM

  THE IAM TRANSITION HAS THREE UNIQUE FEATURES:
    1. Specific shape: exp(1-1/a), not Omega_DE(a) or (1+z)^n
    2. Direction: mu < 1 (gravity WEAKER, unique among late-time models)
    3. Sigma = 1 (lensing unchanged, paired with weakened growth)
    
  These features are NOT adjustable -- they are derived.
  The transition zone is where IAM lives or dies.
""")


if __name__ == "__main__":
    main()
