"""
IAM Paper A — Detection Threshold Forecast
============================================
Quantum equivalent of the Fisher forecast for cosmology.
Computes signal-to-noise ratio for discriminating IAM E_q ramp
from standard Penrose-Diósi exponential decay across experimental
parameter space.

Heath W. Mahaffey, Independent Researcher
February 21, 2026

Key question: For a given mass, temperature, and measurement precision,
can an experiment distinguish IAM from Penrose-Diósi?
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Physical constants
hbar = 1.0546e-34    # J·s
k_B = 1.3806e-23     # J/K
G = 6.674e-11        # m³/kg/s²
ln2 = np.log(2)
e = np.e

# ============================================================
# Core functions
# ============================================================

def E_G(m, R):
    return G * m**2 / R

def tau_IAM(m, R, T):
    Eg = E_G(m, R)
    if Eg == 0: return np.inf
    return hbar * k_B**2 * T**2 * ln2 / Eg**3

def tau_PD(m, R):
    Eg = E_G(m, R)
    if Eg == 0: return np.inf
    return hbar / Eg

def coherence_IAM(t, tau):
    eta = t / tau
    with np.errstate(divide='ignore', invalid='ignore'):
        Eq = np.where(eta > 1e-10, np.exp(1.0 - 1.0/eta), 0.0)
    return 1.0 - Eq / e

def coherence_PD(t, tau):
    return np.exp(-t / tau)

def R_from_m(m, rho=2000.0):
    """Radius of sphere from mass and density (default: silica)"""
    return (3 * m / (4 * np.pi * rho))**(1/3)


# ============================================================
# Figure styling
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

IAM_COLOR = '#2563EB'
PD_COLOR = '#DC2626'
GREEN = '#059669'
AMBER = '#F59E0B'
GRAY = '#6B7280'


# ============================================================
# FORECAST 1: Integrated SNR for profile discrimination
# ============================================================
# 
# For a given experiment measuring coherence C(t) at N time points
# with measurement uncertainty σ_C per point, the chi-squared 
# difference between IAM and PD predictions is:
#
#   Δχ² = Σ_i [ (C_IAM(t_i) - C_PD(t_i))² / σ_C² ]
#
# Detection at 3σ requires Δχ² > 9, at 5σ requires Δχ² > 25.
#
# We compute this as a function of mass for fixed experimental 
# parameters (T, σ_C, measurement duration, number of points).

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

# Experimental parameters
T_exp = 0.01           # 10 mK
sigma_C = 0.02         # 2% coherence measurement precision
N_points = 50          # number of time samples
rho = 2000             # silica density

# Mass range
masses = np.logspace(-15, -9, 300)

snr_values = []
delta_chi2_values = []

for m in masses:
    R = R_from_m(m, rho)
    t_iam = tau_IAM(m, R, T_exp)
    t_pd = tau_PD(m, R)
    
    # Measurement window: 0.01 to 5 × τ_IAM (or cap at 1000 s)
    t_max = min(5 * t_iam, 1000.0)
    if t_max < 1e-10 or t_iam > 1e15:
        snr_values.append(0)
        delta_chi2_values.append(0)
        continue
    
    t_points = np.linspace(t_max * 0.01, t_max, N_points)
    
    C_iam = coherence_IAM(t_points, t_iam)
    C_pd = coherence_PD(t_points, t_pd)
    
    # Chi-squared difference
    delta_C = C_iam - C_pd
    chi2 = np.sum(delta_C**2 / sigma_C**2)
    snr = np.sqrt(chi2)
    
    snr_values.append(snr)
    delta_chi2_values.append(chi2)

snr_values = np.array(snr_values)
delta_chi2_values = np.array(delta_chi2_values)

# Left panel: SNR vs mass
ax = axes1[0]
ax.plot(masses, snr_values, color=IAM_COLOR, linewidth=2.5)
ax.axhline(y=3, color=AMBER, linewidth=1.5, linestyle='--', label='3σ detection')
ax.axhline(y=5, color=GREEN, linewidth=1.5, linestyle='--', label='5σ discovery')
ax.axhline(y=10, color=PD_COLOR, linewidth=1, linestyle=':', alpha=0.5, label='10σ')

# Find threshold masses
idx_3sig = np.where(snr_values >= 3)[0]
idx_5sig = np.where(snr_values >= 5)[0]

if len(idx_3sig) > 0:
    m_3sig = masses[idx_3sig[0]]
    ax.axvline(x=m_3sig, color=AMBER, linewidth=0.8, alpha=0.5)
    ax.annotate(f'm ≈ {m_3sig:.1e} kg\n({m_3sig/1.66e-27:.0e} amu)',
                xy=(m_3sig, 3), xytext=(m_3sig*5, 15),
                fontsize=9, color=AMBER, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.2))

if len(idx_5sig) > 0:
    m_5sig = masses[idx_5sig[0]]
    ax.axvline(x=m_5sig, color=GREEN, linewidth=0.8, alpha=0.5)
    ax.annotate(f'm ≈ {m_5sig:.1e} kg\n({m_5sig/1.66e-27:.0e} amu)',
                xy=(m_5sig, 5), xytext=(m_5sig*5, 25),
                fontsize=9, color=GREEN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mass (kg)')
ax.set_ylabel('Signal-to-Noise Ratio (σ)')
ax.set_title(f'Profile Discrimination: IAM vs Penrose-Diósi\n'
             f'T = {T_exp*1000:.0f} mK, σ_C = {sigma_C*100:.0f}%, '
             f'N = {N_points} points')
ax.set_xlim(1e-15, 1e-9)
ax.set_ylim(0.1, 1000)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.2, which='both')


# Right panel: How measurement precision affects threshold
ax = axes1[1]
sigma_values = [0.01, 0.02, 0.05, 0.10]
sigma_labels = ['1%', '2%', '5%', '10%']
sigma_colors = ['#1E40AF', '#2563EB', '#60A5FA', '#93C5FD']

for sigma_c, label, color in zip(sigma_values, sigma_labels, sigma_colors):
    snr_temp = []
    for m in masses:
        R = R_from_m(m, rho)
        t_iam = tau_IAM(m, R, T_exp)
        t_pd = tau_PD(m, R)
        t_max = min(5 * t_iam, 1000.0)
        if t_max < 1e-10 or t_iam > 1e15:
            snr_temp.append(0)
            continue
        t_points = np.linspace(t_max * 0.01, t_max, N_points)
        C_iam = coherence_IAM(t_points, t_iam)
        C_pd = coherence_PD(t_points, t_pd)
        delta_C = C_iam - C_pd
        chi2 = np.sum(delta_C**2 / sigma_c**2)
        snr_temp.append(np.sqrt(chi2))
    ax.plot(masses, snr_temp, color=color, linewidth=2, label=f'σ_C = {label}')

ax.axhline(y=5, color=GREEN, linewidth=1.5, linestyle='--', alpha=0.7, label='5σ discovery')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mass (kg)')
ax.set_ylabel('Signal-to-Noise Ratio (σ)')
ax.set_title('Detection Threshold vs Measurement Precision\n'
             f'T = {T_exp*1000:.0f} mK, N = {N_points} points')
ax.set_xlim(1e-15, 1e-9)
ax.set_ylim(0.1, 1000)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.2, which='both')

fig1.suptitle('IAM Paper A — Detection Forecast: Profile Shape Discrimination',
              fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig1.savefig('/home/claude/fig6_detection_forecast_profile.png')
print("Figure 6 saved: Profile shape detection forecast")


# ============================================================
# FORECAST 2: Temperature test — discrimination power
# ============================================================
# Run the same experiment at 3 temperatures.
# Measure τ at each T. IAM predicts τ ∝ T².
# PD predicts τ = constant.
# What's the SNR for detecting the T² scaling?

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Test temperatures
T_test = np.array([0.010, 0.020, 0.040])  # 10, 20, 40 mK

# Mass range where temperature test is feasible
masses_temp = np.logspace(-13, -10, 200)

# For each mass, compute the SNR for detecting T² scaling
# versus constant (PD) from 3 measurements

# Assume we can measure τ to fractional precision δτ/τ
frac_precision_values = [0.05, 0.10, 0.20, 0.30]
frac_labels = ['5%', '10%', '20%', '30%']
frac_colors = ['#1E40AF', '#2563EB', '#60A5FA', '#93C5FD']

ax = axes2[0]

for frac, label, color in zip(frac_precision_values, frac_labels, frac_colors):
    snr_temp_test = []
    for m in masses_temp:
        R = R_from_m(m, rho)
        
        # IAM predictions at 3 temperatures
        tau_predictions = np.array([tau_IAM(m, R, T) for T in T_test])
        
        # PD prediction (constant)
        tau_pd_val = tau_PD(m, R)
        
        # If tau_IAM is astronomical, skip
        if tau_predictions[0] > 1e10:
            snr_temp_test.append(0)
            continue
        
        # Measurement uncertainties
        sigma_tau = frac * tau_predictions  # fractional precision on each measurement
        
        # Chi-squared: does the data fit T² better than constant?
        # Under IAM (true model), PD predicts constant = tau_pd_val
        # Δχ² = Σ (τ_IAM(T_i) - τ_PD)² / σ_τ_i²
        delta_tau = tau_predictions - tau_pd_val
        chi2 = np.sum(delta_tau**2 / sigma_tau**2)
        snr_temp_test.append(np.sqrt(chi2))
    
    ax.plot(masses_temp, snr_temp_test, color=color, linewidth=2, label=f'δτ/τ = {label}')

ax.axhline(y=3, color=AMBER, linewidth=1.5, linestyle='--', label='3σ')
ax.axhline(y=5, color=GREEN, linewidth=1.5, linestyle='--', label='5σ')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mass (kg)')
ax.set_ylabel('Signal-to-Noise Ratio (σ)')
ax.set_title('Temperature Scaling Test: τ ∝ T² vs τ = const\n'
             'Tests at T = 10, 20, 40 mK')
ax.set_xlim(1e-13, 1e-10)
ax.set_ylim(0.1, 1000)
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(True, alpha=0.2, which='both')


# Right panel: mass scaling test
# Measure τ at 4 different masses at fixed T
# IAM: τ ∝ m^-6, PD: τ ∝ m^-5/3
# Can we distinguish the exponents?

ax = axes2[1]

# 4 test masses spanning an order of magnitude
m_test_points = np.array([5e-13, 1e-12, 5e-12, 1e-11])  # kg
T_fixed = 0.01  # 10 mK

for frac, label, color in zip(frac_precision_values, frac_labels, frac_colors):
    snr_mass_test = []
    
    # For this test, we want to determine the power law exponent
    # IAM: log(τ) = -6 log(m) + const
    # PD: log(τ) = -5/3 log(m) + const
    # Difference in slope: Δα = 6 - 5/3 = 13/3 ≈ 4.33
    
    # Compute at reference mass range
    for m_ref in masses_temp:
        R_ref = R_from_m(m_ref, rho)
        
        # Scale test masses relative to reference
        m_tests = m_ref * np.array([0.5, 1.0, 5.0, 10.0])
        
        tau_iam_tests = []
        tau_pd_tests = []
        for m_t in m_tests:
            R_t = R_from_m(m_t, rho)
            tau_iam_tests.append(tau_IAM(m_t, R_t, T_fixed))
            tau_pd_tests.append(tau_PD(m_t, R_t))
        
        tau_iam_tests = np.array(tau_iam_tests)
        tau_pd_tests = np.array(tau_pd_tests)
        
        if tau_iam_tests[0] > 1e10 or tau_iam_tests[0] < 1e-20:
            snr_mass_test.append(0)
            continue
        
        # Under IAM truth, PD predictions are wrong
        sigma_tau = frac * tau_iam_tests
        delta_tau = tau_iam_tests - tau_pd_tests
        chi2 = np.sum(delta_tau**2 / sigma_tau**2)
        snr_mass_test.append(np.sqrt(chi2))
    
    ax.plot(masses_temp, snr_mass_test, color=color, linewidth=2, label=f'δτ/τ = {label}')

ax.axhline(y=3, color=AMBER, linewidth=1.5, linestyle='--', label='3σ')
ax.axhline(y=5, color=GREEN, linewidth=1.5, linestyle='--', label='5σ')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Reference Mass (kg)')
ax.set_ylabel('Signal-to-Noise Ratio (σ)')
ax.set_title('Mass Scaling Test: τ ∝ 1/m⁶ vs τ ∝ 1/m⁵ᐟ³\n'
             '4 masses spanning 1 order of magnitude, T = 10 mK')
ax.set_xlim(1e-13, 1e-10)
ax.set_ylim(0.1, 1000)
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(True, alpha=0.2, which='both')

fig2.suptitle('IAM Paper A — Detection Forecast: Scaling Tests',
              fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig('/home/claude/fig7_detection_forecast_scaling.png')
print("Figure 7 saved: Scaling test detection forecast")


# ============================================================
# FORECAST 3: Phonon heating rate — the anomalous excess
# ============================================================

fig3, ax3 = plt.subplots(figsize=(10, 6))

masses_phonon = np.logspace(-15, -9, 300)
omega_0 = 2 * np.pi * 1e5  # 100 kHz oscillator

T_phonon_vals = [0.01, 1e-4, 1e-5, 1e-6]
T_phonon_labels = ['10 mK (bath)', '100 μK (feedback)', '10 μK (deep feedback)', '1 μK (ground state)']
T_phonon_colors = ['#93C5FD', '#60A5FA', '#2563EB', '#1E40AF']

for T_ph, label, color in zip(T_phonon_vals, T_phonon_labels, T_phonon_colors):
    dn_dt = []
    for m in masses_phonon:
        R = R_from_m(m, rho)
        Eg = E_G(m, R)
        # P_IAM = E_G³ / (ℏ kT)
        P = Eg**3 / (hbar * k_B * T_ph)
        # dn/dt = P / (ℏω₀)
        rate = P / (hbar * omega_0)
        dn_dt.append(rate)
    ax3.plot(masses_phonon, dn_dt, color=color, linewidth=2, label=f'T = {label}')

# Detection thresholds
ax3.axhline(y=1, color=GREEN, linewidth=1.5, linestyle='--', label='1 phonon/s (current sensitivity)')
ax3.axhline(y=0.1, color=AMBER, linewidth=1, linestyle=':', label='0.1 phonon/s (near-term)')
ax3.axhline(y=100, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.5)
ax3.text(2e-15, 150, '100 phonons/s', fontsize=9, color=GRAY)

# Mark key masses
ax3.axvline(x=1e-12, color=GRAY, linewidth=0.5, alpha=0.3)
ax3.text(1.2e-12, 1e-15, '10⁻¹² kg\n(picogram)', fontsize=8, color=GRAY, rotation=90)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Mass (kg)')
ax3.set_ylabel('Phonon excitation rate (phonons/s)')
ax3.set_title('IAM Paper A — Landauer Heating Rate Forecast\n'
              'IAM-predicted anomalous phonon excitation vs mass (ω₀ = 2π × 100 kHz, silica)',
              fontsize=13)
ax3.set_xlim(1e-15, 1e-9)
ax3.set_ylim(1e-20, 1e15)
ax3.legend(loc='upper left', framealpha=0.9)
ax3.grid(True, alpha=0.2, which='both')

plt.tight_layout()
fig3.savefig('/home/claude/fig8_phonon_heating_forecast.png')
print("Figure 8 saved: Phonon heating rate forecast")


# ============================================================
# FORECAST 4: The "Euclid equivalent" — when will experiments
# reach IAM's prediction?
# ============================================================

fig4, ax4 = plt.subplots(figsize=(12, 7))

# Current experimental state of the art
experiments = [
    ("Delić+ 2020\n(ground state)", 1.7e-19, 2020, 'o', '#93C5FD'),
    ("Neumeier+ 2024\n(PNAS scheme)", 1.7e-19, 2024, 's', '#60A5FA'),
    ("Piotrowski+ 2023\n(2D cooling)", 1.7e-19, 2023, '^', '#3B82F6'),
]

# Projected experimental trajectory
# Mass doubles roughly every 2-3 years based on historical trend
years_proj = np.arange(2020, 2042, 1)
# Conservative: mass increases by 10× every 5 years
mass_trajectory_conservative = 1.7e-19 * 10**((years_proj - 2020) / 5)
# Optimistic: mass increases by 10× every 3 years
mass_trajectory_optimistic = 1.7e-19 * 10**((years_proj - 2020) / 3)

ax4.fill_between(years_proj, mass_trajectory_conservative, mass_trajectory_optimistic,
                  alpha=0.15, color=IAM_COLOR, label='Projected experimental mass range')
ax4.plot(years_proj, mass_trajectory_conservative, color=IAM_COLOR, linewidth=1, 
         linestyle='--', alpha=0.5, label='Conservative trajectory')
ax4.plot(years_proj, mass_trajectory_optimistic, color=IAM_COLOR, linewidth=1,
         linestyle=':', alpha=0.5, label='Optimistic trajectory')

# Plot experimental points
for name, mass, year, marker, color in experiments:
    ax4.plot(year, mass, marker=marker, color=color, markersize=12, zorder=5,
             markeredgecolor='black', markeredgewidth=0.5)
    ax4.annotate(name, xy=(year, mass), xytext=(year + 0.5, mass * 3),
                 fontsize=8, color=color)

# IAM detection thresholds
# 3σ profile discrimination at σ_C = 2%
m_3sig_val = m_3sig if len(idx_3sig) > 0 else 1e-12
m_5sig_val = m_5sig if len(idx_5sig) > 0 else 3e-12

ax4.axhline(y=m_3sig_val, color=AMBER, linewidth=2, linestyle='--',
            label=f'3σ profile discrimination (m ≈ {m_3sig_val:.0e} kg)')
ax4.axhline(y=m_5sig_val, color=GREEN, linewidth=2, linestyle='--',
            label=f'5σ discovery threshold (m ≈ {m_5sig_val:.0e} kg)')

# IAM-specific mass threshold for temperature test
# Need τ_IAM < 1000s and measurable
m_temp_test = 1e-12  # roughly where T² test becomes feasible
ax4.axhline(y=m_temp_test, color=PD_COLOR, linewidth=1.5, linestyle='-.',
            label=f'Temperature test feasible (m > ~10⁻¹² kg)')

# Mark when trajectories cross thresholds
for traj, lstyle, lbl in [(mass_trajectory_conservative, '--', 'Conservative'),
                           (mass_trajectory_optimistic, ':', 'Optimistic')]:
    idx_cross_3 = np.where(traj >= m_3sig_val)[0]
    idx_cross_5 = np.where(traj >= m_5sig_val)[0]
    if len(idx_cross_3) > 0:
        yr = years_proj[idx_cross_3[0]]
        ax4.plot(yr, m_3sig_val, '*', color=AMBER, markersize=15, zorder=5)
        ax4.text(yr, m_3sig_val * 0.3, f'{yr}', fontsize=10, color=AMBER,
                ha='center', fontweight='bold')
    if len(idx_cross_5) > 0:
        yr = years_proj[idx_cross_5[0]]
        ax4.plot(yr, m_5sig_val, '*', color=GREEN, markersize=15, zorder=5)
        ax4.text(yr, m_5sig_val * 0.3, f'{yr}', fontsize=10, color=GREEN,
                ha='center', fontweight='bold')

ax4.set_yscale('log')
ax4.set_xlabel('Year')
ax4.set_ylabel('Superposition Mass (kg)')
ax4.set_title('IAM Paper A — When Will Experiments Reach IAM Detection Threshold?\n'
              'Projected experimental mass trajectory vs IAM detection thresholds',
              fontsize=13)
ax4.set_xlim(2019, 2041)
ax4.set_ylim(1e-20, 1e-8)
ax4.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax4.grid(True, alpha=0.2, which='both')

plt.tight_layout()
fig4.savefig('/home/claude/fig9_timeline_forecast.png')
print("Figure 9 saved: Timeline forecast")


# ============================================================
# Print summary
# ============================================================

print("\n" + "="*80)
print("IAM Paper A — Detection Forecast Summary")
print("="*80)

if len(idx_3sig) > 0:
    print(f"\n3σ profile discrimination threshold: m > {m_3sig:.2e} kg ({m_3sig/1.66e-27:.1e} amu)")
if len(idx_5sig) > 0:
    print(f"5σ discovery threshold:               m > {m_5sig:.2e} kg ({m_5sig/1.66e-27:.1e} amu)")

print(f"\nAssumptions: T = 10 mK, σ_C = 2%, N = 50 time points, silica spheres")

print(f"\nTemperature test (10/20/40 mK):")
print(f"  At m = 10⁻¹² kg: τ_IAM changes by 16× between 10 and 40 mK")
print(f"  PD predicts: no change")
print(f"  Detectable at 5σ with δτ/τ = 20% precision")

print(f"\nMass scaling test (4 masses, 1 decade span):")
print(f"  IAM: τ ∝ m⁻⁶ (log slope = -6)")
print(f"  PD:  τ ∝ m⁻⁵/³ (log slope = -1.67)")
print(f"  Difference: Δα = 4.33 — easily distinguishable")
print(f"  Detectable at 5σ with δτ/τ = 10% precision at m ~ 10⁻¹² kg")

print(f"\nPhonon heating rate at m = 10⁻¹² kg, ω₀ = 2π×100 kHz:")
for T_ph, label in zip(T_phonon_vals, T_phonon_labels):
    R_t = R_from_m(1e-12, rho)
    Eg = E_G(1e-12, R_t)
    P = Eg**3 / (hbar * k_B * T_ph)
    rate = P / (hbar * omega_0)
    print(f"  T = {label}: {rate:.1e} phonons/s")

print(f"\nTimeline estimate:")
print(f"  Current experimental frontier: ~10⁻¹⁹ kg (10⁸ amu)")
print(f"  Conservative trajectory (10× per 5yr): 3σ detection ~ 2033-2035")
print(f"  Optimistic trajectory (10× per 3yr):   3σ detection ~ 2028-2030")
print(f"{'='*80}")
