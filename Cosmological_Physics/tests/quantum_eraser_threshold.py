"""
IAM Paper B — Quantum Eraser Threshold Formalization
=====================================================
The central prediction: quantum erasure succeeds if and only if
the which-path interaction is thermodynamically reversible.

IAM defines "irreversible" precisely via Landauer's principle:
an interaction is irreversible if it produces entropy ΔS ≥ k_B ln 2
(one bit of which-path information encoded in the environment).

This script derives the threshold, computes it for real experimental
systems, and produces publication-quality figures.

Heath W. Mahaffey, Independent Researcher
February 21, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ============================================================
# Constants
# ============================================================
hbar = 1.0546e-34   # J·s
k_B = 1.3806e-23    # J/K
c = 2.998e8          # m/s
G = 6.674e-11        # m³/kg/s²
ln2 = np.log(2)
eV = 1.602e-19       # J

# ============================================================
# SECTION 1: The Landauer Threshold
# ============================================================
print("=" * 70)
print("QUANTUM ERASER THRESHOLD — IAM PREDICTION")
print("=" * 70)

print("""
THE CORE ARGUMENT:
==================

In the double slit, a photon passes through two slits and hits a screen.
No which-path info → interference pattern (Σ = 1 sector, coherent).
Which-path info → no interference (sector crossing to μ < 1).

The quantum eraser "erases" which-path info and restores interference.
Standard QM: erasure always works if the which-path marking is 
  entangled with an ancilla that can be measured in a conjugate basis.
IAM: erasure works ONLY if the which-path interaction was 
  thermodynamically reversible (ΔS < k_B ln 2).

The threshold is Landauer's bound:
  
  Q_irrev = k_B T ln 2

If the which-path detector dissipates energy Q > Q_irrev into the
environment, one bit of information has been irreversibly encoded.
The entropy is produced. The sector crossing has occurred.
Erasure cannot undo it.

If Q < Q_irrev, the interaction is thermodynamically reversible.
No genuine information was produced. The photon never left the 
Σ = 1 sector. Erasure reveals coherence that was always there.
""")

# ============================================================
# SECTION 2: Formalization — The Erasure Fidelity Function
# ============================================================
print("=" * 70)
print("SECTION 2: ERASURE FIDELITY AS A FUNCTION OF DISSIPATION")
print("=" * 70)

print("""
Define the erasure fidelity F as the visibility of the recovered
interference pattern after attempted erasure:
  F = 1: perfect interference restored (fully reversible)
  F = 0: no interference (fully irreversible, classical)

In standard QM, F depends only on the entanglement structure —
if the which-path marker can be rotated to erase distinguishability,
F = 1 regardless of energy dissipation.

In IAM, F depends on the thermodynamic irreversibility of the
which-path interaction. We model this using the decoherence
function E_q applied to the which-path detector:

  F_IAM = 1 - E_q(Q / Q_Landauer)

where Q is the energy dissipated by the which-path interaction
and Q_Landauer = k_B T ln 2 is the Landauer threshold.

Using E_q(η) = exp(1 - 1/η) / e (normalized to [0,1]):

  F_IAM(Q, T) = 1 - exp(1 - k_B T ln2 / Q) / e

Properties:
  Q → 0:     F → 1  (no dissipation → fully reversible → erasure works)
  Q = Q_L:   F = 1 - 1/e ≈ 0.632  (at Landauer bound, ~63% fidelity lost)
  Q → ∞:     F → 1 - 1 = 0  (fully irreversible → erasure fails completely)
""")

def F_iam(Q, T):
    """IAM erasure fidelity as function of dissipated energy Q and temperature T"""
    Q_L = k_B * T * ln2
    if Q < 1e-30:
        return 1.0
    eta = Q / Q_L
    if eta < 1e-10:
        return 1.0
    return 1.0 - np.exp(1.0 - 1.0/eta) / np.e

def F_standard(Q, T):
    """Standard QM prediction: fidelity depends only on entanglement, not dissipation"""
    return 1.0  # Always recoverable in principle

# Compute for a range of Q/Q_L ratios
eta_range = np.linspace(0.001, 5.0, 1000)

T_lab = 300      # room temperature
T_cryo = 4.0     # liquid helium
T_mK = 0.010     # dilution fridge

Q_L_300 = k_B * T_lab * ln2
Q_L_4 = k_B * T_cryo * ln2
Q_L_mK = k_B * T_mK * ln2

print(f"Landauer threshold Q_L = k_B T ln 2:")
print(f"  At 300 K:   Q_L = {Q_L_300:.3e} J = {Q_L_300/eV:.4e} eV")
print(f"  At 4 K:     Q_L = {Q_L_4:.3e} J = {Q_L_4/eV:.4e} eV")
print(f"  At 10 mK:   Q_L = {Q_L_mK:.3e} J = {Q_L_mK/eV:.4e} eV")
print()

# ============================================================
# SECTION 3: Real Detector Systems — Where Do They Fall?
# ============================================================
print("=" * 70)
print("SECTION 3: REAL WHICH-PATH DETECTORS")
print("=" * 70)

detectors = []

# --- Type 1: Photon polarization rotation (standard eraser) ---
# A half-wave plate rotates polarization. This is a unitary operation.
# Energy dissipated: essentially zero (photon passes through crystal)
# The crystal absorbs negligible energy from a single photon.
Q_pol = 1e-30  # effectively zero
detectors.append({
    'name': 'Polarization rotation\n(half-wave plate)',
    'Q': Q_pol,
    'Q_over_QL': Q_pol / Q_L_300,
    'T': T_lab,
    'F_iam': 1.0,
    'type': 'reversible',
    'notes': 'Unitary rotation, no dissipation'
})

# --- Type 2: BBO crystal entanglement (SPDC) ---
# Spontaneous parametric down-conversion entangles signal/idler.
# The process is coherent — the crystal mediates but doesn't absorb.
# Dissipation: negligible (< 10⁻²² J per photon pair)
Q_bbo = 1e-22
detectors.append({
    'name': 'BBO crystal SPDC\n(entanglement source)',
    'Q': Q_bbo,
    'Q_over_QL': Q_bbo / Q_L_300,
    'T': T_lab,
    'F_iam': F_iam(Q_bbo, T_lab),
    'type': 'reversible',
    'notes': 'Coherent parametric process'
})

# --- Type 3: Beam splitter path marking ---
# A beam splitter directs photon to one of two paths.
# No energy absorbed by the beam splitter itself.
Q_bs = 1e-28
detectors.append({
    'name': 'Beam splitter\n(path marking)',
    'Q': Q_bs,
    'Q_over_QL': Q_bs / Q_L_300,
    'T': T_lab,
    'F_iam': 1.0,
    'type': 'reversible',
    'notes': 'Unitary scattering, no dissipation'
})

# --- Type 4: Atomic which-path detector ---
# An atom placed at one slit gets excited if the photon passes.
# The excitation energy is ~1 eV. If the atom spontaneously emits
# before erasure is attempted, the information is radiated into 
# the environment irreversibly.
# Spontaneous emission: Γ ~ 10⁸/s for optical transitions
# Dissipation: E_photon ~ 1 eV IF spontaneous emission occurs
Q_atom_no_emission = 0  # before emission: reversible (stimulated)
Q_atom_emitted = 1.0 * eV  # after emission: irreversible
detectors.append({
    'name': 'Atomic detector\n(before emission)',
    'Q': Q_atom_no_emission,
    'Q_over_QL': 0,
    'T': T_lab,
    'F_iam': 1.0,
    'type': 'reversible',
    'notes': 'Excited state is reversible until emission'
})
detectors.append({
    'name': 'Atomic detector\n(after spont. emission)',
    'Q': Q_atom_emitted,
    'Q_over_QL': Q_atom_emitted / Q_L_300,
    'T': T_lab,
    'F_iam': F_iam(Q_atom_emitted, T_lab),
    'type': 'irreversible',
    'notes': 'Photon radiated into environment'
})

# --- Type 5: Photodiode / CCD detector ---
# Photon absorbed, electron-hole pair created, current flows.
# Dissipation: full photon energy + readout electronics
# For visible photon: ~2 eV + amplification heat
Q_photodiode = 3.0 * eV  # photon + electronics
detectors.append({
    'name': 'Photodiode / CCD\n(electronic detection)',
    'Q': Q_photodiode,
    'Q_over_QL': Q_photodiode / Q_L_300,
    'T': T_lab,
    'F_iam': F_iam(Q_photodiode, T_lab),
    'type': 'irreversible',
    'notes': 'Photon absorbed, electron cascade'
})

# --- Type 6: Mechanical detector (nanomembrane) ---
# Photon kicks a mechanical resonator, depositing momentum.
# At room T, thermal fluctuations are >> single photon kick.
# But the momentum transfer is real and dissipates into phonons.
# Dissipation: (ℏk)²/(2m) ~ 10⁻²⁸ J for visible photon on 1pg membrane
m_membrane = 1e-12  # 1 pg
k_photon = 2 * np.pi / 500e-9  # visible photon
Q_mechanical = (hbar * k_photon)**2 / (2 * m_membrane)
detectors.append({
    'name': 'Nanomembrane kick\n(mechanical detector)',
    'Q': Q_mechanical,
    'Q_over_QL': Q_mechanical / Q_L_300,
    'T': T_lab,
    'F_iam': F_iam(Q_mechanical, T_lab),
    'type': 'borderline',
    'notes': f'Recoil energy = {Q_mechanical:.2e} J'
})

# --- Type 7: Stern-Gerlach spin measurement ---
# Magnetic field gradient entangles spin with position.
# The field is conservative — no dissipation until detection.
Q_sg = 0
detectors.append({
    'name': 'Stern-Gerlach\n(before screen)',
    'Q': Q_sg,
    'Q_over_QL': 0,
    'T': T_lab,
    'F_iam': 1.0,
    'type': 'reversible',
    'notes': 'Conservative field, no dissipation'
})

# --- Type 8: Fluorescence detection ---
# Atom/molecule excited and fluoresces. Photon emitted into 4π.
# Completely irreversible once the photon is in the field modes.
Q_fluor = 2.5 * eV
detectors.append({
    'name': 'Fluorescence\n(photon into environment)',
    'Q': Q_fluor,
    'Q_over_QL': Q_fluor / Q_L_300,
    'T': T_lab,
    'F_iam': F_iam(Q_fluor, T_lab),
    'type': 'irreversible',
    'notes': 'Photon radiated into 4π steradians'
})

print(f"{'Detector':<32s} {'Q (J)':<12s} {'Q/Q_L':<12s} {'F_IAM':<8s} {'F_QM':<8s} {'Type':<15s}")
print("-" * 90)
for d in detectors:
    q_str = f"{d['Q']:.1e}" if d['Q'] > 0 else "0"
    ql_str = f"{d['Q_over_QL']:.1e}" if d['Q_over_QL'] > 0 else "0"
    f_str = f"{d['F_iam']:.4f}"
    print(f"{d['name'].replace(chr(10),' '):<32s} {q_str:<12s} {ql_str:<12s} {f_str:<8s} {'1.000':<8s} {d['type']:<15s}")

# ============================================================
# SECTION 4: The Critical Prediction
# ============================================================
print()
print("=" * 70)
print("SECTION 4: THE CRITICAL PREDICTION")
print("=" * 70)

print("""
Standard QM says: All these erasers work (F = 1) as long as the 
which-path information is entangled with an accessible degree of 
freedom. Dissipation is irrelevant. Even the photodiode can be 
"erased" in principle by measuring the electron in a conjugate basis.

IAM says: Only the reversible detectors allow erasure (F ≈ 1).
Once dissipation exceeds Q_L = k_B T ln 2:
  - The information has been irreversibly encoded
  - Landauer's principle says the entropy cannot be reversed
  - The sector crossing (Σ=1 → μ<1) has occurred
  - Erasure FAILS regardless of what you do to the ancilla

THE TEST:
=========
Build TWO versions of the same quantum eraser experiment:

Version A (Reversible): Which-path marking via polarization rotation
  or coherent atomic excitation (no spontaneous emission allowed).
  Q << Q_L.  
  Both QM and IAM predict: erasure succeeds, F ≈ 1.

Version B (Irreversible): Which-path marking via fluorescence detection 
  or photodiode absorption. Q >> Q_L.
  Standard QM predicts: erasure still succeeds (F = 1) if ancilla is 
  measured in conjugate basis.
  IAM predicts: erasure FAILS (F ≈ 0).

Same photon source. Same slits. Same screen. Same coincidence counting.
Only the which-path detector changes.

If Version A shows interference and Version B does not, even with 
proper conjugate-basis erasure of the ancilla, IAM is confirmed.

If both show interference, IAM is falsified at this level.
""")

# ============================================================
# SECTION 5: The Atomic Timing Test
# ============================================================
print("=" * 70)
print("SECTION 5: THE ATOMIC TIMING TEST (MOST ELEGANT)")
print("=" * 70)

print("""
The most elegant version uses an ATOMIC which-path detector with 
controllable spontaneous emission timing.

Setup: Atom at slit A. Photon passes → atom excited.
- If atom is in a high-Q cavity: spontaneous emission suppressed.
  The excitation is coherent and reversible. Q < Q_L.
  Erasure should work (both QM and IAM agree).

- If atom is in free space: spontaneous emission occurs in ~10 ns.
  Once the photon is emitted into the environment: Q > Q_L.
  IAM says erasure should fail after emission.
  Standard QM says erasure should still work (the emitted photon 
  is the ancilla; measuring it in a conjugate basis erases which-path).

The KEY: vary the delay between which-path marking and erasure attempt.

Delay < τ_emission (atom hasn't emitted yet):
  Both QM and IAM predict erasure works.

Delay > τ_emission (atom has emitted):
  Standard QM: erasure works if emitted photon is captured and 
  measured conjugately.
  IAM: erasure FAILS because the emission was irreversible.
  The Landauer entropy was produced. The sector crossing occurred.

Plot interference visibility vs delay time.
QM predicts: flat (visibility independent of delay).
IAM predicts: step function dropping at τ_emission.
""")

# Spontaneous emission rate for typical optical transition
gamma_spont = 1e8  # 1/s (typical for optical)
tau_emission = 1.0 / gamma_spont

print(f"Typical spontaneous emission time: τ = {tau_emission*1e9:.1f} ns")
print(f"Landauer threshold at 300 K: Q_L = {Q_L_300:.3e} J = {Q_L_300/eV:.4e} eV")
print(f"Typical photon energy: E_photon = 2 eV = {2*eV:.3e} J")
print(f"E_photon / Q_L = {2*eV/Q_L_300:.0f} (>> 1, deeply irreversible)")
print()

# ============================================================
# SECTION 6: Entanglement entropy vs Landauer entropy
# ============================================================
print("=" * 70)
print("SECTION 6: WHY STANDARD QM AND IAM DISAGREE")
print("=" * 70)

print("""
The disagreement is fundamental and traces to what "information" means.

Standard QM: Information is entanglement entropy. When photon A is 
entangled with detector B, the reduced state of A has von Neumann 
entropy S = -Tr(ρ_A log ρ_A). This entropy is "reversible" — measuring 
B in a conjugate basis disentangles A, restoring S_A = 0.
The key assumption: all processes are fundamentally unitary.
Information is never truly lost, only redistributed.

IAM: Information includes Landauer entropy — the thermodynamic cost 
of irreversible physical processes. When detector B dissipates energy 
Q > k_B T ln 2, a bit of information has been irreversibly encoded 
in the thermal bath. This is NOT the same as entanglement entropy.
It is genuine thermodynamic entropy. It cannot be reversed by any 
measurement on the ancilla because the entropy increase is physical,
not merely informational.

The distinction maps directly to the sector structure:

  Entanglement entropy (reversible) → Σ = 1 sector
  System remains coherent. No sector crossing. 
  Erasure reveals existing coherence.

  Landauer entropy (irreversible) → μ < 1 sector  
  System has decohered. Sector crossing occurred.
  Erasure cannot undo physical entropy production.

This is the same distinction that operates at cosmological scales:
photons (Σ = 1) don't produce Landauer entropy through gravitational
decoherence. Matter (μ < 1) does. The sector structure is fundamentally
about the distinction between reversible and irreversible information.
""")

# ============================================================
# SECTION 7: Quantitative erasure curves
# ============================================================
print("=" * 70)
print("SECTION 7: NUMERICAL PREDICTIONS")
print("=" * 70)

# The erasure fidelity as a function of Q/Q_L
eta_fine = np.linspace(0.001, 10.0, 10000)
F_curve = np.array([F_iam(e * Q_L_300, T_lab) for e in eta_fine])

# Find key thresholds
# 50% fidelity loss
idx_50 = np.argmin(np.abs(F_curve - 0.5))
eta_50 = eta_fine[idx_50]

# 90% fidelity loss  
idx_90 = np.argmin(np.abs(F_curve - 0.1))
eta_90 = eta_fine[idx_90]

# 99% fidelity loss
idx_99 = np.argmin(np.abs(F_curve - 0.01))
eta_99 = eta_fine[idx_99]

print(f"Erasure fidelity thresholds (Q/Q_L values):")
print(f"  F = 0.50 (50% visibility):  Q/Q_L = {eta_50:.3f}")
print(f"  F = 0.10 (10% visibility):  Q/Q_L = {eta_90:.3f}")
print(f"  F = 0.01 (1% visibility):   Q/Q_L = {eta_99:.3f}")
print(f"  At Q = Q_L (Landauer bound): F = {F_iam(Q_L_300, T_lab):.4f}")
print()

# Temperature dependence of the threshold
print("Temperature dependence of Q_L:")
temps = [0.01, 0.1, 1.0, 4.0, 77.0, 300.0]
for T in temps:
    QL = k_B * T * ln2
    print(f"  T = {T:>6.2f} K:  Q_L = {QL:.3e} J = {QL/eV:.3e} eV")

# ============================================================
# FIGURES
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

IAM_C = '#2563EB'
STD_C = '#DC2626'
THRESH_C = '#059669'

# ============================================================
# FIGURE 1: Erasure fidelity vs dissipation
# ============================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Linear scale
ax1.plot(eta_fine, F_curve, color=IAM_C, lw=2.5, label='IAM prediction')
ax1.axhline(y=1.0, color=STD_C, lw=2.5, ls='--', label='Standard QM (always F=1)')
ax1.axvline(x=1.0, color=THRESH_C, lw=1.5, ls=':', alpha=0.7, label=r'Landauer bound $Q = k_BT\ln 2$')
ax1.set_xlabel(r'Dissipated energy $Q / Q_L$')
ax1.set_ylabel('Erasure fidelity F (interference visibility)')
ax1.set_title('(a) Erasure Fidelity vs Dissipation')
ax1.legend(loc='center right', framealpha=0.9)
ax1.set_xlim(0, 5)
ax1.set_ylim(-0.05, 1.1)
ax1.grid(True, alpha=0.2)

# Shade regions
ax1.axvspan(0, 0.5, alpha=0.06, color=IAM_C)
ax1.axvspan(2, 5, alpha=0.06, color=STD_C)
ax1.text(0.25, 0.15, 'Reversible\nregime\n(both agree)', fontsize=10, ha='center',
         color=IAM_C, fontstyle='italic')
ax1.text(3.5, 0.55, 'Irreversible regime\n(QM and IAM disagree)', fontsize=10, ha='center',
         color=STD_C, fontstyle='italic')

# Mark detector types
detector_markers = [
    (0.01, 'Polarization\nrotation', 'reversible'),
    (0.03, 'Beam\nsplitter', 'reversible'),
    (2.0*eV/Q_L_300, 'Fluorescence', 'irreversible'),
    (3.0*eV/Q_L_300, 'Photodiode', 'irreversible'),
]

for q_ql, label, dtype in detector_markers:
    if q_ql < 5:
        f_val = F_iam(q_ql * Q_L_300, T_lab) if q_ql > 0.001 else 1.0
        marker_color = IAM_C if dtype == 'reversible' else STD_C
        ax1.plot(q_ql, f_val, 'o', color=marker_color, markersize=8, zorder=5)

# Right panel: Log scale showing detector positions
ax2.semilogy(eta_fine[F_curve > 1e-4], 1 - F_curve[F_curve > 1e-4], 
             color=IAM_C, lw=2.5, label='IAM: 1 - F (decoherence)')
ax2.axvline(x=1.0, color=THRESH_C, lw=1.5, ls=':', alpha=0.7, label='Landauer bound')
ax2.set_xlabel(r'Dissipated energy $Q / Q_L$')
ax2.set_ylabel('Decoherence (1 - F)')
ax2.set_title('(b) Decoherence vs Dissipation (log scale)')
ax2.legend(loc='lower right', framealpha=0.9)
ax2.set_xlim(0, 5)
ax2.set_ylim(1e-4, 2)
ax2.grid(True, alpha=0.2)

fig1.suptitle('IAM Paper B — Quantum Eraser: Erasure Fidelity Depends on Thermodynamic Irreversibility\n'
              r'$F_{IAM}(Q) = 1 - \exp(1 - Q_L/Q)/e$, where $Q_L = k_BT\ln 2$',
              fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
fig1.savefig('/home/claude/fig_eraser_fidelity.png')
print("\nFigure 1 saved: Erasure fidelity vs dissipation")

# ============================================================
# FIGURE 2: The atomic timing test prediction
# ============================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Visibility vs delay time
delay = np.linspace(0, 50, 1000)  # in ns
# IAM prediction: step function at τ_emission
# Smoothed by the emission probability: P_emitted(t) = 1 - exp(-t/τ_emission)
tau_ns = tau_emission * 1e9  # in ns
P_emitted = 1 - np.exp(-delay / tau_ns)

# Before emission: Q = 0, F = 1
# After emission: Q = E_photon >> Q_L, F ≈ 0
# Weighted: F(t) = (1 - P_emitted) * 1.0 + P_emitted * F_iam(2*eV, 300)
F_post_emission = F_iam(2*eV, T_lab)
V_iam = (1 - P_emitted) * 1.0 + P_emitted * F_post_emission
V_standard = np.ones_like(delay) * 1.0  # QM: always works

ax1.plot(delay, V_iam, color=IAM_C, lw=2.5, label='IAM prediction')
ax1.plot(delay, V_standard, color=STD_C, lw=2.5, ls='--', label='Standard QM prediction')
ax1.axvline(x=tau_ns, color=THRESH_C, lw=1.5, ls=':', alpha=0.7, 
            label=fr'$\tau_{{emission}}$ = {tau_ns:.0f} ns')
ax1.set_xlabel('Delay before erasure attempt (ns)')
ax1.set_ylabel('Recovered interference visibility')
ax1.set_title('(a) Atomic Which-Path Detector: Visibility vs Delay')
ax1.legend(loc='center right', framealpha=0.9)
ax1.set_xlim(0, 50)
ax1.set_ylim(-0.05, 1.15)
ax1.grid(True, alpha=0.2)
ax1.axvspan(0, tau_ns, alpha=0.06, color=IAM_C)
ax1.text(tau_ns/2, 0.15, 'Before emission\n(both agree: F≈1)', 
         fontsize=10, ha='center', color=IAM_C, fontstyle='italic')
ax1.text(30, 0.55, 'After emission\n(QM: F=1, IAM: F≈0)', 
         fontsize=10, ha='center', color=STD_C, fontstyle='italic')

# Right: Cavity-suppressed vs free-space emission
# In a high-Q cavity, emission rate is suppressed by Purcell factor
purcell_factors = [1, 10, 100, 1000]
colors_p = ['#DC2626', '#F97316', '#EAB308', '#22C55E']

for pf, col in zip(purcell_factors, colors_p):
    tau_cav = tau_ns * pf
    P_em_cav = 1 - np.exp(-delay / tau_cav)
    V_cav = (1 - P_em_cav) * 1.0 + P_em_cav * F_post_emission
    label = f'Purcell factor = {pf}' if pf > 1 else 'Free space'
    ax2.plot(delay, V_cav, color=col, lw=2.0, label=label)

ax2.axhline(y=1.0, color=STD_C, lw=2.0, ls='--', alpha=0.5, label='Standard QM')
ax2.set_xlabel('Delay before erasure attempt (ns)')
ax2.set_ylabel('IAM-predicted visibility')
ax2.set_title('(b) Cavity Control of Spontaneous Emission')
ax2.legend(loc='center right', framealpha=0.9, fontsize=9)
ax2.set_xlim(0, 50)
ax2.set_ylim(-0.05, 1.15)
ax2.grid(True, alpha=0.2)

fig2.suptitle('IAM Paper B — The Atomic Timing Test\n'
              'Erasure fidelity drops when which-path atom spontaneously emits',
              fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
fig2.savefig('/home/claude/fig_eraser_timing.png')
print("Figure 2 saved: Atomic timing test")

# ============================================================
# FIGURE 3: Temperature dependence — the cleanest test
# ============================================================
fig3, ax = plt.subplots(1, 1, figsize=(10, 7))

# Fixed which-path interaction energy (say, 0.1 eV infrared photon)
E_interaction = 0.1 * eV  # 0.1 eV

T_range = np.logspace(-2, 3, 500)  # 10 mK to 1000 K
F_vs_T = []
for T in T_range:
    F_vs_T.append(F_iam(E_interaction, T))
F_vs_T = np.array(F_vs_T)

ax.semilogx(T_range, F_vs_T, color=IAM_C, lw=3, label='IAM prediction')
ax.axhline(y=1.0, color=STD_C, lw=2.5, ls='--', label='Standard QM (T-independent)')
ax.set_xlabel('Temperature T (K)')
ax.set_ylabel('Erasure fidelity F')
ax.set_title(f'Quantum Eraser Fidelity vs Temperature\n'
             f'Fixed which-path interaction energy E = {E_interaction/eV:.1f} eV')
ax.legend(loc='center left', framealpha=0.9, fontsize=12)
ax.set_xlim(0.01, 1000)
ax.set_ylim(-0.05, 1.15)
ax.grid(True, alpha=0.2, which='both')

# Mark key temperatures
for T_mark, label in [(0.01, '10 mK\n(dilution)'), (4, '4 K\n(LHe)'), 
                       (77, '77 K\n(LN₂)'), (300, '300 K\n(room)')]:
    QL_mark = k_B * T_mark * ln2
    F_mark = F_iam(E_interaction, T_mark)
    ax.axvline(x=T_mark, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.plot(T_mark, F_mark, 'ko', markersize=8, zorder=5)
    ax.annotate(f'{label}\nF = {F_mark:.3f}', xy=(T_mark, F_mark),
               xytext=(T_mark*2, F_mark - 0.12),
               fontsize=10, ha='left',
               arrowprops=dict(arrowstyle='->', color='black', lw=1))

# Mark the crossover temperature where Q = Q_L
T_cross = E_interaction / (k_B * ln2)
ax.axvline(x=T_cross, color=THRESH_C, lw=2, ls=':', alpha=0.7)
ax.text(T_cross * 1.3, 0.5, f'$T_{{cross}}$ = {T_cross:.0f} K\n($Q = Q_L$)',
        fontsize=11, color=THRESH_C, fontweight='bold')

fig3.text(0.5, -0.02, 
          'IAM predicts erasure fidelity depends on temperature through the Landauer bound.\n'
          'Standard QM predicts no temperature dependence. Run the same eraser at two temperatures.',
          ha='center', fontsize=11, fontstyle='italic')

fig3.savefig('/home/claude/fig_eraser_temperature.png')
print("Figure 3 saved: Temperature dependence")

# ============================================================
# FIGURE 4: The experimental decision tree
# ============================================================
fig4, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'IAM Paper B — Experimental Decision Tree', 
        fontsize=16, fontweight='bold', ha='center', va='center')

# Build tree
boxes = [
    (5, 8.5, 'Run quantum eraser\nwith which-path detector', '#E0E7FF', 'black'),
    (2.5, 7, 'Reversible detector\n(Q << Q_L)', '#DBEAFE', IAM_C),
    (7.5, 7, 'Irreversible detector\n(Q >> Q_L)', '#FEE2E2', STD_C),
    (2.5, 5.5, 'Interference\nrestored?', '#F0FDF4', THRESH_C),
    (7.5, 5.5, 'Interference\nrestored?', '#F0FDF4', THRESH_C),
    (1.0, 3.8, 'YES', '#DCFCE7', THRESH_C),
    (4.0, 3.8, 'NO', '#FEE2E2', STD_C),
    (6.0, 3.8, 'YES', '#FEE2E2', STD_C),
    (9.0, 3.8, 'NO', '#DCFCE7', IAM_C),
    (1.0, 2.3, 'Both QM\nand IAM\nconfirmed', '#E0E7FF', 'black'),
    (4.0, 2.3, 'Both QM\nand IAM\nfalsified', '#FEF3C7', '#B45309'),
    (6.0, 2.3, 'Standard QM\nconfirmed\nIAM falsified', '#FEE2E2', STD_C),
    (9.0, 2.3, 'IAM\nconfirmed\nQM incomplete', '#DBEAFE', IAM_C),
]

for x, y, text, facecolor, edgecolor in boxes:
    bbox = dict(boxstyle='round,pad=0.4', facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=bbox, color=edgecolor if edgecolor != 'black' else '#1a1a1a')

# Arrows
arrow_style = dict(arrowstyle='->', lw=2, color='#666')
from matplotlib.patches import FancyArrowPatch

arrows = [
    (3.5, 8.2, 2.5, 7.5),   # top to reversible
    (6.5, 8.2, 7.5, 7.5),   # top to irreversible
    (2.5, 6.5, 2.5, 6.0),   # reversible to question
    (7.5, 6.5, 7.5, 6.0),   # irreversible to question
    (1.8, 5.1, 1.0, 4.3),   # rev-yes
    (3.2, 5.1, 4.0, 4.3),   # rev-no
    (6.8, 5.1, 6.0, 4.3),   # irrev-yes
    (8.2, 5.1, 9.0, 4.3),   # irrev-no
    (1.0, 3.4, 1.0, 2.8),   # yes to both confirmed
    (4.0, 3.4, 4.0, 2.8),   # no to both falsified
    (6.0, 3.4, 6.0, 2.8),   # yes to QM confirmed
    (9.0, 3.4, 9.0, 2.8),   # no to IAM confirmed
]

for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))

# Bottom note
ax.text(5, 0.8, 'The critical test: same photon source, same slits, same screen.\n'
        'Only the which-path detector changes.\n'
        'Standard QM predicts columns 2+3 give same result. IAM predicts they differ.',
        ha='center', va='center', fontsize=11, fontstyle='italic', color='#555')

fig4.savefig('/home/claude/fig_eraser_decision_tree.png')
print("Figure 4 saved: Decision tree")

# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print("=" * 70)
print("SUMMARY: IAM QUANTUM ERASER PREDICTIONS")
print("=" * 70)
print(f"""
1. ERASURE FIDELITY depends on thermodynamic irreversibility:
   F_IAM(Q,T) = 1 - exp(1 - k_BT ln2 / Q) / e
   Standard QM: F = 1 always (dissipation irrelevant)

2. LANDAUER THRESHOLD at room temperature:
   Q_L = k_B × 300 K × ln 2 = {Q_L_300:.3e} J = {Q_L_300/eV:.4e} eV

3. KEY PREDICTION — The Atomic Timing Test:
   Atom at slit, excitation creates which-path info.
   Before spontaneous emission (t < {tau_ns:.0f} ns): both agree, F ≈ 1
   After spontaneous emission (t > {tau_ns:.0f} ns): 
     Standard QM: F = 1 (capture emitted photon, measure conjugately)
     IAM: F ≈ 0 (emission was irreversible, Landauer entropy produced)

4. TEMPERATURE TEST:
   Same eraser at different temperatures.
   Standard QM: no change (F independent of T)
   IAM: F increases with T (higher T → higher Q_L → harder to be irreversible)
   For E = 0.1 eV interaction:
     At 10 mK:  F = {F_iam(0.1*eV, 0.01):.4f}
     At 4 K:    F = {F_iam(0.1*eV, 4.0):.4f}  
     At 300 K:  F = {F_iam(0.1*eV, 300):.4f}

5. FALSIFIABILITY:
   If irreversible which-path detection STILL allows erasure 
   (F ≈ 1 after dissipation >> Q_L), IAM's Landauer criterion 
   is falsified at the quantum scale.

4 figures generated for Paper B.
""")

