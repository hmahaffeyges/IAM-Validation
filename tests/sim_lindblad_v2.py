"""
IAM Paper A — Modified Lindblad Decoherence Simulation (v2)
============================================================
Works in the INTERACTION PICTURE: removes free Hamiltonian oscillation
and tracks only the decoherence dynamics. This is numerically stable
and physically correct — the interesting physics is the decoherence,
not the oscillation.

Simulates density matrix evolution under:
  1. Standard Lindblad (constant rate Γ₀)
  2. IAM-modified Lindblad (time-dependent Γ(t) from E_q ramp)

Tracks: purity, von Neumann entropy, coherence, phonon number.

Heath W. Mahaffey, Independent Researcher
February 21, 2026
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================
# Physical constants
# ============================================================
hbar = 1.0546e-34
k_B = 1.3806e-23
G = 6.674e-11
ln2 = np.log(2)

# ============================================================
# System: picogram nanosphere at 10 mK
# ============================================================
m = 1e-12
rho_mat = 2000  # silica density
R = (3 * m / (4 * np.pi * rho_mat))**(1/3)
T = 0.01  # 10 mK

E_G = G * m**2 / R
tau_IAM = hbar * k_B**2 * T**2 * ln2 / E_G**3
tau_PD = hbar / E_G

print(f"System: m = {m:.0e} kg, R = {R:.2e} m, T = {T*1000:.0f} mK")
print(f"E_G = {E_G:.3e} J")
print(f"τ_IAM = {tau_IAM:.2f} s")
print(f"τ_PD = {tau_PD:.3e} s")
print()

# ============================================================
# Hilbert space (interaction picture — no Hamiltonian)
# ============================================================
N_fock = 15

# Annihilation operator
a = np.zeros((N_fock, N_fock), dtype=complex)
for n in range(N_fock - 1):
    a[n, n+1] = np.sqrt(n + 1)
a_dag = a.conj().T
n_op = a_dag @ a

# Initial state: coherent |α=2⟩
alpha = 2.0
psi = np.zeros(N_fock, dtype=complex)
for n in range(N_fock):
    psi[n] = (alpha**n / np.sqrt(math.factorial(n))) * np.exp(-abs(alpha)**2 / 2)
psi /= np.linalg.norm(psi)
rho_0 = np.outer(psi, psi.conj())

print(f"Initial state: |α={alpha}⟩, ⟨n⟩ = {np.real(np.trace(n_op @ rho_0)):.2f}")
print(f"Initial purity: {np.real(np.trace(rho_0 @ rho_0)):.6f}")
print()

# ============================================================
# Decoherence rate functions (in units where time is in τ_IAM)
# ============================================================

def gamma_iam_normalized(eta):
    """IAM rate in units of 1/τ_IAM. Zero at η=0, peaks near η=1."""
    if eta < 1e-8:
        return 0.0
    return (1.0 / eta**2) * np.exp(1.0 - 1.0/eta)

def gamma_std_normalized(eta):
    """Constant rate = 1 in normalized units"""
    return 1.0

# ============================================================
# Lindblad dissipator (interaction picture — no Hamiltonian term)
# dρ/dη = Γ(η) × (a ρ a† - ½{a†a, ρ})
# ============================================================

def dissipator(rho_state, gamma):
    """Lindblad dissipator for position dephasing channel.
    
    Gravitational decoherence destroys spatial coherence (superposition)
    WITHOUT removing energy. The correct collapse operator is proportional
    to position: L = x = (a + a†)/√2.
    
    This produces dephasing in position basis: off-diagonal elements 
    decay while diagonal elements (populations) are preserved.
    """
    x = (a + a_dag) / np.sqrt(2)  # position operator
    L = x
    L_dag = x  # x is Hermitian
    LdL = L_dag @ L  # = x²
    L_rho_Ld = L @ rho_state @ L_dag
    anticomm = LdL @ rho_state + rho_state @ LdL
    return gamma * (L_rho_Ld - 0.5 * anticomm)

def rk4_step(rho_state, eta, d_eta, gamma_func):
    """4th-order Runge-Kutta for density matrix evolution"""
    k1 = dissipator(rho_state, gamma_func(eta))
    k2 = dissipator(rho_state + 0.5 * d_eta * k1, gamma_func(eta + 0.5 * d_eta))
    k3 = dissipator(rho_state + 0.5 * d_eta * k2, gamma_func(eta + 0.5 * d_eta))
    k4 = dissipator(rho_state + d_eta * k3, gamma_func(eta + d_eta))
    
    rho_new = rho_state + (d_eta / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Force Hermiticity and trace = 1
    rho_new = 0.5 * (rho_new + rho_new.conj().T)
    rho_new /= np.trace(rho_new)
    
    return rho_new

# ============================================================
# Diagnostics
# ============================================================

def get_purity(rho_state):
    return np.real(np.trace(rho_state @ rho_state))

def get_entropy(rho_state):
    evals = np.real(np.linalg.eigvalsh(rho_state))
    evals = evals[evals > 1e-15]
    return -np.sum(evals * np.log2(evals))

def get_coherence_l1(rho_state):
    """l1 norm of off-diagonal elements"""
    return np.sum(np.abs(rho_state)) - np.sum(np.abs(np.diag(rho_state)))

def get_mean_n(rho_state):
    return np.real(np.trace(n_op @ rho_state))

# ============================================================
# Run simulations
# ============================================================
eta_max = 5.0  # simulate to 5 × τ_IAM
N_steps = 5000
d_eta = eta_max / N_steps
eta_array = np.linspace(0, eta_max, N_steps + 1)

print("Running simulations (RK4 in interaction picture)...")

results = {}
for name, gfunc in [("IAM", gamma_iam_normalized), 
                      ("Standard", gamma_std_normalized)]:
    print(f"  {name}...", end="", flush=True)
    
    pur = [get_purity(rho_0)]
    ent = [get_entropy(rho_0)]
    coh = [get_coherence_l1(rho_0)]
    mn = [get_mean_n(rho_0)]
    gam = [gfunc(0)]
    
    rho_curr = rho_0.copy()
    snapshots = {0.0: rho_0.copy()}
    snap_targets = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    for step in range(N_steps):
        eta = eta_array[step]
        rho_curr = rk4_step(rho_curr, eta, d_eta, gfunc)
        
        pur.append(get_purity(rho_curr))
        ent.append(get_entropy(rho_curr))
        coh.append(get_coherence_l1(rho_curr))
        mn.append(get_mean_n(rho_curr))
        gam.append(gfunc(eta_array[step + 1]))
        
        # Save snapshots
        for st in snap_targets:
            if st not in snapshots and abs(eta_array[step+1] - st) < 1.5 * d_eta:
                snapshots[st] = rho_curr.copy()
    
    results[name] = {
        'purity': np.array(pur),
        'entropy': np.array(ent),
        'coherence': np.array(coh),
        'mean_n': np.array(mn),
        'gamma': np.array(gam),
        'snapshots': snapshots,
    }
    print(f" done. Final purity = {pur[-1]:.4f}")

print()

# ============================================================
# Plotting
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

# ============================================================
# FIGURE 10: Four-panel quantum state evolution
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Purity
ax = axes[0, 0]
ax.plot(eta_array, results['IAM']['purity'], color=IAM_C, lw=2.5, label='IAM (E_q ramp)')
ax.plot(eta_array, results['Standard']['purity'], color=STD_C, lw=2.5, ls='--', label='Standard Lindblad')
ax.set_xlabel(r't / $\tau_{IAM}$')
ax.set_ylabel(r'Purity Tr($\rho^2$)')
ax.set_title('(a) Purity Decay')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.2)
ax.axvspan(0, 0.5, alpha=0.06, color=IAM_C)
ax.text(0.25, 0.95, 'Protected\nregime', fontsize=9, color=IAM_C, 
        fontstyle='italic', ha='center', va='top', transform=ax.get_xaxis_transform())

# (b) Von Neumann Entropy
ax = axes[0, 1]
ax.plot(eta_array, results['IAM']['entropy'], color=IAM_C, lw=2.5, label='IAM')
ax.plot(eta_array, results['Standard']['entropy'], color=STD_C, lw=2.5, ls='--', label='Standard')
ax.set_xlabel(r't / $\tau_{IAM}$')
ax.set_ylabel('Von Neumann Entropy S (bits)')
ax.set_title('(b) Entropy Production')
ax.legend(loc='lower right', framealpha=0.9)
ax.set_xlim(0, 5)
ax.grid(True, alpha=0.2)
ax.axvspan(0, 0.5, alpha=0.06, color=IAM_C)

# (c) Coherence decay
ax = axes[1, 0]
c_iam = results['IAM']['coherence']
c_std = results['Standard']['coherence']
ax.plot(eta_array, c_iam / c_iam[0], color=IAM_C, lw=2.5, label='IAM')
ax.plot(eta_array, c_std / c_std[0], color=STD_C, lw=2.5, ls='--', label='Standard')
ax.set_xlabel(r't / $\tau_{IAM}$')
ax.set_ylabel('Normalized coherence C(t)/C(0)')
ax.set_title('(c) Off-Diagonal Coherence Decay')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.2)
ax.axvspan(0, 0.5, alpha=0.06, color=IAM_C)

# (d) Decoherence rate
ax = axes[1, 1]
ax.plot(eta_array, results['IAM']['gamma'], color=IAM_C, lw=2.5, label=r'IAM $\Gamma(\eta)$')
ax.plot(eta_array, results['Standard']['gamma'], color=STD_C, lw=2.5, ls='--', label=r'Standard $\Gamma_0$')
ax.set_xlabel(r't / $\tau_{IAM}$')
ax.set_ylabel(r'Decoherence rate $\Gamma$ (units of $1/\tau_{IAM}$)')
ax.set_title('(d) Time-Dependent Decoherence Rate')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(0, 5)
ax.grid(True, alpha=0.2)
ax.axvspan(0, 0.5, alpha=0.06, color=IAM_C)
ax.annotate('IAM: rate = 0 at t = 0\n(system protected)',
            xy=(0.05, 0.01), xytext=(1.2, 0.15),
            fontsize=10, color=IAM_C, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=IAM_C, lw=1.5))
ax.annotate('Standard: constant rate\n(immediate decoherence)',
            xy=(0.1, 1.0), xytext=(2.0, 0.7),
            fontsize=10, color=STD_C, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=STD_C, lw=1.5))

fig.suptitle('IAM Paper A — Quantum State Evolution: IAM vs Standard Lindblad Decoherence\n'
             f'System: 1 pg nanosphere, T = 10 mK, coherent state |α = 2⟩, '
             f'τ_IAM = {tau_IAM:.0f} s',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('/home/claude/fig10_lindblad_evolution.png')
print("Figure 10 saved")

# ============================================================
# FIGURE 11: Density matrix snapshots (|ρ_mn|)
# ============================================================
snap_etas = [0.0, 0.5, 1.0, 2.0, 5.0]
snap_labels = ['t = 0', 't = 0.5τ', 't = τ', 't = 2τ', 't = 5τ']

fig2, axes2 = plt.subplots(2, 5, figsize=(20, 7))

for row, (name, color) in enumerate([('IAM', IAM_C), ('Standard', STD_C)]):
    snaps = results[name]['snapshots']
    for col, (se, sl) in enumerate(zip(snap_etas, snap_labels)):
        ax = axes2[row, col]
        if se in snaps:
            rho_snap = snaps[se]
            n_show = 10
            im = ax.imshow(np.abs(rho_snap[:n_show, :n_show]),
                          cmap='magma', vmin=0, vmax=0.45,
                          origin='lower', aspect='equal',
                          extent=[-0.5, n_show-0.5, -0.5, n_show-0.5])
            p = get_purity(rho_snap)
            ax.set_title(f'{sl}\nP = {p:.3f}', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_title(f'{sl}', fontsize=10)
        
        if col == 0:
            ax.set_ylabel(f'{name}\n|ρ_mn|', fontsize=11, color=color, fontweight='bold')
        if row == 1:
            ax.set_xlabel('n', fontsize=9)

# Add colorbar
fig2.subplots_adjust(right=0.92)
cbar_ax = fig2.add_axes([0.93, 0.15, 0.015, 0.7])
fig2.colorbar(im, cax=cbar_ax, label='|ρ_mn|')

fig2.suptitle('IAM Paper A — Density Matrix Evolution: IAM vs Standard Lindblad\n'
              'Fock-space representation (first 10 states), coherent state |α = 2⟩',
              fontsize=14, fontweight='bold', y=1.02)
fig2.savefig('/home/claude/fig11_density_matrices.png')
print("Figure 11 saved")

# ============================================================
# FIGURE 12: Phonon decay + entropy rate + purity difference
# ============================================================
fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# Left: Phonon number
ax1.plot(eta_array, results['IAM']['mean_n'], color=IAM_C, lw=2.5, label='IAM')
ax1.plot(eta_array, results['Standard']['mean_n'], color=STD_C, lw=2.5, ls='--', label='Standard')
ax1.set_xlabel(r't / $\tau_{IAM}$')
ax1.set_ylabel(r'Mean phonon number $\langle n \rangle$')
ax1.set_title('(a) Phonon Number Decay')
ax1.legend(framealpha=0.9)
ax1.set_xlim(0, 5)
ax1.grid(True, alpha=0.2)

# Middle: Entropy production rate
dS_iam = np.gradient(results['IAM']['entropy'], d_eta)
dS_std = np.gradient(results['Standard']['entropy'], d_eta)
ax2.plot(eta_array, dS_iam, color=IAM_C, lw=2.5, label='IAM dS/dη')
ax2.plot(eta_array, dS_std, color=STD_C, lw=2.5, ls='--', label='Standard dS/dη')
ax2.set_xlabel(r't / $\tau_{IAM}$')
ax2.set_ylabel(r'dS/d$\eta$ (bits per $\tau_{IAM}$)')
ax2.set_title('(b) Entropy Production Rate')
ax2.legend(framealpha=0.9)
ax2.set_xlim(0, 5)
ax2.grid(True, alpha=0.2)

# Right: Purity difference (the experimental observable)
purity_diff = results['IAM']['purity'] - results['Standard']['purity']
ax3.plot(eta_array, purity_diff, color='#7C3AED', lw=2.5)
ax3.axhline(y=0, color='gray', lw=0.8, ls=':')
ax3.fill_between(eta_array, 0, purity_diff, 
                  where=(purity_diff > 0), alpha=0.2, color=IAM_C, label='IAM more coherent')
ax3.fill_between(eta_array, 0, purity_diff,
                  where=(purity_diff < 0), alpha=0.2, color=STD_C, label='Standard more coherent')
ax3.set_xlabel(r't / $\tau_{IAM}$')
ax3.set_ylabel(r'$\Delta$Purity (IAM − Standard)')
ax3.set_title('(c) Purity Difference: The Observable')
ax3.legend(framealpha=0.9, fontsize=9)
ax3.set_xlim(0, 5)
ax3.grid(True, alpha=0.2)

# Mark the peak difference
idx_peak = np.argmax(np.abs(purity_diff))
ax3.annotate(f'Peak ΔP = {purity_diff[idx_peak]:.3f}\nat η = {eta_array[idx_peak]:.2f}',
            xy=(eta_array[idx_peak], purity_diff[idx_peak]),
            xytext=(eta_array[idx_peak] + 0.8, purity_diff[idx_peak] * 0.7),
            fontsize=11, fontweight='bold', color='#7C3AED',
            arrowprops=dict(arrowstyle='->', color='#7C3AED', lw=1.5))

fig3.suptitle('IAM Paper A — Information Thermodynamics of Decoherence\n'
              '1 pg nanosphere, 10 mK, |α = 2⟩',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig3.savefig('/home/claude/fig12_phonon_entropy_purity.png')
print("Figure 12 saved")

# ============================================================
# Print summary
# ============================================================
print("\n" + "="*70)
print("SIMULATION RESULTS SUMMARY")
print("="*70)

for name in ['IAM', 'Standard']:
    r = results[name]
    print(f"\n{name} Lindblad:")
    for eta_check in [0.25, 0.5, 1.0, 2.0, 5.0]:
        idx = int(eta_check / eta_max * N_steps)
        idx = min(idx, N_steps)
        print(f"  η = {eta_check}: purity = {r['purity'][idx]:.4f}, "
              f"S = {r['entropy'][idx]:.3f} bits, "
              f"⟨n⟩ = {r['mean_n'][idx]:.3f}, "
              f"C/C₀ = {r['coherence'][idx]/r['coherence'][0]:.4f}")

print(f"\n{'='*70}")
print("KEY DISCRIMINATORS:")
print(f"  At η = 0.5 (halfway to τ_IAM):")
idx_half = int(0.5 / eta_max * N_steps)
p_iam_half = results['IAM']['purity'][idx_half]
p_std_half = results['Standard']['purity'][idx_half]
print(f"    IAM purity:  {p_iam_half:.4f}")
print(f"    Std purity:  {p_std_half:.4f}")
print(f"    Difference:  {p_iam_half - p_std_half:.4f} ({(p_iam_half - p_std_half)/p_std_half*100:.1f}%)")

print(f"\n  Peak purity difference:")
print(f"    ΔP = {purity_diff[idx_peak]:.4f} at η = {eta_array[idx_peak]:.2f}")
print(f"    This is the optimal measurement time for discrimination")

print(f"\n  At η = 1.0 (one τ_IAM):")
idx_one = int(1.0 / eta_max * N_steps)
c_iam_one = results['IAM']['coherence'][idx_one] / results['IAM']['coherence'][0]
c_std_one = results['Standard']['coherence'][idx_one] / results['Standard']['coherence'][0]
print(f"    IAM coherence remaining:  {c_iam_one:.4f} ({c_iam_one*100:.1f}%)")
print(f"    Std coherence remaining:  {c_std_one:.4f} ({c_std_one*100:.1f}%)")
print(f"{'='*70}")
