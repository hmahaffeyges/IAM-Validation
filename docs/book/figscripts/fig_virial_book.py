"""Figures for the virial and gravitational-decoherence chapters:
part2/p2_02_virial.tex, part2/p2_02b_virial_tests.tex, part5/p5_05_gravdec.tex, part5/p5_05b_virial_partners.tex.
Numbers as docs/verification/scripts/verify_virial_papers.py (Planck 2018 Omega_m = 0.3153, beta_m = Omega_m/2; Level 2 chains from
mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv; halo mass-function slopes from docs/verification/virial/NBODY_TRACE_massfunction_slopes.csv).
  fig_virial_mu_E          E(a), mu(z), 1 - mu and the f sigma8 deficit
  fig_virial_halo_slope    published halo virial ratios; d ln F(>M)/d ln D from six mass functions
  fig_virial_h0_census     H0 measurements against the photon and matter rates (worldline rule)
  fig_virial_partition     R(a), energy budget, record share, sector gap, growth vs geometry, record growth vs matter dilution
  fig_gravdec_scaling      tau_IAM against mass at five temperatures and tau_PD; rate profiles
  fig_gravdec_lindblad     Lindblad dephasing (L = x) with the constant rate and with the ramp rate
  fig_gravdec_heating      phonon excitation rate k_B T ln2 / (tau_IAM hbar omega0) against mass
"""
import sys, pathlib, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
from math import factorial
from scipy.integrate import solve_ivp
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
Om = 0.3153; OL = 1 - Om; bm = Om / 2
E = lambda a: np.exp(1 - 1 / a)
H2 = lambda a: Om * a**-3 + OL
mu = lambda a: H2(a) / (H2(a) + bm * E(a))

def grow(m):
    def r(l, y):
        a = np.exp(l); return [y[1], -(2 - 1.5 * Om * a**-3 / H2(a)) * y[1] + 1.5 * (Om * a**-3 / H2(a)) * m(a) * y[0]]
    return solve_ivp(r, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)
Lg, Ig = grow(lambda a: 1.0), grow(mu)
Dg = lambda Sx, a: Sx.sol(np.log(a))[0]; fg = lambda Sx, a: Sx.sol(np.log(a))[1] / Sx.sol(np.log(a))[0]

# ---------------------------------------------------------------- fig_virial_mu_E
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5))
a = np.linspace(0.02, 4, 600)
ax1.plot(a, E(a), color=S.IAM)
ax1.axhline(np.e, color=S.LIGHT, ls="--", lw=0.8); ax1.text(3.95, np.e - 0.08, "e (approached, never reached)", ha="right", va="top", fontsize=6)
ax1.plot([1], [1], "o", color=S.IAM, ms=3.5); ax1.annotate("today, E(1) = 1", (1, 1), xytext=(1.35, 0.55), fontsize=6,
                                                        arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GR))
ax1.plot([0.5], [E(0.5)], "s", color=S.GR, ms=3); ax1.annotate("inflection a = 1/2", (0.5, E(0.5)), xytext=(0.9, 0.12), fontsize=6,
                                                             arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GR))
ax1.set_xlabel("scale factor a"); ax1.set_ylabel("E(a) = exp(1 − 1/a)"); ax1.set_xlim(0, 4); ax1.set_ylim(0, 3)
S.panel_letter(ax1, "a")
z = np.linspace(0, 2.5, 400); az = 1 / (1 + z)
ax2.plot(z, 100 * (1 - mu(az)), color=S.IAM, label="coupling deficit 1 − μ(z)")
dfs = [100 * (1 - fg(Ig, x) * Dg(Ig, x) / (fg(Lg, x) * Dg(Lg, x))) for x in az]
ax2.plot(z, dfs, color=S.ALT, ls="--", label="f σ8 deficit (same early amplitude)")
for zz, lab in ((0, "13.6 %"), (0.3, "7.8 %")):
    ax2.plot([zz], [100 * (1 - mu(1 / (1 + zz)))], "o", color=S.IAM, ms=3)
    ax2.text(zz + 0.06, 100 * (1 - mu(1 / (1 + zz))) + 0.4, lab, fontsize=6)
ax2.text(0.06, 4.25 + 0.4, "4.25 %", fontsize=6, color=S.ALT)
ax2.set_xlabel("redshift z"); ax2.set_ylabel("per cent"); ax2.set_xlim(0, 2.5); ax2.set_ylim(0, 15)
ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 0.55)); S.panel_letter(ax2, "b")
S.save(fig, "part2", "fig_virial_mu_E")

# ---------------------------------------------------------------- fig_virial_halo_slope
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6))
pubs = [("Bett et al. 2007", 11.4, 15, 1.2, 1.3), ("Neto et al. 2007", 12, 15, 1.12, 1.26), ("Power et al. 2012", 12, 15, 1.15, 1.25),
        ("Klypin et al. 2016", 12, 15, 1.1, 1.4), ("Klypin, surface term", 12, 15, 1.02, 1.17)]
cols = [S.IAM, S.ALT, S.GOLD, S.DATA, S.ALT2]
for i, (nm, m0, m1, lo, hi) in enumerate(pubs):
    ax1.fill_between([m0, m1], [lo / 2] * 2, [hi / 2] * 2, color=cols[i], alpha=0.25, lw=0)
    ax1.text(m0 + 0.05, hi / 2 + 0.004, nm, fontsize=5.5, color=cols[i]) if i in (0, 3, 4) else None
ax1.axhline(0.5, color=S.GR, ls="--", lw=0.8); ax1.text(15, 0.497, "equilibrium K/|W| = 1/2", fontsize=6, ha="right", va="top")
ax1.set_xlabel("log₁₀ halo mass (h⁻¹ M⊙)"); ax1.set_ylabel("K/|W| (published 2T/|U| ÷ 2)"); ax1.set_xlim(11, 15.2); ax1.set_ylim(0.45, 0.75)
S.panel_letter(ax1, "a")
rows = list(csv.DictReader(open(S.REPO / "docs/verification/virial/NBODY_TRACE_massfunction_slopes.csv")))
names = {"PS74": "Press–Schechter", "ST99": "Sheth–Tormen", "Jenkins01": "Jenkins et al.", "Reed03": "Reed et al.",
         "Tinker08_D200m": "Tinker et al.", "Watson13_FOF": "Watson et al."}
for i, (key, lab) in enumerate(names.items()):
    rr = [r for r in rows if r["mf"] == key]
    m = np.log10([float(r["M_hinv_Msun"]) for r in rr]); s = [float(r["dlnF_dlnD"]) for r in rr]
    ax2.plot(m, s, color=[S.GR, S.IAM, S.ALT, S.GOLD, S.DATA, S.ALT2][i], label=lab, lw=1.0)
ax2.set_yscale("log"); ax2.set_xlabel("log₁₀ M_min (h⁻¹ M⊙)"); ax2.set_ylabel("d ln F(>M)/d ln D")
ax2.legend(loc="upper left", fontsize=5.5); S.panel_letter(ax2, "b")
S.save(fig, "part2", "fig_virial_halo_slope")

# ---------------------------------------------------------------- fig_virial_h0_census
ch = {r["chain"]: r for r in csv.DictReader(open(S.REPO / "mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv"))}
Hg = float(ch["iam_level2_runA"]["H0"]); Hm = Hg * np.sqrt(1 + bm)
data = [("Planck 2018 CMB", 67.36, 0.54, "photon"), ("ACT DR4 + WMAP", 67.6, 1.1, "photon"), ("H0LiCOW time delays", 73.3, 1.8, "photon"),
        ("SH0ES Cepheids", 73.04, 1.04, "matter"), ("Megamasers (MCP)", 73.9, 3.0, "matter"), ("Surface brightness fluct.", 73.3, 2.5, "matter")]
fig, ax = plt.subplots(figsize=(S.TEXTW * 0.75, 2.4))
for i, (nm, h, s, sec) in enumerate(data):
    yv = len(data) - i
    ax.errorbar(h, yv, xerr=s, fmt="o", ms=3.5, color=S.IAM if sec == "photon" else S.DATA, capsize=2, lw=0.9)
    ax.text(62.6, yv, nm, va="center", fontsize=6)
ax.axvline(Hg, color=S.IAM, ls="--", lw=0.9); ax.axvline(Hm, color=S.DATA, ls="--", lw=0.9)
ax.text(Hg - 0.15, 6.75, f"photon {Hg:.2f}", color=S.IAM, fontsize=6, ha="right")
ax.text(Hm + 0.15, 6.75, f"matter {Hm:.2f}", color=S.DATA, fontsize=6)
ax.set_xlim(62.5, 78); ax.set_ylim(0.4, 7.2); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("H₀ (km s⁻¹ Mpc⁻¹)")
S.save(fig, "part2", "fig_virial_h0_census")

# ---------------------------------------------------------------- fig_virial_partition
fig, axs = plt.subplots(2, 3, figsize=(S.TEXTW, 4.0))
z = np.linspace(0, 2.5, 500); az = 1 / (1 + z)
ax = axs[0, 0]; ax.semilogy(z, Om * az**-3 / (bm * E(az)), color=S.IAM); ax.axhline(2, color=S.GR, ls="--", lw=0.8)
ax.text(2.45, 2.4, "R = 2 today (by construction)", fontsize=5.5, ha="right"); ax.set_xlabel("z"); ax.set_ylabel("R(a) = Ωm a⁻³ / βm E(a)")
ax.axvspan(0.3, 0.7, color=S.SKY, alpha=0.2, lw=0); S.panel_letter(ax, "a")
ax = axs[0, 1]; ax.plot(z, 100 * Om * az**-3, color=S.GR, label="matter Ωm a⁻³"); ax.plot(z, 100 * OL + 0 * z, color=S.ALT, ls=":", label="vacuum ΩΛ")
ax.plot(z, 100 * bm * E(az), color=S.IAM, label="record βm E(a)"); ax.plot(z, 100 * (OL + bm * E(az)), color=S.GOLD, ls="--", label="ΩΛ + βm E(a)")
zq = 0.361; ax.plot([zq], [100 * Om * (1 + zq)**3], "o", ms=3, color=S.GOLD); ax.text(zq + 0.12, 100 * Om * (1 + zq)**3 + 4, "z = 0.361", fontsize=5.5)
ax.set_ylim(0, 150); ax.set_xlabel("z"); ax.set_ylabel("% of critical density today"); ax.legend(fontsize=5, loc="upper left")
ax.axvspan(0.3, 0.7, color=S.SKY, alpha=0.2, lw=0); S.panel_letter(ax, "b")
ax = axs[0, 2]; sh = 100 * bm * E(az) / (OL + bm * E(az)); ax.plot(z, sh, color=S.IAM)
for zz in (0, 0.3, 0.7, 1.5):
    v = 100 * bm * E(1 / (1 + zz)) / (OL + bm * E(1 / (1 + zz))); ax.plot([zz], [v], "o", ms=3, color=S.IAM); ax.text(zz + 0.07, v + 0.6, f"{v:.1f} %", fontsize=5.5)
ax.set_xlabel("z"); ax.set_ylabel("record share of dark energy (%)"); ax.set_ylim(0, 22); ax.axvspan(0.3, 0.7, color=S.SKY, alpha=0.2, lw=0); S.panel_letter(ax, "c")
ax = axs[1, 0]; ax.plot(z, 100 * (1 / np.sqrt(mu(az)) - 1), color=S.DATA); ax.set_xlabel("z"); ax.set_ylabel("H_m/H − 1, per cent")
ax.text(0.05, 100 * (1 / np.sqrt(mu(1.0)) - 1) - 0.6, f"{100*(1/np.sqrt(mu(1.0))-1):.2f} % today", fontsize=5.5, va="top")
ax.axvspan(0.3, 0.7, color=S.SKY, alpha=0.2, lw=0); S.panel_letter(ax, "d")
ax = axs[1, 1]; ax.plot(z, 100 * Om + 0 * z, color=S.GR, ls="--", label="geometric Ωm"); ax.plot(z, 100 * Om * mu(az), color=S.IAM, label="growth-only Ωm μ(z)")
ax.set_xlabel("z"); ax.set_ylabel("Ωm, per cent"); ax.set_ylim(26, 33); ax.legend(fontsize=5.5, loc="lower right")
ax.axvspan(0.3, 0.7, color=S.SKY, alpha=0.2, lw=0); S.panel_letter(ax, "e")
ax = axs[1, 2]; ax.plot(z, 100 * E(az) * az**2 / 6, color=S.IAM); ax.set_xlabel("z"); ax.set_ylabel("record growth ÷ matter dilution (%)")
for zz in (0.3, 0.7, 1.5):
    v = 100 * E(1 / (1 + zz)) / (1 + zz)**2 / 6; ax.plot([zz], [v], "o", ms=3, color=S.IAM); ax.text(zz + 0.07, v + 0.4, f"{v:.1f} %", fontsize=5.5)
ax.axvspan(0.3, 0.7, color=S.SKY, alpha=0.2, lw=0); S.panel_letter(ax, "f")
fig.tight_layout()
S.save(fig, "part5", "fig_virial_partition")

# ---------------------------------------------------------------- fig_gravdec_scaling
G, hbar, k = C.G, C.hbar, C.k; rho = 2200.0; ln2 = np.log(2)
def EG(m): return G * m**2 / (3 * m / (4 * np.pi * rho))**(1 / 3)
tI = lambda m, T: hbar * (k * T)**2 * ln2 / EG(m)**3; tD = lambda m: hbar / EG(m)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6))
m = np.logspace(-16, -8, 300)
for T, cc in zip((0.001, 0.01, 0.1, 1.0, 300.0), (S.IAM, S.ALT, S.GOLD, S.ALT2, S.GR)):
    ax1.loglog(m, tI(m, T), color=cc, lw=1.0, label=f"τ_IAM, {T*1e3:g} mK" if T < 1 else f"τ_IAM, {T:g} K")
ax1.loglog(m, tD(m), color=S.DATA, ls="--", label="τ_PD = ħ/E_G")
mx = 2.23e-10; ax1.plot([mx], [tD(mx)], "*", color=S.DATA, ms=6)
ax1.set_xlabel("mass (kg), silica 2200 kg m⁻³"); ax1.set_ylabel("coherence time (s)"); ax1.set_ylim(1e-12, 1e25)
ax1.legend(fontsize=5.5, loc="lower left"); S.panel_letter(ax1, "a")
eta = np.linspace(1e-3, 4, 800)
ax2.plot(eta, np.exp(1 - 1 / eta) / eta**2 / np.e, color=S.IAM, label="ramp: −dC/dη = η⁻² e^(−1/η)  (conjecture)")
ax2.plot(eta, np.exp(-eta), color=S.DATA, ls="--", label="exponential: −dC/dη = e^(−η)")
ax2.axvline(0.5, color=S.LIGHT, lw=0.8); ax2.text(0.55, 0.05, "η = 1/2", fontsize=6)
ax2.set_xlabel("η = t/τ"); ax2.set_ylabel("normalised rate of loss of coherence"); ax2.set_ylim(0, 1.05); ax2.legend(fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 1.08))
S.panel_letter(ax2, "b")
S.save(fig, "part5", "fig_gravdec_scaling")

# ---------------------------------------------------------------- fig_gravdec_lindblad
N = 40; aop = np.diag(np.sqrt(np.arange(1, N)), 1); X = (aop + aop.T) / np.sqrt(2); nop = aop.T @ aop
psi = np.array([np.exp(-2.0) * 2.0**n / np.sqrt(float(factorial(n))) for n in range(N)]); rho0 = np.outer(psi, psi).astype(complex)
Dm = lambda r: X @ r @ X - 0.5 * (X @ X @ r + r @ X @ X)
def evolve(gam, tmax=5.0, dt=2e-3):
    r = rho0.copy(); out = []; t = 0.0
    while t <= tmax + 1e-12:
        out.append((t, np.real(np.trace(r @ r)), np.real(np.trace(nop @ r)), abs(r[0, 4]) / abs(rho0[0, 4]), gam(t)))
        k1 = gam(t) * Dm(r); k2 = gam(t + dt / 2) * Dm(r + dt / 2 * k1); k3 = gam(t + dt / 2) * Dm(r + dt / 2 * k2); k4 = gam(t + dt) * Dm(r + dt * k3)
        r = r + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
    return np.array(out)
std = evolve(lambda t: 1.0); rmp = evolve(lambda t: 0.0 if t <= 1e-9 else np.exp(1 - 1 / t) / t**2)
fig, axs = plt.subplots(1, 4, figsize=(S.TEXTW, 1.9))
for j, (col, lab) in enumerate(((1, "purity Tr ρ²"), (3, "|ρ₀₄| / |ρ₀₄(0)|"), (2, "⟨n⟩"), (4, "rate Γ(η)"))):
    ax = axs[j]; ax.plot(rmp[:, 0], rmp[:, col], color=S.IAM, label="ramp rate"); ax.plot(std[:, 0], std[:, col], color=S.DATA, ls="--", label="constant rate")
    ax.set_xlabel("η = t/τ"); ax.set_ylabel(lab); S.panel_letter(ax, "abcd"[j], dx=-0.22)
axs[0].legend(fontsize=5.5, loc="upper right")
fig.tight_layout()
S.save(fig, "part5", "fig_gravdec_lindblad")
dP = rmp[:, 1] - std[:, 1]; i = int(np.argmax(dP)); print(f"purity difference peak eta = {rmp[i,0]:.3f}, dP = {dP[i]:.3f}")

# ---------------------------------------------------------------- fig_gravdec_heating
w0 = 2 * np.pi * 1e5
fig, ax = plt.subplots(figsize=(S.TEXTW * 0.6, 2.4))
m = np.logspace(-14, -10, 200)
for T, cc in zip((0.01, 0.02, 0.04, 0.3), (S.IAM, S.ALT, S.GOLD, S.GR)):
    ax.loglog(m, EG(m)**3 / (hbar * k * T) / (hbar * w0), color=cc, label=f"{T*1e3:g} mK")
ax.axhline(1.0, color=S.LIGHT, ls="--", lw=0.8); ax.text(1.2e-14, 1.3, "1 phonon s⁻¹", fontsize=6)
ax.set_xlabel("mass (kg), silica 2200 kg m⁻³"); ax.set_ylabel("dn/dt (phonons s⁻¹)"); ax.set_ylim(1e-8, 1e8)
ax.legend(fontsize=6, loc="upper left")
S.save(fig, "part5", "fig_gravdec_heating")
