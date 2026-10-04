"""Part 2, Chapters 'The cosmological constant' (p2_12_lambda.tex), 'The baryon density' (p2_13_baryon.tex), 'Records at the quantum scale'
(p4_14_quantum_records.tex) and 'The particle scale' (p2_15_particle_masses.tex).
fig_cc_relation: (Omega_b/Omega_m) / ((3/16) sqrt(Omega_L)) on each chain (Cosmological_Physics/mgcamb_validation/chains, 30 % burn-in, weighted; as
verify_cc_and_baryon.py) and on the Planck 2018 values the chapter quotes (Ob 0.0493, Om 0.3153, OL 0.6846).
fig_eta: eta = 273.9e-10 Omega_b h^2 on the same chains, and the two inversions of the cosmological-constant expressions at Omega_m h^2 = 0.1430.
fig_neff_bottomup: bottom-up exponent n_eff(z) from halo mass functions (colossus; method and parameters of verify_bottom_up_exponent.py).
fig_tau_mass: tau_IAM = hbar (k_B T)^2 ln2/E_G^3 against tau_PD = hbar/E_G for silica spheres (rho = 2200 kg m^-3, fused silica, as in Part 5), E_G = G m^2/R.
fig_koide: the charge-orbit encoding sqrt(m) = x (1 + sqrt2 cos(phi + delta)) with the measured masses, and the positivity count by n.
fig_electron_h0: the electron fixed point m_e(H0)/m_e - 1 against H0 (verify_electron_mass.py)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
import _bookstyle as S, _chains as CH
import matplotlib.pyplot as plt

S.apply()
MG = CH.MG
stems = [("18th chain (baryon test, CMB only)", "iam_baryon_test"), ("$\\Lambda$CDM, Planck", "lcdm_baseline"),
         ("$\\Lambda$CDM, Planck + BAO", "planck_bao_lcdm_baseline"), ("$\\Lambda$CDM, Planck + Pantheon+", "planck_pantheon_lcdm_baseline")]
res = []
for lab, st in stems:
    X = CH.load(MG + st + ".1.txt"); w = X.weight.values; h = X.H0.values / 100; ob = X.ombh2.values; om = 1 - X.omegal.values
    q = ob / h**2 / om / (3 / 16 * np.sqrt(X.omegal.values))
    res.append((lab, *CH.wmean_sd(q, w), *CH.wmean_sd(273.9 * ob, w)))
    print(f"{lab:36s} ratio {res[-1][1]:.4f} +/- {res[-1][2]:.4f}   eta {res[-1][3]:.3f} +/- {res[-1][4]:.3f}")
p18 = (0.0493 / 0.3153) / (3 / 16 * np.sqrt(0.6846)); print("Planck 2018 values: ratio", round(p18, 4))
fig, ax = plt.subplots(figsize=(0.75 * S.TEXTW, 1.9))
for i, (lab, m, s_, e, es) in enumerate(res[::-1]):
    ax.errorbar(m, i, xerr=s_, fmt="o", color=S.IAM if i == len(res) - 1 else S.GR, ms=4, capsize=2, lw=0.9)
    ax.annotate(f"{m:.4f} ± {s_:.4f}", (m, i), xytext=(0, 6), textcoords="offset points", fontsize=7, ha="center")
ax.plot(p18, len(res), "s", color=S.DATA, ms=4); ax.annotate(f"{p18:.4f}", (p18, len(res)), xytext=(0, 6), textcoords="offset points", fontsize=7, ha="center")
ax.axvline(1, color=S.GR, lw=0.8, ls="--")
ax.set_yticks(range(len(res) + 1)); ax.set_yticklabels([r[0] for r in res[::-1]] + ["Planck 2018 values"], fontsize=7)
ax.set_xlim(0.985, 1.03); ax.set_ylim(-0.6, len(res) + 0.7)
ax.set_xlabel("$(\\Omega_b/\\Omega_m)\\,/\\,(\\frac{3}{16}\\sqrt{\\Omega_\\Lambda})$")
ax.set_title("The present-epoch relation holds to 0.5–1.1 %")
S.save(fig, "part2", "fig_cc_relation")

hbar, c, G = C.hbar, C.c, C.G; Mpc = 3.0857e22; lP = np.sqrt(hbar * G / c**3); EP = np.sqrt(hbar * c**5 / G)
rvac = EP**4 / (hbar * c)**3; H = 67.36e3 / Mpc; omh2 = 0.3153 * 0.6736**2
o = 0.6847 * 3 * H**2 / (8 * np.pi * G) * c**2 / rvac
base = 2 / np.pi * (lP / (c / H))**2 / omh2; corr = base * np.sqrt(0.6847)
eta_b, eta_c = 273.9 * o / base, 273.9 * o / corr
print(f"inverted: baseline eta {eta_b:.3f}e-10, with sqrt(OL) {eta_c:.3f}e-10")
fig, ax = plt.subplots(figsize=(0.75 * S.TEXTW, 2.0))
rows = [(r[0], r[3], r[4], S.IAM if k == 0 else S.GR, "o") for k, r in enumerate(res)] + \
       [("inverted, with $\\sqrt{\\Omega_\\Lambda}$", eta_c, 0, S.ALT, "D"), ("inverted, baseline", eta_b, 0, S.ALT2, "D")]
for i, (lab, v, e, col, mk) in enumerate(rows[::-1]):
    ax.errorbar(v, i, xerr=e if e else None, fmt=mk, color=col, ms=4, capsize=2, lw=0.9)
    ax.annotate(f"{v:.3f}" + (f" ± {e:.3f}" if e else ""), (v, i), xytext=(0, 6), textcoords="offset points", fontsize=7, ha="center")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7)
ax.set_xlim(4.8, 6.4); ax.set_ylim(-0.6, len(rows) - 0.3)
ax.set_xlabel("$\\eta$ ($10^{-10}$)")
ax.set_title("Every CMB fit returns the same $\\eta$")
S.save(fig, "part2", "fig_eta")

# ---------------- bottom-up exponent -----------------
from colossus.cosmology import cosmology
from colossus.lss import mass_function
P18 = {'flat': True, 'H0': 67.36, 'Om0': 0.3153, 'Ob0': 0.0493, 'sigma8': 0.8111, 'ns': 0.9649, 'relspecies': True}
cc = cosmology.setCosmology('p18chains', params=P18, persistence='')
lna = np.linspace(np.log(0.02), 0, 240); a = np.exp(lna); z = 1 / a - 1
lnM = np.linspace(np.log(1e4), np.log(1e17), 320); M = np.exp(lnM)
Ez = cc.Ez(z); D = cc.growthFactor(z); f = np.gradient(np.log(D), lna); Omz = cc.Om(z); Dc = 18 * np.pi**2 + 82 * (Omz - 1) - 39 * (Omz - 1)**2
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.7))
for (model, mdef), col, lab in zip((("press74", "fof"), ("sheth99", "fof"), ("tinker08", "200m")), (S.GOLD, S.IAM, S.ALT),
                                   ("Press–Schechter", "Sheth–Tormen", "Tinker et al. 2008")):
    dn = np.array([mass_function.massFunction(M, zz, q_in='M', q_out='dndlnM', mdef=mdef, model=model) for zz in z])
    U = np.trapezoid(dn * M**(5 / 3) * ((Dc * Ez**2)[:, None])**(1 / 3), lnM, axis=1)
    ne = np.gradient(np.log(np.gradient(U, lna) / f), np.log(D))
    cr = z[np.where(np.diff(np.sign(ne - 3.5)))[0]]
    print(f"{model}: n_eff z=4 {np.interp(-4, -z, ne):.2f}, z=2 {np.interp(-2, -z, ne):.2f}; equals 7/2 at z {cr.round(2)}")
    sel = (z <= 9) & (z >= 0.1)
    ax.plot(z[sel], ne[sel], color=col, lw=1.4, label=lab)
ax.axhline(3.5, color=S.GR, lw=0.9, ls="--"); ax.text(8.9, 3.55, "$n=7/2$ (top-down)", fontsize=7, color=S.GR, ha="right", va="bottom")
ax.set_xlim(0, 9); ax.set_ylim(0.8, 6.3)
ax.set_xlabel("redshift $z$"); ax.set_ylabel("bottom-up exponent $n_{\\rm eff}$")
ax.legend(loc="upper left", fontsize=7)
ax.set_title("The bottom-up exponent passes 7/2 at $z\\approx3$–4")
S.save(fig, "part4", "fig_neff_bottomup")

# ---------------- tau vs mass -----------------
hb, kB = C.hbar, C.k; rho = 2200.
EG = lambda m: G * m * m / ((3 * m / (4 * np.pi * rho))**(1 / 3))
tI = lambda m, T: hb * (kB * T)**2 * np.log(2) / EG(m)**3; tPD = lambda m: hb / EG(m)
from scipy.optimize import brentq
mx = 10**brentq(lambda l: np.log(tI(10**l, 0.01) / tPD(10**l)), -20, -5)
print(f"1e-12 kg, 10 mK: tau_IAM {tI(1e-12,0.01):.1f} s, tau_PD {tPD(1e-12)*1e6:.2f} us; crossing {mx:.2e} kg")
mm = np.logspace(-15, -8, 200)
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.7))
for T, ls in ((0.01, "-"), (0.02, "--"), (0.04, ":")):
    ax.plot(mm, tI(mm, T), color=S.IAM, lw=1.4, ls=ls, label=f"$\\tau_{{\\rm IAM}}$, {T*1e3:.0f} mK")
ax.plot(mm, tPD(mm), color=S.GR, lw=1.4, label="$\\tau_{\\rm PD}=\\hbar/E_G$")
ax.plot(1e-12, tI(1e-12, 0.01), "o", color=S.IAM, ms=4); ax.annotate(f"{tI(1e-12,0.01):.0f} s", (1e-12, tI(1e-12, 0.01)), xytext=(6, 2), textcoords="offset points", fontsize=7)
ax.plot(1e-12, tPD(1e-12), "o", color=S.GR, ms=4); ax.annotate(f"{tPD(1e-12)*1e6:.1f} µs", (1e-12, tPD(1e-12)), xytext=(6, -8), textcoords="offset points", fontsize=7)
ax.plot(mx, tPD(mx), "x", color="black", ms=5); ax.annotate(f"cross at {S.sci(mx, 2)} kg", (mx, tPD(mx)), xytext=(-6, -12), textcoords="offset points", fontsize=7, ha="right")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-15, 1e-8); ax.set_ylim(1e-14, 1e18)
ax.set_xticks([1e-14, 1e-12, 1e-10, 1e-8]); ax.set_yticks([1e-12, 1e-6, 1, 1e6, 1e12, 1e18])
ax.set_xlabel("mass of a silica sphere (kg)"); ax.set_ylabel("coherence time (s)")
ax.legend(loc="upper right", fontsize=7)
ax.set_title("Eight orders apart at the nanogram scale")
S.save(fig, "part4", "fig_tau_mass")

# ---------------- Koide -----------------
me, mmu, mt = 0.51099895, 105.6583755, 1776.86; sq = np.sqrt([me, mmu, mt]); x = sq.sum() / 3
Q = (me + mmu + mt) / sq.sum()**2
d = brentq(lambda t: (1 + np.sqrt(2) * np.cos(t)) - sq[2] / x, 0, 1)
print(f"Q = {Q:.8f}; x^2 = {x**2:.3f} MeV; delta = {d:.5f}")
dl = np.linspace(0, 2 * np.pi, 200001); fracs = []
for n in range(2, 8):
    mn = (1 + np.sqrt(2) * np.cos(dl[:, None] + 2 * np.pi * np.arange(n) / n)).min(1); fracs.append(np.mean(mn > 1e-9))
print("fraction of offsets allowed, n = 2..7:", [round(v, 3) for v in fracs])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.35, width_ratios=[1.4, 1]))
ph = np.linspace(0, 2 * np.pi, 400)
a1.plot(ph, x * (1 + np.sqrt(2) * np.cos(ph + d)), color=S.IAM, lw=1.4)
a1.axhline(0, color=S.GR, lw=0.6)
for k in range(3):
    p = 2 * np.pi * k / 3; v = x * (1 + np.sqrt(2) * np.cos(p + d))
    nm = min((("$\\tau$", mt), ("$\\mu$", mmu), ("$e$", me)), key=lambda t: abs(np.sqrt(t[1]) - v))[0]
    a1.plot(p, v, "o", color=S.DATA, ms=5); a1.annotate(f"{nm}: $\\sqrt{{m}}$ = {v:.3f}", (p, v), xytext=(7, 4), textcoords="offset points", fontsize=7)
a1.set_xticks([0, 2 * np.pi / 3, 4 * np.pi / 3, 2 * np.pi]); a1.set_xticklabels(["0", "$2\\pi/3$", "$4\\pi/3$", "$2\\pi$"])
a1.set_xlim(-0.2, 2 * np.pi + 0.2); a1.set_ylim(-20, 50)
a1.set_xlabel("phase on the charge orbit $\\phi$"); a1.set_ylabel("$\\sqrt{m}$ (MeV$^{1/2}$)")
a1.set_title(f"Equal channels: $Q$ = 2/3 (measured {Q:.5f})")
S.panel_letter(a1, "a", dx=-0.14)
ns = list(range(2, 8))
a2.plot(ns, fracs, "o", color=S.IAM, ms=5)
for n, v in zip(ns, fracs):
    a2.plot([n, n], [0, v], color=S.LIGHT, lw=1.0, zorder=1)
    a2.annotate(f"{v:.2f}", (n, v), xytext=(0, 5), textcoords="offset points", fontsize=7, ha="center")
a2.set_xticks(ns); a2.set_xlim(1.5, 7.5); a2.set_ylim(-0.03, 0.62)
a2.set_xlabel("number of phases $n$"); a2.set_ylabel("fraction of offsets with all $m>0$")
a2.set_title("At most three generations")
S.panel_letter(a2, "b", dx=-0.2)
S.save(fig, "part2", "fig_koide")

# ---------------- electron fixed point -----------------
al, mE = C.alpha, C.m_e; mP = np.sqrt(hbar * c / G); MpcE = 3.0856775814913673e22
mfix = lambda H0: (2 * np.pi)**-0.1 * (hbar * (H0 * 1e3 / MpcE) * np.log(2) * mP**1.5 / (al**2.5 * c**2))**0.4
HH = np.linspace(66, 75, 300)
for h in (67.36, 67.4, 73.04):
    print(f"H0 {h}: deviation {100*(mfix(h)/mE-1):+.4f} %")
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.5))
ax.axvspan(67.36 - 0.54, 67.36 + 0.54, color=S.DATA, alpha=0.15, lw=0)
ax.plot(HH, 100 * (mfix(HH) / mE - 1), color=S.IAM, lw=1.5)
ax.axhline(0, color=S.GR, lw=0.8, ls="--")
for h, lab, col in ((67.36, "Planck 67.36", S.DATA), (73.04, "SH0ES 73.04", S.DATA)):
    v = 100 * (mfix(h) / mE - 1); ax.plot(h, v, "s", color=col, ms=4)
    ax.annotate(f"{lab}: {v:+.2f} %", (h, v), xytext=(-6, 6) if h > 70 else (8, -10), textcoords="offset points", fontsize=7, ha="right" if h > 70 else "left")
ax.set_xlim(66, 75); ax.set_ylim(-1.5, 4.5)
ax.set_xlabel("$H_0$ (km s$^{-1}$ Mpc$^{-1}$)"); ax.set_ylabel("$m_e$(fixed point)$/m_e - 1$ (%)")
ax.set_title("$m_e\\propto H_0^{2/5}$: as precise as $H_0$")
S.save(fig, "part2", "fig_electron_h0")
