"""Part 2, Chapter 'The cosmological coupling beta_m = Omega_m/2' (p2_02_virial.tex).
fig_record_history: (a) R(a) = Omega_m a^-3/(beta_m E(a)); (b) the record term's share of Omega_L + beta_m E(a) in the matter-sector rate.
fig_growth_vs_geometry: (a) Omega_m inferred from growth alone in LambdaCDM, Omega_m mu(z), against the geometric Omega_m;
(b) the E_G statistic, E_G(IAM)/E_G(GR) - 1 = f_LCDM/f_IAM - 1 (Sigma = 1), from the linear growth equation (_cosmo.py)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import _bookstyle as S, _cosmo as K
import matplotlib.pyplot as plt

S.apply()
z = np.linspace(0, 3, 400); a = 1 / (1 + z)
R = K.Om * a**-3 / (K.bm * K.E(a))
share = 100 * K.bm * K.E(a) / (K.OL + K.bm * K.E(a))
for zz in (0, 0.3, 0.7, 1.5, 2.0):
    aa = 1 / (1 + zz); print(f"z {zz}: R {K.Om*aa**-3/(K.bm*K.E(aa)):.1f}  share {100*K.bm*K.E(aa)/(K.OL+K.bm*K.E(aa)):.1f} %")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.plot(z, R, color=S.IAM, lw=1.5); a1.set_yscale("log")
for zz in (0, 0.3, 0.7, 2.0):
    aa = 1 / (1 + zz); v = K.Om * aa**-3 / (K.bm * K.E(aa)); a1.plot(zz, v, "o", color=S.IAM, ms=4)
    a1.annotate(f"{v:.0f}", (zz, v), xytext=(5, -3), textcoords="offset points", fontsize=7, va="top")
a1.axhline(2, color=S.LIGHT, lw=0.8, ls=":")
a1.set_xlim(-0.05, 3); a1.set_ylim(1, 1e4); a1.set_yticks([1, 10, 100, 1000, 10000]); a1.set_yticklabels(["1", "10", "100", "1k", "10k"])
a1.set_xlabel("redshift $z$"); a1.set_ylabel("$R=\\Omega_m a^{-3}/\\beta_mE(a)$")
a1.set_title("Matter over record term")
S.panel_letter(a1, "a", dx=-0.16)
a2.plot(z, share, color=S.IAM, lw=1.5)
for zz in (0, 0.3, 0.7, 1.5):
    aa = 1 / (1 + zz); v = 100 * K.bm * K.E(aa) / (K.OL + K.bm * K.E(aa)); a2.plot(zz, v, "o", color=S.IAM, ms=4)
    a2.annotate(f"{v:.1f} %", (zz, v), xytext=(5, 2), textcoords="offset points", fontsize=7)
a2.set_xlim(-0.05, 3); a2.set_ylim(0, 21)
a2.set_xlabel("redshift $z$"); a2.set_ylabel("record share of $\\Omega_\\Lambda+\\beta_mE$ (%)")
a2.set_title("The record term switches on late")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_record_history")

zz = np.linspace(0, 2, 300); aa = 1 / (1 + zz)
Omg = K.Om * K.mu(aa)
EG = 100 * (K.f(K.LCDM, aa) / K.f(K.IAM, aa) - 1)
print("Omega_m growth at z=0.5:", round(K.Om * K.mu(1 / 1.5), 4), " deficit %:", round(100 * (1 - K.mu(1 / 1.5)), 2))
print("E_G change z=0, 0.295:", [round(float(100 * (K.f(K.LCDM, 1/(1+q)) / K.f(K.IAM, 1/(1+q)) - 1)), 2) for q in (0, 0.295)])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.axhline(K.Om, color=S.GR, lw=1.0, ls="--"); a1.text(1.95, K.Om + 0.001, "geometry (BAO, CMB): $\\Omega_m$ = 0.3153", fontsize=7, ha="right", va="bottom", color=S.GR)
a1.plot(zz, Omg, color=S.IAM, lw=1.5); a1.text(0.55, 0.287, "growth read in $\\Lambda$CDM: $\\Omega_m\\mu(z)$", fontsize=7, color=S.IAM)
p = K.Om * K.mu(1 / 1.5); a1.plot(0.5, p, "o", color=S.IAM, ms=4); a1.annotate(f"{p:.3f}", (0.5, p), xytext=(-5, 4), textcoords="offset points", fontsize=7, ha="right")
a1.set_xlim(0, 2); a1.set_ylim(0.265, 0.325); a1.set_yticks([0.27, 0.29, 0.31])
a1.set_xlabel("redshift $z$"); a1.set_ylabel("inferred $\\Omega_m$")
a1.set_title("Growth and geometry give different $\\Omega_m$")
S.panel_letter(a1, "a", dx=-0.17)
a2.plot(zz, EG, color=S.IAM, lw=1.5); a2.axhline(0, color=S.GR, lw=0.8, ls="--")
a2.text(1.95, 0.12, "general relativity", fontsize=7, ha="right", color=S.GR)
for q in (0, 0.295):
    v = 100 * (K.f(K.LCDM, 1 / (1 + q)) / K.f(K.IAM, 1 / (1 + q)) - 1); a2.plot(q, v, "o", color=S.IAM, ms=4)
    a2.annotate(f"+{v:.1f} %", (q, v), xytext=(6, 0), textcoords="offset points", fontsize=7, va="center")
a2.set_xlim(-0.03, 2); a2.set_ylim(-0.3, 4.2)
a2.set_xlabel("redshift $z$"); a2.set_ylabel("$E_G$ change (%)")
a2.set_title("$E_G$ lies above general relativity")
S.panel_letter(a2, "b", dx=-0.17)
S.save(fig, "part2", "fig_growth_vs_geometry")
