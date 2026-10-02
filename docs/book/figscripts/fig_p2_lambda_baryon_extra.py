"""Part 2, Chapters 'The cosmological constant' (p2_12_lambda.tex) and 'The baryon density' (p2_13_baryon.tex).
Constants and inputs as docs/verification/scripts/verify_cc_and_baryon.py (CODATA via scipy.constants; Planck 2018: H0 67.4,
Ob 0.0493, Om 0.3153, OL 0.6846). Chains read with _chains.py (30 % burn-in, weighted).
fig_cc_factors: (a) the ratio of the expression to the measured rho_Lambda/rho_vac as each factor is applied in turn:
  (l_P/l_H)^2, x Ob/Om, x 2/pi, x sqrt(OL); the identity factor 3 OL/8 pi closes it exactly. (b) The same ratio for
  (2/pi)(l_P/l_H)^2 (Ob/Om) OL^p against the exponent p; p = 0.521 closes it, p = 1/2 leaves +0.79 %.
fig_baryon_posterior: (a) the Omega_b h^2 posterior of the 18th chain (flat range 0.010-0.040, shaded) and of the three
  LambdaCDM chains; (b) the posterior of (Ob/Om)/((3/16) sqrt(OL)) on the same chains.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
import _bookstyle as S
import _chains as Ch
import matplotlib.pyplot as plt

S.apply()
hbar, c, G = C.hbar, C.c, C.G; Mpc = 3.0857e22
lP = np.sqrt(hbar * G / c**3); EP = np.sqrt(hbar * c**5 / G)
H0 = 67.4e3 / Mpc; lH = c / H0; Ob, Om, OL = 0.0493, 0.3153, 0.6846
rvac = EP**4 / (hbar * c)**3; rc = 3 * H0**2 / (8 * np.pi * G) * c**2; obs = OL * rc / rvac
steps = [("$(l_P/l_H)^2$", (lP / lH)**2), ("$\\times\\,\\Omega_b/\\Omega_m$", (lP / lH)**2 * Ob / Om),
         ("$\\times\\,2/\\pi$", 2 / np.pi * (lP / lH)**2 * Ob / Om), ("$\\times\\,\\sqrt{\\Omega_\\Lambda}$", 2 / np.pi * (lP / lH)**2 * Ob / Om * np.sqrt(OL))]
ratios = [v / obs for _, v in steps]
pclose = np.log(obs / steps[2][1]) / np.log(OL)
print("observed ratio %.4e; step ratios %s; exponent closing %.3f; identity %.4e" % (obs, [round(r, 4) for r in ratios], pclose, 3 * OL / (8 * np.pi) * (lP / lH)**2))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.33, width_ratios=[1.15, 1]))
x = np.arange(len(steps))
a1.plot(x, ratios, "o-", color=S.IAM, ms=5, lw=1.2)
for i, rv in enumerate(ratios):
    a1.annotate(f"{rv:.3f}" if rv < 2 else f"{rv:.2f}", (i, rv), xytext=(6, 4), textcoords="offset points", fontsize=7, color=S.IAM)
a1.axhline(1, color=S.DATA, lw=0.9, ls="--"); a1.text(-0.3, 1.06, "measured", fontsize=7, color=S.DATA, ha="left", va="bottom")
a1.set_yscale("log"); a1.set_ylim(0.6, 25); a1.set_xlim(-0.4, 3.5)
a1.set_xticks(x); a1.set_xticklabels([s for s, _ in steps], fontsize=7)
a1.set_yticks([1, 2, 5, 10, 20]); a1.set_yticklabels(["1", "2", "5", "10", "20"])
a1.set_ylabel("expression / measured $\\rho_\\Lambda/\\rho_{\\rm vac}$")
a1.set_title("Each factor brings the ratio towards 1")
S.panel_letter(a1, "a", dx=-0.14)
p = np.linspace(0, 1.2, 241); rp = steps[2][1] * OL**p / obs
a2.plot(p, 100 * (rp - 1), color=S.IAM, lw=1.6)
a2.axhline(0, color=S.DATA, lw=0.9, ls="--")
a2.plot(0.5, 100 * (OL**0.5 * steps[2][1] / obs - 1), "o", color=S.IAM, ms=4.5)
a2.annotate(f"$p$ = 1/2: {100*(ratios[3]-1):+.2f} %", (0.5, 100 * (ratios[3] - 1)), xytext=(8, 8), textcoords="offset points", fontsize=7, ha="left")
a2.plot(pclose, 0, "s", color=S.DATA, ms=4)
a2.annotate(f"closes at $p$ = {pclose:.3f}", (pclose, 0), xytext=(8, -10), textcoords="offset points", fontsize=7, color=S.DATA)
a2.set_xlim(0, 1.2); a2.set_ylim(-25, 25)
a2.set_xlabel("exponent $p$ of $\\Omega_\\Lambda$"); a2.set_ylabel("offset from measured (%)")
a2.set_title("The temperature factor as an exponent")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_cc_factors")

# ---------------- baryon posterior -----------------
MG = Ch.MG
sets = (("18th chain (CMB only, wide range)", [MG + "iam_baryon_test.1.txt"], S.IAM, "-"),
        ("$\\Lambda$CDM, Planck", [MG + "lcdm_baseline.1.txt"], S.GR, "--"),
        ("$\\Lambda$CDM, Planck + BAO", [MG + "planck_bao_lcdm_baseline.1.txt"], S.ALT, "--"),
        ("$\\Lambda$CDM, Planck + Pantheon+", [MG + "planck_pantheon_lcdm_baseline.1.txt"], S.ALT2, ":"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.33))
a1.axvspan(0.010, 0.040, color=S.LIGHT, alpha=0.3, lw=0); a1.axvspan(0.020, 0.025, color=S.LIGHT, alpha=0.5, lw=0)
a1.text(0.0105, 0.97, "18th-chain range 0.010-0.040", transform=a1.get_xaxis_transform(), fontsize=6.5, color=S.GR, va="top")
a1.text(0.02025, 0.30, "range of the other runs", transform=a1.get_xaxis_transform(), fontsize=6, color=S.GR, rotation=90, va="bottom")
zin = a1.inset_axes([0.60, 0.30, 0.37, 0.48])
for nm, fl, col, ls in sets:
    X = Ch.load(*fl); w = X.weight.values; ob = X.ombh2.values; h = X.H0.values / 100; OLx = X.omegal.values
    rr = ob / h**2 / (1 - OLx) / (3 / 16 * np.sqrt(OLx))
    m, s = Ch.wmean_sd(ob, w); q, qs = Ch.wmean_sd(rr, w)
    print(f"{nm}: rows {len(X)}  ombh2 {m:.5f} +/- {s:.5f}  eta {273.9*m:.3f}  ratio {q:.4f} +/- {qs:.4f}")
    hb, e = np.histogram(ob, bins=np.linspace(0.0215, 0.0232, 41), weights=w, density=True)
    a1.step(0.5 * (e[1:] + e[:-1]), hb / hb.max(), where="mid", color=col, lw=1.2, ls=ls, label=nm)
    zin.step(0.5 * (e[1:] + e[:-1]), hb / hb.max(), where="mid", color=col, lw=0.9, ls=ls)
    hr, e2 = np.histogram(rr, bins=np.linspace(0.97, 1.05, 41), weights=w, density=True)
    a2.step(0.5 * (e2[1:] + e2[:-1]), hr / hr.max(), where="mid", color=col, lw=1.2, ls=ls, label=nm)
zin.set_xlim(0.0217, 0.0230); zin.set_ylim(0, 1.1); zin.set_yticks([]); zin.set_xticks([0.0220, 0.0230])
zin.set_xticklabels(["0.0220", "0.0230"], fontsize=5.5); zin.set_title("zoom", fontsize=6)
a1.set_xlim(0.0098, 0.0402); a1.set_ylim(0, 1.25)
a1.set_xlabel("$\\Omega_bh^2$"); a1.set_ylabel("posterior (peak = 1)")
a1.set_title("The peaks fix $\\Omega_bh^2$ within any range")
S.panel_letter(a1, "a", dx=-0.14)
a2.axvline(1, color=S.DATA, lw=0.9, ls="--")
a2.set_xlim(0.97, 1.05); a2.set_ylim(0, 1.6)
a2.set_xlabel("$(\\Omega_b/\\Omega_m)\\,/\\,[(3/16)\\sqrt{\\Omega_\\Lambda}]$"); a2.set_ylabel("posterior (peak = 1)")
a2.legend(loc="upper left", fontsize=6)
a2.set_title("The relation on each chain")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_baryon_posterior")
