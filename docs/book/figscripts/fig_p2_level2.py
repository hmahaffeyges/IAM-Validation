"""Part 2, Chapter 'The mechanism in the Boltzmann code' (p2_06_dual_sector_perturbation.tex).
fig_sector_rates: H(z) for photons (LambdaCDM background) and H_m(z) for matter, H_m^2 = H^2 + beta_m E(a) H0^2, at the Run A posterior
(H0 = 67.161, Omega_m = 0.3166; mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv); (b) their ratio.
fig_param_shifts: parameter shifts Run A (informational term) minus Run C (LambdaCDM, same code), in units of Run C's posterior sd,
from camb_validation/chains (30 % burn-in, weighted)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, csv
import _bookstyle as S, _chains as CH
import matplotlib.pyplot as plt

bm = 0.15765
rows = {r["chain"]: r for r in csv.DictReader(open(S.REPO / "mgcamb_validation" / "CHAIN_EXTRACTION_FINAL.csv"))}
H0, Om = float(rows["iam_level2_runA"]["H0"]), float(rows["iam_level2_runA"]["omegam"])
H = lambda z: H0 * np.sqrt(Om * (1 + z)**3 + 1 - Om)
Hm = lambda z: np.sqrt(H(z)**2 + bm * np.exp(-z) * H0**2)
for z in (0, 0.5, 1, 2, 3):
    print(f"z {z}: H {H(z):.2f}  H_m {Hm(z):.2f}  ratio {Hm(z)/H(z):.4f}")
S.apply()
z = np.linspace(0, 3, 300)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.plot(z, H(z) / (1 + z), color=S.GR, lw=1.4); a1.plot(z, Hm(z) / (1 + z), color=S.IAM, lw=1.6)
a1.text(1.55, 57.5, "photons: $H$\n($\\Lambda$CDM background)", fontsize=7, color=S.GR, va="center")
a1.text(0.25, 71.5, "matter: $H_m$", fontsize=7, color=S.IAM, va="bottom")
a1.set_xlim(0, 3); a1.set_ylim(55, 76)
a1.set_xlabel("redshift $z$"); a1.set_ylabel("$H(z)/(1+z)$ (km s$^{-1}$ Mpc$^{-1}$)")
a1.set_title("Matter feels a faster expansion, late only")
S.panel_letter(a1, "a", dx=-0.16)
a2.plot(z, Hm(z) / H(z), color=S.IAM, lw=1.6); a2.axhline(1, color=S.GR, lw=0.8, ls="--")
for q in (0, 0.5, 1, 2):
    r = Hm(q) / H(q); a2.plot(q, r, "o", color=S.IAM, ms=4)
    a2.annotate(f"{r:.4f}", (q, r), xytext=(6, 3), textcoords="offset points", fontsize=7)
a2.set_xlim(-0.05, 3); a2.set_ylim(0.99, 1.09)
a2.set_xlabel("redshift $z$"); a2.set_ylabel("$H_m/H$")
a2.set_title("$\\sqrt{1+\\beta_m}=1.0759$ today")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_sector_rates")

A = CH.load(*CH.L2["A"]); Cc = CH.load(*CH.L2["C"])
pars = [("H0", "$H_0$"), ("sigma8", "$\\sigma_8$"), ("S8", "$S_8$"), ("ombh2", "$\\omega_b$"), ("omch2", "$\\omega_c$"),
        ("tau", "$\\tau$"), ("ns", "$n_s$"), ("logA", "$\\ln10^{10}A_s$"), ("omegam", "$\\Omega_m$")]
sh = []
for p, lab in pars:
    ma, sa = CH.wmean_sd(A[p].values, A.weight.values); mc, sc = CH.wmean_sd(Cc[p].values, Cc.weight.values)
    sh.append((lab, (ma - mc) / sc)); print(f"{p:7s} C {mc:.5f} +/- {sc:.5f}  A {ma:.5f} +/- {sa:.5f}  shift {(ma-mc)/sc:+.2f} sigma")
fig, ax = plt.subplots(figsize=(0.6 * S.TEXTW, 2.6))
y = np.arange(len(sh))[::-1]
for yi, (lab, v) in zip(y, sh):
    ax.plot([0, v], [yi, yi], color=S.LIGHT, lw=1.0); ax.plot(v, yi, "o", color=S.IAM if abs(v) > 0.5 else S.GR, ms=5)
    ax.annotate(f"{v:+.2f}" if abs(v) >= 0.005 else "0.00", (v, yi), xytext=(-6 if v < 0 else 6, 0), textcoords="offset points", fontsize=7, va="center", ha="right" if v < 0 else "left")
ax.axvline(0, color="black", lw=0.6)
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in sh], fontsize=7)
ax.set_xlim(-2.0, 0.6); ax.set_xlabel("shift, informational term minus $\\Lambda$CDM ($\\sigma$)")
ax.set_title("Only the growth amplitude moves")
S.save(fig, "part2", "fig_param_shifts")
