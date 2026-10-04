"""Part 1, Chapter 'One half at every scale' (p1_03_virial_law.tex), Table tab:virial_domains.
fig_virial_domains: the virial share that the 1/r theorem fixes at 1/2, by system size: hydrogen (<K>/|<V>|, exact),
Hartree-Fock atoms and molecules (1/2 by construction), simulated halos (published 2K/|W| = 1.1-1.3, 1.02-1.17 with surface pressure,
plotted as K/|W|), Schwarzschild horizons (T_H S/(M c^2), computed), cosmic horizon (beta_m/Omega_m: the fixed coupling over the
Level 2 Run A posterior Omega_m, Cosmological_Physics/mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv).
fig_binding_ledger: (a) hydrogen captured from rest; (b) the Sun's contraction (Kelvin-Helmholtz) time against its age.
Numbers as docs/verification/scripts/verify_virial_atoms_to_horizon.py."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C, csv
import _bookstyle as S
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

G, c, hbar, k = C.G, C.c, C.hbar, C.k
Msun, Rsun, Lsun, yr = 1.98847e30, 6.957e8, 3.828e26, 3.15576e7
Eh = C.physical_constants["Hartree energy in eV"][0]; a0 = C.physical_constants["Bohr radius"][0]
K_H, V_H = Eh / 2, -Eh
ratio_H = K_H / abs(V_H)
bhs = []
for m in (1.0, 10.0, 1e6, 4.3e6, 1e9, 6.5e9):
    M = m * Msun; T = hbar * c**3 / (8 * np.pi * G * M * k); Sbh = k * 4 * np.pi * G * M**2 / (hbar * c)
    bhs.append((2 * G * M / c**2, T * Sbh / (M * c**2)))
rowsC = {r["chain"]: r for r in csv.DictReader(open(S.REPO / "Cosmological_Physics/mgcamb_validation" / "CHAIN_EXTRACTION_FINAL.csv"))}
Om_A, Om_A_sd = float(rowsC["iam_level2_runA"]["omegam"]), float(rowsC["iam_level2_runA"]["omegam_sd"])
bm = 0.15765; r_cos = bm / Om_A; r_cos_sd = bm * Om_A_sd / Om_A**2
R_H = c / (67.36e3 / 3.0857e22)
print(f"hydrogen K/|V| = {ratio_H:.4f};  BH T S/Mc^2 = {[round(b[1], 10) for b in bhs]}")
print(f"cosmic: beta_m/Omega_m(Run A) = {r_cos:.4f} +/- {r_cos_sd:.4f}  (Omega_m {Om_A:.4f})")
print("halos K/|W| from 2K/|W| 1.1-1.3:", 0.55, 0.65, "; with surface pressure 1.02-1.17:", 0.51, 0.585)

S.apply()
fig, ax = plt.subplots(figsize=(S.TEXTW, 3.0))
ax.axhline(0.5, color=S.GR, lw=0.8, ls="--", zorder=1)
ax.text(2e-12, 0.503, "1/2", fontsize=7, color=S.GR, va="bottom")
ax.plot(a0, ratio_H, "o", color=S.IAM, ms=5, zorder=3)
ax.annotate("hydrogen atom\n$\\langle K\\rangle/|\\langle V\\rangle|$, exact", (a0, ratio_H), xytext=(4, -14), textcoords="offset points", fontsize=7, ha="left", va="top")
ax.plot([1e-10, 1e-9], [0.5, 0.5], "s", mfc="white", mec=S.IAM, ms=5, zorder=3)
ax.annotate("20 atoms, 10 molecules\n(Hartree–Fock, $T/|V|$;\nexact by construction)", (1e-9, 0.5), xytext=(4, 22), textcoords="offset points", fontsize=7, ha="left", va="bottom")
ax.add_patch(Rectangle((3e21, 0.55), 3e23 - 3e21, 0.10, fc=S.DATA, alpha=0.25, ec="none", zorder=2))
ax.add_patch(Rectangle((3e21, 0.51), 3e23 - 3e21, 0.075, fc="none", ec=S.DATA, hatch="////", lw=0.6, zorder=2))
ax.text(1.5e21, 0.655, "simulated halos, $K/|W|$", fontsize=7, ha="right", va="center", color=S.DATA)
ax.text(1.5e21, 0.60, "within $r_{\\rm vir}$", fontsize=7, ha="right", va="center", color=S.DATA)
ax.text(1.5e21, 0.545, "with surface pressure (hatched)", fontsize=7, ha="right", va="center", color=S.DATA)
xs, ys = zip(*bhs)
ax.plot(xs, ys, "D", color=S.IAM, ms=4, zorder=3)
ax.annotate("Schwarzschild horizons\n$T_HS/Mc^2$ (Smarr)", (np.sqrt(xs[0] * xs[-1]), 0.5), xytext=(0, -26), textcoords="offset points", fontsize=7, ha="center", va="top")
ax.errorbar(R_H, r_cos, yerr=r_cos_sd, fmt="o", color=S.DATA, ms=4, capsize=2, lw=0.8, zorder=3)
ax.plot(R_H, 0.5, "o", mfc="white", mec=S.IAM, ms=7, mew=1.0, zorder=2)
ax.annotate("cosmic horizon\n$\\beta_m/\\Omega_m$: predicted 1/2 (open);\nfixed $\\beta_m$ over Planck $\\Omega_m$ (point)", (R_H, 0.48), xytext=(0, -24), textcoords="offset points", fontsize=7, ha="right", va="top")
ax.set_xscale("log"); ax.set_xlim(1e-12, 1e28); ax.set_ylim(0.36, 0.70)
ax.set_xticks([1e-10, 1e-5, 1, 1e5, 1e10, 1e15, 1e20, 1e25])
ax.set_xlabel("size of the bound system (m)")
ax.set_ylabel("share held by the kinetic half")
ax.set_title("The 1/r virial half holds from the atom to the cosmic horizon")
S.save(fig, "part1", "fig_virial_domains")

# ---------------- binding ledger -----------------
U = -1.5 * G * Msun**2 / Rsun; tKH = abs(U) / 2 / Lsun / yr / 1e6
print(f"hydrogen: V {V_H:.4f}, K {K_H:.4f}, E {-K_H:.4f}, photon {K_H:.4f} eV;  Sun: |U|/2 = {abs(U)/2:.3e} J, t_KH = {tKH:.1f} Myr")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.2), gridspec_kw=dict(wspace=0.85, width_ratios=[1.25, 1]))
labs = ["potential $\\langle V\\rangle$", "kinetic $\\langle K\\rangle$", "total $E$", "photon emitted on capture"]
vals = [V_H, K_H, -K_H, K_H]; cols = [S.GR, S.IAM, S.GR, S.DATA]
y = np.arange(4)[::-1]
a1.barh(y, vals, color=cols, height=0.6)
for yi, v in zip(y, vals):
    if v > 0:
        a1.text(v + 0.8, yi, f"{v:+.2f} eV", va="center", ha="left", fontsize=7)
    else:
        a1.text(v - 0.8, yi, f"{v:+.2f} eV", va="center", ha="right", fontsize=7)
a1.axvline(0, color="black", lw=0.6)
a1.set_yticks(y); a1.set_yticklabels(labs, fontsize=7); a1.set_xlim(-50, 26); a1.set_xticks([-40, -20, 0, 20])
a1.set_xlabel("energy (eV)")
a1.set_title("Hydrogen: heat released = $\\langle K\\rangle$")
S.panel_letter(a1, "a", dx=-0.62)
t_sun = 4568  # Sun's age in Myr, Bouvier & Wadhwa 2010 (\cite{Bouvier2010} in the caption)
a2.barh([1, 0], [tKH, t_sun], color=[S.IAM, S.GR], height=0.55)
a2.plot(tKH, 1, "o", color=S.IAM, ms=4)
a2.text(tKH + 150, 1, f"{tKH:.1f} Myr", va="center", fontsize=7)
a2.text(t_sun - 80, 0, f"{t_sun:,} Myr", va="center", ha="right", fontsize=7, color="white")
a2.set_yticks([1, 0]); a2.set_yticklabels(["contraction alone, $|U|/2L$", "the Sun's age"], fontsize=7)
a2.set_xlim(0, 5000); a2.set_xlabel("time (Myr)")
a2.set_title("The Sun cannot run on contraction")
S.panel_letter(a2, "b", dx=-0.75)
S.save(fig, "part1", "fig_binding_ledger")
