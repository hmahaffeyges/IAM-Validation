"""Free-coupling fit to three H0 values and seven f sigma8 points (Chapter ch:dual).
(a) chi2 against beta: total, H0 entries and f sigma8 entries, each from its own minimum.
(b) the f sigma8 points against LambdaCDM and the fixed coupling beta_m = Omega_m/2, same early amplitude.
Numbers: docs/verification/scripts/verify_summary_papers.py."""
import numpy as np
from scipy.integrate import solve_ivp
import _bookstyle as bs
import matplotlib.pyplot as plt

bs.apply()
Om = 0.3153; OL = 1 - Om; bm = Om / 2
H2 = lambda a: Om * a**-3 + OL
E = lambda a: np.exp(1 - 1 / a)
def growth(beta):
    def rhs(l, y):
        a = np.exp(l); h2 = H2(a); mu = h2 / (h2 + beta * E(a))
        return [y[1], -(2 - 1.5 * Om * a**-3 / h2) * y[1] + 1.5 * Om * a**-3 / h2 * mu * y[0]]
    return solve_ivp(rhs, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-13).sol
solL = growth(0.0); DL1 = solL(0)[0]
def fs8(sol, z):
    y = sol(np.log(1 / (1 + z))); return y[1] * 0.811 / DL1
H0d = [(67.40, 0.50, 0), (73.04, 1.04, 1), (70.39, 1.94, 1)]
fsd = np.array([[0.067, 0.423, 0.055], [0.150, 0.530, 0.160], [0.380, 0.497, 0.045], [0.510, 0.459, 0.038],
                [0.700, 0.473, 0.041], [0.850, 0.315, 0.095], [1.480, 0.462, 0.045]])
betas = np.linspace(0, 0.40, 81)
cH, cG = [], []
for b in betas:
    cH.append(sum(((v - (67.4 * np.sqrt(1 + b) if m else 67.4)) / s) ** 2 for v, s, m in H0d))
    sol = growth(b); cG.append(np.sum(((fsd[:, 1] - np.array([fs8(sol, z) for z in fsd[:, 0]])) / fsd[:, 2]) ** 2))
cH, cG = np.array(cH), np.array(cG); cT = cH + cG

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(bs.TEXTW, 2.4))
ax1.plot(betas, cT - cT.min(), color=bs.IAM, label="all ten entries")
ax1.plot(betas, cH - cH.min(), color=bs.DATA, ls="--", label="three $H_0$ entries")
ax1.plot(betas, cG - cG.min(), color=bs.ALT, ls=":", label=r"seven $f\sigma_8$ points")
for lev in (1, 4):
    ax1.axhline(lev, color=bs.LIGHT, lw=0.6, zorder=0)
ax1.axvline(bm, color=bs.GR, lw=0.8)
ax1.text(bm + 0.004, 5.0, r"$\Omega_m/2$", fontsize=6, color=bs.GR)
ax1.set_xlim(0, 0.40); ax1.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); ax1.set_ylim(0, 10)
ax1.set_xlabel(r"coupling $\beta$"); ax1.set_ylabel(r"$\Delta\chi^2$ from minimum")
ax1.legend(loc="upper right", fontsize=6)
bs.panel_letter(ax1, "a")
zz = np.linspace(0, 1.6, 200); solI = growth(bm)
ax2.plot(zz, [fs8(solL, z) for z in zz], color=bs.GR, label=r"$\Lambda$CDM")
ax2.plot(zz, [fs8(solI, z) for z in zz], color=bs.IAM, label=r"$\beta_m=\Omega_m/2$")
ax2.errorbar(fsd[:, 0], fsd[:, 1], yerr=fsd[:, 2], fmt="o", ms=2.5, color=bs.DATA, elinewidth=0.7, capsize=1.5, label="data")
ax2.set_xlabel("redshift $z$"); ax2.set_ylabel(r"$f\sigma_8$"); ax2.set_ylim(0.2, 0.72)
ax2.legend(loc="upper right", fontsize=6)
bs.panel_letter(ax2, "b")
fig.tight_layout()
bs.save(fig, "part2", "fig_freecoupling_fit")
