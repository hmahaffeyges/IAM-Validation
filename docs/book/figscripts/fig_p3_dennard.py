"""part5/fig_dennard (Chapter 'Landauer at the transistor gate: CMOS and the Dennard wall', Figure fig:dennard).

Relative power density P/A = C V^2 f / A over node steps kappa = sqrt(2)^k (Dennard et al. 1974, IEEE JSSC 9, 256):
constant-field scaling (C ~ 1/kappa, V ~ 1/kappa, f ~ kappa, A ~ 1/kappa^2) keeps it at 1; with the voltage held fixed it grows as kappa^2.
Derived from the scaling rules; no data. Run: python docs/book/figscripts/fig_p3_dennard.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()

k = np.sqrt(2) ** np.arange(7)
pd_cf = (1 / k) * (1 / k) ** 2 * k / (1 / k ** 2)      # C V^2 f / A, constant field
pd_fv = (1 / k) * 1.0 * k / (1 / k ** 2)                # voltage fixed
fig, ax = plt.subplots(figsize=(0.7 * S.TEXTW, 2.8))
ax.plot(k, pd_cf, "o-", color=S.IAM, ms=4, label=r"constant field (Dennard 1974): $V\propto1/\kappa$")
ax.plot(k, pd_fv, "o-", color=S.DATA, ms=4, label=r"voltage held fixed: $\propto\kappa^2$")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel(r"linear scaling factor $\kappa$"); ax.set_ylabel("power density (relative)")
ax.legend(loc="upper left"); ax.set_title("Power density once voltage stops scaling")
S.save(fig, "part5", "fig_dennard")
