"""fig_eps0_T: copy-error floor against body temperature (Part 4, temperature chapter).
E_hold = 3.41 kT at 310.15 K (CANON/iam_canon.json). Solid: fixed holding energy, eps0(T) = 1/(1+exp(E_hold/kT)).
Dashed: error held fixed in units of kT (eps0 constant). Floor plotted as H(eps0(T))/H(eps0(310.15 K))."""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
E, T0 = 3.41, 310.15
H = lambda x: -(x*np.log2(x) + (1-x)*np.log2(1-x))
eps = lambda T: 1/(1+np.exp(E*T0/T))
T = np.linspace(273.15+0, 273.15+42, 300); r = H(eps(T))/H(eps(T0))
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.plot(T-273.15, r, color="#1F5FA8", lw=1.6, label="fixed holding energy")
ax.axhline(1, color="#888888", lw=1.0, ls="--", label=r"error fixed in units of $k_BT$")
for t, lab, dx, dy in ((10, "salmonid 10 °C", 2.0, -0.05), (37, "human 37 °C", -11.0, -0.045), (38.5, "dog 38.5 °C", -12.5, 0.022)):
    y = H(eps(t+273.15))/H(eps(T0)); ax.plot(t, y, "o", color="#C0392B", ms=4)
    ax.annotate(f"{lab}: {y:.3f}×", (t, y), xytext=(t+dx, y+dy), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.4, color="#555555"))
ax.set_ylim(0.68, 1.08)
ax.set_xlabel("body temperature (°C)"); ax.set_ylabel("floor relative to 37 °C"); ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig("figures/part6/fig_eps0_T.pdf"); fig.savefig("figures/part6/fig_eps0_T.png", dpi=200)
print(round(eps(283.15), 4), round(H(eps(283.15))/H(eps(T0)), 3), round(H(eps(311.65))/H(eps(T0)), 3))
