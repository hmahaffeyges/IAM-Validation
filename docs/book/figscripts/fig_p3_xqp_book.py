"""Chapter ch:xqp (part3/p3_02_xqp.tex), figures added in the line-for-line carriage (2026-10-03).
fig_xqp_ladder : energy scales at an Al/AlOx/Al junction written as temperatures (T_CMB, T_gap = Delta/k_B, T_c, T_fridge) and the
                 junction cross-section with high-transmission sites. Delta = 182 ueV, T_c = 1.20 K.
fig_xqp_london : integrand 4 pi r^2 exp(-2r/lambda_L) and its running integral, which reaches pi lambda_L^3 (lambda_L = 50 nm).
fig_xqp_tau    : x_qp = 2 N tau_qp/(tau_TLS n_cp V) against tau_TLS for N/V = 0.006, 0.06, 0.6 um^-3; n_cp = 4e6 um^-3, tau_qp = 100 us.
Numbers: docs/verification/scripts/verify_xqp_book.py."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
import _bookstyle as S
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

S.apply()
k = C.k; Delta = 182e-6 * C.e; Tgap = Delta / k

# ---------------- ladder of scales + junction sketch ----------------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw={"width_ratios": [1.0, 1.1]})
levels = [(2.725, "cosmic microwave background, 2.725 K", S.GR), (Tgap, f"gap scale $\\Delta/k_B$ = {Tgap:.2f} K", S.IAM),
          (1.20, "Al critical temperature $T_c$ = 1.20 K", S.ALT), (0.015, "mixing chamber, 15 mK", S.DATA)]
for (T, lab, c), ty in zip(levels, (3.6, 1.95, 1.0, 0.015)):
    ax.axhline(T, xmin=0.05, xmax=0.25, color=c, lw=2.0)
    ax.annotate(lab, (0.25, T), xytext=(0.32, ty), xycoords=ax.get_yaxis_transform(), textcoords=ax.get_yaxis_transform(),
                color=c, fontsize=6.5, va="center", arrowprops=dict(arrowstyle="-", color=c, lw=0.5))
ax.set_yscale("log"); ax.set_ylim(8e-3, 6); ax.set_xlim(0, 1); ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
ax.set_ylabel("temperature, or energy / $k_B$ (K)")
ax.set_title("Scales at the junction")
S.panel_letter(ax, "a")
# junction cross-section
bx.add_patch(Rectangle((0, 0.58), 1, 0.30, color=S.SKY, alpha=0.55, lw=0))
bx.add_patch(Rectangle((0, 0.12), 1, 0.30, color=S.SKY, alpha=0.55, lw=0))
bx.add_patch(Rectangle((0, 0.42), 1, 0.16, color=S.LIGHT, lw=0))
bx.text(0.5, 0.80, "Al electrode", ha="center", fontsize=7)
bx.text(0.5, 0.20, "Al electrode", ha="center", fontsize=7)
bx.text(0.03, 0.50, "AlO$_x$ barrier, 1–2 nm", fontsize=6.5, va="center")
for xs in (0.60, 0.76, 0.92):
    bx.add_patch(Circle((xs, 0.50), 0.025, color=S.DATA, lw=0))
    bx.add_patch(Circle((xs, 0.50), 0.075, fill=False, ec=S.IAM, lw=0.8, ls="--"))
bx.text(0.76, 0.98, "fluctuator / high-transmission sites", ha="center", fontsize=6.5, color=S.DATA)
bx.text(0.76, 0.02, "dashed: phase-restoration region, radius ~ $\\lambda_L$", ha="center", fontsize=6.5, color=S.IAM)
bx.set_xlim(0, 1); bx.set_ylim(-0.05, 1.05); bx.axis("off")
bx.set_title("Junction cross-section (not to scale)", loc="center")
S.panel_letter(bx, "b", dx=0.0, dy=1.10)
S.save(fig, "part3", "fig_xqp_ladder")
print(f"T_gap = {Tgap:.3f} K; T_CMB - T_gap = {2.725-Tgap:.3f} K; T_gap - 0.015 = {Tgap-0.015:.3f} K")

# ---------------- London volume ----------------
lam = 50.0   # nm
r = np.linspace(0, 400, 4001)
integ = 4 * np.pi * r**2 * np.exp(-2 * r / lam)
cum = np.concatenate([[0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(r))])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.4))
ax.plot(r, np.exp(-r / lam), color=S.GR, lw=1.2, ls="--", label="envelope $e^{-r/\\lambda_L}$")
ax.plot(r, integ / integ.max(), color=S.IAM, lw=1.4, label="$4\\pi r^2 e^{-2r/\\lambda_L}$ (scaled)")
ax.axvline(lam, color=S.LIGHT, lw=0.8); ax.text(lam + 4, 0.05, "$r=\\lambda_L$", fontsize=7)
ax.set_xlabel("distance from the site $r$ (nm)"); ax.set_ylabel("relative size"); ax.set_xlim(0, 300); ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", fontsize=6.5); S.panel_letter(ax, "a")
bx.plot(r, cum / (np.pi * lam**3), color=S.IAM, lw=1.4)
bx.axhline(1, color=S.GR, lw=0.8, ls="--"); bx.text(310, 0.93, "$\\pi\\lambda_L^3$", fontsize=7, ha="right", va="top")
bx.set_xlabel("upper limit $R$ (nm)"); bx.set_ylabel("$V(R)/\\pi\\lambda_L^3$"); bx.set_xlim(0, 320); bx.set_ylim(0, 1.08)
S.panel_letter(bx, "b", dx=-0.13)
S.save(fig, "part3", "fig_xqp_london")
print(f"numerical V(400 nm)/pi lam^3 = {cum[-1]/(np.pi*lam**3):.5f}; pi lam^3 = {np.pi*lam**3:.4e} nm^3")

# ---------------- x_qp vs tau_TLS ----------------
ncp, tq = 4e6, 100.0
tT = np.logspace(-1, 4, 300)
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.7))
ax.axhspan(1e-8, 1e-6, color=S.DATA, alpha=0.15, lw=0); ax.axhline(1e-7, color=S.DATA, lw=0.8, ls="--")
ax.axvspan(1, 100, color=S.LIGHT, alpha=0.35, lw=0)
ax.text(1.2, 2e-11, "working range\n1–100 µs", fontsize=6.5, color=S.GR)
ax.text(9e3, 1.25e-8, "measured background", fontsize=6.5, color=S.DATA, va="bottom", ha="right")
for NV, ls in ((0.006, ":"), (0.06, "-"), (0.6, "--")):
    x = 2 * NV * tq / (tT * ncp)
    ts = 2 * NV * tq / (ncp * 1e-7)
    ax.plot(tT, x, color=S.IAM, lw=1.4, ls=ls, label=f"$N/V$ = {NV} µm$^{{-3}}$: $10^{{-7}}$ at {ts:.0f} µs")
    ax.plot(ts, 1e-7, "o", color=S.IAM, ms=3.5)
    print(f"N/V {NV}: x = 1e-7 at tau_TLS = {ts:.1f} us")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.1, 1e4); ax.set_ylim(1e-11, 1e-4)
ax.set_xlabel("fluctuator switching time $\\tau_{\\rm TLS}$ (µs)"); ax.set_ylabel("quasiparticle fraction $x_{\\rm qp}$")
ax.legend(loc="upper right", fontsize=6.3)
ax.set_title("Floor against switching time")
S.save(fig, "part3", "fig_xqp_tau")
