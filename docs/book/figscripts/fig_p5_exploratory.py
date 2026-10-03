"""Part 5, Chapter 'Exploratory' (part5/p5_02_exploratory.tex): four figures, each computed from the chapter's own formulas.
fig_exploratory_transit:   (a) xi = D/(c t) for a one-way transit to the Pleiades (136 pc) against t; (b) tidal residual 2GML/r^3 at the
                           Earth's surface against body length L.                                                   [calculated / derived]
fig_exploratory_recession: v_rec/c = H0 D/c against proper distance for the photon-sector (67.16) and matter-sector (72.26) H0.  [calculated]
fig_exploratory_twin:      Earth-elapsed and ship-elapsed time for a through-space voyage to the Pleiades at constant v (special relativity),
                           and the flat displaced region, where both equal D/(xi c) to O(Delta Phi/c^2).               [calculated]
fig_exploratory_steering:  cycle-averaged |Phi_drive|^2 in the x-z plane for three coplanar and four non-coplanar phase-locked sources,
                           with an ILLUSTRATIVE isotropic wave kernel G = exp(ikr)/r (the true kernel is unknown).        [derived symmetry; kernel assumed]
Numbers are checked in docs/verification/scripts/verify_exploratory.py."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
c = C.c
Mpc, ly, pc = 3.0856775814913673e22, 9.4607304725808e15, 3.0856775814913673e16
Dly = 136 * pc / ly                                   # Pleiades, 136 pc (VLBI parallax)

# ---------------- transit ----------------
t = np.logspace(-1, 4, 200)                           # days
xi = Dly / (t / 365.25); x7 = Dly / (7 / 365.25)
print(f"Pleiades {Dly:.1f} ly; xi(7 d) = {x7:.4e}")
GMe, Re, g0 = 3.986004418e14, 6.371e6, 9.80665
L = np.logspace(0, 3, 50); tid = 2 * GMe * L / Re**3
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5))
a1.plot(t, xi, color=S.GR); a1.plot(7, x7, "o", color=S.DATA, ms=4)
a1.annotate(f"7 days: ξ = {S.sci(x7, 2)}", (7, x7), xytext=(8, 2), textcoords="offset points", fontsize=6.5)
a1.axhline(1, color=S.LIGHT, lw=0.8, ls="--"); a1.text(0.12, 1.3, "ξ = 1", fontsize=6, color=S.GR)
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel("one-way transit time to the Pleiades (days)")
a1.set_ylabel(r"required $\xi$ (rate in units of $c$)"); a1.set_title(f"{Dly:.0f} ly in a given time"); S.panel_letter(a1, "a")
a2.plot(L, tid / g0, color=S.GR); a2.set_xscale("log"); a2.set_yscale("log")
for LL in (10, 100):
    v = 2 * GMe * LL / Re**3 / g0; a2.plot(LL, v, "o", color=S.DATA, ms=3.5)
    a2.annotate(f"{LL} m: {S.sci(v, 2)} g", (LL, v), xytext=(6, -8), textcoords="offset points", fontsize=6)
    print(f"tidal {LL} m: {v:.3e} g")
a2.set_xlabel("length of the body $L$ (m)"); a2.set_ylabel(r"tidal residual $2GML/r^3$ in units of $g$")
a2.set_title("What a body in free fall feels at the Earth's surface"); S.panel_letter(a2, "b", dx=-0.2)
fig.tight_layout(); S.save(fig, "part5", "fig_exploratory_transit")

# ---------------- recession ----------------
D = np.logspace(1, 11, 300)
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
for h, ls, col, lab in ((67.16, "-", S.IAM, "photon sector"), (72.26, "--", S.ALT, "matter sector")):
    DH = c / (h * 1e3 / Mpc) / ly
    ax.plot(D, D / DH, color=col, lw=1.3, ls=ls, label=f"{lab}, $H_0$ = {h:g}: $D_H$ = {DH/1e10:.2f}$\\times10^{{10}}$ ly")
    print(f"H0 {h}: D_H = {DH:.4e} ly; Pleiades fraction {Dly/DH:.2e}")
ax.axhline(1, color=S.GR, lw=0.8, ls=":"); ax.text(1e5, 1.6, "$v_{\\rm rec}=c$", fontsize=7, color=S.GR, ha="right")
DHp = c / (67.16e3 / Mpc) / ly
ax.plot(Dly, Dly / DHp, "o", color=S.DATA, ms=4)
ax.annotate(f"Pleiades, {Dly:.0f} ly: $3\\times10^{{-8}}$ of $D_H$", (Dly, Dly / DHp), xytext=(-4, 10), textcoords="offset points", fontsize=7, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(10, 1e11); ax.set_ylim(1e-10, 10)
ax.set_xlabel("proper distance $D$ (ly)"); ax.set_ylabel("recession speed $H_0D/c$")
ax.legend(loc="lower right", fontsize=6)
ax.set_title("Recession exceeds $c$ beyond the Hubble radius")
S.save(fig, "part5", "fig_exploratory_recession")

# ---------------- twin comparison ----------------
beta = np.linspace(0.05, 0.9999, 400); gam = 1 / np.sqrt(1 - beta**2)
tE = Dly / beta; tS = tE / gam
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
ax.plot(beta, tE, color=S.GR, label="through space: elapsed at the origin, $D/v$")
ax.plot(beta, tS, color=S.DATA, label=r"through space: elapsed aboard, $D/(\gamma v)$")
ax.axhline(Dly / x7, color=S.IAM, lw=1.2, ls="--", label="flat displaced region, ξ = 2.3×10⁴: both 7 days")
for b in (0.9, 0.99, 0.999):
    g = 1 / np.sqrt(1 - b**2); print(f"v = {b}c: Earth {Dly/b:.1f} yr, ship {Dly/b/g:.1f} yr")
ax.set_yscale("log"); ax.set_xlim(0, 1); ax.set_ylim(5e-3, 2e4)
ax.set_xlabel("speed through space $v/c$"); ax.set_ylabel("one-way elapsed time (yr)")
ax.legend(loc="center left", fontsize=6); ax.set_title(f"One way to the Pleiades ({Dly:.0f} ly)")
S.save(fig, "part5", "fig_exploratory_twin")

# ---------------- steering geometry ----------------
k = 2 * np.pi / 1.0
tgt = np.array([0.4, 0.0, 1.2])
S3 = np.array([[3., 0, 0], [-1.5, 2.6, 0], [-1.5, -2.6, 0]])
S4 = np.vstack([S3, [0, 0, 3.5]])
xx, zz = np.meshgrid(np.linspace(-2.5, 2.5, 321), np.linspace(-2.5, 2.5, 321))
pts = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], 1)
def I(src):
    tot = np.zeros(len(pts), complex)
    for s in src:
        R0 = np.linalg.norm(tgt - s); R = np.linalg.norm(pts - s, axis=1)
        tot += R0 * np.exp(1j * k * (R - R0)) / R
    return (np.abs(tot)**2 / len(src)**2).reshape(xx.shape)
fig, axs = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw={"wspace": 0.38})
for ax, src, ttl, let in ((axs[0], S3, "3 coplanar sources (z = 0)", "a"), (axs[1], S4, "4 non-coplanar sources", "b")):
    im = ax.imshow(I(src), extent=(-2.5, 2.5, -2.5, 2.5), origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax.plot(*tgt[[0, 2]], "+", color=S.DATA, ms=7, mew=1.2)
    ax.plot(tgt[0], -tgt[2], "x", color=S.DATA, ms=5, mew=1.0)
    ax.axhline(0, color=S.LIGHT, lw=0.6, ls=":")
    ax.set_xlabel("x (wavelengths)"); ax.set_ylabel("z (wavelengths)"); ax.set_title(ttl); S.panel_letter(ax, let, dx=-0.18)
    It = I(src)[np.argmin(abs(zz[:, 0] - tgt[2])), np.argmin(abs(xx[0] - tgt[0]))]
    Im = I(src)[np.argmin(abs(zz[:, 0] + tgt[2])), np.argmin(abs(xx[0] - tgt[0]))]
    print(f"{ttl}: target {It:.3f}, mirror {Im:.3f}")
cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02); cb.set_label(r"$\langle|\Phi_{\rm drive}|^2\rangle$ / focal maximum", fontsize=7)
S.save(fig, "part5", "fig_exploratory_steering")
