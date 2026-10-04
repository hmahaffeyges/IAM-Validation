"""Figures for Part 2 chapters p4_15a_lepton_koide.tex, p4_15b_electron_mass.tex and p4_22b_higgs_record.tex.

fig_koide_flavour : (a) the square-root mass vector (sqrt m_e, sqrt m_mu, sqrt m_tau) in flavour space with the (1,1,1) direction and the
                    circle of all vectors at 45 deg to it (Q = 2/3) at the measured scale x; (b) the same circle in the plane sqrt m_e + sqrt m_mu
                    + sqrt m_tau = 3x, with the triangle of positive roots.
fig_koide_orbit   : (a) the charge orbit S^1 with the three phases phi_k = delta + 2 pi k/3 at the measured offset; (b) sqrt m(phi)/x = 1 + sqrt2 cos phi.
fig_koide_sweep   : the offset sweep: (a-e) the orbit at five offsets; (f) the three masses against delta at fixed x.
fig_electron_fp   : (a) the two sides of m c^2 = E_bit N(m)/f(alpha); (b) the fixed point against H0; (c) against the exponent of alpha.
fig_higgs_proper  : (a) rest masses acquired at the electroweak crossover and their Compton times; (b) E = exp(1 - 1/a) = exp(-z) from the
                    crossover to today.
Inputs: PDG 2024 masses (Navas et al. 2024); CODATA (scipy.constants); Planck 2018; book sector values H0 = 67.16 / 72.26.
Every plotted number is computed here; the same numbers are printed by docs/verification/scripts/verify_particle_book.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as C
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import _bookstyle as S

S.apply()
# ---------------------------------------------------------------- lepton numbers (PDG 2024)
me, mm, mt = 0.51099895000, 105.6583755, 1776.93
s = np.sqrt([me, mm, mt])
x = s.sum() / 3
ph = 2 * np.pi * np.arange(3) / 3                      # k = 0 tau, 1 e, 2 mu
so = np.sqrt([mt, me, mm])
d = np.arctan2(-(2/3) * np.sum(so * np.sin(ph)), (2/3) * np.sum(so * np.cos(ph)))
r2 = np.sqrt(2)
print(f"x = {x:.5f}, delta = {d:.6f}, point = {np.round(s, 4)}")
LAB = {0: r"$\tau$", 1: r"$e$", 2: r"$\mu$"}

def vec(dd):
    """(sqrt m_e, sqrt m_mu, sqrt m_tau) for offset dd at scale x, y/x = sqrt2 (e at k=1, mu at k=2, tau at k=0)."""
    return x * np.array([1 + r2*np.cos(dd + ph[1]), 1 + r2*np.cos(dd + ph[2]), 1 + r2*np.cos(dd)])

# ---------------------------------------------------------------- fig_koide_flavour
fig = plt.figure(figsize=(S.TEXTW, 3.1))
ax = fig.add_axes([0.0, 0.08, 0.46, 0.86], projection="3d")
D = np.linspace(0, 2*np.pi, 721)
V = np.array([vec(t) for t in D])
pos = (V > 0).all(1)
ax.plot(*V.T, color=S.LIGHT, lw=0.8, ls="--")
Vp = V.copy(); Vp[~pos] = np.nan
ax.plot(*Vp.T, color=S.IAM, lw=1.4, label="45° circle, all roots > 0")
t = np.linspace(0, 1.25*x, 2)
ax.plot(t, t, t, color=S.GR, lw=1.0, label="(1,1,1) direction")
ax.plot([0, s[0]], [0, s[1]], [0, s[2]], color=S.DATA, lw=0.8)
ax.scatter(*s, color=S.DATA, s=18, depthshade=False, label=r"leptons $(\sqrt{m_e},\sqrt{m_\mu},\sqrt{m_\tau})$")
v0 = vec(0.0)
ax.scatter(*v0, color=S.GR, marker="s", s=12, depthshade=False, label=r"$\delta=0$ ($m_e=m_\mu$)")
ax.set_xlabel(r"$\sqrt{m_e}$ (MeV$^{1/2}$)", labelpad=-4); ax.set_ylabel(r"$\sqrt{m_\mu}$ (MeV$^{1/2}$)", labelpad=-4)
ax.set_zlabel(r"$\sqrt{m_\tau}$ (MeV$^{1/2}$)", labelpad=-4)
lim = (-10, 45)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
ax.set_box_aspect((1, 1, 1))
ax.tick_params(pad=-2, labelsize=5)
ax.view_init(elev=24, azim=28)
ax.legend(loc="upper left", fontsize=5.5, bbox_to_anchor=(-0.05, 1.08))
ax.text2D(0.0, 1.06, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
# (b) plane sum sqrt m = 3x
ax = fig.add_axes([0.6, 0.14, 0.38, 0.78])
e1 = np.array([1, -1, 0]) / np.sqrt(2); e2 = np.array([1, 1, -2]) / np.sqrt(6)
P = lambda v: (np.dot(v - x, e1) / x, np.dot(v - x, e2) / x)   # coordinates in units of x
tri = np.array([[3*x, 0, 0], [0, 3*x, 0], [0, 0, 3*x], [3*x, 0, 0]])
T = np.array([P(v) for v in tri])
ax.fill(T[:, 0], T[:, 1], color=S.SKY, alpha=0.18, lw=0)
ax.plot(T[:, 0], T[:, 1], color=S.GR, lw=0.7)
C2 = np.array([P(v) for v in V])
ax.plot(C2[:, 0], C2[:, 1], color=S.LIGHT, lw=0.8, ls="--")
C2p = C2.copy(); C2p[~pos] = np.nan
ax.plot(C2p[:, 0], C2p[:, 1], color=S.IAM, lw=1.4)
pl = P(s); p0 = P(v0)
ax.plot(*pl, "o", color=S.DATA, ms=4); ax.plot(*p0, "s", color=S.GR, ms=3.5)
ax.annotate(f"leptons, δ = {d:.4f}", pl, xytext=(0.5, -1.95), fontsize=6, color=S.DATA,
            arrowprops=dict(arrowstyle="-", color=S.DATA, lw=0.5))
ax.annotate("δ = 0", p0, xytext=(-1.9, -1.2), fontsize=6, color=S.GR, arrowprops=dict(arrowstyle="-", color=S.GR, lw=0.5))
for v, lab, off in [(tri[0], r"$e$ only", (-0.45, 0.12)), (tri[1], r"$\mu$ only", (-0.1, 0.12)), (tri[2], r"$\tau$ only", (0.1, -0.05))]:
    q = P(v); ax.text(q[0] + off[0], q[1] + off[1], lab, fontsize=6, color=S.GR)
ax.text(0.0, 0.0, "(1,1,1)", fontsize=6, ha="center", va="center", color=S.GR)
ax.text(-2.15, -2.65, r"radius $\sqrt{3}\,x$: the 45° circle" + "\n" + r"(dashed: a root $<0$)", fontsize=6, color=S.IAM)
ax.set_aspect("equal"); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.75, 1.5)
ax.set_xlabel(r"$(\sqrt{m}-x)\cdot\hat e_1/x$"); ax.set_ylabel(r"$(\sqrt{m}-x)\cdot\hat e_2/x$")
S.panel_letter(ax, "b", dx=-0.16)
S.save(fig, "part4", "fig_koide_flavour")

# ---------------------------------------------------------------- fig_koide_orbit
fig, axs = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(width_ratios=[1, 1.5], wspace=0.35))
ax = axs[0]
f = np.linspace(0, 2*np.pi, 721)
ax.plot(np.cos(f), np.sin(f), color=S.GR, lw=0.8)
bad = np.linspace(3*np.pi/4, 5*np.pi/4, 100)
ax.plot(np.cos(bad), np.sin(bad), color=S.DATA, lw=4, alpha=0.25, solid_capstyle="butt")
ax.text(-1.45, 0.0, "root < 0\n" + r"($|\phi-\pi|<\pi/4$)", fontsize=5.5, color=S.DATA, ha="center", va="center")
for k in range(3):
    a = d + ph[k]; val = 1 + r2*np.cos(a)
    ax.plot([0, np.cos(a)], [0, np.sin(a)], color=S.IAM, lw=0.6)
    ax.plot(np.cos(a), np.sin(a), "o", color=S.IAM, ms=4)
    ax.text(1.42*np.cos(a), 1.42*np.sin(a), LAB[k] + f"\n{val:.3f}", fontsize=6, ha="center", va="center")
    a0 = ph[k]; ax.plot(np.cos(a0), np.sin(a0), "s", mfc="none", mec=S.GR, ms=3.5)
ax.plot([0, 1.0], [0, 0], color=S.LIGHT, lw=0.5, ls=":")
ax.text(0.5, -0.12, r"$\phi=0$", fontsize=5.5, ha="center", color=S.GR)
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-2.0, 1.75); ax.set_ylim(-1.6, 1.6)
ax.text(-2.0, 1.6, "a", fontsize=9, fontweight="bold", va="top")
ax = axs[1]
yv = 1 + r2*np.cos(f)
ax.axhspan(-0.6, 0, color=S.DATA, alpha=0.08, lw=0)
ax.axhline(0, color=S.GR, lw=0.5)
ax.plot(f, yv, color=S.IAM)
for k in range(3):
    a = (d + ph[k]) % (2*np.pi); val = 1 + r2*np.cos(a)
    ax.plot(a, val, "o", color=S.DATA, ms=4)
    ax.annotate(LAB[k] + rf": $\sqrt{{m}}/x={val:.3f}$", (a, val), xytext=(a + 0.25, val + 0.28), fontsize=6)
    a0 = ph[k]; ax.plot(a0, 1 + r2*np.cos(a0), "s", mfc="none", mec=S.GR, ms=3.5)
ax.text(np.pi, -0.35, r"$1+\sqrt{2}\cos\phi<0$", ha="center", fontsize=6, color=S.DATA)
ax.set_xticks(np.arange(0, 2.01*np.pi, np.pi/2)); ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_xlim(0, 2*np.pi); ax.set_ylim(-0.6, 2.9)
ax.set_xlabel(r"phase on the charge orbit $\phi$"); ax.set_ylabel(r"$\sqrt{m(\phi)}/x=1+\sqrt{2}\cos\phi$")
ax.text(4.3, 2.55, f"filled: δ = {d:.4f} rad\nopen: δ = 0", fontsize=6, color=S.GR)
S.panel_letter(ax, "b")
S.save(fig, "part4", "fig_koide_orbit")

# ---------------------------------------------------------------- fig_koide_sweep
fig = plt.figure(figsize=(S.TEXTW, 3.5))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.35], hspace=0.45, wspace=0.15)
deltas = [0.0, 0.10, d, np.pi/12, 0.40]
names = ["δ = 0", "δ = 0.10", f"δ = {d:.4f}\n(measured)", "δ = π/12\n(edge)", "δ = 0.40"]
for i, (dd, nm) in enumerate(zip(deltas, names)):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(np.cos(f), np.sin(f), color=S.GR, lw=0.6)
    ax.plot(np.cos(bad), np.sin(bad), color=S.DATA, lw=3, alpha=0.25, solid_capstyle="butt")
    for k in range(3):
        a = dd + ph[k]; val = 1 + r2*np.cos(a)
        col = S.IAM if val > 1e-9 else S.DATA
        ax.plot(np.cos(a), np.sin(a), "o", color=col, ms=3)
        ax.text(1.38*np.cos(a), 1.38*np.sin(a), LAB[k], fontsize=6, ha="center", va="center")
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_title(nm, fontsize=6, loc="center")
    if i == 0: ax.text(-1.6, 1.9, "a", fontsize=9, fontweight="bold")
ax = fig.add_subplot(gs[1, :])
dd = np.linspace(-np.pi/3, np.pi/3, 2001)
cols = {0: S.GR, 1: S.IAM, 2: S.ALT}
for k in range(3):
    rt = x * (1 + r2*np.cos(dd + ph[k])); mk = rt**2
    ok = rt > 0
    ax.plot(dd[ok], mk[ok], color=cols[k], lw=1.2)
    i0 = np.argmin(abs(dd - (-0.55 if k != 2 else 0.6)))
    ax.text(dd[i0], mk[i0]*(2.2 if k == 0 else 0.35), LAB[k], color=cols[k], fontsize=7, ha="center")
    bd = mk.copy(); bd[ok] = np.nan
    ax.plot(dd, bd, color=cols[k], lw=0.8, ls=":")
ax.axvspan(-np.pi/12, np.pi/12, color=S.SKY, alpha=0.15, lw=0)
for v, nm in [(d, "measured"), (0, "δ = 0")]:
    ax.axvline(v, color=S.DATA if nm == "measured" else S.GR, lw=0.7, ls="--")
for mv in (me, mm, mt):
    ax.plot(d, mv, "o", color=S.DATA, ms=3.5)
ax.set_yscale("log"); ax.set_ylim(1e-3, 6e3); ax.set_xlim(-np.pi/3, np.pi/3)
ax.set_xticks([-np.pi/3, -np.pi/6, -np.pi/12, 0, np.pi/12, np.pi/6, np.pi/3])
ax.set_xticklabels([r"$-\pi/3$", r"$-\pi/6$", r"$-\pi/12$", "0", r"$\pi/12$", r"$\pi/6$", r"$\pi/3$"])
ax.set_xlabel(r"offset $\delta$ (rad), scale fixed at $x^2=$" + f"{x**2:.2f} MeV"); ax.set_ylabel("mass (MeV)")
ax.text(-0.13, 2.5e-3, "all three roots > 0", fontsize=6, color=S.IAM, ha="center", bbox=dict(fc="white", ec="none", pad=0.5))
ax.text(0.40, 3e-2, "dotted: the root is negative;\nits square is still positive", fontsize=6, color=S.GR)
S.panel_letter(ax, "b", dx=-0.06)
S.save(fig, "part4", "fig_koide_sweep")

# ---------------------------------------------------------------- fig_electron_fp
hbar, c, G, al, mE = C.hbar, C.c, C.G, C.alpha, C.m_e
mP = np.sqrt(hbar*c/G); Mpc = 3.0856775814913673e22
Ebit = lambda H0: hbar*(H0*1e3/Mpc)*np.log(2)/(2*np.pi)
def rhs(mu, H0=67.4, p=2.5, coef=1.0):           # E_bit N/f in units of m_e c^2, mu = m/m_e
    return coef*Ebit(H0)*(mP/(mu*mE))**1.5/al**p/(mE*c**2)
B = lambda H0, p=2.5: (hbar*(H0*1e3/Mpc)*np.log(2)*mP**1.5/(al**p*c**2))**0.4
mfix = lambda H0, p=2.5, pref=(2*np.pi)**-0.1: pref*B(H0, p)/mE
coef_id = (2*np.pi)**0.75
fig, axs = plt.subplots(1, 3, figsize=(S.TEXTW, 2.25), gridspec_kw=dict(width_ratios=[1.15, 1.1, 0.9], wspace=0.5))
ax = axs[0]
mu = np.logspace(-1, 0.6, 200)
ax.plot(mu, mu, color=S.GR, label=r"rest energy $mc^2$")
ax.plot(mu, rhs(mu), color=S.ALT, ls="--", label=r"$E_{\rm bit}N/f$, as derived")
ax.plot(mu, rhs(mu, coef=coef_id), color=S.IAM, label=r"$\times(2\pi)^{3/4}$ (identified)")
m1 = (2*np.pi)**-0.4*B(67.4)/mE; m2 = mfix(67.4)
ax.plot(m1, m1, "o", color=S.ALT, ms=3.5); ax.plot(m2, m2, "o", color=S.IAM, ms=3.5)
ax.annotate(f"{m1:.3f}", (m1, m1), xytext=(m1*0.42, m1*1.9), fontsize=6, color=S.ALT)
ax.annotate(f"{m2:.5f}", (m2, m2), xytext=(m2*1.25, m2*0.55), fontsize=6, color=S.IAM)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.1, 4); ax.set_ylim(0.03, 30)
ax.set_xticks([0.1, 0.3, 1, 3]); ax.set_xticklabels(["0.1", "0.3", "1", "3"]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel(r"trial mass $m/m_e$"); ax.set_ylabel(r"energy / $m_ec^2$")
ax.legend(fontsize=5.5, loc="upper center")
S.panel_letter(ax, "a", dx=-0.2)
ax = axs[1]
H = np.linspace(64, 75, 200)
ax.axvspan(67.36-0.54, 67.36+0.54, color=S.LIGHT, alpha=0.4, lw=0)
ax.plot(H, 100*(mfix(H)-1), color=S.IAM)
ax.axhline(0, color=S.GR, lw=0.5)
for Hv, nm, dx, dy, ha in [(67.16, "67.16\nphoton sector", 0.25, -1.6, "left"), (67.36, "Planck\n67.36", -0.35, 0.6, "right"), (72.26, "72.26\nmatter sector", 0.3, -1.6, "left"), (73.04, "SH0ES\n73.04", -0.3, 0.3, "right")]:
    yv = 100*(mfix(Hv)-1); ax.plot(Hv, yv, "o", color=S.DATA, ms=3)
    ax.text(Hv + dx, yv + dy, nm + f"\n{yv:+.2f} %", fontsize=5, ha=ha)
ax.set_xlim(64, 75); ax.set_ylim(-2.5, 4.5)
ax.set_xlabel(r"$H_0$ (km s$^{-1}$ Mpc$^{-1}$)"); ax.set_ylabel(r"$m/m_e-1$ (%)")
S.panel_letter(ax, "b", dx=-0.22)
ax = axs[2]
ps = np.array([1.5, 2.0, 2.5, 3.0, 3.5]); vals = np.array([mfix(67.4, p) for p in ps])
ax.bar(ps, vals, width=0.32, color=[S.IAM if p == 2.5 else S.LIGHT for p in ps])
for p, v in zip(ps, vals):
    ax.text(p, v*1.12, f"{v:.2f}", ha="center", fontsize=5.5)
ax.axhline(1, color=S.GR, lw=0.5, ls="--")
ax.set_yscale("log"); ax.set_ylim(0.08, 15)
ax.set_xticks(ps); ax.set_xticklabels(["3/2", "2", "5/2", "3", "7/2"])
ax.set_xlabel(r"exponent $p$ in $f=\alpha^{p}$"); ax.set_ylabel(r"$m/m_e$")
S.panel_letter(ax, "c", dx=-0.3)
S.save(fig, "part4", "fig_electron_fp")

# ---------------------------------------------------------------- fig_higgs_proper
GeV = 1e9*C.e
parts = [("e", me*1e-3), (r"$\mu$", mm*1e-3), (r"$\tau$", mt*1e-3), ("W", 80.3692), ("Z", 91.1880), ("H", 125.20), ("t", 172.57)]
fig, axs = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5), gridspec_kw=dict(wspace=0.45))
ax = axs[0]
yy = np.arange(len(parts))
ax.barh(yy, [m_*1e9 for _, m_ in parts], left=1e-19, color=S.IAM, height=0.55)
ax.barh(len(parts), 1e-18, left=1e-19, color=S.LIGHT, height=0.55)
ax.annotate("", xy=(1e-21, len(parts)), xytext=(1e-18, len(parts)), arrowprops=dict(arrowstyle="->", color=S.GR, lw=0.7))
ax.set_yticks(list(yy) + [len(parts)]); ax.set_yticklabels([n for n, _ in parts] + [r"$\gamma$ (bound)"])
ax.set_xscale("log"); ax.set_xlim(1e-21, 1e13)
ax.axvline(159.5e9, color=S.DATA, lw=0.8, ls="--")
ax.text(159.5e9*0.6, -0.9, r"$T_c$", color=S.DATA, fontsize=6, ha="right")
for i, (n, m_) in enumerate(parts):
    ax.text(m_*1e9*3, i, f"{C.hbar/(m_*GeV):.1e} s", va="center", fontsize=5)
ax.text(2e-17, len(parts), r"$m_\gamma<10^{-18}$ eV: no proper time", va="center", fontsize=5.5, color=S.GR)
ax.set_xlabel("rest energy (eV); label: Compton time ħ/mc²")
S.panel_letter(ax, "a", dx=-0.12)
ax = axs[1]
T0 = C.k*2.7255/C.e                                   # eV
def gs_(T):                                           # entropy degrees of freedom, step approximation (T in eV)
    return np.select([T > 1.7e11, T > 8e10, T > 5e9, T > 1.5e9, T > 1.5e8, T > 1e8, T > 5e5], [106.75, 96.25, 86.25, 75.75, 61.75, 17.25, 10.75], 3.938)
Tg = np.logspace(np.log10(T0), np.log10(2e11), 600)
z = Tg/T0*(gs_(Tg)/3.938)**(1/3) - 1
ax.plot(Tg, np.maximum(z, 1e-3), color=S.IAM)
marks = [(159.5e9, "electroweak crossover", (-0.15, 0.5)), (1.6e8, "QCD", (-0.15, 0.5)), (1e5, "nucleosynthesis", (0.3, -0.9)), (T0*1090.9, "recombination", (0.3, -0.9)), (T0*31, "z = 30", (0.35, -0.75))]
for Tm, nm, (ox, oy) in marks:
    zm = Tm/T0*(gs_(np.array(Tm))/3.938)**(1/3) - 1
    ax.plot(Tm, zm, "o", color=S.DATA, ms=3)
    ax.text(Tm*10**ox, zm*10**oy, f"{nm}\n" + r"$-\ln E$" + f" = {S.sci(float(zm), 2)}", fontsize=5, ha="right" if ox < 0 else "left")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-4, 1e13); ax.set_ylim(0.3, 1e17)
ax.set_xlabel("photon temperature (eV)"); ax.set_ylabel(r"$-\ln E(a)=1/a-1=z$")
S.panel_letter(ax, "b", dx=-0.16)
S.save(fig, "part4", "fig_higgs_proper")
