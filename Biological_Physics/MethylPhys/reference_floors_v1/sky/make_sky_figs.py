"""Figures for the cellular book chapter 'Tools from the sky' (2026-10-02).
Inputs: cmb_planck2018_nside256.npz (CAMB 2.0.4, Planck 2018 TT,TE,EE+lowE+lensing best fit, healpy synfast seed 42);
sky_neut6_maps.npz / sky_neut6_stats.csv (box job fcb79e1e, remote_jobs/sky6/sky_neut6.py)."""
import numpy as np, pandas as pd, healpy as hp, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "font.family": "DejaVu Sans", "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
D = np.load(SKY); T = pd.read_csv(STATS); C = np.load(CMB)["map"]; NS = int(D["nside"])
def st(a, m):
    r = T[(T["array"] == a) & (T["map"] == m)].iloc[0]; return r.MetA_6000, r.C_6000
def proj(m, xs=1600):
    P = hp.projector.MollweideProj(xsize=xs); ns = hp.npix2nside(len(m))
    img = P.projmap(np.where(np.isfinite(m), m, hp.UNSEEN), lambda x, y, z: hp.vec2pix(ns, x, y, z))
    return np.ma.masked_where((img == hp.UNSEEN) | ~np.isfinite(img) | (img < -1e20), img)
def moll(ax, m, cmap, vmin, vmax, xs=1600):
    im = ax.imshow(proj(m, xs), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest", extent=(-2, 2, -1, 1))
    e = mpl.patches.Ellipse((0, 0), 4, 2, transform=ax.transData, fc="none", ec="#555555", lw=0.4); ax.add_patch(e); im.set_clip_path(e)
    ax.set_xlim(-2.02, 2.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect("equal"); ax.set_axis_off(); return im
def letter(ax, s): ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
DIV = plt.get_cmap("RdBu_r").copy(); DIV.set_bad("#d9d9d9")

# ---------- Figure 1: the CMB and a healthy neutrophil, same projection, same kind of quantity
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
im = moll(axs[0], C, DIV, -300, 300); letter(axs[0], "a")
axs[0].set_title("The sky: CMB temperature minus its mean", loc="left")
cb = fig.colorbar(im, ax=axs[0], orientation="horizontal", fraction=0.05, pad=0.04, ticks=[-300, 0, 300]); cb.set_label(r"$\Delta T$ ($\mu$K)")
im = moll(axs[1], D["GSM2998021_healthy"], DIV, -4, 4); letter(axs[1], "b")
A, c = st("GSM2998021", "healthy")
axs[1].set_title(f"The cell: one healthy neutrophil minus five others", loc="left")
cb = fig.colorbar(im, ax=axs[1], orientation="horizontal", fraction=0.05, pad=0.04, ticks=[-4, 0, 4]); cb.set_label("pixel residual $z$")
fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.16, wspace=0.10)
fig.savefig("fig_sky_cmb_vs_neutrophil.png"); fig.savefig("fig_sky_cmb_vs_neutrophil.pdf"); plt.close(fig)

# ---------- Figure 2: the genome on the sphere
ch = D["pix_chrom"].astype(float); ch[ch < 0] = np.nan
cm = mpl.colors.ListedColormap(["#4c72b0", "#9fb8d9"] * 11 + ["#c44e52", "#e8a33d"])
fig = plt.figure(figsize=(7.2, 3.2)); ax = fig.add_axes([0.02, 0.03, 0.80, 0.84])
moll(ax, ch, cm, 0.5, 24.5)
ax.set_title(f"{int(D['n_probes'])/1e3:.0f}k EPIC CpGs laid in genome order around the HEALPix rings ({D['probes_per_pixel'].mean():.1f} per pixel)", loc="left")
P = hp.projector.MollweideProj(xsize=1600)
for c_, lab in ((1, "chr1"), (6, "chr6"), (12, "chr12"), (19, "chr19"), (23, "chrX")):
    pix = np.where(D["pix_chrom"] == c_)[0]; th, ph = hp.pix2ang(NS, np.array([pix[len(pix) // 2]]))
    x, y = P.ang2xy(th, ph); x, y = float(np.ravel(x)[0]), float(np.ravel(y)[0])
    ax.annotate(lab, (x, y), xytext=(2.12, y), fontsize=7, va="center", annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=0.5, color="#333333"))
fig.savefig("fig_sky_genome_on_sphere.png"); fig.savefig("fig_sky_genome_on_sphere.pdf"); plt.close(fig)

# ---------- Figure 3: what the sky sees that one number misses
pan = [("GSM2998021_healthy", "Healthy array", ("GSM2998021", "healthy")),
       ("GSM2998021_blur2pct", "Known damage spread everywhere (2 % blur)", ("GSM2998021", "blur2pct")),
       ("GSM2998021_local5pct", "Known damage in 10 genome regions (5 % blur, 5 % of sites)", ("GSM2998021", "local5pct")),
       ("GSM2998057_healthy", "Healthy female array read against five male arrays", ("GSM2998057", "healthy"))]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.4))
for k, (ax, (key, title, s)) in enumerate(zip(axs.flat, pan)):
    im = moll(ax, D[key], DIV, -4, 4); letter(ax, "abcd"[k]); ax.set_title(title, loc="left")
    A, c = st(*s); ax.text(0.5, -0.06, f"Met-A {A:.3f}   C-score {c:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=7)
cax = fig.add_axes([0.35, 0.095, 0.30, 0.018]); cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-4, 0, 4])
cb.set_label("pixel residual $z$")
fig.subplots_adjust(left=0.03, right=0.99, top=0.94, bottom=0.19, hspace=0.32, wspace=0.06)
fig.savefig("fig_sky_what_it_sees.png"); fig.savefig("fig_sky_what_it_sees.pdf"); plt.close(fig)

# ---------- Figure 4: the C-score, unrolled along the genome
h = D["GSM2998021_healthy"]; l = D["GSM2998021_local5pct"]
def runmean(v, w=25):
    v = np.where(np.isfinite(v), v, 0.0); n = np.isfinite(v).astype(float); k = np.ones(w)
    return np.convolve(v, k, "same") / np.maximum(np.convolve(np.isfinite(v), k, "same"), 1)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 3.0), sharex=True)
x = np.arange(len(h)) / len(h)
for ax, v, lab, s in ((axs[0], h, "Healthy array", ("GSM2998021", "healthy")), (axs[1], l, "Same array, damage in 10 regions", ("GSM2998021", "local5pct"))):
    ax.plot(x, v, ",", color="#9a9a9a", alpha=0.6, rasterized=True)
    ax.plot(x, runmean(v), color="#4c72b0" if v is h else "#c44e52", lw=0.8)
    A, c = st(*s); ax.set_title(f"{lab}: Met-A {A:.3f}, C-score {c:.2f}", loc="left"); ax.set_ylim(-6, 6); ax.set_ylabel("pixel $z$")
    ax.axhline(0, color="black", lw=0.4)
pc = D["pix_chrom"]; starts = {c_: np.argmax(pc == c_) / len(pc) for c_ in (1, 5, 10, 15, 20, 23)}
axs[1].set_xticks(list(starts.values())); axs[1].set_xticklabels(["chr1", "chr5", "chr10", "chr15", "chr20", "chrX"])
axs[1].set_xlabel("position along the genome (pixel order)")
fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.14, hspace=0.35)
fig.savefig("fig_sky_cscore_unrolled.png"); fig.savefig("fig_sky_cscore_unrolled.pdf"); plt.close(fig)
print("ok")
