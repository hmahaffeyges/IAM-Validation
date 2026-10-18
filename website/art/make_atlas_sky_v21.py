"""IAM-Atlas v2.1 on the sphere for the public page (2026-10-18; replaces the v2 sky art: make_atlas_sky.py is kept for the record).
Reads the v2.1 maps made by Biological_Physics/MethylPhys/atlas/tools/sky_maps_v21.py (HEALPix NSIDE 256, NESTED, genome order: every cell's
best value and the half-width of its 95 % range), and the measured Planck 2018 SMICA map (website/art/data, as before).
Writes to website/art/out/: sky_atlas_globes.jpg and sky_atlas_plate.jpg (the measured sky beside v2.1 neutrophils), sky_atlas_cells.jpg
(six cell types and how sure the atlas is), sky_v21_plate.jpg (all 79), cells/<slug>.jpg (each cell: value beside uncertainty) and
cells/index.json (per-cell numbers for the cell pages). Nothing is fitted; pictures only.
Usage: python3 website/art/make_atlas_sky_v21.py MAPS_DIR   (MAPS_DIR holds sky_v21_mu_nside256_nest.npz and sky_v21_halfwidth_nside256_nest.npz)"""
import os, sys, re, json, numpy as np
os.environ.setdefault("ASTROPY_CONFIGDIR", os.path.join(os.getcwd(), ".astro")); os.makedirs(os.environ["ASTROPY_CONFIGDIR"], exist_ok=True)
import healpy as hp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); CO = os.path.join(OUT, "cells"); os.makedirs(CO, exist_ok=True)
BG, FG = "#0b0d11", "#E6E3DC"; CM = ListedColormap(np.loadtxt(os.path.join(HERE, "data", "Planck_Parchment_RGB.txt")) / 255.0, name="planck")
UC = plt.get_cmap("magma"); D = sys.argv[1]
Z = np.load(os.path.join(D, "sky_v21_mu_nside256_nest.npz")); W = np.load(os.path.join(D, "sky_v21_halfwidth_nside256_nest.npz")); cells = [str(c) for c in Z["cells"]]
def m_(Dz, c):
    v = Dz[f"m{cells.index(c):02d}"].astype(np.float64); return np.where(np.isfinite(v), v, hp.UNSEEN)
def pretty(c):
    s = re.sub(r"\s+", " ", c.replace("_", " ")).strip(); s = s[:1].upper() + s[1:]
    for a, b in (("Cd4", "CD4"), ("Cd8", "CD8"), ("cd4", "CD4"), ("cd8", "CD8"), ("Nk ", "NK "), ("nk ", "NK "), ("cd56", "CD56"), ("cd28", "CD28"), ("Cmp", "CMP"), ("th1", "Th1"), ("Th1", "Th1"), ("temra", "TEMRA")): s = s.replace(a, b)
    return s
def slug(c): return re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-")
def panel(fig, sub, m, title, cmap, lo, hi, unit, ticks, labels=None, nest=True, fs=13):
    hp.mollview(m, nest=nest, fig=fig.number, sub=sub, title="", cmap=cmap, min=lo, max=hi, cbar=False, badcolor="#2a2d33", bgcolor=BG)
    ax = plt.gca(); ax.set_title(title, color=FG, fontsize=fs, pad=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi)); cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, aspect=40, shrink=0.8, ticks=ticks)
    if labels: cb.ax.set_xticklabels(labels)
    cb.set_label(unit, color=FG, fontsize=10); cb.ax.tick_params(colors=FG, labelsize=9); cb.outline.set_edgecolor("#444")
MU = ("methylation: 0 never tagged, 1 always tagged", [0, 0.25, 0.5, 0.75, 1]); UU = ("uncertainty: dark = certain (0), bright = less sure", [0, 0.025, 0.05], ["0  certain", "\u00b10.025", "\u00b10.05 or more"])
# 2026-10-18 (author: the cell pages' certainty maps were "all too pale when they should be dark"): the 0-0.05 scale above was set for
# neutrophils, the atlas's most certain cell (median half-width 0.006); 38 of the 79 cells have half their sites beyond it, so their maps
# saturated. Every cell page now uses ONE shared scale, 0 to 0.15 (nine in ten sites inside it for 76 of 79 cells), so each map shows its own
# certainty and cells can be compared with each other. The neutrophil tile on the front page keeps 0-0.05 for its fine detail.
UHI = 0.15; UC_ = ("uncertainty: dark = certain (0), bright = less sure (same scale for every cell)", [0, 0.05, 0.10, 0.15], ["0  certain", "\u00b10.05", "\u00b10.10", "\u00b10.15"])
f = os.path.join(HERE, "data", "planck_smica_pr3_nside256.npz"); C0 = hp.smoothing(np.load(f)["map"].astype(np.float64), fwhm=np.radians(0.5))
NM = m_(Z, "neutrophils"); nN = int((NM != hp.UNSEEN).sum())
# globes
fig = plt.figure(figsize=(16, 8.2), facecolor=BG)
for k, (m, t, lo, hi, nest) in enumerate([(C0, "The microwave sky, measured by Planck (2018)", -300, 300, False), (NM, "IAM-Atlas v2.1: healthy neutrophils", 0, 1, True)]):
    hp.orthview(m, fig=fig.number, sub=(1, 2, k + 1), half_sky=True, rot=(0, 20), title="", cmap=CM, min=lo, max=hi, cbar=False, nest=nest, badcolor=BG, bgcolor=BG)
    plt.gca().set_title(t, color=FG, fontsize=15, pad=6)
fig.savefig(os.path.join(OUT, "sky_atlas_globes.jpg"), dpi=130, facecolor=BG, bbox_inches="tight", pil_kwargs={"quality": 88}); plt.close(fig)
# flat pair
fig = plt.figure(figsize=(16, 5.2), facecolor=BG)
panel(fig, (1, 2, 1), C0, "The microwave sky, measured by Planck (2018)", CM, -300, 300, "temperature minus its mean (\u03bcK)", [-300, -150, 0, 150, 300], nest=False)
panel(fig, (1, 2, 2), NM, "IAM-Atlas v2.1: healthy neutrophils, every site in genome order", CM, 0, 1, *MU)
fig.savefig(os.path.join(OUT, "sky_atlas_plate.jpg"), dpi=130, facecolor=BG, bbox_inches="tight", pil_kwargs={"quality": 88}); plt.close(fig)
# six cells and the neutrophil uncertainty
SIX = [("neutrophils", "Neutrophils (blood)"), ("hepatocyte", "Liver cells"), ("colon epithelium", "Colon lining"), ("lung alveolar epithelium", "Lung air-sac cells"),
       ("breast luminal epithelium", "Breast lining"), ("cortical neurons", "Brain neurons")]
SIX = [(c, n) for c, n in SIX if c in cells]
tiles = []
for c, n in SIX + [("neutrophils", None)]:
    fg = plt.figure(figsize=(8, 4.9), facecolor=BG)
    if n: panel(fg, (1, 1, 1), m_(Z, c), n, CM, 0, 1, *MU)
    else: panel(fg, (1, 1, 1), m_(W, c), "Neutrophils: how sure the atlas is", UC, 0, 0.05, *UU)
    t = os.path.join(OUT, f"_t{len(tiles)}.png"); fg.savefig(t, dpi=105, facecolor=BG, bbox_inches="tight"); plt.close(fg); tiles.append(t)
ims = [Image.open(t).convert("RGB") for t in tiles]; w = max(i.width for i in ims); h = max(i.height for i in ims); g = 24; rows = (len(ims) + 1) // 2
S = Image.new("RGB", (2 * w + g, rows * h + (rows - 1) * g), BG)
for k, im in enumerate(ims): S.paste(im, ((k % 2) * (w + g) + (w - im.width) // 2, (k // 2) * (h + g)))
S.save(os.path.join(OUT, "sky_atlas_cells.jpg"), quality=86, optimize=True)
for t in tiles: os.remove(t)
# every cell: value beside uncertainty, and its numbers
idx = []
for c in cells:
    v = Z[f"m{cells.index(c):02d}"].astype(float); u = W[f"m{cells.index(c):02d}"].astype(float)
    fg = plt.figure(figsize=(14, 4.4), facecolor=BG)
    panel(fg, (1, 2, 1), m_(Z, c), f"{pretty(c)}: the value at each site", CM, 0, 1, *MU, fs=12)
    panel(fg, (1, 2, 2), m_(W, c), "How sure the atlas is of that value", UC, 0, UHI, *UC_, fs=12)
    fg.savefig(os.path.join(CO, slug(c) + ".jpg"), dpi=95, facecolor=BG, bbox_inches="tight", pil_kwargs={"quality": 80}); plt.close(fg)
    idx.append(dict(cell=c, name=pretty(c), slug=slug(c), pixels_measured=int(np.isfinite(v).sum()), pixels=int(v.size),
                    mean_methylation=round(float(np.nanmean(v)), 3), halfwidth_median=round(float(np.nanmedian(u)), 4), halfwidth_p90=round(float(np.nanpercentile(u, 90)), 4), uncertainty_scale=[0, UHI]))
json.dump(dict(atlas="IAM-Atlas v2.1", nside=256, order="NESTED", source="Biological_Physics/MethylPhys/atlas/tools/sky_maps_v21.py", cells=idx),
          open(os.path.join(CO, "index.json"), "w"), indent=1)
print("cells", len(idx), "| neutrophil pixels", nN)
# thumbnails for the gallery: the value map only (left half of each cell's picture)
for c in idx:
    im = Image.open(os.path.join(CO, c["slug"] + ".jpg")).convert("RGB"); w, h = im.size
    t = im.crop((0, 0, w // 2, h)); t.thumbnail((360, 240)); t.save(os.path.join(CO, c["slug"] + "_thumb.jpg"), quality=78, optimize=True)
print("thumbnails", len(idx))
