"""IAM-Atlas v2 on the sky (art for the front page; author 2026-10-12: "a real HEALPix skymap of the atlas ... next to the CMB itself").
Redraws CPG Plates 1 and 3 ("The Cosmic Microwave Methylome", "Grandaddy" plate; class-era atlas, retired) from atlas v2. Every CpG in genome
order (chr1..chr22, chrX, chrY, then position) goes to pixel floor(i * npix / n) at HEALPix NSIDE 256 in NESTED order: 786,432 pixels for
784,950 placed sites, so one site per pixel (a few pixels hold two, averaged), at the resolution of the simulated sky beside it. Nested order
keeps genome neighbors together in small patches of the sphere (as Plate 3 did); the chain's grid contract (NSIDE 128, ring order,
atlas/healpix_mapping/) is for the development stage and is not used here. Mollweide projection; zooms are gnomonic, 15 degrees across. Coordinates from the Illumina EPIC v1 B4 and 450K
manifests (hg19). Values: the atlas v2 posterior mean beta (and its posterior SD) of each cell type, read from the files that
atlas/tools/extract_posterior.py writes from the published atlas (DOI 10.5281/zenodo.23325865).
The microwave sky beside it is SIMULATED: CAMB lensed TT spectrum at the Planck 2018 best fit, healpy synfast NSIDE 256, seed 42, the same map
the book's sky figures use (docs/book/figscripts/fig_p4_sky.py builds it). It is labeled as such.
This is a PICTURE of the atlas. No reading, fit or statistic uses the sky map (book tool board row 15: built, not in the chain; atlas plan U5).
Writes website/art/out/sky_atlas_hero.jpg (simulated sky | neutrophils), sky_atlas_plate.jpg (both skies and a zoom of each) and sky_atlas_cells.jpg (six cell types, mean; neutrophils' SD).
Usage: python3 website/art/make_atlas_sky.py POSTDIR MANIFEST_DIR"""
import os, sys, numpy as np, pandas as pd
os.environ.setdefault("ASTROPY_CONFIGDIR", os.path.join(os.getcwd(), ".astro")); os.makedirs(os.environ["ASTROPY_CONFIGDIR"], exist_ok=True)
import healpy as hp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out"); NSIDE = 256; NPIX = hp.nside2npix(NSIDE); BG = "#0b0d11"; FG = "#E6E3DC"
POST, MAN = sys.argv[1], sys.argv[2]
CELLS = [("neutrophils", "Neutrophils"), ("hepatocyte", "Liver cells"), ("colon_epithelium", "Colon lining"),
         ("lung_alveolar_epithelium", "Lung air-sac cells"), ("breast_luminal_epithelium", "Breast lining"), ("prostate_epithelium", "Prostate lining")]

def coords():
    a = pd.read_csv(os.path.join(MAN, "MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz"), usecols=["IlmnID", "CHR", "MAPINFO"], low_memory=False)
    b = pd.read_csv(os.path.join(MAN, "HumanMethylation450k_15017482_v3.csv.gz"), usecols=["IlmnID", "CHR", "MAPINFO"], low_memory=False)
    c = pd.concat([a, b]).dropna().drop_duplicates("IlmnID").set_index("IlmnID")
    c["CHR"] = c.CHR.astype(str).str.replace("chr", "", regex=False); order = {str(i): i for i in range(1, 23)} | {"X": 23, "Y": 24}
    c = c[c.CHR.isin(order)]; c["k"] = c.CHR.map(order); c["MAPINFO"] = c.MAPINFO.astype(float)
    return c.sort_values(["k", "MAPINFO"])
C = coords()
def sky(cell, col="mean"):
    d = pd.read_parquet(os.path.join(POST, f"atlas_{cell}_posterior.parquet")).set_index("cpg")[col].dropna()   # sites with no value are left out
    ids = C.index[C.index.isin(d.index)]; v = d.loc[ids].to_numpy(float); n = len(v)
    pix = (np.arange(n) * NPIX // n).astype(np.int64); s = np.bincount(pix, v, NPIX); k = np.bincount(pix, minlength=NPIX)
    m = np.full(NPIX, hp.UNSEEN); m[k > 0] = s[k > 0] / k[k > 0]; return m, n
def cmb():
    f = os.path.join(REPO, "docs", "book", "figscripts", "_data", "cmb_planck2018_nside256.npz")
    if not os.path.exists(f):
        import importlib.util; spec = importlib.util.spec_from_file_location("fs", os.path.join(REPO, "docs", "book", "figscripts", "fig_p4_sky.py"))
        fs = importlib.util.module_from_spec(spec); spec.loader.exec_module(fs); fs.build_cmb()
    return np.load(f)["map"]
def panel(fig, sub, m, title, cmap, lo, hi, unit, ticks=None, nest=True, zoom=None):
    if zoom is None: hp.mollview(m, fig=fig.number, sub=sub, title="", cmap=cmap, min=lo, max=hi, cbar=False, badcolor=BG, bgcolor=BG, nest=nest)
    else: hp.gnomview(m, fig=fig.number, sub=sub, title="", cmap=cmap, min=lo, max=hi, cbar=False, badcolor=BG, bgcolor=BG, nest=nest, rot=zoom, xsize=600, reso=1.5, notext=True)
    ax = plt.gca(); ax.set_title(title, color=FG, fontsize=13, pad=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi)); cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, aspect=40, shrink=0.8, ticks=ticks)
    cb.set_label(unit, color=FG, fontsize=10); cb.ax.tick_params(colors=FG, labelsize=9); cb.outline.set_edgecolor("#444")
os.makedirs(OUT, exist_ok=True); CM = "RdYlBu_r"
fig = plt.figure(figsize=(16, 5.2), facecolor=BG)
C0 = cmb(); panel(fig, (1, 2, 1), C0, "The microwave sky (simulated from Planck's measured spectrum)", CM, -300, 300, "temperature minus its mean (\u03bcK)", [-300, -150, 0, 150, 300], nest=False)
m, n = sky("neutrophils"); panel(fig, (1, 2, 2), m, f"IAM-Atlas v2: healthy neutrophils, {n:,} CpGs in genome order", CM, 0, 1, "methylation (posterior mean)", [0, 0.25, 0.5, 0.75, 1])
fig.savefig(os.path.join(OUT, "sky_atlas_hero.jpg"), dpi=150, facecolor=BG, bbox_inches="tight", pil_kwargs={"quality": 88}); plt.close(fig)
# Plate: both skies, then a 15-degree zoom of each (same patch on the sphere)
NM, _ = sky("neutrophils"); Z = (40, 20)
from PIL import Image
def sheet(specs, out, cols=2):
    ts = []
    for k, (m, title, cmap, lo, hi, unit, ticks, nest, zoom) in enumerate(specs):
        f = plt.figure(figsize=(8, 4.9) if zoom is None else (8, 8.4), facecolor=BG); panel(f, (1, 1, 1), m, title, cmap, lo, hi, unit, ticks, nest=nest, zoom=zoom)
        t = os.path.join(OUT, f"_p{k}.png"); f.savefig(t, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(f); ts.append(t)
    ims = [Image.open(t).convert("RGB") for t in ts]; w = max(i.width for i in ims); g = 24; rows = (len(ims) + cols - 1) // cols
    hs = [max(ims[r * cols + c].height for c in range(cols) if r * cols + c < len(ims)) for r in range(rows)]
    S = Image.new("RGB", (cols * w + (cols - 1) * g, sum(hs) + (rows - 1) * g), BG); y = 0
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            if k < len(ims): S.paste(ims[k], (c * (w + g) + (w - ims[k].width) // 2, y))
        y += hs[r] + g
    S.save(os.path.join(OUT, out), quality=88, optimize=True)
    for t in ts: os.remove(t)
TU = ("temperature minus its mean (\u03bcK)", [-300, -150, 0, 150, 300]); MU = ("methylation (posterior mean)", [0, 0.25, 0.5, 0.75, 1])
sheet([(C0, "The microwave sky (simulated from Planck's measured spectrum)", CM, -300, 300, *TU, False, None),
       (NM, "IAM-Atlas v2: healthy neutrophils, one CpG per pixel", CM, 0, 1, *MU, True, None),
       (C0, "Zoom, 15 degrees across", CM, -300, 300, *TU, False, Z),
       (NM, "Zoom, 15 degrees across: about 4,300 neighboring sites", CM, 0, 1, *MU, True, Z)], "sky_atlas_plate.jpg")
tiles = []
def tile(m, title, cmap, lo, hi, unit, ticks):
    f = plt.figure(figsize=(8, 4.9), facecolor=BG); panel(f, (1, 1, 1), m, title, cmap, lo, hi, unit, ticks)
    t = os.path.join(OUT, f"_tile{len(tiles)}.png"); f.savefig(t, dpi=110, facecolor=BG, bbox_inches="tight"); plt.close(f); tiles.append(t)
for cell, name in CELLS:
    m, n = sky(cell); tile(m, name, CM, 0, 1, "methylation (posterior mean)", [0, 0.25, 0.5, 0.75, 1])
m, n = sky("neutrophils", "sd"); tile(m, "Neutrophils: how sure the atlas is (posterior SD)", "magma", 0, 0.06, "posterior SD", [0, 0.02, 0.04, 0.06])
ims = [Image.open(t).convert("RGB") for t in tiles]; w = max(i.width for i in ims); h = max(i.height for i in ims); g = 24
sheet = Image.new("RGB", (2 * w + g, 4 * h + 3 * g), BG)
for k, im in enumerate(ims): sheet.paste(im, ((k % 2) * (w + g) + (w - im.width) // 2, (k // 2) * (h + g)))
sheet.save(os.path.join(OUT, "sky_atlas_cells.jpg"), quality=86, optimize=True)
for t in tiles: os.remove(t)
print("wrote sky_atlas_hero.jpg, sky_atlas_cells.jpg | CpGs placed:", n)
