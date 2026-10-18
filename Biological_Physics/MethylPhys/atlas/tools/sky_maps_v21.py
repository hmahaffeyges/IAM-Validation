#!/usr/bin/env python3
"""IAM-Atlas v2.1 on the sphere (release items M02 atlas maps and M03 uncertainty maps; written 2026-10-18).

The instrument's job: draw every cell type of the commissioned atlas on one HEALPix sphere, site for site, so any two cells (and a sample's
own map) can be laid side by side.
 - layout: every atlas site placed in genome order (chr1..chr22, chrX, chrY, then position; Illumina EPIC v1 B4 and 450K manifests, hg19) at
   pixel floor(i * npix / n), HEALPix NSIDE 256, NESTED order (genome neighbors share small patches), as the v2 page art
   (website/art/make_atlas_sky.py); a pixel that holds two sites shows their mean; a pixel with no measured site is masked (grey).
 - values: the atlas's best value per site and cell (compact copy atlas/v2.1/mu_part*.parquet; uint16 x 1e-4, 65535 = not measured) and its
   uncertainty, the half-width of the 95 % range, (mu_hi - mu_lo) / 2.
 - writes (OUT): sky_v21_mu_nside256_nest.npz and sky_v21_halfwidth_nside256_nest.npz (one float16 map per cell, NaN = masked), the 79-cell
   plate (atlas_v21_plate.jpg), the uncertainty plate (atlas_v21_uncertainty_plate.jpg) and a manifest with checksums.
 - masks and apodization: the masks are the unmeasured pixels. No statistic is computed here, so no apodization is applied; it enters with the
   power-spectrum work after the gift (atlas story).
A picture of the atlas: no reading, fit or score uses these maps. Usage: python3 sky_maps_v21.py ATLAS_DIR MANIFEST_DIR OUT | --synthetic"""
import os, sys, json, hashlib
for _k, _v in (("ASTROPY_CONFIGDIR", "/tmp/astro"), ("XDG_CONFIG_HOME", "/tmp/xdg")): os.environ.setdefault(_k, _v); os.makedirs(os.environ[_k], exist_ok=True)
import numpy as np, pandas as pd
NSIDE = 256; NPIX = 12 * NSIDE * NSIDE; NA = 65535

def genome_order(man, ids):
    a = pd.read_csv(os.path.join(man, "MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz"), usecols=["IlmnID", "CHR", "MAPINFO"], low_memory=False)
    b = pd.read_csv(os.path.join(man, "HumanMethylation450k_15017482_v3.csv.gz"), usecols=["IlmnID", "CHR", "MAPINFO"], low_memory=False)
    c = pd.concat([a, b]).dropna().drop_duplicates("IlmnID").set_index("IlmnID")
    c["CHR"] = c.CHR.astype(str).str.replace("chr", "", regex=False); k = {str(i): i for i in range(1, 23)} | {"X": 23, "Y": 24}
    c = c[c.CHR.isin(k)]; c["k"] = c.CHR.map(k); c["MAPINFO"] = c.MAPINFO.astype(float)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    c = c[c.index.isin(pos.index)].sort_values(["k", "MAPINFO"])
    return pos.loc[c.index].to_numpy(), int(len(ids) - len(c))     # row numbers of the atlas tables, in genome order; sites without a position

def decode(u):
    u = np.asarray(u); v = u.astype(np.float32) / 1e4; v[u == NA] = np.nan; return v

def to_sky(v):
    """v: one value per site, already in genome order -> NSIDE 256 NESTED map (mean per pixel; NaN where no measured site)."""
    n = len(v); pix = (np.arange(n) * NPIX // n).astype(np.int64); ok = np.isfinite(v)
    s = np.bincount(pix[ok], v[ok], NPIX); k = np.bincount(pix[ok], minlength=NPIX)
    m = np.full(NPIX, np.nan, np.float32); m[k > 0] = s[k > 0] / k[k > 0]; return m

_UP = {"cd4": "CD4", "cd8": "CD8", "nk": "NK", "th1": "Th1", "cd56bright": "CD56bright", "cd28-": "CD28-", "t": "T", "b": "B", "temra": "TEMRA"}
def pretty(c):   # display name only (the data keep the atlas's own names)
    w = [(_UP.get(x.lower(), x) if x.lower() in _UP or i else x[:1].upper() + x[1:]) for i, x in enumerate(c.replace("(cd28-)", "(CD28-)").split(" "))]
    return " ".join(w)[:36]

def synthetic():
    """planted: decoding, genome-order placement and masking, before any atlas value is read."""
    T = {}
    u = np.array([0, 10000, 5000, NA, 1234], np.uint16); d = decode(u)
    T["decode"] = bool(d[0] == 0 and d[1] == 1 and abs(d[2] - 0.5) < 1e-6 and np.isnan(d[3]) and abs(d[4] - 0.1234) < 1e-6)
    n = 813_951; v = np.repeat(np.arange(24, dtype=np.float32), n // 24 + 1)[:n]    # chromosome number as the value, in genome order
    m = to_sky(v); T["one_value_per_chromosome_block"] = bool(np.nanmax(np.abs(np.diff(m[np.isfinite(m)]))) <= 1 and np.all(np.diff(m[np.isfinite(m)]) >= 0))
    v2 = v.copy(); v2[:n // 4] = np.nan; m2 = to_sky(v2)
    T["masked_where_unmeasured"] = bool(np.isnan(m2[: NPIX // 4 - 2]).all() and np.isfinite(m2[NPIX // 4 + 2:]).all())
    T["pixels_filled"] = int(np.isfinite(m).sum()); T["pixels_total"] = NPIX
    T["met"] = all(x for k, x in T.items() if isinstance(x, bool)); return T

def main(atlas, man, out):
    import pyarrow.parquet as pq, healpy as hp, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    os.makedirs(out, exist_ok=True)
    ids = pq.read_table(os.path.join(atlas, "sites.parquet")).column("cpg").to_pylist(); order, unplaced = genome_order(man, ids)
    MU, HW = {}, {}
    for k in (1, 2, 3):
        a, lo, hi = (pq.read_table(os.path.join(atlas, f"{t}_part{k}.parquet")) for t in ("mu", "mu_lo", "mu_hi"))
        for c in a.column_names:
            mu = decode(a.column(c).to_numpy())[order]; MU[c] = to_sky(mu)
            HW[c] = to_sky((decode(hi.column(c).to_numpy()) - decode(lo.column(c).to_numpy()))[order] / 2)
        del a, lo, hi
    cells = sorted(MU)
    for name, D in (("sky_v21_mu_nside256_nest.npz", MU), ("sky_v21_halfwidth_nside256_nest.npz", HW)):
        np.savez_compressed(os.path.join(out, name), cells=np.array(cells), nside=NSIDE, order="NESTED", **{f"m{i:02d}": D[c].astype(np.float16) for i, c in enumerate(cells)})
    here = os.path.dirname(os.path.abspath(__file__)); rgb = os.path.join(here, "..", "..", "..", "..", "website", "art", "data", "Planck_Parchment_RGB.txt")
    CM = ListedColormap(np.loadtxt(rgb) / 255.0, name="planck"); CM.set_bad("#d9d9d9")
    def thumb(m):   # plate thumbnails only: the mean of the measured pixels in each NSIDE 64 pixel (16 of NSIDE 256, NESTED); the saved maps stay NSIDE 256
        g = m.reshape(-1, 16); ok = np.isfinite(g); s_ = np.where(ok, g, 0).sum(1); k_ = ok.sum(1)
        return np.where(k_ > 0, s_ / np.maximum(k_, 1), hp.UNSEEN)
    def plate(D, fn, lo_, hi_, cmap, unit, title):
        cols = 9; rows = int(np.ceil(len(cells) / cols)); fig = plt.figure(figsize=(cols * 2.3, rows * 1.5 + 1.6), facecolor="white")
        for i, c in enumerate(cells):
            hp.mollview(thumb(D[c]), nest=True, fig=fig.number, sub=(rows + 1, cols, i + 1), title="", cmap=cmap, min=lo_, max=hi_,
                        cbar=False, badcolor="#d9d9d9", bgcolor="white", xsize=420)
            plt.gca().set_title(pretty(c), fontsize=6.8, pad=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo_, hi_)); cax = fig.add_axes([0.35, 0.06, 0.3, 0.012])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_label(unit, fontsize=9); cb.ax.tick_params(labelsize=8)
        fig.suptitle(title, fontsize=12, y=1.035); fig.savefig(os.path.join(out, fn), dpi=130, bbox_inches="tight", pil_kwargs={"quality": 90}); plt.close(fig)
    plate(MU, "atlas_v21_plate.jpg", 0, 1, CM, "methylation (best value)",
          f"IAM-Atlas v2.1: {len(cells)} healthy human cell types, {len(order):,} sites each, in genome order on one HEALPix sphere\n"
          "Each map: one cell type's best value at every site; neighbors on the genome are neighbors on the sphere. Thumbnails at NSIDE 64; the released maps are NSIDE 256.")
    UC = plt.get_cmap("viridis").copy(); UC.set_bad("#d9d9d9")
    plate(HW, "atlas_v21_uncertainty_plate.jpg", 0, 0.15, UC, "half-width of the 95 % range",
          "IAM-Atlas v2.1: where each cell's reference is sure and where it is not (half-width of the 95 % range, site by site)")
    M = dict(cells=len(cells), sites=len(ids), placed=len(order), without_position=unplaced, nside=NSIDE, order="NESTED", pixels=NPIX,
             files={f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest() for f in sorted(os.listdir(out)) if f.endswith((".npz", ".jpg"))},
             median_pixels_filled=int(np.median([np.isfinite(MU[c]).sum() for c in cells])))
    json.dump(M, open(os.path.join(out, "sky_v21_manifest.json"), "w"), indent=1); print(json.dumps({k: v for k, v in M.items() if k != "files"}))

if __name__ == "__main__":
    if sys.argv[1] == "--synthetic": print(json.dumps(synthetic(), indent=1))
    else: main(*sys.argv[1:4])
