"""The report's turning globe (2026-10-21, to-do step 13d; author: "remove all but the differences and see the 3D version of the
differences from the healthy cell map, manipulated by the reviewer").

A picture, not a reading: nothing here is scored, and no instrument reads from it. Met-A and the C-Score keep their commissioned reference
(the purified-array reference); this globe shows the sample against IAM-Atlas v2.1, named on the page.

Three layers on one globe, every atlas site on the same pixel as the atlas's cell pages and the public globe (the frozen genome-order layout,
Runtime Matrices/Atlas_V21/sky_order_v21.npz, built by atlas/tools/sky_order_v21.py; site i of n at NESTED pixel floor(i * NPIX / n),
NSIDE 256; a pixel holding two sites shows their mean):
  sample      the sample's methylation at each atlas site;
  reference   the healthy reference the sample is compared with, chosen by specimen:
                an isolated cell type -> that cell in the atlas;
                whole blood or buffy coat -> the atlas's healthy blood for this person's own mix: each of the six panel cells weighted by its
                share from Stage D21 (the IAM-Deconvolver), the shares scaled to sum to 1. Comparing blood with one cell would light up every
                site where blood differs from that cell;
  difference  sample minus reference, per pixel. The reader sets a size below which differences fade into the background, so only the
              larger ones stay lit. The page prints the share of measured pixels above the chosen size, from the pixel values themselves.
  departures  only Met-A's 6,000 neutrophil identity sites: the C-Score's departure map (conductor_v3, unchanged), centered and scaled by the
              sample's own median and spread at those sites, exactly as the report's overlay (changes_sky_png); the reader sets the size. A small loss
              spread over many sites is not visible on the whole-genome layer; this layer shows the instrument's own view of it.
  purified    Met-A's own reference (2026-10-21, author: "the purified cell reference globe ... the appropriate one for the appropriate test"):
              at Met-A's 6,000 identity sites only, the purified-array group profiles (Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json)
              weighted by this blood's Stage A fractions (conductor_v3.stage_a_composition), or the neutrophil profile for isolated neutrophils;
              with the sample's difference from it. Dots, as the departures layer.
  cells       each Stage D21 cell as its own layer, named with this sample's share: the cell's healthy map in IAM-Atlas v2.1.
Sex: in women one X chromosome is held methylated, so a woman's X differs from a reference that pools both sexes (seen on the first real
blood, 2026-10-21: the last 2 % of the genome order, where X sits, differs 7 to 11 times as often as the rest). The page names it.
Known offset (DEV-REFERENCES-01, ATL-070): the atlas blends 21 sources and sits about 0.006 closer to 0.5 than a single array at most sites,
so a healthy array shows a faint even difference everywhere; that is the reference's make, not biology.
build(beta, specimen, shares, out_html) -> record; the page fragment is written to out_html and also returned in the record."""
import os, io, json, base64, numpy as np, pandas as pd
from globe_viewer import dedupe, fragment, page, write_files, VIEWER_JS   # the one viewer

HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(HERE, "..")
ATLAS = os.path.join(CH, "..", "atlas", "v2.1")
ORDER = os.path.join(CH, "Runtime Matrices", "Atlas_V21", "sky_order_v21.npz")
W, HT = 1024, 512; NA = 65535; R_DOT = 3; BLOCK = 10   # the C-Score's block (neutrophil_reference_v1_2.json clustering_block)
def FW_LINE(n):
    """The fixed line for n blocks: the size pure chance (a normal block) crosses less than once in 20 genomes, two-sided, 5 % family-wise."""
    from scipy.stats import norm
    return float(norm.isf(0.05 / max(n, 1) / 2))
PANEL = {"neutrophils": "neutrophils", "monocytes": "monocytes", "b cells": "b cells", "cd4 t cells": "cd4 t cells", "cd8 t cells": "cd8 t cells", "nk cells": "nk cells"}
BLOOD = ("whole blood", "buffy coat", "peripheral blood")

def _to_sky(v, npix):
    """One value per site, in genome order -> NESTED map, mean per pixel, NaN where no site was measured (sky_maps_v21.to_sky, the same rule)."""
    n = len(v); pix = (np.arange(n) * npix // n).astype(np.int64); ok = np.isfinite(v)
    s = np.bincount(pix[ok], v[ok], npix); k = np.bincount(pix[ok], minlength=npix)
    return np.where(k > 0, s / np.maximum(k, 1), np.nan)

def _mu(cells, stem="mu"):
    """Atlas values for `cells` at every atlas site (atlas row order); NaN where the atlas has none. stem: mu (best value), mu_lo / mu_hi
    (the 95 % range from the fit)."""
    import pyarrow.parquet as pq
    out = {}
    for k in (1, 2, 3):
        p = os.path.join(ATLAS, f"{stem}_part{k}.parquet"); have = [c for c in cells if c in pq.read_schema(p).names]
        if not have: continue
        t = pq.read_table(p, columns=have)
        for c in have:
            u = t.column(c).to_numpy(); v = u.astype(np.float32) / 1e4; v[u == NA] = np.nan; out[c] = v
    return out

def _to_sky_extreme(v, npix):
    """One value per site -> NESTED map, per pixel the value of largest size (sign kept): a changed site is never averaged away by an
    unchanged neighbor in the same pixel. 0 where every site of the pixel is unchanged; NaN where none was measured."""
    n = len(v); pix = (np.arange(n) * npix // n).astype(np.int64); ok = np.isfinite(v)
    hi = np.full(npix, -np.inf); lo = np.full(npix, np.inf); np.maximum.at(hi, pix[ok], v[ok]); np.minimum.at(lo, pix[ok], v[ok])
    out = np.where(np.isfinite(hi), np.where(hi >= -lo, hi, lo), np.nan); return out

def _lookup(nside):
    """For every texture pixel (equirectangular W x HT), the NESTED sky pixel it shows (nearest pixel center)."""
    from scipy.spatial import cKDTree
    import sys; sys.path.insert(0, CH); from stage_v4 import nest_pix2ang
    th, ph = nest_pix2ang(nside, np.arange(12 * nside * nside)); xyz = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    tt = np.pi * (np.arange(HT) + 0.5) / HT; pp = 2 * np.pi * (np.arange(W) + 0.5) / W; T, P = np.meshgrid(tt, pp, indexing="ij")
    return cKDTree(xyz).query(np.c_[(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()])[1]

def _png_b64(arr, mode):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(arr, mode).save(b, "PNG", optimize=True); return base64.b64encode(b.getvalue()).decode()

def _jpg_b64(rgb):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=84, optimize=True); return base64.b64encode(b.getvalue()).decode()

def reference(specimen, shares):
    """(name, {atlas cell: weight}) of the healthy reference for this specimen, or (None, reason)."""
    s = str(specimen or "").strip().lower()
    if s in BLOOD:
        w = {PANEL[c]: float(v) for c, v in (shares or {}).items() if c in PANEL and v is not None and np.isfinite(v) and v > 0}
        if not w: return None, "no cell shares from Stage D21, so the healthy blood for this person's mix cannot be drawn"
        t = sum(w.values()); return "healthy blood for this person's own cell mix (IAM-Atlas v2.1, Stage D21 shares)", {c: v / t for c, v in w.items()}
    c = s[9:] if s.startswith("isolated ") else s
    c = {"neutrophil": "neutrophils", "cd8": "cd8 t cells", "cd8 t cell": "cd8 t cells", "cd4 t cell": "cd4 t cells", "monocyte": "monocytes", "b cell": "b cells", "nk cell": "nk cells"}.get(c, c)
    return (f"healthy {c} (IAM-Atlas v2.1)", {c: 1.0}) if c else (None, "no specimen given")

def _dots(v):
    """A sparse texture (most pixels NaN) drawn as dots: every pixel within R_DOT of a site takes that site's value (nearest site)."""
    from scipy import ndimage
    miss = ~np.isfinite(v)
    if miss.all(): return v
    d, (i, j) = ndimage.distance_transform_edt(miss, return_indices=True)
    return np.where(d <= R_DOT, v[i, j], np.nan)

def purified(fractions, specimen):
    """Met-A's own reference at its 6,000 sites: Series by site, or None. Blood: the Stage A fractions times the purified group profiles."""
    B = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "blood_composition_EPIC_v1.json")))
    S = B["neutrophil_sites"]; P = {g: np.asarray(v, float) for g, v in B["profiles_at_neutrophil_sites"].items()}
    if str(specimen or "").strip().lower() in BLOOD:
        f = {g: float(x) for g, x in (fractions or {}).items() if g in P and x is not None}
        if not f or sum(f.values()) <= 0: return None
        t = sum(f.values()); return pd.Series(sum(P[g] * (x / t) for g, x in f.items()), index=S)
    return pd.Series(P["NEU"], index=S) if "neutrophil" in str(specimen).lower() else None

def build(beta, specimen, shares, out_html=None, z=None, fractions=None):
    import pyarrow.parquet as pq
    from matplotlib.colors import ListedColormap
    if not os.path.exists(ORDER): return {"drawn": False, "reason": f"the frozen layout {os.path.relpath(ORDER, CH)} is missing"}
    name, w = reference(specimen, shares)
    if name is None: return {"drawn": False, "reason": w}
    Z = np.load(ORDER); order = Z["order"]; nside = int(Z["nside"]); npix = 12 * nside * nside
    ids = np.array(pq.read_table(os.path.join(ATLAS, "sites.parquet")).column("cpg").to_pylist(), dtype=object)
    import hashlib
    if hashlib.sha256(open(os.path.join(ATLAS, "sites.parquet"), "rb").read()).hexdigest() != str(Z["sites_sha256"]):
        return {"drawn": False, "reason": "the atlas site list differs from the one the frozen layout was built on"}
    mu = _mu(list(w)); missing = [c for c in w if c not in mu]
    if missing: return {"drawn": False, "reason": f"cell(s) not in IAM-Atlas v2.1: {', '.join(missing)}"}
    ref = sum(w[c] * mu[c] for c in w)                                   # NaN where any weighted cell has no atlas value
    smp = pd.Series(beta, dtype="float64").reindex(ids).to_numpy()
    ref_s, smp_s = _to_sky(ref[order], npix), _to_sky(smp[order], npix)
    both = np.isfinite(ref[order]) & np.isfinite(smp[order]); d = np.where(both, smp[order] - ref[order], np.nan); dif_s = _to_sky(d, npix)
    # 2026-10-21 (author: "slide out anything but the change and the change has a fixed ... number"): the change is fixed by the measurement,
    # never by the reader. Each site's difference is divided by that site's healthy width in the atlas (the healthy person-to-person spread,
    # donor_sd, and the atlas's own uncertainty, half the 95 % range over 1.96; blood: weighted by the person's shares), then centered and scaled
    # by the sample's own median and spread (MAD x 1.4826), which cancels the atlas-to-array offset and the array's own noise level. A site has
    # CHANGED beyond 3: the C-Score's rule, on the whole genome. (Tried first and dropped the same day: "outside the atlas's 95 % range" flagged
    # 84 % of a healthy blood, because that range is how sure the atlas is of the healthy average, not how much a healthy person varies; with
    # the person-to-person spread added, still 18.5 % beyond 2 widths, because an array's own noise is not in the atlas.)
    lo_, hi_, sd_ = _mu(list(w), "mu_lo"), _mu(list(w), "mu_hi"), _mu(list(w), "donor_sd")
    un = sum(w[c] * (hi_[c] - lo_[c]) / 3.92 for c in w); width = np.sqrt(sum(w[c] * sd_[c] ** 2 for c in w) + un ** 2)[order]
    x = np.where(both & np.isfinite(width) & (width > 0), d / np.where(width > 0, width, np.nan), np.nan); xv = x[np.isfinite(x)]
    med = float(np.median(xv)); mad = float(np.median(np.abs(xv - med))) * 1.4826 or 1.0; zg = (x - med) / mad
    # Blocks, not single sites (2026-10-21, author: "no way the entire globe is lit up"): one site beyond 3 lit 2.1 % of a healthy blood's
    # sites across the whole globe. The change is read in blocks of 10 neighboring measured sites in genome order (the C-Score's block), each
    # block's mean in the sample's own block spread (neighboring CpGs move together, so a healthy block's spread is about twice that of
    # independent sites); a block has changed beyond the line pure chance crosses less than once in 20 genomes (5 % family-wise over all blocks,
    # FW_LINE). Healthy blood: 45 of 77,355 blocks; 200 planted runs of 30 sites: 229 of their 756 blocks found.
    mi = np.where(np.isfinite(zg))[0]; nb = len(mi) // BLOCK; bm_ = zg[mi[:nb * BLOCK]].reshape(nb, BLOCK).mean(1)
    c_ = float(np.median(bm_)); s_ = float(np.median(np.abs(bm_ - c_))) * 1.4826 or 1.0; bz = (bm_ - c_) / s_; line = FW_LINE(nb)
    ex = np.where(np.isfinite(zg), 0.0, np.nan); ex[mi[:nb * BLOCK]] = np.repeat(np.where(np.abs(bz) > line, bz, 0.0), BLOCK)
    n_changed = int((np.abs(bz) > line).sum()); n_ranged = int(nb); ex_s = _to_sky_extreme(ex, npix)
    look = _lookup(nside)
    cm = ListedColormap(np.loadtxt(os.path.join(HERE, "Planck_Parchment_RGB.txt")) / 255.0)
    def rgb(m):
        v = m[look].reshape(HT, W); x = (cm(np.clip(np.nan_to_num(v, nan=0), 0, 1))[..., :3] * 255).astype(np.uint8); x[~np.isfinite(v)] = (42, 45, 51); return x
    dv = ex_s[look].reshape(HT, W)   # the change code: 0 no site, 128 unchanged, 128 +/- 127 x size/0.5 changed (at least one step from 128)
    def _code(v, full):
        # 2026-10-21: a browser may color-manage an image and shift its values by a few steps (seen in Safari: every "unchanged" 128 drawn as a
        # faint change), so the code keeps wide gaps: 0 no site; 128 unchanged; a change 128 +/- (48 + 79 x size/full), never within 48 of 128.
        # The page reads below 24 as no site, within 24 of 128 as unchanged.
        a = np.clip(np.abs(v) / full, 0, 1); c = np.where(np.isfinite(v), np.where(v != 0, 128 + np.sign(v) * np.round(48 + 79 * a), 128), 0)
        c = np.clip(c, 0, 255).astype(np.uint8); return np.repeat(c[..., None], 3, -1)   # stored as RGB, read the same way by every browser
    code = _code(dv, 10)
    zcode, zabove = None, None
    if z is not None and len(z):
        z = pd.Series(z, dtype="float64"); med = float(z.median()); mad = float((z - med).abs().median()) * 1.4826 or float(z.std()) or 1.0
        z = (z - med) / mad   # in the sample's own spread at these sites, exactly as the report's overlay (report_figs_v4.changes_sky_png)
        # 2026-10-21 (author: "shouldn't they still be in clumps ... if they are truly other than aging"): the change at Met-A's sites is read as
        # the C-Score reads it, in blocks of 10 neighboring measured sites in genome order (neutrophil_reference_v1_2.json clustering_block):
        # a block has changed when its mean departure, in standard errors (mean x sqrt(10)), is beyond 3, and every site of a changed block is
        # lit with the block's value. Scattered wear seldom moves a whole block; a clustered loss does (healthy blood: 7 of 597 blocks;
        # the planted clustered loss: 57). A single site is not judged alone: a small loss spread along a run never passes 3 at one site.
        _R = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "neutrophil_reference_v1_2.json"))); _k = int(_R["clustering_block"])
        _zo = z.reindex(_R["sites_ordered"]).dropna(); _nb = len(_zo) // _k
        _bz = _zo.to_numpy()[:_nb * _k].reshape(_nb, _k).mean(1) * np.sqrt(_k); _ml = FW_LINE(_nb); _lit = np.repeat(np.where(np.abs(_bz) > _ml, _bz, 0.0), _k)
        z = pd.Series(np.r_[_lit, np.zeros(len(_zo) - _nb * _k)], index=_zo.index); blocks = (int((np.abs(_bz) > _ml).sum()), int(_nb))
        zz = z.reindex(ids[order]).to_numpy(); zs = _to_sky(zz, npix); zv = zs[look].reshape(HT, W)
        # 6,000 sites hold about one texture pixel each, too small to see: each is drawn as a dot of radius R_DOT pixels at its own place, the
        # larger departure winning where two dots meet, so a run of neighboring departing sites shows as a patch. Positions are unchanged.
        from scipy import ndimage
        fp = np.hypot(*np.mgrid[-R_DOT:R_DOT + 1, -R_DOT:R_DOT + 1]) <= R_DOT
        pos = ndimage.maximum_filter(np.where(zv > 0, zv, -np.inf), footprint=fp, mode="wrap")
        neg = ndimage.maximum_filter(np.where(zv < 0, -zv, -np.inf), footprint=fp, mode="wrap")
        has = ndimage.maximum_filter(np.isfinite(zv).astype(np.uint8), footprint=fp, mode="wrap") > 0
        zv = np.where(has, np.where(pos >= neg, np.maximum(pos, 0), -neg), np.nan); zv[has & ~np.isfinite(zv)] = 0.0
        zcode = _code(np.where(np.isfinite(zv) & (np.abs(zv) > 3), zv, np.where(np.isfinite(zv), 0.0, np.nan)), 10)   # changed: beyond 3 spreads, the C-Score's fixed rule
        zabove = blocks   # (changed blocks, blocks)
    pcode = prgb = pabove = None; pe = purified(fractions, specimen)
    if pe is not None:
        pv = pe.reindex(ids[order]).to_numpy(); sv = pd.Series(beta, dtype="float64").reindex(ids[order]).to_numpy()
        ps = _to_sky(pv, npix); pdif = _to_sky(np.where(np.isfinite(pv) & np.isfinite(sv), sv - pv, np.nan), npix)
        pt = _dots(ps[look].reshape(HT, W)); prgb = (cm(np.clip(np.nan_to_num(pt, nan=0), 0, 1))[..., :3] * 255).astype(np.uint8); prgb[~np.isfinite(pt)] = (42, 45, 51)
        pcode = True
    cellmaps = []
    for c, wt in sorted((shares or {}).items(), key=lambda x: -(x[1] or 0)):
        if c in PANEL and wt is not None:
            mc = _mu([PANEL[c]])[PANEL[c]]; cellmaps.append(dict(name=f"{c}: {100 * wt:.0f} % of this sample", t=_jpg_b64(rgb(_to_sky(mc[order], npix)))))
    m = np.abs(dif_s[np.isfinite(dif_s)]); steps = np.round(np.arange(0, 0.51, 0.01), 2)
    above = [round(float((m > t).mean()), 4) if len(m) else None for t in steps]   # share of measured sky pixels above each size (equal-area)
    rec = {"drawn": True, "reference": name, "weights": {c: round(v, 4) for c, v in w.items()}, "sites_compared": int(both.sum()),
           "pixels_compared": int(len(m)), "median_abs_difference": round(float(np.median(m)), 4) if len(m) else None,
           "share_above": {"0.05": above[5], "0.10": above[10], "0.20": above[20], "0.30": above[30]},
           "change_rule": "blocks of 10 neighboring sites, each site's difference in its healthy width in IAM-Atlas v2.1, the block mean in the sample's own block spread, beyond the 5 % family-wise line", "sites_changed": n_changed, "sites_with_range": n_ranged,
           "layout": "IAM-Atlas v2.1 genome-order sky, NSIDE 256 NESTED (sky_order_v21.npz)", "scored": False}
    J = lambda b: "data:image/jpeg;base64," + b; P_ = lambda b: "data:image/png;base64," + b
    R_ = J(_jpg_b64(rgb(ref_s)))
    L = [dict(name="the change from the healthy reference", kind="change", src=P_(_png_b64(code, "RGB")), base=R_, full=10,
              hint="Slide right to quiet the healthy background: its colors drain to gray, then to dark, until only the change is left lit. Slide left to bring the full map back.",
              fixed=f"Changed: {n_changed:,} of {n_ranged:,} blocks of 10 neighboring sites. This count is fixed; the slider never changes it.",
              note=(f"Against {name}. Each site's difference is measured in that site's healthy width (how much healthy people vary there, and how "
                    "sure the atlas is), in this sample's own spread, and read in blocks of 10 neighboring sites, as the C-Score reads them. A block "
                    "has changed beyond the line pure chance crosses less than once in 20 genomes. Ordinary wear is scattered and seldom moves a whole "
                    "block, so a healthy sample stays nearly dark; a change that runs along the genome lights up in clumps. Red: more methylation "
                    "than healthy; blue: less.")),
         dict(name="the sample", kind="image", src=J(_jpg_b64(rgb(smp_s))), note="The sample's methylation at every atlas site, 0 to 1."),
         dict(name="the healthy reference", kind="image", src=R_, note=f"{name}, 0 to 1.")]
    if zcode is not None:
        L.append(dict(name="the change at Met-A's 6,000 sites (against its purified reference)", kind="change", src=P_(_png_b64(zcode, "RGB")), full=10,
                      hint="Slide right to quiet the unchanged sites to dark, until only the changed ones are left lit. Slide left to bring them back.",
                      fixed=f"Changed: {zabove[0]:,} of {zabove[1]:,} blocks of 10 neighboring sites. This count is fixed; the slider never changes it.",
                      note=("The C-Score's departure map against Met-A's commissioned purified-array reference, read as the C-Score reads it: in blocks "
                            "of 10 neighboring sites along the genome, a block changed beyond the line pure chance crosses less than once in 20 genomes. "
                            "Ordinary wear is scattered and seldom moves a whole block; a loss that runs along the genome does, so it shows as clumps.")))
    if pcode is not None:
        Pr = J(_jpg_b64(prgb))
        L += [dict(name="Met-A's purified-array reference", kind="image", src=Pr, note="Met-A's own healthy reference at its 6,000 sites, weighted by this blood's Stage A fractions; the change against it is the layer above.")]
    groups = [dict(name="this sample against its references", layers=L)]
    if cellmaps: groups.append(dict(name="the cells the IAM-Deconvolver read", layers=[dict(name="healthy " + c["name"], kind="image", src=J(c["t"]),
                                     note="This cell's healthy map in IAM-Atlas v2.1; the share is the IAM-Deconvolver's reading of this sample.") for c in cellmaps]))
    rec["layers"] = [x["name"] for g in groups for x in g["layers"]]
    if zabove: rec["metaa_blocks_changed"] = {"rule": "blocks of 10 neighboring sites, mean departure in standard errors beyond the 5 % family-wise line (the C-Score's blocks)", "changed": zabove[0], "blocks": zabove[1]}
    spec = dedupe(dict(id="sample", name="This sample", about=ABOUT.replace("__REF__", name).replace("__N__", f"{rec['sites_compared']:,}"), groups=groups))
    rec["fragment"] = fragment([spec])
    if out_html: open(out_html, "w", encoding="utf-8").write(page([spec], "The sample on the globe")); rec["html"] = os.path.abspath(out_html)
    rec["prose"] = " ".join([spec["about"]] + [g["name"] for g in groups] + [x.get(k, "") for g in groups for x in g["layers"] for k in ("name", "note", "share_text")])
    rec["spec"] = spec
    return rec

ABOUT = ("Every site of IAM-Atlas v2.1 has one fixed place on this globe, the same as on the atlas's cell pages. Drag to turn it. The healthy "
         "reference is <b>__REF__</b>, compared at __N__ sites. The change is fixed by the measurement, never by the reader: each site's difference "
         "is measured in that site's healthy width and the sample's own spread, and read in blocks of 10 neighboring sites, as the C-Score reads "
         "them; a block has changed beyond the line pure chance crosses less than once in 20 genomes. Ordinary wear is scattered and seldom moves a "
         "whole block, so a healthy sample stays nearly dark; a change that runs along the genome lights up in clumps. The slider only quiets the "
         "background. A picture: nothing is scored from it; Met-A and the C-Score read against their own commissioned reference, which is also a "
         "layer. In a woman one X chromosome is held methylated, so her X can differ from a reference that pools both sexes.")

PAIR_ABOUT = ("<b>DEVELOPMENT: one person's two samples, not commissioned by our standards.</b> Every site of IAM-Atlas v2.1 has one fixed place on "
              "this globe. The sample is compared with the same person's own __REF__, at __N__ sites both arrays measured: self-reference, no "
              "other person enters. The change is fixed by the measurement: each site's difference in the pair's own spread, read in blocks of "
              "10 neighboring sites, a block changed beyond the line pure chance crosses less than once in 20 genomes. A change that runs along "
              "the genome lights up in clumps; scattered differences stay dark. The slider only quiets the background. A picture: nothing is "
              "scored from it.")

def build_pair(beta, ref_beta, ref_name, out_html=None, title="This sample against the same person's own other sample"):
    """DEVELOPMENT (2026-10-21, author: test the globe on the breast and colon tissue results): one person's sample (a tumor) against the same
    person's own other sample (its adjacent normal), on the atlas's layout, by the globe's block rule without the atlas: each site's
    difference, centered and scaled by the pair's own spread (MAD x 1.4826), blocks of BLOCK neighboring measured sites, the block mean in its
    own block spread, changed beyond FW_LINE. Layers: the change (slider quiets the reference sample's map), the sample, the reference."""
    import pyarrow.parquet as pq
    ids = np.array(pq.read_table(os.path.join(ATLAS, "sites.parquet")).column("cpg").to_pylist(), dtype=object)
    Z = np.load(ORDER); order = Z["order"]; nside = int(Z["nside"]); npix = 12 * nside * nside; look = _lookup(nside)
    smp = pd.Series(beta, dtype="float64").reindex(ids).to_numpy(); ref = pd.Series(ref_beta, dtype="float64").reindex(ids).to_numpy()
    d = (smp - ref)[order]; ok = np.isfinite(d); v = d[ok]; med = float(np.median(v)); mad = float(np.median(np.abs(v - med))) * 1.4826 or 1.0
    zg = np.where(ok, (d - med) / mad, np.nan); mi = np.where(ok)[0]; nb = len(mi) // BLOCK
    bm_ = zg[mi[:nb * BLOCK]].reshape(nb, BLOCK).mean(1); c_ = float(np.median(bm_)); s_ = float(np.median(np.abs(bm_ - c_))) * 1.4826 or 1.0
    bz = (bm_ - c_) / s_; line = FW_LINE(nb); ex = np.where(ok, 0.0, np.nan); ex[mi[:nb * BLOCK]] = np.repeat(np.where(np.abs(bz) > line, bz, 0.0), BLOCK)
    dv = _to_sky_extreme(ex, npix)[look].reshape(HT, W)
    a = np.clip(np.abs(dv) / 10, 0, 1); c = np.where(np.isfinite(dv), np.where(dv != 0, 128 + np.sign(dv) * np.round(48 + 79 * a), 128), 0)
    code = np.repeat(np.clip(c, 0, 255).astype(np.uint8)[..., None], 3, -1)
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(np.loadtxt(os.path.join(HERE, "Planck_Parchment_RGB.txt")) / 255.0)
    def rgb(m):
        x = (cm(np.clip(np.nan_to_num(m, nan=0), 0, 1))[..., :3] * 255).astype(np.uint8); x[~np.isfinite(m)] = (42, 45, 51); return x
    sm = _to_sky(smp[order], npix)[look].reshape(HT, W); rm = _to_sky(ref[order], npix)[look].reshape(HT, W)
    J = lambda b: "data:image/jpeg;base64," + b; R_ = J(_jpg_b64(rgb(rm))); n_ch = int((np.abs(bz) > line).sum())
    L = [dict(name="the change from the same person's own " + ref_name, kind="change", src="data:image/png;base64," + _png_b64(code, "RGB"), base=R_, full=10,
              hint="Slide right to quiet the reference sample's map: its colors drain to gray, then to dark, until only the change is left lit.",
              fixed=f"Changed: {n_ch:,} of {nb:,} blocks of 10 neighboring sites. This count is fixed; the slider never changes it.",
              note="Red: more methylation than the same person's own " + ref_name + "; blue: less. Development: not commissioned by our standards."),
         dict(name="the sample", kind="image", src=J(_jpg_b64(rgb(sm))), note="The sample's methylation at every atlas site its array measured, 0 to 1."),
         dict(name="the same person's own " + ref_name, kind="image", src=R_, note="The reference sample's methylation, 0 to 1.")]
    spec = dedupe(dict(id="pair", name=title, about=PAIR_ABOUT.replace("__REF__", ref_name).replace("__N__", f"{int(ok.sum()):,}"),
                       groups=[dict(name="this person", layers=L)]))
    rec = dict(drawn=True, mode="pair (development)", sites_compared=int(ok.sum()), blocks=int(nb), blocks_changed=n_ch, line=round(line, 3),
               change_rule="the pair's own difference, blocks of 10 neighboring sites, 5 % family-wise line", flag="development: not commissioned by our standards")
    if out_html: open(out_html, "w", encoding="utf-8").write(page([spec], title)); rec["html"] = os.path.abspath(out_html)
    rec["spec"] = spec; return rec

if __name__ == "__main__":
    # python3 report_globe_v4.py pair SAMPLE_BETAS REFERENCE_BETAS "reference name" OUT.html ["title"]   (betas: chain Stage 1, cpg_id,beta)
    import sys as _s
    if len(_s.argv) >= 6 and _s.argv[1] == "pair":
        rd = lambda f: (pd.read_parquet(f).iloc[:, 0] if f.endswith(".parquet") else pd.read_csv(f, index_col=0).iloc[:, 0])
        r = build_pair(rd(_s.argv[2]), rd(_s.argv[3]), _s.argv[4], _s.argv[5], *(_s.argv[6:7]))
        print(json.dumps({k: v for k, v in r.items() if k != "spec"}))
    else: print(__doc__)

