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
W, HT = 1024, 512; NA = 65535; R_DOT = 3
PANEL = {"neutrophils": "neutrophils", "monocytes": "monocytes", "b cells": "b cells", "cd4 t cells": "cd4 t cells", "cd8 t cells": "cd8 t cells", "nk cells": "nk cells"}
BLOOD = ("whole blood", "buffy coat", "peripheral blood")

def _to_sky(v, npix):
    """One value per site, in genome order -> NESTED map, mean per pixel, NaN where no site was measured (sky_maps_v21.to_sky, the same rule)."""
    n = len(v); pix = (np.arange(n) * npix // n).astype(np.int64); ok = np.isfinite(v)
    s = np.bincount(pix[ok], v[ok], npix); k = np.bincount(pix[ok], minlength=npix)
    return np.where(k > 0, s / np.maximum(k, 1), np.nan)

def _mu(cells):
    """Atlas best values for `cells` at every atlas site (atlas row order); NaN where the atlas has none."""
    import pyarrow.parquet as pq
    out = {}
    for k in (1, 2, 3):
        p = os.path.join(ATLAS, f"mu_part{k}.parquet"); have = [c for c in cells if c in pq.read_schema(p).names]
        if not have: continue
        t = pq.read_table(p, columns=have)
        for c in have:
            u = t.column(c).to_numpy(); v = u.astype(np.float32) / 1e4; v[u == NA] = np.nan; out[c] = v
    return out

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
    look = _lookup(nside)
    cm = ListedColormap(np.loadtxt(os.path.join(HERE, "Planck_Parchment_RGB.txt")) / 255.0)
    def rgb(m):
        v = m[look].reshape(HT, W); x = (cm(np.clip(np.nan_to_num(v, nan=0), 0, 1))[..., :3] * 255).astype(np.uint8); x[~np.isfinite(v)] = (42, 45, 51); return x
    dv = dif_s[look].reshape(HT, W)
    code = np.where(np.isfinite(dv), np.clip(np.round(128 + np.clip(dv, -0.5, 0.5) * 254), 1, 255), 0).astype(np.uint8)
    zcode, zabove = None, None
    if z is not None and len(z):
        z = pd.Series(z, dtype="float64"); med = float(z.median()); mad = float((z - med).abs().median()) * 1.4826 or float(z.std()) or 1.0
        z = (z - med) / mad   # in the sample's own spread at these sites, exactly as the report's overlay (report_figs_v4.changes_sky_png)
        zz = z.reindex(ids[order]).to_numpy(); zs = _to_sky(zz, npix); zv = zs[look].reshape(HT, W)
        # 6,000 sites hold about one texture pixel each, too small to see: each is drawn as a dot of radius R_DOT pixels at its own place, the
        # larger departure winning where two dots meet, so a run of neighboring departing sites shows as a patch. Positions are unchanged.
        from scipy import ndimage
        fp = np.hypot(*np.mgrid[-R_DOT:R_DOT + 1, -R_DOT:R_DOT + 1]) <= R_DOT
        pos = ndimage.maximum_filter(np.where(zv > 0, zv, -np.inf), footprint=fp, mode="wrap")
        neg = ndimage.maximum_filter(np.where(zv < 0, -zv, -np.inf), footprint=fp, mode="wrap")
        has = ndimage.maximum_filter(np.isfinite(zv).astype(np.uint8), footprint=fp, mode="wrap") > 0
        zv = np.where(has, np.where(pos >= neg, np.maximum(pos, 0), -neg), np.nan); zv[has & ~np.isfinite(zv)] = 0.0
        zcode = np.where(np.isfinite(zv), np.clip(np.round(128 + np.clip(zv / 10, -0.5, 0.5) * 254), 1, 255), 0).astype(np.uint8)
        zm = np.abs(zs[np.isfinite(zs)]); zabove = [round(float((zm > t / 10).mean()), 4) for t in range(51)]
    pcode = prgb = pabove = None; pe = purified(fractions, specimen)
    if pe is not None:
        pv = pe.reindex(ids[order]).to_numpy(); sv = pd.Series(beta, dtype="float64").reindex(ids[order]).to_numpy()
        ps = _to_sky(pv, npix); pdif = _to_sky(np.where(np.isfinite(pv) & np.isfinite(sv), sv - pv, np.nan), npix)
        pt = _dots(ps[look].reshape(HT, W)); prgb = (cm(np.clip(np.nan_to_num(pt, nan=0), 0, 1))[..., :3] * 255).astype(np.uint8); prgb[~np.isfinite(pt)] = (42, 45, 51)
        pdv = _dots(pdif[look].reshape(HT, W))
        pcode = np.where(np.isfinite(pdv), np.clip(np.round(128 + np.clip(pdv, -0.5, 0.5) * 254), 1, 255), 0).astype(np.uint8)
        pm = np.abs(pdif[np.isfinite(pdif)]); pabove = [round(float((pm > t / 100).mean()), 4) for t in range(51)]
    cellmaps = []
    for c, wt in sorted((shares or {}).items(), key=lambda x: -(x[1] or 0)):
        if c in PANEL and wt is not None:
            mc = _mu([PANEL[c]])[PANEL[c]]; cellmaps.append(dict(name=f"{c}: {100 * wt:.0f} % of this sample", t=_jpg_b64(rgb(_to_sky(mc[order], npix)))))
    m = np.abs(dif_s[np.isfinite(dif_s)]); steps = np.round(np.arange(0, 0.51, 0.01), 2)
    above = [round(float((m > t).mean()), 4) if len(m) else None for t in steps]   # share of measured sky pixels above each size (equal-area)
    rec = {"drawn": True, "reference": name, "weights": {c: round(v, 4) for c, v in w.items()}, "sites_compared": int(both.sum()),
           "pixels_compared": int(len(m)), "median_abs_difference": round(float(np.median(m)), 4) if len(m) else None,
           "share_above": {"0.05": above[5], "0.10": above[10], "0.20": above[20], "0.30": above[30]},
           "layout": "IAM-Atlas v2.1 genome-order sky, NSIDE 256 NESTED (sky_order_v21.npz)", "scored": False}
    J = lambda b: "data:image/jpeg;base64," + b; P_ = lambda b: "data:image/png;base64," + b
    R_ = J(_jpg_b64(rgb(ref_s)))
    L = [dict(name="difference from the healthy reference", kind="diff", src=P_(_png_b64(code, "L")), base=R_, scale=1, above=above, start=10,
              note=f"Against {name}, at {int(both.sum()):,} sites. Red: more methylation than healthy; blue: less; grey: within the chosen size (the reference's own shade)."),
         dict(name="the sample", kind="image", src=J(_jpg_b64(rgb(smp_s))), note="The sample's methylation at every atlas site, 0 to 1."),
         dict(name="the healthy reference", kind="image", src=R_, note=f"{name}, 0 to 1.")]
    if zcode is not None:
        L.append(dict(name="departures at Met-A's 6,000 sites", kind="diff", src=P_(_png_b64(zcode, "L")), scale=10, unit="spreads", above=zabove, start=20,
                      share_text="of the pixels holding Met-A's sites depart by more than this", note="The C-Score's departure map, in the sample's own spread, as in the report's overlay."))
    if pcode is not None:
        Pr = J(_jpg_b64(prgb))
        L += [dict(name="Met-A's purified-array reference", kind="image", src=Pr, note="Met-A's own healthy reference at its 6,000 sites, weighted by this blood's Stage A fractions."),
              dict(name="difference from Met-A's purified reference", kind="diff", src=P_(_png_b64(pcode, "L")), base=Pr, scale=1, above=pabove, start=5,
                   share_text="of Met-A's sites differ by more than this", note="Against the reference Met-A and the C-Score read against.")]
    groups = [dict(name="this sample against its references", layers=L)]
    if cellmaps: groups.append(dict(name="the cells the IAM-Deconvolver read", layers=[dict(name="healthy " + c["name"], kind="image", src=J(c["t"]),
                                     note="This cell's healthy map in IAM-Atlas v2.1; the share is the IAM-Deconvolver's reading of this sample.") for c in cellmaps]))
    rec["layers"] = [x["name"] for g in groups for x in g["layers"]]
    if zabove: rec["departure_share_above"] = {"2 spreads": zabove[20], "3 spreads": zabove[30]}
    spec = dedupe(dict(id="sample", name="This sample", about=ABOUT.replace("__REF__", name).replace("__N__", f"{rec['sites_compared']:,}"), groups=groups))
    rec["fragment"] = fragment([spec])
    if out_html: open(out_html, "w", encoding="utf-8").write(page([spec], "The sample on the globe")); rec["html"] = os.path.abspath(out_html)
    rec["prose"] = " ".join([spec["about"]] + [g["name"] for g in groups] + [x.get(k, "") for g in groups for x in g["layers"] for k in ("name", "note", "share_text")])
    rec["spec"] = spec
    return rec

ABOUT = ("Every site of IAM-Atlas v2.1 has one fixed place on this globe, the same as on the atlas's cell pages. Drag to turn it. The healthy "
         "reference is <b>__REF__</b>; the difference is shown at __N__ sites. Choose the smallest difference to show: everything smaller fades into the "
         "reference's own grey, so only the larger differences stay lit. A picture: nothing is scored from it; Met-A and the C-Score read against "
         "their own commissioned reference, which is also a layer. The atlas blends 21 sources and sits slightly closer to 0.5 than one array, so a "
         "healthy sample shows a faint even difference. In a woman one X chromosome is held methylated, so her X shows as a patch against a reference "
         "that pools both sexes. A small loss spread over many sites shows on the departures layer, not the whole genome.")

