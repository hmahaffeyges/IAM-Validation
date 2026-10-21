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
Sex: in women one X chromosome is held methylated, so a woman's X differs from a reference that pools both sexes (seen on the first real
blood, 2026-10-21: the last 2 % of the genome order, where X sits, differs 7 to 11 times as often as the rest). The page names it.
Known offset (DEV-REFERENCES-01, ATL-070): the atlas blends 21 sources and sits about 0.006 closer to 0.5 than a single array at most sites,
so a healthy array shows a faint even difference everywhere; that is the reference's make, not biology.
build(beta, specimen, shares, out_html) -> record; the page fragment is written to out_html and also returned in the record."""
import os, io, json, base64, numpy as np, pandas as pd

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

def build(beta, specimen, shares, out_html=None, z=None):
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
    m = np.abs(dif_s[np.isfinite(dif_s)]); steps = np.round(np.arange(0, 0.51, 0.01), 2)
    above = [round(float((m > t).mean()), 4) if len(m) else None for t in steps]   # share of measured sky pixels above each size (equal-area)
    rec = {"drawn": True, "reference": name, "weights": {c: round(v, 4) for c, v in w.items()}, "sites_compared": int(both.sum()),
           "pixels_compared": int(len(m)), "median_abs_difference": round(float(np.median(m)), 4) if len(m) else None,
           "share_above": {"0.05": above[5], "0.10": above[10], "0.20": above[20], "0.30": above[30]},
           "layout": "IAM-Atlas v2.1 genome-order sky, NSIDE 256 NESTED (sky_order_v21.npz)", "scored": False}
    data = dict(s=_jpg_b64(rgb(smp_s)), r=_jpg_b64(rgb(ref_s)), d=_png_b64(code, "L"), above=above, ref=name,
                z=_png_b64(zcode, "L") if zcode is not None else None, zabove=zabove)
    if zabove: rec["departure_share_above"] = {"2 spreads": zabove[20], "3 spreads": zabove[30]}
    html = PAGE.replace("__DATA__", json.dumps(data)).replace("__REF__", name).replace("__N__", f"{rec['sites_compared']:,}")
    if out_html: open(out_html, "w", encoding="utf-8").write(html); rec["html"] = os.path.abspath(out_html)
    rec["fragment"] = html
    return rec

PAGE = r"""<section id="sec-globe"><h2>The sample on the globe, against its healthy reference</h2>
<p>Every site of IAM-Atlas v2.1 has one fixed place on this globe, the same as on the atlas's cell pages. Drag to turn it. Three layers: the
sample, its healthy reference (<b>__REF__</b>), and their difference at __N__ sites. In the difference layer, choose the smallest difference to
show: everything smaller fades into the background, so only the larger differences stay lit, red where the sample holds more methylation than
healthy and blue where it holds less. A picture: nothing is scored from it. Met-A and the C-Score read against their own commissioned
reference. The atlas blends 21 sources and sits slightly closer to 0.5 than one array, so a healthy sample shows a faint even difference.
In a woman one X chromosome is held methylated, so her X shows as a patch of difference against a reference that pools both sexes. The last
layer shows only Met-A's 6,000 neutrophil identity sites, each as its departure in the sample's own spread (the C-Score's view, as in the overlay above): a small loss spread over
many sites is easier to see there than on the whole genome.</p>
<div style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start"><canvas id="gl" width="520" height="520" style="max-width:100%;touch-action:none;cursor:grab"></canvas>
<div style="min-width:240px"><p><label>Layer <select id="gl-layer"><option value="d">difference</option><option value="s">sample</option><option value="r">healthy reference</option><option value="z">departures at Met-A's sites</option></select></label></p>
<p><label>Show differences larger than <input id="gl-t" type="range" min="0" max="50" value="10"> <span id="gl-tv">0.10</span></label></p>
<p id="gl-share"></p><p><button id="gl-spin" type="button">Stop or start turning</button></p>
<p style="font-size:0.9em">Methylation 0 to 1 (sample and reference layers, the atlas pages' scale); grey: no measured site.</p></div></div>
<script>(function(){
const D = __DATA__, cv = document.getElementById('gl'), cx = cv.getContext('2d'), L = document.getElementById('gl-layer'), T = document.getElementById('gl-t');
let yaw = 0, pitch = 0.35, spin = true, drag = null; const tex = {}; let TW = 0, TH = 0;
function load(k, src) { return new Promise(r => { const im = new Image(); im.onload = () => { const o = document.createElement('canvas'); o.width = TW = im.width; o.height = TH = im.height;
  const g = o.getContext('2d'); g.drawImage(im, 0, 0); tex[k] = g.getImageData(0, 0, TW, TH).data; r(); }; im.src = src; }); }
function share() { const t = +T.value, zl = L.value === 'z';
  document.getElementById('gl-tv').textContent = zl ? (t / 10).toFixed(1) + ' spreads' : (t / 100).toFixed(2);
  const a = zl ? (D.zabove || [])[t] : D.above[t];
  document.getElementById('gl-share').textContent = a == null ? '' : zl ? (100 * a).toFixed(1) + " % of the pixels holding Met-A's sites depart by more than " + (t / 10).toFixed(1) + ' of the sample\'s own spreads.'
    : (100 * a).toFixed(1) + ' % of the measured globe differs from the reference by more than ' + (t / 100).toFixed(2) + '.'; }
function px(t, k) { const L_ = L.value;
  if (L_ === 's' || L_ === 'r') { const s = tex[L_]; return [s[t], s[t + 1], s[t + 2]]; }
  const c = (L_ === 'z' ? tex.z : tex.d)[t]; if (c === 0) return [42, 45, 51]; const v = (c - 128) / 254, a = Math.abs(v), th = L_ === 'z' ? +T.value / 500 : +T.value / 100;
  if (a <= th) return [58, 60, 66]; const f = Math.min(1, 0.35 + 0.65 * a / 0.5);
  return v > 0 ? [255 * f, 70 * (1 - f) + 40, 50] : [50, 90 * (1 - f) + 60, 255 * f]; }
function draw() { if (!tex.d) return; const N = cv.width, img = cx.createImageData(N, N), d = img.data, ca = Math.cos(yaw), sa = Math.sin(yaw), cb = Math.cos(pitch), sb = Math.sin(pitch);
  for (let py = 0; py < N; py++) for (let qx = 0; qx < N; qx++) { const sx = 2 * (qx + 0.5) / N - 1, sy = 1 - 2 * (py + 0.5) / N, r2 = sx * sx + sy * sy, k = 4 * (py * N + qx);
    if (r2 > 1) { d[k + 3] = 0; continue; } const x = Math.sqrt(1 - r2), y = sx, z = sy, x1 = x * cb - z * sb, z1 = x * sb + z * cb, x2 = x1 * ca - y * sa, y2 = x1 * sa + y * ca;
    const th = Math.acos(Math.max(-1, Math.min(1, z1))); let ph = Math.atan2(y2, x2); if (ph < 0) ph += 2 * Math.PI;
    const i = Math.min(TH - 1, Math.floor(th / Math.PI * TH)), j = Math.min(TW - 1, Math.floor(ph / (2 * Math.PI) * TW)), t = 4 * (i * TW + j), c = px(t), sh = 0.55 + 0.45 * x;
    d[k] = c[0] * sh; d[k + 1] = c[1] * sh; d[k + 2] = c[2] * sh; d[k + 3] = 255; }
  cx.putImageData(img, 0, 0); }
cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY, yaw, pitch]; spin = false; cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointermove', e => { if (!drag) return; yaw = drag[2] - (e.clientX - drag[0]) * 0.01; pitch = Math.max(-1.5, Math.min(1.5, drag[3] + (e.clientY - drag[1]) * 0.01)); draw(); });
cv.addEventListener('pointerup', () => drag = null); document.getElementById('gl-spin').onclick = () => { spin = !spin; };
if (!D.z) L.querySelector('option[value="z"]').remove(); L.onchange = () => { share(); draw(); }; T.oninput = () => { share(); draw(); }; share();
Promise.all([load('s', 'data:image/jpeg;base64,' + D.s), load('r', 'data:image/jpeg;base64,' + D.r), load('d', 'data:image/png;base64,' + D.d)].concat(D.z ? [load('z', 'data:image/png;base64,' + D.z)] : [])).then(draw);
(function tick() { if (spin) { yaw += 0.006; draw(); } requestAnimationFrame(tick); })();
})();</script></section>
"""
