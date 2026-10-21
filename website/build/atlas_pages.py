"""IAM-Atlas v2.1 cell pages for the site (2026-10-18, author: every cell on its own page). One gallery page (atlas/index.html) and one
page per cell (atlas/<slug>.html): the cell's healthy map beside how sure the atlas is, with its numbers, read from
website/art/out/cells/index.json (written by website/art/make_atlas_sky_v21.py). Pictures and numbers only; nothing is fitted."""
import os, json, html, csv, collections
E = html.escape
# 2026-10-18 (author: "Why is this colon cell so unsure?? ... it doesnt match what the words say"): the maps matched the atlas, but the page never
# said what sets a cell's certainty: how many healthy samples the atlas has of it (rank correlation -0.82 over the 79 cells). Each page now
# states its samples, studies and platforms, read at build time from the atlas's own sample list.
SAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "Biological_Physics", "MethylPhys", "doors", "data", "DEV_ATLAS_V21_01", "final_samples_v21.csv")
PLAT = {"WGBS": "whole-genome bisulfite sequencing", "WGBS-pooled": "pooled whole-genome bisulfite sequencing", "array": "methylation arrays"}
def built_from():
    n, src, pl = collections.Counter(), collections.defaultdict(set), collections.defaultdict(set)
    for r in csv.DictReader(open(SAMPLES, encoding="utf-8")):
        c = r["cell"].lower(); n[c] += 1; src[c].add(r["source"]); pl[c].add(PLAT.get(r["platform"], r["platform"]))
    return {c: (n[c], len(src[c]), " and ".join(sorted(pl[c]))) for c in n}
HOW = ("Every site of the atlas keeps one place on a HEALPix sphere, the pixel grid the Planck and WMAP missions used to map the microwave sky, "
       "set by where the site sits in the genome (chromosome 1 through Y). The layout is the same for every cell, so any two cells, and a person's "
       "sample, can be compared site for site.")

SPHERE_JS = r"""
const C = JSON.parse(document.getElementById('cells').textContent), cv = document.getElementById('globe'), cx = cv.getContext('2d');
const sel = document.getElementById('cell'), mode = document.getElementById('mode'), note = document.getElementById('note');
let yaw = 0, pitch = 0.35, spin = true, tex = null, TW = 0, TH = 0, drag = null;
for (const c of C) { const o = document.createElement('option'); o.value = c.slug; o.textContent = c.name; sel.appendChild(o); }
const q = new URLSearchParams(location.search).get('cell'); sel.value = C.some(c => c.slug === q) ? q : 'neutrophils';
function load() {
  const im = new Image(); im.onload = () => { const o = document.createElement('canvas'); o.width = TW = im.width; o.height = TH = im.height;
    const g = o.getContext('2d'); g.drawImage(im, 0, 0); tex = g.getImageData(0, 0, TW, TH).data; draw(); };
  im.src = ART + 'sphere/' + sel.value + (mode.value === 'u' ? '_u.jpg' : '_v.jpg');
  const c = C.find(c => c.slug === sel.value); note.textContent = c.name + ': ' + c.pixels_measured.toLocaleString() + ' of ' + PIX.toLocaleString() + ' pixels hold a measured site.';
}
function draw() {
  if (!tex) return; const N = cv.width, img = cx.createImageData(N, N), d = img.data, ca = Math.cos(yaw), sa = Math.sin(yaw), cb = Math.cos(pitch), sb = Math.sin(pitch);
  for (let py = 0; py < N; py++) for (let px = 0; px < N; px++) {
    const sx = 2 * (px + 0.5) / N - 1, sy = 1 - 2 * (py + 0.5) / N, r2 = sx * sx + sy * sy, k = 4 * (py * N + px);
    if (r2 > 1) { d[k + 3] = 0; continue; }
    const x = Math.sqrt(1 - r2), y = sx, z = sy;                       // viewer on +x; screen right = +y, up = +z
    const x1 = x * cb - z * sb, z1 = x * sb + z * cb;                   // tilt
    const x2 = x1 * ca - y * sa, y2 = x1 * sa + y * ca;                 // turn about the poles
    const th = Math.acos(Math.max(-1, Math.min(1, z1))); let ph = Math.atan2(y2, x2); if (ph < 0) ph += 2 * Math.PI;
    const i = Math.min(TH - 1, Math.floor(th / Math.PI * TH)), j = Math.min(TW - 1, Math.floor(ph / (2 * Math.PI) * TW)), t = 4 * (i * TW + j);
    const sh = 0.55 + 0.45 * x;                                         // light from the viewer, so the curve reads
    d[k] = tex[t] * sh; d[k + 1] = tex[t + 1] * sh; d[k + 2] = tex[t + 2] * sh; d[k + 3] = 255;
  }
  cx.putImageData(img, 0, 0);
}
cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY, yaw, pitch]; spin = false; cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointermove', e => { if (!drag) return; yaw = drag[2] - (e.clientX - drag[0]) * 0.01;
  pitch = Math.max(-1.5, Math.min(1.5, drag[3] + (e.clientY - drag[1]) * 0.01)); draw(); });
cv.addEventListener('pointerup', () => drag = null);
document.getElementById('spin').onclick = () => { spin = !spin; };
sel.onchange = load; mode.onchange = load; load();
(function tick() { if (spin) { yaw += 0.006; draw(); } requestAnimationFrame(tick); })();
"""

def write_sphere(out, art_rel, css_tag, style, back, names):
    """atlas/sphere.html: every cell's map wrapped on a globe the reader can turn (to-do step 13). Written only when the textures exist."""
    f = os.path.join(os.path.dirname(__file__), "..", "art", "out", "sphere", "sphere.json")
    if not os.path.exists(f): return 0
    S = json.load(open(f)); cells = [dict(slug=c["slug"], name=names.get(c["slug"], c["cell"]), pixels_measured=c["pixels_measured"]) for c in S["cells"]]
    cells.sort(key=lambda c: c["name"].lower())
    import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rgb = np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "art", "data", "Planck_Parchment_RGB.txt"))
    pv = ",".join("rgb(%d,%d,%d)" % tuple(rgb[int(round(i * (len(rgb) - 1) / 10))]) for i in range(11))
    pu = ",".join("rgb(%d,%d,%d)" % tuple(int(255 * x) for x in plt.get_cmap("magma")(i / 10)[:3]) for i in range(11))
    d = os.path.join(out, "atlas"); os.makedirs(d, exist_ok=True)
    html_ = (f'<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
             f'<title>IAM-Atlas v2.1 on the globe</title>{css_tag}<style>{style}#globe{{width:min(560px,92vw);height:min(560px,92vw);touch-action:none;cursor:grab;display:block;margin:12px auto}}'
             f'.bar{{height:12px;border-radius:3px;max-width:560px;margin:4px auto}}.ctl{{display:flex;gap:12px;flex-wrap:wrap;justify-content:center;align-items:center}}'
             f'.lab{{display:flex;justify-content:space-between;max-width:560px;margin:0 auto;font-size:13px}}</style></head><body><section class="wrap">'
             f'<p><a href="index.html">&larr; all {len(cells)} cells</a> &middot; <a href="{back}">front page</a></p><h1>IAM-Atlas v2.1 on the globe</h1>'
             f'<p>{E(HOW)} Here the same map is wrapped on a globe. Drag it to turn it; choose a cell and whether to see its methylation or how sure '
             f'the atlas is of it. Grey pixels hold no measured site.</p>'
             f'<div class="ctl"><select id="cell" aria-label="cell"></select><select id="mode" aria-label="map"><option value="v">methylation</option>'
             f'<option value="u">how sure the atlas is</option></select><button id="spin" type="button">turn / stop</button></div>'
             f'<canvas id="globe" width="560" height="560" aria-label="the selected cell on a globe"></canvas><p id="note" style="text-align:center"></p>'
             f'<div class="bar" style="background:linear-gradient(90deg,{pv})"></div><div class="lab"><span>methylation 0 (never tagged)</span><span>1 (always tagged)</span></div>'
             f'<div class="bar" style="background:linear-gradient(90deg,{pu})"></div><div class="lab"><span>certain</span><span>&plusmn;0.15 or wider</span></div>'
             f'<p>The layout is HEALPix NSIDE {S["nside"]}, {S["order"]} order ({S["pixels"]:,} pixels) for {S["sites"]:,} sites; each pixel shows the mean of the sites it '
             f'holds. The globe is drawn in the browser from one picture per cell, made by <code>website/art/make_sphere_3d.py</code>. A picture of the atlas: '
             f'nothing is read or scored from it.</p>'
             f'<script id="cells" type="application/json">{json.dumps(cells)}</script><script>const ART = {json.dumps(art_rel)}, PIX = {S["pixels"]};' + SPHERE_JS
             + '</script></section></body></html>')
    open(os.path.join(d, "sphere.html"), "w", encoding="utf-8").write(html_); return len(cells)

def write(out, art_rel, css_tag, style, back):
    I = json.load(open(os.path.join(os.path.dirname(__file__), "..", "art", "out", "cells", "index.json")))["cells"]
    I = sorted(I, key=lambda c: c["name"].lower()); d = os.path.join(out, "atlas"); os.makedirs(d, exist_ok=True)
    BF = built_from(); missing = [c["cell"] for c in I if c["cell"].lower() not in BF]
    assert not missing, "atlas_pages: cells with no samples in the atlas sample list: " + ", ".join(missing)
    ref = next(c for c in I if c["cell"].lower() == "neutrophils"); rn, rk, _ = BF["neutrophils"]
    globe = write_sphere(out, art_rel, css_tag, style, back, {c["slug"]: c["name"] for c in I}) > 0
    def why(c):
        n, k, pl = BF[c["cell"].lower()]
        return ("<tr><th>Built from</th><td>" + f"{n} healthy sample{'s' if n != 1 else ''} from {k} stud{'ies' if k != 1 else 'y'} ({E(pl)})</td></tr>",
                "<p>How sure the atlas can be of a cell depends mostly on how many healthy samples of it the atlas holds. Neutrophils, built from "
                + f"{rn} samples from {rk} studies, have half their sites within &plusmn;{ref['halfwidth_median']:.3f} and read nearly black; a cell built from "
                + "two or three samples of one study reads paler, because a few samples pin each value down less tightly. More healthy samples of this "
                + "cell would make its map darker.</p>")
    head = lambda t: (f'<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                      f'<title>{E(t)}</title>{css_tag}<style>{style}.cg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px}}'
                      f'.cg a{{text-decoration:none;color:inherit}}.cg img{{width:100%;border-radius:4px}}.cg span{{font-size:14px}}</style></head><body><section class="wrap">')
    g = [f'<a href="{c["slug"]}.html"><img loading="lazy" src="{art_rel}cells/{c["slug"]}_thumb.jpg" alt="{E(c["name"])} on the sphere"><span>{E(c["name"])}</span></a>' for c in I]
    open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(head("IAM-Atlas v2.1: every cell on the sky")
        + f'<p><a href="{back}">&larr; IAMPerformance front page</a></p><h1>IAM-Atlas v2.1: all {len(I)} healthy cell types on the sky</h1><p>{E(HOW)} '
        + 'Choose a cell to see its map beside how sure the atlas is at every site.' + (' Or <a href="sphere.html">turn any cell on the globe</a>.' if globe else '')
        + '</p><div class="cg">' + "".join(g) + "</div></section></body></html>")
    for k, c in enumerate(I):
        p, n = I[k - 1], I[(k + 1) % len(I)]
        body = (f'<p><a href="index.html">&larr; all {len(I)} cells</a> &middot; <a href="{back}">front page</a></p><h1>{E(c["name"])}</h1>'
                f'<figure><img src="{art_rel}cells/{c["slug"]}.jpg" alt="{E(c["name"])}: value and uncertainty on the sphere" style="width:100%">'
                f'<figcaption>Left: healthy {E(c["name"].lower())} from IAM-Atlas v2.1, every site in its genome place, from never tagged (blue) to always tagged (red). '
                f'Right: how sure the atlas is of each value. Dark means certain (0 would mean the value is known exactly); the brighter a point, the wider its 95&nbsp;% range. Every cell&rsquo;s map uses the same scale, 0 to &plusmn;0.15, so cells can be compared with each other.</figcaption></figure>'
                f'<table><tr><th>Average methylation</th><td>{c["mean_methylation"]:.3f}</td></tr>'
                f'<tr><th>How sure (half-width of the 95&nbsp;% range)</th><td>half the sites within &plusmn;{c["halfwidth_median"]:.3f}; nine in ten within &plusmn;{c["halfwidth_p90"]:.3f}</td></tr>'
                f'<tr><th>Pixels with a measured site</th><td>{c["pixels_measured"]:,} of {c["pixels"]:,}</td></tr>' + why(c)[0] + '</table>' + why(c)[1] +
                f'<p>{E(HOW)}</p>' + (f'<p><a href="sphere.html?cell={c["slug"]}">Turn this cell on the globe</a></p>' if globe else '') + f'<p><a href="{p["slug"]}.html">&larr; {E(p["name"])}</a> &middot; <a href="{n["slug"]}.html">{E(n["name"])} &rarr;</a></p>')
        open(os.path.join(d, c["slug"] + ".html"), "w", encoding="utf-8").write(head(f'{c["name"]}: IAM-Atlas v2.1 on the sky') + body + "</section></body></html>")
    return len(I)
