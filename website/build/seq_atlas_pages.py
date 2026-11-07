"""The IAM Sequencing Atlas's cell pages for the site: one gallery (seq-atlas/index.html) and one page per cell (seq-atlas/<slug>.html), the
cell's healthy map at every CpG beside how sure the atlas is, with its numbers. Every number is read at build time from a committed file:
the maps and their numbers from the chain's own sphere build (website/art/out/seq_cells/index.json, written by run_sample.py --build
seq-atlas-sphere), the donors from the atlas's input list and donor traits, each donor's library floor from input_floors_v1.parquet, and
the cell's healthy readings from the 57-cell healthy pass. Pictures and numbers only; nothing is fitted.
Usage (the site build calls write()); python3 website/build/seq_atlas_pages.py --check   (exit 1 naming any cell without its maps or page data)"""
import os, sys, json, html, csv, collections, statistics
E = html.escape
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MP = os.path.join(REPO, "Biological_Physics", "MethylPhys"); SD = os.path.join(MP, "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas")
ART = os.path.join(REPO, "website", "art", "out", "seq_cells")
READINGS = os.path.join(MP, "doors", "data", "DEV_METAA_ATLAS_CURVE_01", "seq_atlas_01", "healthy_fullchain7_readings.csv")
FLOORS = os.path.join(MP, "chain", "Runtime Matrices", "Seq_Atlas", "input_floors_v1.parquet")
SEXW = {"F": "female", "M": "male", "not recorded": "with sex not recorded"}
SHOW = (("Met-A sorted cell", "Met-A, read as this cell"), ("territory islands", "the islands"), ("territory domains", "the domains"), ("C-Score whole genome", "the C-Score"))
PHYS = ("Why a map like this can be a ruler: a methyl mark is a record a cell keeps against heat, at a price Landauer's bound sets, so the "
        "reading scale's ends come from physics (a mark held at the holding energy, and a mark lost entirely, one bit of disorder). This map "
        "is where healthy cells of this kind sit between those ends at every CpG, counted from healthy donors' reads with nothing fitted. A "
        "sequenced sample of this cell is read against it site by site, at the sample's own number of reads.")
HOW = ("The atlas holds every CpG; its globe is drawn on IAM-Atlas v2.1's layout of 813,951 sites, each keeping one place on a HEALPix sphere, the pixel grid the Planck and WMAP missions used to map the microwave sky, "
       "set by where the site sits in the genome (chromosome 1 through Y). The layout is the same for every cell, so any two cells, and a person's "
       "sample, can be compared site for site.")

def data():
    I = json.load(open(os.path.join(ART, "index.json"), encoding="utf-8"))
    don = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(SD, "seq_atlas_inputs.csv"), encoding="utf-8")): don[r["cell"].strip()].append((r["gsm"].strip(), r["half"].strip().upper()))
    sex = {r["gsm"].strip(): (r.get("sex") or "").strip() for r in csv.DictReader(open(os.path.join(SD, "donor_traits.csv"), encoding="utf-8"))}
    import pyarrow.parquet as pq
    T = pq.read_table(FLOORS).to_pydict(); floor = dict(zip(T["gsm"], T["floor"]))
    rd = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(open(READINGS, encoding="utf-8")):
        if r["A"] not in ("", "nan"): rd[r["cell"].strip()][r["reading"]][r["gsm"]] = float(r["A"])
    return I, don, sex, floor, rd

def problems():
    I, don, sex, floor, rd = data(); P = []
    names = {c["cell"] for c in I["cells"]}
    for c in sorted(don):
        if c not in names: P.append(f"{c}: in the atlas's inputs but not in the sphere build")
    for c in I["cells"]:
        for k in ("_v.jpg", "_u.jpg", "_thumb.jpg"):
            if not os.path.exists(os.path.join(ART, c["slug"] + k)): P.append(f"{c['cell']}: no {c['slug'] + k}")
        if c["cell"] not in don: P.append(f"{c['cell']}: in the sphere build but not in the atlas's inputs")
        if c.get("mean_methylation") is None: P.append(f"{c['cell']}: no numbers in the sphere build")
        if not rd.get(c["cell"]) and len({h for _, h in don.get(c["cell"], [])}) > 1: P.append(f"{c['cell']}: no healthy readings in the 57-cell pass")
        if rd.get(c["cell"]) and len({h for _, h in don.get(c["cell"], [])}) == 1: P.append(f"{c['cell']}: read in the healthy pass though its donors sit in one half")
    return P

NZD = os.path.join(os.path.dirname(ART), "seq_noise")
def noise_fig(c, art_rel):
    """2026-11-06 (plan 61.4): the cell's two noise maps, drawn by the chain (run_sample.py --build seq-noise-sphere), with what their null test found
    for this cell (noise_null.json, run_sample.py --build seq-noise-null); a map that does not exist (a cell from one donor has no donor halves) is left out."""
    SD = os.path.join(REPO, "Biological_Physics", "MethylPhys", "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas")
    NN = {(r["cell"], r["map"][:4]): r for r in json.load(open(os.path.join(SD, "noise_null.json")))["rows"]}
    have = [k for k in (1, 2) if os.path.exists(os.path.join(NZD, f'{c["slug"]}_n{k}.jpg'))]
    if not have: return ""
    def said(k):
        r = NN.get((c["cell"], f"map{k}"))
        if r is None: return ""
        n = len(r["bands_above_null"])
        return (f" Against its shuffled null: {n} of {len(r['ratio'])} bands above it" + (", where differences between healthy people show." if k == 1 and n > 1 else
                (", as pure counting noise should." if k == 2 and n <= 1 else ".")))
    cap = {1: "donor halves: the cell's donors split into two sealed halves, each counted alone, and the difference drawn; it shows how much healthy people of this cell differ.",
           2: "read halves: each donor's reads split at random into two halves, counted alone, and the difference drawn; it shows the counting noise alone."}
    imgs = "".join(f'<img src="{art_rel}seq_noise/{c["slug"]}_n{k}.jpg" alt="{E(c["cell"])}: noise map, {cap[k].split(":")[0]}">' for k in have)
    txt = " ".join(f"{'Left' if i == 0 and len(have) == 2 else ('Right' if len(have) == 2 else 'The map')}, {cap[k]}{said(k)}" for i, k in enumerate(have))
    return (f'<figure><div class="two">{imgs}</div><figcaption>Its noise maps, on the same layout (difference from &minus;0.3 to +0.3). {E(txt)}'
            + ("" if 1 in have else " There is no donor-halves map: its donors do not fill two sealed halves, so there is no second half to compare.") + "</figcaption></figure>")

def write(out, art_rel, css_tag, style, back):
    I, don, sex, floor, rd = data(); C = sorted(I["cells"], key=lambda c: c["cell"].lower())
    d = os.path.join(out, "seq-atlas"); os.makedirs(d, exist_ok=True); v21 = set()
    _ig = os.path.join(REPO, "Biological_Physics", "MethylPhys", "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", "globe", "seq_globe.html")   # 2026-11-05: its information globe, as the one command drew it
    if os.path.exists(_ig): open(os.path.join(d, "information_globe.html"), "w", encoding="utf-8").write(   # the old address now opens the sphere page on this atlas
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../atlas/information_sphere_sequencing.html"><a href="../atlas/information_sphere_sequencing.html">The information sphere</a>')
    v21i = os.path.join(REPO, "website", "art", "out", "cells", "index.json")
    if os.path.exists(v21i): v21 = {c["slug"] for c in json.load(open(v21i, encoding="utf-8"))["cells"]}
    _UP = {"cd4": "CD4", "cd8": "CD8", "cd14": "CD14", "cd34": "CD34", "nk": "NK", "t": "T", "b": "B", "dc": "DC", "hspc": "HSPC", "mep": "MEP"}   # cell names as written in the field
    _FIX = {"fallopien tubes epithelium": "Fallopian tube epithelium"}   # display only; the atlas keeps its source's label
    nm = lambda c: _FIX.get(c["cell"].lower()) or " ".join(_UP.get(w, w) if (w in _UP and (i or w not in ("b", "t"))) or w in ("cd4", "cd8", "cd14", "cd34", "nk") else (w[:1].upper() + w[1:] if i == 0 else w) for i, w in enumerate(c["cell"].split(" ")))
    lownm = lambda c: (lambda x: x if (x[:2].isupper() or x.split(' ')[0] in ('B', 'T', 'NK')) else x[:1].lower() + x[1:])(nm(c))   # 2026-11-06: the name inside a sentence
    head = lambda t: (f'<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                      f'<title>{E(t)}</title>{css_tag}<style>{style}.cg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px}}'
                      f'.cg a{{text-decoration:none;color:inherit}}.cg img{{width:100%;border-radius:4px}}.cg span{{font-size:14px}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}.two img{{width:100%}}</style></head><body><section class="wrap">')
    g = [f'<a href="{c["slug"]}.html"><img loading="lazy" src="{art_rel}seq_cells/{c["slug"]}_thumb.jpg" alt="{E(nm(c))} on the sphere"><span>{E(nm(c))}</span></a>' for c in C]
    open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(head("The IAM Sequencing Atlas: every cell at every CpG")
        + f'<p><a href="{back}">&larr; IAMPerformance front page</a> &middot; <a href="../atlas/index.html">IAM-Atlas v2.1 (arrays)</a> &middot; '
        + '<a href="../atlas/sphere.html?globe=seq-atlas">Sequencing Atlas Globe</a> &middot; <a href="globe.html">The globe at every CpG, part by part</a>' + (' &middot; <a href="../atlas/information_sphere_sequencing.html">Sequencing Information Sphere</a>' if os.path.exists(_ig) else '') + '</p>'
        + f'<h1>The IAM Sequencing Atlas: {len(C)} healthy cell types at every CpG</h1><p><b>In development.</b> The ruler for sequenced samples, measured with '
        + "sequencing's own chemistry; IAM-Atlas v2.1 stays the ruler for arrays. " + E(PHYS) + '</p><p>The whole atlas, every cell at every CpG with its runtime tables and a checksum for every piece, is on Zenodo: <a href="https://doi.org/10.5281/zenodo.24029328">doi.org/10.5281/zenodo.24029328</a>.</p><p>' + E(HOW) + ' Choose a cell.</p><div class="cg">' + "".join(g) + "</div></section></body></html>")
    for k, c in enumerate(C):
        p, n = C[k - 1], C[(k + 1) % len(C)]; dn = don.get(c["cell"], [])
        halves = collections.Counter(h for _, h in dn); sx = collections.Counter(sex.get(g_, "") or "not recorded" for g_, _ in dn)
        fl = [floor[g_] for g_, _ in dn if g_ in floor]
        rows = [f'<tr><th>Built from</th><td>{len(dn)} healthy donor{"s" if len(dn) != 1 else ""}, ' + (f'sealed into two halves ({halves.get("A", 0)} in half A, {halves.get("B", 0)} in half B); ' if len(halves) > 1 else f'all in sealed half {next(iter(halves), "")}; ')
                + ", ".join(f"{v} {E(SEXW.get(s, s))}" for s, v in sorted(sx.items())) + "</td></tr>",
                f'<tr><th>Average methylation</th><td>{c["mean_methylation"]:.3f}</td></tr>',
                f'<tr><th>How sure (half-width of the 95&nbsp;% range of the average)</th><td>half the sites within &plusmn;{c["halfwidth_median"]:.3f}; nine in ten within &plusmn;{c["halfwidth_p90"]:.3f}</td></tr>',
                f'<tr><th>Pixels with a measured site</th><td>{c["pixels_measured"]:,} of {c["pixels"]:,}</td></tr>']
        if fl: rows.append(f'<tr><th>Its donors&rsquo; library floor</th><td>{statistics.median(fl):.4f} (false &ldquo;methylated&rdquo; calls at the sites no healthy cell marks; a property of each library&rsquo;s chemistry, measured by the chain)</td></tr>')
        R = rd.get(c["cell"], {})
        for key, lab in SHOW:
            v = list(R.get(key, {}).values())
            if v: rows.append(f'<tr><th>Its own healthy inputs read ({E(lab)})</th><td>median {statistics.median(v):.4f} over {len(v)} (healthy reads 1)</td></tr>')
        if not R: rows.append('<tr><th>Its own healthy inputs read</th><td>not read in the healthy pass: all its donors sit in one sealed half, so there is no other half of the same cell to read them against</td></tr>')
        body = (f'<p><a href="index.html">&larr; all {len(C)} cells</a> &middot; <a href="{back}">front page</a>'
                + f' &middot; <a href="../atlas/sphere.html?globe=seq-atlas&amp;cell={c["slug"]}">this cell on the Sequencing Atlas Globe</a>' + (f' &middot; <a href="../atlas/{c["slug"]}.html">this cell in IAM-Atlas v2.1</a>' if c["slug"] in v21 else '') + f'</p><h1>{E(nm(c))}</h1>'
                f'<p><b>In development.</b> A cell is commissioned when healthy samples from laboratories the atlas has never held read Normal against it.</p>'
                f'<figure><div class="two"><img src="{art_rel}seq_cells/{c["slug"]}_v.jpg" alt="{E(nm(c))}: healthy level on the sphere"><img src="{art_rel}seq_cells/{c["slug"]}_u.jpg" alt="{E(nm(c))}: how sure the atlas is"></div>'
                f'<figcaption>Left: healthy {E(lownm(c))} at each of the 813,951 layout sites in its genome place, from never tagged to always tagged. Right: how sure the atlas is of each value, '
                f'dark certain, brighter a wider 95&nbsp;% range of the average (0 to &plusmn;0.15, the same scale for every cell). Grey: no reads.</figcaption></figure>'
                + noise_fig(c, art_rel) +
                f'<table>{"".join(rows)}</table><p>{E(PHYS)}</p><p>{E(HOW)}</p>'
                f'<p><a href="{p["slug"]}.html">&larr; {E(nm(p))}</a> &middot; <a href="{n["slug"]}.html">{E(nm(n))} &rarr;</a></p>')
        open(os.path.join(d, c["slug"] + ".html"), "w", encoding="utf-8").write(head(f'{nm(c)}: the IAM Sequencing Atlas') + body + "</section></body></html>")
    return len(C)

if __name__ == "__main__":
    P = problems()
    if P: print("seq_atlas_pages: " + "; ".join(P[:20]) + (f" (+{len(P) - 20} more)" if len(P) > 20 else "")); sys.exit(1)
    print("seq_atlas_pages: ok"); sys.exit(0)
