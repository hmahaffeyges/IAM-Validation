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

# the globe viewer lives in chain/MethylPhys_Interface/globe_viewer.py (2026-10-21), one code for every globe

def write_sphere(out, art_rel, css_tag, style, back, names):
    """atlas/sphere.html: the Globe Explorer (2026-10-21, author: "the webpage to show ALL the versions"). Every globe is drawn by the chain's one
    viewer (chain/MethylPhys_Interface/globe_viewer.py), the same code as a report's globe; each globe's pictures load only when it is chosen.
    The first globe is IAM-Atlas v2.1 (every cell, its methylation or how sure the atlas is); then every spec in website/art/out/explorer/*.json
    (website/art/make_explorer.py: Met-A's purified-array reference, the example samples; noise maps when built). Written only when the atlas
    textures exist."""
    import glob, sys as _s
    H_ = os.path.dirname(os.path.abspath(__file__)); ART_ = os.path.join(H_, "..", "art", "out")
    f = os.path.join(ART_, "sphere", "sphere.json")
    if not os.path.exists(f): return 0
    _s.path.insert(0, os.path.join(H_, "..", "..", "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface")); import globe_viewer as V
    S = json.load(open(f)); cells = sorted(({"slug": c["slug"], "name": names.get(c["slug"], c["cell"])} for c in S["cells"]), key=lambda c: c["name"].lower())
    atlas = dict(id="atlas", name="Array Atlas Globe: IAM-Atlas v2.1, 79 healthy cell types", groups=[dict(id=c["slug"], name=c["name"], layers=[
        dict(name="methylation", kind="image", src=f"{art_rel}sphere/{c['slug']}_v.jpg", mask=f"{art_rel}sphere/{c['slug']}_vm.png", vscale=[0, 1], vword="at least this methylated:",
             hint="Slide right to keep only the sites at least this methylated; the rest go dark. Slide back to see every site.",
             note="The atlas's best value at every site, 0 (never tagged) to 1 (always tagged)."),
        dict(name="how sure the atlas is", kind="image", src=f"{art_rel}sphere/{c['slug']}_u.jpg", mask=f"{art_rel}sphere/{c['slug']}_um.png", vscale=[0, 0.15], vword="less sure than +/-",
             hint="Slide right to keep only the sites where the atlas is least sure; the sure ones go dark. Slide back to see every site.",
             note="Half-width of the 95 % range at every site: dark is certain, bright is +/-0.15 or wider.")]) for c in cells],
        about=(f"{E(HOW)} Every one of the {S['sites']:,} atlas sites has one fixed place, the same on every globe here and in every report: HEALPix "
               f"NSIDE {S['nside']}, {S['order']} order, sites in genome order. Its benefits as a map: every pixel the same area, no edges, and change that "
               f"runs along the genome shows as a patch while scattered noise shows as speckle. A picture of the atlas: nothing is read or scored from it."))
    specs = [atlas]
    # 2026-11-05 (author: "make sure the atlas globes are both linked and represented for BOTH atlas's"): the IAM Sequencing Atlas, every cell on the
    # same viewer, from the chain's own sphere build (website/art/out/seq_cells/index.json, run_sample.py --build seq-atlas-sphere); no masks are drawn
    fs = os.path.join(ART_, "seq_cells", "index.json")
    if os.path.exists(fs):
        Q = json.load(open(fs)); qc = sorted(({"slug": c["slug"], "name": c["cell"][:1].upper() + c["cell"][1:]} for c in Q["cells"]), key=lambda c: c["name"].lower())
        specs.append(dict(id="seq-atlas", name=f"Sequencing Atlas Globe: the IAM Sequencing Atlas, {len(qc)} healthy cell types at every CpG", groups=[dict(id=c["slug"], name=c["name"], layers=[
            dict(name="methylation", kind="image", src=f"{art_rel}seq_cells/{c['slug']}_v.jpg", note="The healthy level at every CpG, counted from the donors' reads, 0 (never tagged) to 1 (always tagged)."),
            dict(name="how sure the atlas is", kind="image", src=f"{art_rel}seq_cells/{c['slug']}_u.jpg", note="Half-width of the 95 % range of the level: dark is certain, bright is +/-0.15 or wider.")]) for c in qc],
            about=(f"The ruler for sequenced samples, measured with sequencing's own chemistry. Every CpG of the genome has one fixed place: HEALPix NSIDE "
                   f"{Q['nside']}, {Q['order']} order, in genome order, the same grid as IAM-Atlas v2.1's globe, so the two atlases can be compared cell for cell. "
                   f"A picture of the atlas: nothing is read or scored from it. Every cell's page: seq-atlas/index.html.")))
    for g in sorted(glob.glob(os.path.join(ART_, "explorer", "*.json"))): specs.append(json.load(open(g)))
    intro = (f'<p><a href="index.html">&larr; all {len(cells)} cells</a> &middot; <a href="{back}">front page</a></p><p>Choose a globe, then what to show. '
             'Drag to turn it. Each globe brings its own layers and sliders. Gray or dark pixels hold no measured site.</p>'
             + ('<p>Two atlases are here: IAM-Atlas v2.1, the ruler for arrays, and the IAM Sequencing Atlas, the ruler for sequenced samples '
                '(<a href="sphere.html?globe=seq-atlas">the Sequencing Atlas Globe</a>; <a href="../seq-atlas/index.html">its cell pages</a>; '
                '<a href="information_sphere.html?atlas=seq">the Sequencing Information Sphere</a>).</p>' if os.path.exists(fs) else ''))
    isp = os.path.join(ART_, "information_sphere.html")   # 2026-10-23: the information sphere, as the one command built it (run_sample.py --build info-globe)
    if os.path.exists(isp): intro += ('<p><b>New: <a href="information_sphere.html">the information sphere</a></b> (development). These globes place every site by where '
                                      'it sits on the DNA; the information sphere places it by how it behaves across the 79 cell types.</p>')
    d = os.path.join(out, "atlas"); os.makedirs(d, exist_ok=True)
    # 2026-11-05 (author: "They each should have a link to the appropriate sphere ... a dropdown option on one information sphere to toggle between
    # the two"): each atlas's sphere keeps its own file, as the one command drew it; information_sphere.html chooses between them. Counts are read
    # from each sphere's build record; the sequencing atlas's sphere caption is v2.1's fixed text until the builder writes it from the atlas given.
    import shutil as _sh
    gd = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Biological_Physics", "MethylPhys", "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", "globe")
    SPH = []
    if os.path.exists(isp):
        _sh.copy(isp, os.path.join(d, "info_sphere_v21.html")); m = json.load(open(os.path.join(gd, "v21_globe.json")))
        SPH.append(("v21", "Array Information Sphere (IAM-Atlas v2.1)", "info_sphere_v21.html", m))
    sq = os.path.join(gd, "seq_globe.html")
    if os.path.exists(sq):
        _sh.copy(sq, os.path.join(d, "info_sphere_seq.html")); m = json.load(open(os.path.join(gd, "seq_globe.json")))
        SPH.append(("seq", "Sequencing Information Sphere (IAM Sequencing Atlas)", "info_sphere_seq.html", m))
    if SPH:
        opts = "".join(f'<option value="{k}">{nm}</option>' for k, nm, f, m in SPH)
        info = {k: dict(src=f, text=f"{nm}: {m['sites']:,} sites placed, from {m['cell_types']} healthy cell types (sites measured in every cell type that differ between them); "
                                     f"{m['sites_clear']:,} stand apart in one cell type and are colored.") for k, nm, f, m in SPH}
        note = ("" if not any(k == "seq" for k, *_ in SPH) else
                "<p class=\"note\">The sequencing atlas&rsquo;s sphere was drawn from 56 of its 57 cell types (before the granulocytes were added), and its "
                "&ldquo;What you are looking at&rdquo; panel still carries IAM-Atlas v2.1&rsquo;s wording; the points drawn are the sequencing atlas&rsquo;s. "
                "A rebuild with all 57 and its own wording is next.</p>")
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Array and Sequencing Information Spheres</title><style>body{margin:0;font-family:system-ui,sans-serif;background:#0b0e14;color:#e8e8e8}'
                '.bar{padding:10px 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}select{font-size:16px;padding:6px 10px;border-radius:6px}'
                'a{color:#9cc3ff}.note{margin:0 16px 8px;font-size:13px;color:#c8b98a}#t{font-size:14px;color:#cfd6e4}iframe{border:0;width:100%;height:calc(100vh - 110px)}</style></head><body>'
                f'<div class="bar"><b>Information spheres</b><label>Show: <select id="s">{opts}</select></label><span id="t"></span>'
                '<a href="sphere.html">the Globe Viewer (the atlas globes: every site by its place on the DNA)</a> &middot; <a href="index.html">IAM-Atlas v2.1 cells</a> &middot; '
                '<a href="../seq-atlas/index.html">IAM Sequencing Atlas cells</a></div>' + note + '<iframe id="f" title="information sphere"></iframe>'
                '<script>const I=' + json.dumps(info) + ';const s=document.getElementById("s");'
                'function go(k){if(!I[k])k=Object.keys(I)[0];s.value=k;document.getElementById("f").src=I[k].src;document.getElementById("t").textContent=I[k].text;'
                'history.replaceState(null,"","?atlas="+k)}s.onchange=()=>go(s.value);go(new URLSearchParams(location.search).get("atlas")||"v21");</script></body></html>')
        open(os.path.join(d, "information_sphere.html"), "w", encoding="utf-8").write(page)
    open(os.path.join(d, "sphere.html"), "w", encoding="utf-8").write(V.page(specs, "The IAM Globe Viewer", intro=intro))
    return len(cells)

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
    open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(head("IAM-Atlas v2.1: every cell on the Array Atlas Globe")
        + f'<p><a href="{back}">&larr; IAMPerformance front page</a> &middot; <a href="../seq-atlas/index.html">the IAM Sequencing Atlas (sequencing)</a></p><h1>IAM-Atlas v2.1: all {len(I)} healthy cell types on the Array Atlas Globe</h1><p>{E(HOW)} '
        + 'Choose a cell to see its map beside how sure the atlas is at every site.' + (' Or <a href="sphere.html">open any cell in the Globe Viewer</a>.' if globe else '')
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
