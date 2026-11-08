"""IAM-Atlas v2.1 cell pages for the site (2026-10-18, author: every cell on its own page). One gallery page (atlas/index.html) and one
page per cell (atlas/<slug>.html): the cell's healthy map beside how sure the atlas is, with its numbers, read from
website/art/out/cells/index.json (written by website/art/make_atlas_sky_v21.py). Pictures and numbers only; nothing is fitted."""
import os, re, json, html, csv, collections
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

# 2026-11-05: cell names as written in the field (CD4 T cells, NK cells, B cells), for every page; the atlases keep their own labels
_UPW = {"cd4": "CD4", "cd8": "CD8", "cd14": "CD14", "cd34": "CD34", "cd56bright": "CD56bright", "cd56dim": "CD56dim", "nk": "NK", "temra": "TEMRA", "(cd28-)": "(CD28-)", "th1": "Th1", "th17": "Th17", "treg": "Treg"}
_FIXN = {"fallopien tubes epithelium": "Fallopian tube epithelium"}
def cellname(s):
    if s.lower() in _FIXN: return _FIXN[s.lower()]
    w = s.split(" "); out = []
    for i, x in enumerate(w):
        lx = x.lower()
        if lx in _UPW: out.append(_UPW[lx])
        elif lx in ("t", "b") and i + 1 < len(w) and w[i + 1].lower().startswith("cell"): out.append(lx.upper())
        else: out.append(x[:1].upper() + x[1:] if i == 0 else x)
    return " ".join(out)

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
    S = json.load(open(f)); cells = sorted(({"slug": c["slug"], "name": cellname(names.get(c["slug"], c["cell"]))} for c in S["cells"]), key=lambda c: c["name"].lower())
    atlas = dict(id="atlas", name="Array Atlas Globe (IAM-Atlas v2.1, 79 cell types)", groups=[dict(id=c["slug"], name=c["name"], layers=[
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
        # 2026-11-06 (plan 61.4): the two noise maps, drawn by the chain (run_sample.py --build seq-noise-sphere), with what their null test found
        # (run_sample.py --build seq-noise-null); every number read from the committed records at build time
        _SD = os.path.join(H_, "..", "..", "Biological_Physics", "MethylPhys", "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas")
        _NS = json.load(open(os.path.join(_SD, "noise_sphere.json"))); _NN = json.load(open(os.path.join(_SD, "noise_null.json")))["rows"]
        _m1 = [r for r in _NN if r["map"].startswith("map1")]; _m2 = [r for r in _NN if r["map"].startswith("map2")]
        _lo, _hi = _NS["scale"]
        NZ = {1: ("noise map: donor halves", f"The cell's donors split into two sealed halves, each counted alone, and the difference drawn ({_lo:+.1f} to {_hi:+.1f}). It shows how much "
                 f"healthy people of this cell differ from one another. In {sum(1 for r in _m1 if len(r['bands_above_null']) > 1)} of the {len(_m1)} cells with donors to split, it stands above "
                 "its shuffled null in more than one band: real differences between healthy people, not noise. Cells from a single donor have no such map."),
              2: ("noise map: read halves", f"Each donor's reads split at random into two halves, counted alone, and the difference drawn ({_lo:+.1f} to {_hi:+.1f}). It shows the "
                 f"counting noise alone. It fits its shuffled null (at most one band above it) in {sum(1 for r in _m2 if len(r['bands_above_null']) <= 1)} of {len(_m2)} cells: what pure counting noise looks like.")}
        Q = json.load(open(fs)); qc = sorted(({"slug": c["slug"], "name": cellname(c["cell"])} for c in Q["cells"]), key=lambda c: c["name"].lower())
        # 2026-11-07 (author: the viewer must show the new globe, at every CpG): the Sequencing Atlas Globe is the globe at every CpG, drawn by the one
        # command from the published textures (run_sample.py --build seq-wg-sphere; website/art/out/seq_wg/index.json); the same atlas on the array layout
        # stays as a second globe, beside its noise maps, for comparing the two atlases cell for cell
        _fw = os.path.join(ART_, "seq_wg", "index.json")
        if os.path.exists(_fw):
            _flw = os.path.join(ART_, "seq_wg", "layers.json"); _LY = {x["slug"]: x for x in json.load(open(_flw))["cells"]} if os.path.exists(_flw) else {}   # 2026-11-08: the CMB-tool layers
            WG = json.load(open(_fw)); wc = sorted(({"slug": c["slug"], "name": cellname(c["cell"])} for c in WG["cells"]), key=lambda c: c["name"].lower())
            specs.append(dict(id="seq-atlas", name=f"Sequencing Atlas Globe, every CpG (IAM Sequencing Atlas, {len(wc)} cell types)", groups=[dict(id=c["slug"], name=c["name"], layers=[
                dict(name="methylation (the sky map)", kind="image", src=f"{art_rel}seq_wg/{c['slug']}_v.jpg", mask=f"{art_rel}seq_wg/{c['slug']}_vm.png", vscale=[0, 1], vword="at least this methylated:",
                     hint="Slide right to keep only the places at least this methylated; the rest go dark. Slide back to see every place.",
                     note="The healthy level, counted from the donors' reads, 0 (never tagged) to 1 (always tagged); each pixel the average of the CpGs it holds."),
                dict(name="how sure the atlas is (the error map)", kind="image", src=f"{art_rel}seq_wg/{c['slug']}_u.jpg", mask=f"{art_rel}seq_wg/{c['slug']}_um.png", vscale=[0, 0.15], vword="less sure than +/-",
                     hint="Slide right to keep only the places where the atlas is least sure; the sure ones go dark.",
                     note="Half-width of the 95 % range of the level: dark is certain, bright is +/-0.15 or wider.")]
                + ([dict(name="noise map 1: donor halves (half-mission difference)", kind="image", src=f"{art_rel}seq_wg/{c['slug']}_n1.jpg",
                     note="The cell's donors split in two halves, each half counted on its own, and the half-difference drawn: how much healthy people of this cell differ, region by region. The same test a sky survey makes by splitting its observing time. Blue below 0, red above, the scale +/-0.05 the same for every cell.")] if _LY.get(c["slug"], {}).get("n1") else [])
                + [dict(name="noise map 2: read halves (counting noise)", kind="image", src=f"{art_rel}seq_wg/{c['slug']}_n2.jpg",
                     note="Each read sent at random to one of two halves, each half counted on its own, and the half-difference drawn: the counting noise alone, nothing biological. Blue below 0, red above, scale +/-0.05."),
                   dict(name="the null (shuffled noise map 2)", kind="image", src=f"{art_rel}seq_wg/{c['slug']}_null.jpg",
                     note="Noise map 2's own values shuffled among the places the tools use: what the noise looks like with no position in the genome left in it. Structure in the real maps that this one lacks is what the null test measures."),
                   dict(name="the mask", kind="image", src=f"{art_rel}seq_wg/mask.jpg",
                     note="The places the tools leave out, as a sky survey masks its galaxy: the sex chromosomes (dark), the few places no cell measured (gray), and the places used (light).")]) for c in wc],
                about=(f"The ruler for sequenced samples, measured with sequencing's own chemistry, at every one of the genome's {WG['cpgs']:,} CpGs, each in one fixed place in genome order "
                       f"(HEALPix, {WG['order']} order). The globe itself is NSIDE 2048, about 2 CpGs a pixel; shown here at NSIDE {WG['nside']}, each pixel the average of the CpGs it holds "
                       f"(about 36). Published on Zenodo ({WG['doi']}); how each part of it works: seq-atlas/globe.html. A picture of the atlas: nothing is read or scored from it.")))
        specs.append(dict(id=("seq-atlas-layout" if os.path.exists(_fw) else "seq-atlas"), name=f"Sequencing Atlas on the array layout (813,951 sites, {len(qc)} cell types)", groups=[dict(id=c["slug"], name=c["name"], layers=[
            dict(name="methylation", kind="image", src=f"{art_rel}seq_cells/{c['slug']}_v.jpg", note="The healthy level at each of the layout's sites, counted from the donors' reads, 0 (never tagged) to 1 (always tagged)."),
            dict(name="how sure the atlas is", kind="image", src=f"{art_rel}seq_cells/{c['slug']}_u.jpg", note="Half-width of the 95 % range of the level: dark is certain, bright is +/-0.15 or wider.")]
            + [dict(name=NZ[k][0], kind="image", src=f"{art_rel}seq_noise/{c['slug']}_n{k}.jpg", note=NZ[k][1]) for k in (1, 2) if os.path.exists(os.path.join(ART_, "seq_noise", f"{c['slug']}_n{k}.jpg"))]) for c in qc],
            about=(f"The ruler for sequenced samples, measured with sequencing's own chemistry. Drawn on IAM-Atlas v2.1's layout of 813,951 sites, each with one fixed place: HEALPix NSIDE "
                   f"{Q['nside']}, {Q['order']} order, in genome order, the same grid as the Array Atlas Globe, so the two atlases can be compared cell for cell, with the atlas's two "
                   f"noise maps. The globe at every CpG is the Sequencing Atlas Globe above. A picture of the atlas: nothing is read or scored from it. Every cell's page: seq-atlas/index.html.")))
    SHORT = {"healthy-blood": "Healthy whole blood (GSM7508429)", "planted-blood": "Same blood, planted loss", "purified": "Met-A purified reference",
             "noise": "Array atlas uncertainty"}   # menu names short enough for the menu box; each globe's own page text is unchanged
    for g in sorted(glob.glob(os.path.join(ART_, "explorer", "*.json"))):
        sp = json.load(open(g)); sp["name"] = SHORT.get(sp.get("id"), sp.get("name")); specs.append(sp)
    intro = (f'<p><a href="{back}">&larr; front page</a></p><p>Choose a globe, then what to show. '
             'Drag to turn it. Each globe brings its own layers and sliders. Gray or dark pixels hold no measured site.</p>'
             + ('<p>Two atlases are here, each on two kinds of map. <b>Atlas globes</b> place every site by where it sits on the DNA: '
                '<a href="sphere.html">the Array Atlas Globe</a> (IAM-Atlas v2.1, the ruler for arrays) and '
                '<a href="sphere.html?globe=seq-atlas">the Sequencing Atlas Globe</a> (the IAM Sequencing Atlas, the ruler for sequenced samples, at every one of the genome\'s 28.2 million CpGs; '
                '<a href="../seq-atlas/globe.html">how each part of it works</a>; the same atlas <a href="sphere.html?globe=seq-atlas-layout">on the array layout</a>, to compare the two atlases cell for cell). <b>Information spheres</b> place every site by how it behaves '
                'across the cell types: <a href="information_sphere.html">the Array Information Sphere</a> and '
                '<a href="information_sphere_sequencing.html">the Sequencing Information Sphere</a>. Cell pages: '
                '<a href="index.html">IAM-Atlas v2.1</a> &middot; <a href="../seq-atlas/index.html">the IAM Sequencing Atlas</a>.</p>' if os.path.exists(fs) else ''))
    isp = os.path.join(ART_, "information_sphere.html")   # the Array Information Sphere, as the one command built it (run_sample.py --build info-globe)
    d = os.path.join(out, "atlas"); os.makedirs(d, exist_ok=True)
    # 2026-11-05 (author: "They each should have a link to the appropriate sphere ... a dropdown option on one information sphere to toggle between
    # the two"): each atlas's sphere keeps its own file, as the one command drew it; information_sphere.html chooses between them. Counts are read
    # from each sphere's build record; the sequencing atlas's sphere caption is v2.1's fixed text until the builder writes it from the atlas given.
    # The two information spheres, each a full page as the one command drew it, with one more menu in its own side panel (same label and
    # select style as the page's other menus) to switch atlas. The sequencing sphere's title, subtitle and "What you are looking at" panel
    # carry the array atlas's fixed text from the builder; its own numbers are written in from its build record (seq_globe.json) until the
    # builder writes them itself. The build stops if any array-atlas number is left on the sequencing page.
    import shutil as _sh
    gd = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Biological_Physics", "MethylPhys", "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", "globe")
    sq = os.path.join(gd, "seq_globe.html")
    if os.path.exists(isp) and os.path.exists(sq):
        A1 = json.load(open(os.path.join(gd, "v21_globe.json"))); A2 = json.load(open(os.path.join(gd, "seq_globe.json")))
        F1, F2 = "information_sphere.html", "information_sphere_sequencing.html"
        def menu(cur):
            o = "".join(f'<option value="{f}"{" selected" if f == cur else ""}>{nm}</option>' for f, nm in (
                (F1, "Array Information Sphere (IAM-Atlas v2.1)"), (F2, "Sequencing Information Sphere (IAM Sequencing Atlas)")))   # counts are in each page's subtitle
            return f'<label for="atlas">Atlas</label><select id="atlas" onchange="location.href=this.value">{o}</select>'
        def put_menu(h, cur, title):
            h = re.sub(r"<h1>The information sphere", "<h1>" + title, h, count=1)
            k = h.index('<div class="sub">', h.index('<div class="side">')); k = h.index("</div>", k) + len("</div>")
            return h[:k] + menu(cur) + h[k:]
        h1 = open(isp, encoding="utf-8").read()
        h1 = put_menu(h1, F1, "Array Information Sphere")
        h1 = h1.replace("<script>", "<script>if(new URLSearchParams(location.search).get('atlas')==='seq')location.replace('" + F2 + "');", 1)   # older links
        open(os.path.join(d, F1), "w", encoding="utf-8").write(h1)
        h2 = open(sq, encoding="utf-8").read(); s0 = h2.index("<script>"); v, rest = h2[:s0], h2[s0:]
        n2, c2, k2 = f"{A2['sites']:,}", str(A2["cell_types"]), f"{A2['sites_clear']:,}"
        for a_, b_ in (("placed by how it behaves across IAM-Atlas v2.1's 79 healthy cell types.", f"placed by how it behaves across the IAM Sequencing Atlas's {c2} healthy cell types."),
                       ("one of 191,458 that IAM-Atlas v2.1 measured in all 79 of its healthy cell types", f"one of {n2} that the IAM Sequencing Atlas measured in all {c2} of its healthy cell types"),
                       ("its methylation in the 79 cell types.", f"its methylation in the {c2} cell types."),
                       ("computed from its 79 measured levels alone", f"computed from its {c2} measured levels alone"),
                       ("and the 79 cell types are grouped into 15 families", f"and the {c2} cell types are grouped into {A2['sites_by_family']} families"),
                       ("41,241 of the 191,458.", f"{k2} of the {n2}."),
                       ("how it behaves across the 79 cell types</td>", f"how it behaves across the {c2} cell types</td>"),
                       ("On the atlas's real healthy cells, 11 of 12", "The construction was tested on IAM-Atlas v2.1's healthy cells (not yet on the sequencing atlas's): there, 11 of 12")):
            assert v.count(a_) == 1, "sequencing sphere text changed: " + a_[:50]; v = v.replace(a_, b_)
        left = [x for x in ("191,458", "41,241", "IAM-Atlas v2.1's 79", " 79 ") if x in v]
        assert not left, f"array-atlas numbers left on the sequencing sphere: {left}"
        h2 = put_menu(v + rest, F2, "Sequencing Information Sphere")
        open(os.path.join(d, F2), "w", encoding="utf-8").write(h2)
    elif os.path.exists(isp): _sh.copy(isp, os.path.join(d, "information_sphere.html"))
    open(os.path.join(d, "sphere.html"), "w", encoding="utf-8").write(V.page(specs, "The IAM Globe Viewer", intro=intro))
    return len(cells)

def write(out, art_rel, css_tag, style, back):
    I = json.load(open(os.path.join(os.path.dirname(__file__), "..", "art", "out", "cells", "index.json")))["cells"]
    I = sorted(({**c, "name": cellname(c["name"])} for c in I), key=lambda c: c["name"].lower()); d = os.path.join(out, "atlas"); os.makedirs(d, exist_ok=True)
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
                f'<figcaption>Left: healthy {E(c["name"][:1].lower() + c["name"][1:] if not c["name"][:2].isupper() else c["name"])} from IAM-Atlas v2.1, every site in its genome place, from never tagged (blue) to always tagged (red). '
                f'Right: how sure the atlas is of each value. Dark means certain (0 would mean the value is known exactly); the brighter a point, the wider its 95&nbsp;% range. Every cell&rsquo;s map uses the same scale, 0 to &plusmn;0.15, so cells can be compared with each other.</figcaption></figure>'
                f'<table><tr><th>Average methylation</th><td>{c["mean_methylation"]:.3f}</td></tr>'
                f'<tr><th>How sure (half-width of the 95&nbsp;% range)</th><td>half the sites within &plusmn;{c["halfwidth_median"]:.3f}; nine in ten within &plusmn;{c["halfwidth_p90"]:.3f}</td></tr>'
                f'<tr><th>Pixels with a measured site</th><td>{c["pixels_measured"]:,} of {c["pixels"]:,}</td></tr>' + why(c)[0] + '</table>' + why(c)[1] +
                f'<p>{E(HOW)}</p>' + (f'<p><a href="sphere.html?cell={c["slug"]}">Turn this cell on the globe</a></p>' if globe else '') + f'<p><a href="{p["slug"]}.html">&larr; {E(p["name"])}</a> &middot; <a href="{n["slug"]}.html">{E(n["name"])} &rarr;</a></p>')
        open(os.path.join(d, c["slug"] + ".html"), "w", encoding="utf-8").write(head(f'{c["name"]}: IAM-Atlas v2.1 on the sky') + body + "</section></body></html>")
    return len(I)
