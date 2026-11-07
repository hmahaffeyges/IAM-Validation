"""The Sequencing Atlas Globe showpiece page (plan 60.23.3), generated only from CANON/globe_wg_story.json (plan 60.23.2: one source for the
page, the book section, the SOP and the catalogue). Each part of the globe stands beside its cosmic-microwave-background counterpart and the
name a geneticist uses, with one physics sentence from the canon, what it shows, its result and the committed file behind it. Images are the
ones the site already draws from the published globe (website/art/out/seq_cells, seq_noise), shown for one example cell.
Usage (the site build calls write()); python3 website/build/globe_wg_page.py --check   (exit 1 naming any part missing a field or an image)"""
import os, sys, json, html
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STORY = os.path.join(REPO, "CANON", "globe_wg_story.json"); ART = os.path.join(REPO, "website", "art", "out")
GH = "https://github.com/hmahaffeyges/IAM-Validation/blob/main/"
FIELDS = ("name", "cmb", "geneticist", "physics", "shows", "result", "record", "how")
EX = ("b-cells", "B cells")   # the example cell for the pictures
IMG = {"The level map": ("seq_cells", "_v.jpg"), "The range map": ("seq_cells", "_u.jpg"),
       "Noise map 1: donor halves": ("seq_noise", "_n1.jpg"), "Noise map 2: read halves": ("seq_noise", "_n2.jpg")}
E = lambda s: html.escape(str(s))

def problems():
    S = json.load(open(STORY, encoding="utf-8")); P = []
    for p in S["parts"]:
        P += [f"{p.get('name')}: no {f}" for f in FIELDS if not p.get(f)]
        if p.get("name") in IMG:
            d, suf = IMG[p["name"]]
            if not os.path.exists(os.path.join(ART, d, EX[0] + suf)): P.append(f"{p['name']}: image {d}/{EX[0]}{suf} missing")
    for k in ("title", "opening", "facts", "rebuild", "closing"):
        if not S.get(k): P.append(f"story: no {k}")
    return P

def links(rec):
    out = []
    for r in [x.strip() for x in str(rec).split(";") if x.strip()]:
        out.append(f'<a href="{GH}{E(r)}"><code>{E(os.path.basename(r))}</code></a>')
    return ", ".join(out)

def write(out, art_rel, css_tag, style, back):
    S = json.load(open(STORY, encoding="utf-8")); F = S["facts"]
    d = os.path.join(out, "seq-atlas"); os.makedirs(d, exist_ok=True)
    css = (style + ".part{border-top:1px solid rgba(128,128,128,.35);padding:18px 0}.part h2{margin:0 0 8px}"
           ".names{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:8px 0}.names div{padding:10px;border-radius:6px;background:rgba(128,128,128,.10)}"
           ".names b{display:block;font-size:13px;text-transform:uppercase;letter-spacing:.04em;opacity:.75}"
           ".part img{max-width:100%;border-radius:4px;margin:8px 0}.facts td{padding:3px 12px 3px 0}"
           "@media (max-width:700px){.names{grid-template-columns:1fr}}")
    H = [f'<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
         f'<title>{E(S["title"])}</title>{css_tag}<style>{css}</style></head><body><section class="wrap">',
         f'<p><a href="{back}">&larr; IAMPerformance front page</a> &middot; <a href="index.html">The IAM Sequencing Atlas</a> &middot; '
         '<a href="../atlas/sphere.html?globe=seq-atlas">Open the Globe Viewer</a> &middot; <a href="../atlas/information_sphere_sequencing.html">Sequencing Information Sphere</a></p>',
         f'<h1>{E(S["title"])}</h1>'] + [f"<p>{E(x)}</p>" for x in S["opening"]]
    H.append('<table class="facts">'
             f'<tr><td>CpGs on the globe</td><td>{F["cpgs"]:,}</td></tr>'
             f'<tr><td>Cell types</td><td>{F["cells"]} (megakaryocytes provisional)</td></tr>'
             f'<tr><td>Resolution</td><td>HEALPix NSIDE {F["nside"]}: one CpG to a pixel</td></tr>'
             f'<tr><td>Check against the array-layout globe</td><td>{F["values_equal"]:,} of {F["values_checked_against_array_layout_globe"]:,} values equal</td></tr>'
             f'<tr><td>The globe on Zenodo</td><td><a href="https://doi.org/{E(F["doi"])}">doi.org/{E(F["doi"])}</a></td></tr>'
             f'<tr><td>The atlas it is built from</td><td><a href="https://doi.org/{E(F["atlas_doi"])}">doi.org/{E(F["atlas_doi"])}</a></td></tr></table>')
    H.append("<h2>Each part, by its name in cosmology and in genetics</h2>")
    for p in S["parts"]:
        H.append(f'<div class="part"><h2>{E(p["name"])}</h2><div class="names"><div><b>In cosmology</b>{E(p["cmb"])}</div>'
                 f'<div><b>In genetics</b>{E(p["geneticist"])}</div></div>'
                 f'<p><b>The physics.</b> {E(p["physics"])}</p><p><b>What it shows.</b> {E(p["shows"])}</p><p><b>Result.</b> {E(p["result"])}</p>')
        if p["name"] in IMG:
            dd, suf = IMG[p["name"]]
            H.append(f'<img loading="lazy" src="{art_rel}{dd}/{EX[0]}{suf}" alt="{E(EX[1])}: {E(p["name"].lower())}"><br><small>Shown for {E(EX[1])}; every cell has its own on its page in the atlas.</small>')
        H.append(f'<p><small>Record: {links(p["record"])}. Built by: <code>{E(p["how"])}</code></small></p></div>')
    H.append('<div class="part"><h2>Build it again</h2><p>Everything needed is on Zenodo and GitHub.</p><ol>' + "".join(f"<li>{E(x)}</li>" for x in S["rebuild"]) + "</ol></div>")
    H.append(f'<p>{E(S["closing"])}</p></section></body></html>')
    open(os.path.join(d, "globe.html"), "w", encoding="utf-8").write("".join(H))
    return len(S["parts"])

if __name__ == "__main__" and "--check" in sys.argv:
    P = problems()
    if P: print("globe_wg_page: " + "; ".join(P)); sys.exit(1)
    print("globe_wg_page: ok"); sys.exit(0)
