"""IAM-Atlas v2.1 cell pages for the site (2026-10-18, author: every cell on its own page). One gallery page (atlas/index.html) and one
page per cell (atlas/<slug>.html): the cell's healthy map beside how sure the atlas is, with its numbers, read from
website/art/out/cells/index.json (written by website/art/make_atlas_sky_v21.py). Pictures and numbers only; nothing is fitted."""
import os, json, html
E = html.escape
HOW = ("Every site of the atlas keeps one place on a HEALPix sphere, the pixel grid the Planck and WMAP missions used to map the microwave sky, "
       "set by where the site sits in the genome (chromosome 1 through Y). The layout is the same for every cell, so any two cells, and a person's "
       "sample, can be compared site for site.")
def write(out, art_rel, css_tag, style, back):
    I = json.load(open(os.path.join(os.path.dirname(__file__), "..", "art", "out", "cells", "index.json")))["cells"]
    I = sorted(I, key=lambda c: c["name"].lower()); d = os.path.join(out, "atlas"); os.makedirs(d, exist_ok=True)
    head = lambda t: (f'<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                      f'<title>{E(t)}</title>{css_tag}<style>{style}.cg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px}}'
                      f'.cg a{{text-decoration:none;color:inherit}}.cg img{{width:100%;border-radius:4px}}.cg span{{font-size:14px}}</style></head><body><section class="wrap">')
    g = [f'<a href="{c["slug"]}.html"><img loading="lazy" src="{art_rel}cells/{c["slug"]}_thumb.jpg" alt="{E(c["name"])} on the sphere"><span>{E(c["name"])}</span></a>' for c in I]
    open(os.path.join(d, "index.html"), "w", encoding="utf-8").write(head("IAM-Atlas v2.1: every cell on the sky")
        + f'<p><a href="{back}">&larr; IAMPerformance front page</a></p><h1>IAM-Atlas v2.1: all {len(I)} healthy cell types on the sky</h1><p>{E(HOW)} '
        + 'Choose a cell to see its map beside how sure the atlas is at every site.</p><div class="cg">' + "".join(g) + "</div></section></body></html>")
    for k, c in enumerate(I):
        p, n = I[k - 1], I[(k + 1) % len(I)]
        body = (f'<p><a href="index.html">&larr; all {len(I)} cells</a> &middot; <a href="{back}">front page</a></p><h1>{E(c["name"])}</h1>'
                f'<figure><img src="{art_rel}cells/{c["slug"]}.jpg" alt="{E(c["name"])}: value and uncertainty on the sphere" style="width:100%">'
                f'<figcaption>Left: healthy {E(c["name"].lower())} from IAM-Atlas v2.1, every site in its genome place, from never tagged (blue) to always tagged (red). '
                f'Right: how sure the atlas is of each value. Dark means certain (0 would mean the value is known exactly); the brighter a point, the wider its 95&nbsp;% range. Every cell&rsquo;s map uses the same scale, 0 to &plusmn;0.15, so cells can be compared with each other.</figcaption></figure>'
                f'<table><tr><th>Average methylation</th><td>{c["mean_methylation"]:.3f}</td></tr>'
                f'<tr><th>How sure (half-width of the 95&nbsp;% range)</th><td>half the sites within &plusmn;{c["halfwidth_median"]:.3f}; nine in ten within &plusmn;{c["halfwidth_p90"]:.3f}</td></tr>'
                f'<tr><th>Pixels with a measured site</th><td>{c["pixels_measured"]:,} of {c["pixels"]:,}</td></tr></table>'
                f'<p>{E(HOW)}</p><p><a href="{p["slug"]}.html">&larr; {E(p["name"])}</a> &middot; <a href="{n["slug"]}.html">{E(n["name"])} &rarr;</a></p>')
        open(os.path.join(d, c["slug"] + ".html"), "w", encoding="utf-8").write(head(f'{c["name"]}: IAM-Atlas v2.1 on the sky') + body + "</section></body></html>")
    return len(I)
