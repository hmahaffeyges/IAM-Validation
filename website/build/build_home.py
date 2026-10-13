"""Front page for followers (PREVIEW; author request 2026-10-12). Writes SITE/preview/index.html from the committed records only:
  CANON/now_running.json   the Now running stories (newest update each; full story in FIELD_NOTES.md)
  CANON/results_register.json   results in plain words, with status and record
  Biological_Physics/MethylPhys/RUNS.md   actual AWS charges (month totals, daily line)
  CANON/funding.json   the fundraiser link
Nothing on the page is typed here except headings and the fixed explanation of the method, which states each cosmology tool with its book
tool-board status. The page text is checked with the field-notes voice and hype rules (CANON/now_running.py) and every repository link must
exist; --check runs those checks without writing. --standalone OUT.html writes one self-contained file (CSS and images inlined) for review.
Usage: python3 website/build/build_home.py SITE_DIR | --check | --standalone OUT.html"""
import os, re, sys, json, html, base64, datetime, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GH = "https://github.com/hmahaffeyges/IAM-Validation/blob/main/"
SITE_URL = "https://hmahaffeyges.github.io/IAM-Validation/"
J = lambda p: json.load(open(os.path.join(REPO, p), encoding="utf-8"))
spec = importlib.util.spec_from_file_location("nr", os.path.join(REPO, "CANON", "now_running.py")); NR = importlib.util.module_from_spec(spec); spec.loader.exec_module(NR)
E = html.escape
STCLS = {"getting ready": "st-b", "running": "st-a", "on hold": "st-a", "result": "st-g"}
TOOLS = [  # (cosmology tool, what it does here, tool-board status), from docs/book/part6/16a_skytools_map79.tex
    ("One hierarchical fit with an offset per source, sampled by MCMC", "combines many laboratories' healthy cells into one atlas, each lab's offset measured", "partial: the atlas is a posterior; readings carry none yet"),
    ("Component separation", "reads past the mix of cell types to the signal underneath", "in use, one solver; a second is built, not in the chain"),
    ("Split maps and null tests", "checks the map on data it never saw", "in use"),
    ("HEALPix pixelization", "lays the genome out the way the sky is laid out", "built, not in the chain; its statistics are withheld until their null test passes"),
]

def rec(p): return GH + p
def costs():
    t = open(os.path.join(REPO, "Biological_Physics/MethylPhys/RUNS.md"), encoding="utf-8").read()
    months = re.findall(r"^\| (\d{4}-\d{2}) \| \$([\d,.]+) \| (.*?) \|$", t, re.M)
    daily = re.search(r"^Daily: (.*)$", t, re.M); days = re.findall(r"(\d{2}-\d{2}) \$([\d,.]+)", daily.group(1)) if daily else []
    assert months and days, "RUNS.md billing section not found"
    week = days[-7:]
    return months, days, week, sum(float(v.replace(",", "")) for _, v in week)

def page(art):
    NRJ = J("CANON/now_running.json"); REG = J("CANON/results_register.json"); FUND = J("CANON/funding.json")
    months, days, week, wsum = costs()
    stories = []
    for s in NRJ["stories"]:
        u = sorted(enumerate(s["updates"]), key=lambda t: (t[1]["date"], t[0]))[-1][1]
        stories.append(f'''<article class="card"><span class="chip {STCLS[s["status"]]}">{E(NR.STATUS[s["status"]])}</span>
<h3>{E(s["headline"])}</h3><p>{E(s["why"])}</p>
<p class="hr"><b>What we hope:</b> {E(s["hope"])}</p><p class="hr"><b>What could go wrong:</b> {E(s["risk"])}</p>
<p class="latest"><span>{E(u["date"])}</span> {E(u["text"])}</p>
<p class="more"><a href="{rec("FIELD_NOTES.md")}#{s["id"].replace("_", "-")}">The full story, in plain words</a> · <a href="{rec(u["record"])}">record</a></p></article>''')
    regs = sorted([e for e in REG["entries"] if e.get("status") != "withdrawn"], key=lambda e: (e["status"] != "commissioned", e["date"]), reverse=False)
    regs = sorted(regs, key=lambda e: e["date"], reverse=True); regs = [e for e in regs if e["status"] == "commissioned"] + [e for e in regs if e["status"] != "commissioned"]
    results = "\n".join(f'''<article class="res"><span class="chip {"st-g" if e["status"] == "commissioned" else "st-b"}">{"Commissioned" if e["status"] == "commissioned" else "In development"}</span>
<h3>{E(e.get("title") or e["id"])}</h3><p>{E(e.get("plain", ""))}</p><p class="more"><span>{E(e["date"])}</span> · <a href="{rec(e["record"])}">record</a></p></article>''' for e in regs)
    tools = "\n".join(f"<tr><td>{E(a)}</td><td>{E(b)}</td><td>{E(c)}</td></tr>" for a, b, c in TOOLS)
    cur = months[-1]
    body = f'''<header class="hero" style="background-image:linear-gradient(rgba(10,12,16,.55),rgba(10,12,16,.85)),url({art("web_banner.jpg")})">
<div class="wrap"><p class="kicker">IAMPerformance · research built in public</p>
<h1>Two telescopes.<br>One sky map.</h1>
<p class="lead">We read the methylation of human DNA with the same mathematics cosmologists use to map the sky, to measure how faithfully a cell holds its information. The goal is early cancer detection from a blood or stool sample. Every test is written down before its data is read, and every result is posted here, win or lose.</p>
<p class="cta"><a class="btn" href="#now">Follow the work</a><a class="btn ghost" href="../">Read the book</a><a class="btn ghost" href="{E(FUND["url"])}">Help fund it</a></p></div></header>
<main>
<section id="now" class="wrap"><h2>Now running</h2><p class="sub">Live experiments, in plain words. Each links to its full story and its record.</p><div class="grid">{"".join(stories)}</div></section>
<section class="wrap pic"><h2>The sky and the cell</h2>
<figure><img src="{art("sky_pair.jpg")}" alt="Left: the microwave sky, temperature minus its mean. Right: one healthy neutrophil's methylation read against five others, drawn on the same kind of map."><figcaption>Left, the microwave sky. Right, one healthy white blood cell read against five others, drawn the same way. Healthy cells look like noise: nothing stands out.</figcaption></figure>
<figure><img src="{art("sky_damage.jpg")}" alt="Four maps: a healthy array; known damage spread everywhere; known damage in ten genome regions; a female array read against male arrays."><figcaption>Known damage, planted on purpose. Spread everywhere, it warms the whole map. Packed into ten regions, it shows as stripes, which the clustering reading catches when the average barely moves. A female sample read against male ones lights up the X chromosome at the bottom, as it should.</figcaption></figure></section>
<section class="wrap"><h2>Our instruments are the telescopes. The atlas is the sky map.</h2>
<p>Met-A and IAM-A are the two instruments; each is commissioned once, on its own tests. The atlas, a map of what healthy human cells look like, is the sky map they read against. Like each release of the microwave-sky maps, every atlas version is refitted whole, gets its own DOI and is checked again by the same tests on data it never saw. This is not a metaphor: it is the same mathematics applied to a different signal. Each tool, with where it stands today:</p>
<table class="tools"><thead><tr><th>The cosmology tool</th><th>What it does here</th><th>Where it stands</th></tr></thead><tbody>{tools}</tbody></table>
<p class="more">Every tool, built or not yet built: the <a href="../book/">book's sky-tools chapter</a>. The atlas itself, free to download: <a href="https://doi.org/10.5281/zenodo.23325865">DOI 10.5281/zenodo.23325865</a>.</p></section>
<div class="art-band" style="background-image:url({art("bh_banner.jpg")})" role="img" aria-label="Art from the book"></div>
<section class="wrap"><h2>Results so far</h2><p class="sub">Each in plain words, with its record. "Commissioned" means it passed every test written for it before its data was read.</p><div class="grid">{results}</div></section>
<section class="wrap"><h2>What it costs</h2>
<div class="grid costs"><div class="card"><p class="big">${E(f"{wsum:,.2f}")}</p><p>rented computing, last 7 billed days ({E(week[0][0])} to {E(week[-1][0])})</p></div>
<div class="card"><p class="big">${E(cur[1])}</p><p>rented computing, {E(cur[0])} so far</p><p class="small">{E(cur[2])}</p></div>
<div class="card"><p>Every run, its purpose, its estimate before it starts and its actual cost: <a href="{rec("Biological_Physics/MethylPhys/RUNS.md")}">the run ledger</a>. Figures come from Amazon's billing, by type.</p></div></div></section>
<section class="wrap"><h2>Check it yourself</h2><div class="grid links">
<a class="tile" href="../">The book, every number checked by a script</a><a class="tile" href="../book/">Run every check in your browser</a>
<a class="tile" href="{rec("FIELD_NOTES.md")}">Field notes: every experiment's story</a><a class="tile" href="{rec("development/METHYLPHYS_DEVELOPMENT_LOG.md")}">The development log, every result good or bad</a>
<a class="tile" href="https://github.com/hmahaffeyges/IAM-Validation">The repository: every script and record</a><a class="tile" href="{E(FUND["url"])}">The fundraiser</a></div></section>
</main>
<footer class="wrap"><p>IAMPerformance. Independent research. Generated from the repository's records on {datetime.date.today().isoformat()}; nothing on this page is typed by hand except its headings and the method description above.</p></footer>'''
    return body

CSS = """
:root{--max:1080px}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
.wrap{max-width:var(--max);margin:0 auto;padding:0 22px}
a{color:var(--link)}
.hero{background-size:cover;background-position:center;color:#F2F0EA;padding:110px 0 90px}
.hero .kicker{letter-spacing:.14em;text-transform:uppercase;font-size:13px;opacity:.85;margin:0 0 10px}
.hero h1{font-size:clamp(40px,7vw,76px);line-height:1.02;margin:0 0 18px;font-weight:800;letter-spacing:-.02em}
.hero .lead{max-width:720px;font-size:19px;opacity:.95}
.cta{margin-top:26px}.btn{display:inline-block;padding:11px 20px;margin:0 10px 10px 0;border-radius:6px;background:#8DB8F2;color:#10141a;text-decoration:none;font-weight:600}
.btn.ghost{background:transparent;color:#F2F0EA;border:1px solid rgba(242,240,234,.6)}
section{padding:56px 0 8px} h2{font-size:30px;margin:0 0 6px;letter-spacing:-.01em} .sub{color:var(--muted);margin:0 0 22px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:18px}
.card,.res{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:20px 22px}
.card h3,.res h3{margin:10px 0 8px;font-size:19px;line-height:1.3}
.chip{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:3px 9px;border-radius:999px;color:var(--st-fg)}
.st-g{background:var(--st-g)}.st-b{background:var(--st-b)}.st-a{background:var(--st-a)}.st-r{background:var(--st-r)}
.hr{margin:6px 0;font-size:15px}.latest{border-left:3px solid var(--link);padding-left:10px;margin:14px 0 8px;font-size:15px}
.latest span,.more span{color:var(--muted);font-variant-numeric:tabular-nums}.more{font-size:14px;margin:6px 0 0}
.pic figure{margin:0 0 28px}.pic img{width:100%;border-radius:8px;background:#fff}figcaption{color:var(--muted);font-size:15px;margin-top:8px}
table.tools{width:100%;border-collapse:collapse;margin:14px 0}.tools th,.tools td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--rule);vertical-align:top;font-size:15px}
.tools th{color:var(--muted);font-weight:600}
.art-band{height:220px;background-size:cover;background-position:center;margin:56px 0 0}
.big{font-size:34px;font-weight:800;margin:0;font-variant-numeric:tabular-nums}.small{color:var(--muted);font-size:13px}
.links .tile{display:block;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:18px 20px;text-decoration:none;color:var(--fg);font-weight:600}
.links .tile:hover,.card:hover{border-color:var(--link)}
footer{padding:50px 22px 60px;color:var(--muted);font-size:14px}
"""

def problems(body):
    txt = re.sub(r"<[^>]+>", " ", body); P = []
    for rx, why in ((NR.VOICE, "voice"), (NR.HYPE, "hype word")):
        m = rx.search(txt)
        if m: P.append(f"{why} '{m.group(0)}'")
    for p in re.findall(re.escape(GH) + r"([^\"#]+)", body):
        if not os.path.exists(os.path.join(REPO, p)): P.append(f"record not found: {p}")
    return P

def full(art, css_tag):
    body = page(art)
    head = f'''<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IAMPerformance: two telescopes, one sky map</title>{css_tag}<style>{CSS}</style></head><body>'''
    return head + body + "</body></html>", body

if __name__ == "__main__":
    ART = os.path.join(REPO, "website", "art", "out")
    if sys.argv[1] == "--check":
        _, b = full(lambda n: "../art/" + n, ""); P = problems(b)
        print("build_home: " + ("ok" if not P else "REFUSED: " + "; ".join(P))); sys.exit(1 if P else 0)
    if sys.argv[1] == "--standalone":
        b64 = lambda n: "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(ART, n), "rb").read()).decode()
        themes = open(os.path.join(REPO, "website", "static", "themes.css"), encoding="utf-8").read()
        doc, b = full(b64, f"<style>{themes}</style>"); P = problems(b)
        if P: raise SystemExit("build_home: " + "; ".join(P))
        open(sys.argv[2], "w", encoding="utf-8").write(doc); print("build_home: standalone written", sys.argv[2]); sys.exit(0)
    site = sys.argv[1]; os.makedirs(os.path.join(site, "preview"), exist_ok=True)
    doc, b = full(lambda n: "../art/" + n, '<link rel="stylesheet" href="../static/themes.css">'); P = problems(b)
    if P: raise SystemExit("build_home: " + "; ".join(P))
    open(os.path.join(site, "preview", "index.html"), "w", encoding="utf-8").write(doc); print("build_home: preview/index.html written")
