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
PDF = "pdf/IAMs_Law_and_Order.pdf"   # the live PDF; page numbers below verified against it 2026-10-13
CH = {"astro": (48, 437, "Astro-Genetics: the cell read as a thermometer"), "landauer": (49, 446, "Landauer's cost at body temperature"),
      "gauge": (53, 464, "One gauge for a cell"), "meta": (54, 470, "Met-A: the cell's own reference on arrays"),
      "iama": (55, 474, "IAM-A: copy error on single molecules"), "cscore": (56, 478, "The C-score: where the departures sit"),
      "translation": (58, 484, "The translation principle"), "separation": (60, 492, "Component separation: finding the cells"),
      "reference": (61, 495, "The reference: purified healthy cells"), "identity": (62, 503, "Identity sites: where a cell is read"),
      "skytools": (63, 506, "Tools from the sky"), "sky": (64, 517, "The sky: what the reference cannot explain"),
      "discipline": (66, 522, "Discipline: nulls, simulations, pre-registration"), "established": (71, 537, "What is established, and what is not")}
def ch(k, label=None):
    n, pg, t = CH[k]; return f'<a href="{PDF}#page={pg}">Chapter {n}, {html.escape(label or t)}</a>'
SITE_URL = "https://hmahaffeyges.github.io/IAM-Validation/"
J = lambda p: json.load(open(os.path.join(REPO, p), encoding="utf-8"))
spec = importlib.util.spec_from_file_location("nr", os.path.join(REPO, "CANON", "now_running.py")); NR = importlib.util.module_from_spec(spec); spec.loader.exec_module(NR)
E = html.escape
STCLS = {"getting ready": "st-b", "running": "st-a", "on hold": "st-m", "result": "st-g"}
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
    sys.path.insert(0, os.path.join(REPO, "CANON")); import results_to_tex as RT
    def vals(e):
        try: v = dict(RT.compute(e))
        except Exception: v = {}
        if "n" in v: v["exp"] = round(0.05 * v["n"])
        if "k" in v and v.get("exp"): v["xchance"] = round(v["k"] / (0.05 * v["n"]))
        return v
    regs = sorted([e for e in REG["entries"] if e.get("status") != "withdrawn"], key=lambda e: e["win"]["rank"])
    def card(e):
        w, v = e["win"], vals(e); f = lambda t: E(t.format(**v))
        chip = '<span class="chip st-g">Commissioned</span>' if e["status"] == "commissioned" else '<span class="chip st-b">Passed its sealed test</span>'
        return f'''<article class="res win">{chip}<p class="stat">{f(w["stat"])}</p><h3>{f(w["head"])}</h3><p>{f(w["sub"])}</p><p class="more"><span>{E(e["date"])}</span> · <a href="{rec(e["record"])}">the record</a></p></article>'''
    results = "\n".join(card(e) for e in regs)
    losses = "\n".join(f'<li><b>{E(x["date"])}</b> {E(x["plain"])} <a href="{rec(x["record"])}">the record</a></li>' for x in REG["losses"])
    mouse = next(e for e in REG["entries"] if e["id"] == "DEV-SAM-LEVER-01"); mv = vals(mouse)
    going = "; ".join(E(s_["headline"]) for s_ in NRJ["stories"] if s_["status"] != "result")
    NJ = J("CANON/next_jobs.json"); AT = J("CANON/atlas_release.json")
    CO = J("CANON/costs_snapshot.json"); mkey = max(CO["aws_by_month"]); A = CO["aws_by_month"][mkey]
    s3c = f"${A.get('S3 storage', 0):,.2f}"
    names = {"machines": "rented computers", "machine disks and snapshots": "their disks", "S3 storage": "data storage", "data transfer": "data transfer", "other": "other Amazon services"}
    mlines = "; ".join([f"{names.get(k, k)} ${v:,.2f}" for k, v in A.items() if v >= 0.01] + [f"{f['item']} ${f['monthly_usd']:,.0f} a month ({f['note']})" for f in CO["fixed"]])
    mtot = sum(A.values()) + sum(f["monthly_usd"] for f in CO["fixed"])
    road = "\n".join(f'<li><b>{E(j["plain"])}</b><br><span class="meta">Needs {E(j["needs"])} · {E(j["estimate"])} · {E(j["stands"])} · <a href="{rec(j["record"])}">record</a></span></li>' for j in NJ["jobs"])
    later = "\n".join(f'<li>{E(x["plain"])}</li>' for x in NJ.get("later", []))
    ideas = "\n".join(f'<li>{E(x["plain"])}<br><span class="meta">Needs {E(x["needs"])} · {E(x["stands"])}</span></li>' for x in NJ.get("ideas", []))
    buys = "\n".join(f'<li><b>{E(j["estimate"])}</b> {E(j["plain"].split(".")[0])}.</li>' for j in NJ["jobs"] if j.get("cost") == "box")
    A = json.load(open(os.path.join(REPO, "CANON", "author.json"), encoding="utf-8"))
    apar = "\n".join(f"<p>{E(x)}</p>" for x in A["paragraphs"])
    amaybe = "\n".join(f'<li>{E(x)}</li>' for x in A["maybes"])
    helpw = "\n".join(f'<li><b>{E(h["ask"])}</b> {E(h["why"])} <a href="{rec(h["record"])}">record</a></li>' for h in NJ["help_wanted"])
    tools = "\n".join(f"<tr><td>{E(a)}</td><td>{E(b)}</td><td>{E(c)}</td></tr>" for a, b, c in TOOLS)
    cur = months[-1]
    body = f'''<header class="hero" style="background-image:linear-gradient(rgba(10,12,16,.55),rgba(10,12,16,.85)),url({art("web_banner.jpg")})">
<div class="wrap"><p class="kicker">IAMPerformance · research built in public</p>
<h1>Two telescopes.<br>One sky map.</h1>
<p class="lead">We read the methylation of human DNA with the same mathematics cosmologists use to map the sky, to measure how faithfully a cell holds its information. The goal is early cancer detection from a blood or stool sample. Every test is written down before its data is read, and every result is posted here, win or lose.</p>
<p class="cta"><a class="btn" href="#now">Follow the work</a><a class="btn ghost" href="read/">Read the book</a><a class="btn ghost" href="{E(FUND["url"])}">Help fund it</a></p></div></header>
<main>
<section id="now" class="wrap"><h2>Now running</h2><p class="sub">Live experiments, in plain words. Each links to its full story and its record.</p><div class="grid">{"".join(stories)}</div></section>
<section class="wrap" id="why"><h2>Why a researcher should look</h2><p class="sub">Five things anyone can check in ten minutes. None of them asks you to trust us.</p>
<ol class="why">
<li><b>A prediction committed before the data existed, then met.</b> We wrote the window the reading had to fall in ({mv["low"]:.3f} to {mv["high"]:.3f}) into a dated file before any knockout mouse was read. The mice read {mv["ratio"]:.3f}. The commit is older than the measurement, and anyone can check that. <a href="{rec(mouse["record"])}">the record</a></li>
<li><b>Our failures are listed below, not buried.</b> Four lung attempts, one instrument's commissioning on hold tonight, tests not met, a drug experiment with no result.</li>
<li><b>One command on your own laptop</b> recomputes every number in the book from the data: <code>python3 docs/book/verify_book.py</code>. <a href="book/">Or run it in your browser.</a></li>
<li><b>Scored once, and the code refuses otherwise.</b> Every test has a committed sealed rule and a scorer whose fingerprint is checked before scoring. The push gate refuses a rule written after the data was read, a number with no script behind it, or a result missing from the record. <a href="{rec("CANON/repro_check.py")}">read the gate</a></li>
<li><b>Built with an AI, and that is exactly why nothing rests on anyone's word.</b> An AI wrote most of this code and makes mistakes; ours are logged by date in <a href="{rec("Biological_Physics/MethylPhys/LESSONS.md")}">the lessons file</a>, each with the rule or check added so it cannot happen again.</li>
</ol></section>
<section class="wrap" id="wins"><h2>What we have accomplished</h2><p class="sub">Every one was a test written down before its data was read, and scored once. Each links to its record.</p><div class="grid">{results}</div>
<p class="continues">Work continues: {going}. <a href="#road">The road ahead</a>.</p></section>
<section class="wrap pic"><h2>The sky and the cell</h2>
<figure><img src="{art("sky_atlas_globes.jpg")}" alt="Two globes: the microwave sky measured by Planck, and healthy neutrophil methylation from the atlas drawn the same way."><figcaption>Two skies, drawn the same way. Left, the microwave sky as the Planck satellite measured it (ESA and the Planck Collaboration, 2018); the pale band is the plane of our own galaxy, where Planck's processing fills in what dust and gas hide. Right, healthy neutrophils from our atlas, every site in genome order on the same kind of sphere.</figcaption></figure>
<details class="fold"><summary>See more sky maps: the flat maps and zooms, six cell types, and planted damage (4 maps)</summary>
<figure><img src="{art("sky_atlas_plate.jpg")}" alt="Top: the microwave sky measured by Planck, and the atlas's healthy neutrophils with one CpG per pixel. Bottom: a 15-degree zoom of each."><figcaption>Top left, the same measured sky laid flat, smoothed half a degree as ESA's own image is. Top right, the atlas itself: healthy neutrophils, all 784,950 sites in genome order, one site per pixel on the same kind of sphere (HEALPix, the sky's grid) and at the same resolution. Genome neighbors sit next to each other, so the patches are stretches of DNA methylated alike. The atlas is mostly dark red and blue because DNA methylation comes in two sharp levels: most sites are almost fully methylated (switched off) or almost unmethylated (gene switches), while the microwave sky clusters smoothly around its average. Below, a zoom of each: the sky is soft because it is smoothed; every square of the atlas is one real site, unsmoothed, and the few black specks are pixels with no site. This is a picture of the map: no reading uses the sphere yet, and its statistics are withheld until they pass their own test.</figcaption></figure>
<figure><img src="{art("sky_atlas_cells.jpg")}" alt="Six healthy cell types from the atlas drawn on the sphere, and the atlas's uncertainty for neutrophils."><figcaption>Six of the atlas's 74 healthy cell types, each its own sky. The last map is how sure the atlas is about neutrophils at every site: dark is very sure. The brighter band at the bottom is the Y chromosome, where the atlas has the fewest donors.</figcaption></figure>
<figure><img src="{art("sky_pair.jpg")}" alt="Left: the microwave sky, temperature minus its mean. Right: one healthy neutrophil's methylation read against five others, drawn on the same kind of map."><figcaption>Left, the microwave sky. Right, one healthy white blood cell read against five others, drawn the same way. Healthy cells look like noise: nothing stands out.</figcaption></figure>
<figure><img src="{art("sky_damage.jpg")}" alt="Four maps: a healthy array; known damage spread everywhere; known damage in ten genome regions; a female array read against male arrays."><figcaption>Known damage, planted on purpose. Spread everywhere, it warms the whole map. Packed into ten regions, it shows as stripes, which the clustering reading catches when the average barely moves. A female sample read against male ones lights up the X chromosome at the bottom, as it should.</figcaption></figure>
<p class="more"><b>All of this, in full, in the book:</b> {ch("skytools")} lists every cosmology tool and what each does here; {ch("sky")} is the sky reading itself and why its statistics stay withheld; {ch("reference")} is what the atlas is and how it is built; {ch("translation")} is why the tools carry over at all; {ch("astro")} is the idea in one chapter.</p>
</details></section>
<section class="wrap"><h2>Our instruments are the telescopes. The atlas is the sky map.</h2>
<details class="fold"><summary>How it works, and every cosmology tool we use</summary>
<p>Met-A and IAM-A are the two instruments; each is commissioned once, on its own tests. The atlas, a map of what healthy human cells look like, is the sky map they read against. Like each release of the microwave-sky maps, every atlas version is refitted whole, gets its own DOI and is checked again by the same tests on data it never saw. This is not a metaphor: it is the same mathematics applied to a different signal. Each tool, with where it stands today:</p>
<table class="tools"><thead><tr><th>The cosmology tool</th><th>What it does here</th><th>Where it stands</th></tr></thead><tbody>{tools}</tbody></table>
<p class="more">Read it in full in the book: {ch("astro")} and {ch("landauer")} are the physics, {ch("translation")} is why cosmology's tools apply, {ch("skytools")} is the tool list. Sky data: Planck 2018 SMICA map from the Planck Legacy Archive (ESA and the Planck Collaboration, Planck 2018 results IV, A&A 641, A4, 2020), colors from the Planck Collaboration's own color scale. Every tool, built or not yet built: the <a href="read/">book's sky-tools chapter</a>. The atlas itself, free to download: <a href="https://doi.org/{E(AT["doi"])}">DOI {E(AT["doi"])}</a> ({AT["total_bytes"] / 1e9:.2f} GB in {AT["files"]} files, every file checked against its fingerprint).</p></details></section>
<section class="wrap" id="road"><h2>The road ahead</h2><p class="sub">Every job still in front of us, in order: what it needs and what it is expected to cost. This is why support matters.</p>
<details class="fold"><summary>See the {len(NJ["jobs"])} jobs ahead, and where a researcher could help</summary>
<ol class="road">{road}</ol>
<h3>Further down the road</h3><p class="sub">Whole areas we mean to reach, each needing the steps above first.</p>
<ul class="road">{later}</ul>
<h3>Ideas we want to test</h3><p class="sub">Not designed yet, written down so they are not lost. Each becomes a test with its rule fixed in advance, or it is dropped.</p>
<ul class="road">{ideas}</ul>
<h3>What support pays for</h3><p class="sub">The jobs above that need a rented computer, each with the estimate written before it starts; everything else runs on one laptop at no cost. The fundraiser's goal is $25,000, all of it research: rented computing, the data work, and a workstation to replace rented hours.</p>
<ul class="road">{buys}</ul>
<h3 class="help">Where a researcher could help</h3><p class="sub">The gaps public data cannot close. If you hold data like this, or can point to it, open an issue or a discussion on the repository. You do not need to introduce yourself to follow along.</p>
<ul class="road help">{helpw}</ul>
<p class="more">The same list on GitHub: <a href="{rec("ROADMAP.md")}">ROADMAP.md</a>.</p></details></section>
<div class="art-band" style="background-image:url({art("bh_banner.jpg")})" role="img" aria-label="Art from the book"></div>
<section class="wrap" id="asks"><h2>The questions a researcher asks first</h2><p class="sub">Short answers, each with the chapter that gives the full one.</p>
<details class="fold"><summary>See the nine questions and their answers</summary>
<dl class="qa">
<dt>What exactly do you measure?</dt><dd>How faithfully a cell holds its methylation pattern against a limit physics sets, not how much methylation it has. Two instruments: one reads arrays, one reads single molecules. {ch("gauge")}, {ch("meta")}, {ch("iama")}</dd>
<dt>Where does the limit come from?</dt><dd>The energy it takes to hold one bit at body temperature, which fixes the floor, with the healthy height measured in between. {ch("landauer")}</dd>
<dt>Isn't this just methylation entropy, which is decades old?</dt><dd>The entropy itself is old and we say so. What is ours is the gauge: a physics-set floor and a measured healthy reference, so a reading means something absolute rather than a comparison between groups. {ch("gauge")}</dd>
<dt>How do you handle the cell mixture in a tissue?</dt><dd>Component separation, the same job the microwave-sky maps do with dust and galaxy glow, plus sites chosen so that a change in the mixture cannot move them. {ch("separation")}, {ch("identity")}</dd>
<dt>What is the reference, and is it circular?</dt><dd>A fixed Bayesian map of 74 purified healthy cell types, from seven public sources on arrays and sequencing, with an interval at every site. No IAM physics enters the fit, and it is never refitted to the sample being read. {ch("reference")}</dd>
<dt>Why cosmology's tools, and which ones do you actually use?</dt><dd>Because the data have the same shape: faint structure under foregrounds, many instruments, one map. The book marks each of 79 tools built, partly built or not built. {ch("translation")}, {ch("skytools")}</dd>
<dt>What stops you from fitting your own answer?</dt><dd>Every test's rule is committed before its data is read, scored once, and the push gate refuses a late rule, an untraceable number or a missing result. {ch("discipline")}</dd>
<dt>What is established and what is not?</dt><dd>One instrument is commissioned on isolated white blood cells; everything else is development. The book has a table of every claim with its status. {ch("established")}</dd>
<dt>Where do the departures sit along the genome?</dt><dd>The clustering reading asks whether the leftover disorder clumps or scatters, which separates scattered wear from the block losses cancer leaves. {ch("cscore")}</dd>
</dl></details></section>
<section class="wrap" id="lost"><h2>What has failed</h2><p class="sub">Every one was written down in advance and reported on ourselves. This is why the results above mean something.</p>
<ul class="road lost">{losses}</ul>
<p class="more"><b>What these results are not.</b> Each sealed cancer test asked whether tumors leave the healthy range more often than chance across a group of patients. None is a per-patient test and none is a diagnostic: there is no sensitivity, no specificity and no detection limit yet. That test is on the road ahead, and no early-detection claim is made before it.</p>
<p class="more">Every result, good or bad, day by day: <a href="{rec("development/METHYLPHYS_DEVELOPMENT_LOG.md")}">the development log</a>.</p></section>
<section class="wrap" id="author"><h2>Who is doing this, and why</h2>
<!-- AUTHOR:BEGIN (the author's own words, from CANON/author.json; the only first-person block on the site) -->
<p class="sub">{E(A["name"])} &mdash; {E(A["role"])}</p>
{apar}
<p class="sub">{E(A["maybes_intro"])}</p>
<ul class="road maybe">{amaybe}</ul>
<p class="closer">{E(A["closer"])}</p>
<!-- AUTHOR:END -->
</section>
<section class="wrap"><h2>What it costs</h2><p class="sub">Every dollar, from Amazon's own billing and our storage, as of {E(CO["as_of"])}. Everything is paid by us and by people who choose to help.</p>
<details class="fold"><summary>See every cost: this month ${mtot:,.2f} so far</summary>
<div class="grid costs"><div class="card"><p class="big">${mtot:,.2f}</p><p>this month so far ({E(mkey)}), all costs</p><p class="small">{E(mlines)}</p></div>
<div class="card"><p class="big">{CO["s3"]["bytes"] / 1e12:.2f} TB</p><p>of public data stored for our tests ({CO["s3"]["objects"]:,} files), {E(s3c)} this month</p><p class="small">Amazon moves files we have not touched to cheaper storage on its own.</p></div>
<div class="card"><p class="big">${wsum:,.2f}</p><p>rented computing, last 7 billed days ({E(week[0][0])} to {E(week[-1][0])})</p><p class="small">Every run, its purpose, its estimate before it starts and its actual cost: <a href="{rec("Biological_Physics/MethylPhys/RUNS.md")}">the run ledger</a>.</p></div></div></details></section>
<section class="wrap"><h2>Check it yourself</h2><div class="grid links">
<a class="tile" href="read/">The book, every number checked by a script</a><a class="tile" href="book/">Run every check in your browser</a>
<a class="tile" href="{rec("FIELD_NOTES.md")}">Field notes: every experiment's story</a><a class="tile" href="{rec("development/METHYLPHYS_DEVELOPMENT_LOG.md")}">The development log, every result good or bad</a>
<a class="tile" href="https://github.com/hmahaffeyges/IAM-Validation">The repository: every script and record</a><a class="tile" href="{E(FUND["url"])}">The fundraiser</a></div>
<p class="more"><b>Cite this work.</b> The book: Mahaffey, H. W. <i>IAM's Law and Order: The Actualization of Reality</i>, version 1.0, 2026, DOI <a href="https://doi.org/10.5281/zenodo.23151068">10.5281/zenodo.23151068</a>. The atlas: IAM-Atlas v2, DOI <a href="https://doi.org/10.5281/zenodo.23325865">10.5281/zenodo.23325865</a>. Code and records carry the MIT license, in the repository. A preprint follows the first replication kit.</p>
<p class="more"><b>Questions, or think something here is wrong?</b> Open an <a href="https://github.com/hmahaffeyges/IAM-Validation/issues">issue</a> or a <a href="https://github.com/hmahaffeyges/IAM-Validation/discussions">discussion</a>. Finding an error in our numbers is the most useful thing anyone can do with this page.</p>
<p class="more"><b>For researchers:</b> everything for the methylation work, the chain, every test's sealed rule and outcome, the atlas and the operating manual, is in the <a href="https://github.com/hmahaffeyges/IAM-Validation/tree/main/Biological_Physics">Biological Physics folder</a>; start with its <a href="{rec("Biological_Physics/README.md")}">README</a>. The status of every instrument, stage by stage: <a href="{rec("Biological_Physics/MethylPhys/STATUS.md")}">STATUS</a>.</p></section>
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
.st-g{background:var(--st-g)}.st-b{background:var(--st-b)}.st-a{background:var(--st-a)}.st-r{background:var(--st-r)}.st-m{background:transparent;color:var(--muted);border:1px solid var(--muted)}
.hr{margin:6px 0;font-size:15px}.latest{border-left:3px solid var(--link);padding-left:10px;margin:14px 0 8px;font-size:15px}
.latest span,.more span{color:var(--muted);font-variant-numeric:tabular-nums}.more{font-size:14px;margin:6px 0 0}
.pic figure{margin:0 0 28px}.pic img{width:100%;border-radius:8px;background:#fff}figcaption{color:var(--muted);font-size:15px;margin-top:8px}
table.tools{width:100%;border-collapse:collapse;margin:14px 0}.tools th,.tools td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--rule);vertical-align:top;font-size:15px}
.tools th{color:var(--muted);font-weight:600}
.art-band{height:220px;background-size:cover;background-position:center;margin:56px 0 0}
.big{font-size:34px;font-weight:800;margin:0;font-variant-numeric:tabular-nums}.small{color:var(--muted);font-size:13px}
.links .tile{display:block;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:18px 20px;text-decoration:none;color:var(--fg);font-weight:600}
.links .tile:hover,.card:hover{border-color:var(--link)}
ol.road,ul.road,ol.why{padding-left:22px}dl.qa dt{font-weight:700;margin:16px 0 4px}#author p{max-width:62ch}.maybe li{margin:0 0 8px}p.closer{max-width:62ch;font-style:italic;margin-top:18px}dl.qa dd{margin:0;color:var(--muted)}ol.why li{margin:0 0 14px}.lost li{margin:0 0 14px}code{background:var(--panel);padding:2px 6px;border-radius:4px;font-size:14px}.road li{margin:0 0 14px}.road .meta{color:var(--muted);font-size:14px}h3.help{margin:30px 0 4px;font-size:22px}
details.fold{margin:6px 0 10px}details.fold>summary{cursor:pointer;display:inline-block;list-style:none;padding:12px 20px;border:1px solid var(--link);border-radius:8px;color:var(--link);font-weight:700;font-size:16px}
details.fold>summary::-webkit-details-marker{display:none}details.fold>summary::before{content:"\25B8  Click to open: "}details.fold[open]>summary::before{content:"\25BE  Close: "}details.fold[open]>summary{margin-bottom:20px}
.win .stat{font-size:38px;font-weight:800;margin:10px 0 0;line-height:1.05;font-variant-numeric:tabular-nums;color:var(--fg)}.win h3{font-size:20px}
.continues{color:var(--muted);font-size:14px;margin:22px 0 0}
footer{padding:50px 22px 60px;color:var(--muted);font-size:14px}
"""

def problems(body):
    txt = re.sub(r"<[^>]+>", " ", re.sub(r"<!-- AUTHOR:BEGIN.*?<!-- AUTHOR:END -->", "", body, flags=re.S)); P = []
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
    site = sys.argv[1]
    live = "--preview" not in sys.argv              # live since 2026-10-13 (author approved): the project front page IS the site root
    out = site if live else os.path.join(site, "preview"); os.makedirs(out, exist_ok=True)
    pre = "" if live else "../"
    doc, b = full(lambda n: pre + "art/" + n, f'<link rel="stylesheet" href="{pre}static/themes.css">'); P = problems(b)
    if P: raise SystemExit("build_home: " + "; ".join(P))
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(doc); print("build_home: " + ("index.html" if live else "preview/index.html") + " written")
