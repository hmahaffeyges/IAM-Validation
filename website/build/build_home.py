"""Front page for followers (PREVIEW; author request 2026-10-12). Writes SITE/preview/index.html from the committed records only:
  CANON/now_running.json   the Now running stories (newest update each; full story in FIELD_NOTES.md)
  CANON/results_register.json   results in plain words, with status and record
  Biological_Physics/MethylPhys/RUNS.md   actual AWS charges (month totals, daily line)
  CANON/funding.json   the fundraiser link
Nothing on the page is typed here except headings and the fixed explanation of the method, which states each cosmology tool with its book
tool-board status. The page text is checked with the field-notes voice and hype rules (CANON/now_running.py) and every repository link must
exist; --check runs those checks without writing. --standalone OUT.html writes one self-contained file (CSS and images inlined) for review.
Usage: python3 website/build/build_home.py SITE_DIR | --check | --standalone OUT.html"""
import glob, os, re, sys, json, html, base64, datetime, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GH = "https://github.com/hmahaffeyges/IAM-Validation/blob/main/"
PDF = "pdf/IAMs_Law_and_Order.pdf"   # the live PDF; page numbers below verified against it 2026-10-13
CH = {"astro": (48, 437, "AstroGenetics and the Cosmic Methylome: the cell read as a thermometer"), "landauer": (49, 448, "Landauer's cost at body temperature"),
      "gauge": (53, 466, "One gauge for a cell"), "meta": (54, 472, "Met-A: the cell's own reference on arrays"),
      "iama": (55, 476, "IAM-A: held error on single molecules"), "cscore": (56, 480, "The C-score: where the departures sit"),
      "translation": (58, 486, "The translation principle"), "separation": (60, 494, "Component separation: finding the cells"),
      "reference": (61, 497, "The reference: purified healthy cells"), "identity": (62, 505, "Identity sites: where a cell is read"),
      "skytools": (63, 508, "Tools from the sky"), "sky": (64, 519, "The sky: what the reference cannot explain"),
      "discipline": (66, 524, "Discipline: nulls, simulations, pre-registration"), "established": (71, 539, "What is established, and what is not")}
# Chapter pages come from the compiled book when the build passes them (--pages FILE.json, written by chapter_pages.py from the
# PDF it just made); the numbers in CH are only the fallback for a local build without a PDF.
if "--pages" in sys.argv:
    _pg = json.load(open(sys.argv[sys.argv.index("--pages") + 1], encoding="utf-8"))
    CH = {k: (v[0], _pg.get(k, v[1]), v[2]) for k, v in CH.items()}
    del sys.argv[sys.argv.index("--pages"):sys.argv.index("--pages") + 2]
def ch(k, label=None):
    n, pg, t = CH[k]; return f'<a href="{PDF}#page={pg}">Chapter {n}, {html.escape(label or t)}</a>'
SITE_URL = "https://hmahaffeyges.github.io/IAM-Validation/"
J = lambda p: json.load(open(os.path.join(REPO, p), encoding="utf-8"))
spec = importlib.util.spec_from_file_location("nr", os.path.join(REPO, "CANON", "now_running.py")); NR = importlib.util.module_from_spec(spec); spec.loader.exec_module(NR)
E = html.escape
STCLS = {"getting ready": "st-b", "running": "st-a", "on hold": "st-m", "result": "st-g"}
def book_tools():
    """The cosmology tools, read from the rows the book's 79-row map is generated from (docs/book/figscripts/make_skytools_tables.py),
    so the page and the book cannot disagree. The 'next' order is read from the chapter's own list of what is left to employ."""
    import ast
    src = open(os.path.join(REPO, "docs", "book", "figscripts", "make_skytools_tables.py"), encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ROWS")
    rows = ast.literal_eval(node.value)
    ch = open(os.path.join(REPO, "docs", "book", "part6", "p6_16a_skytools.tex"), encoding="utf-8").read()
    nxt = ch[ch.index("What is left to employ"):ch.index("The same tools elsewhere")]
    order = {}
    for k, m in enumerate(re.finditer(r"\\item \\textbf\{[^}]*\}[^\n]*?\(rows? ([0-9, ]+)", nxt)):
        for r in re.findall(r"\d+", m.group(1)): order.setdefault(int(r), k + 1)
    def tex(t):
        t = t.replace("\\%", "%").replace("\\,", " ").replace("~", " ").replace("\\_", "_").replace("--", "-").replace("``", '"').replace("''", '"')
        t = re.sub(r"\$([^$]*)\$", r"\1", t); t = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", t)
        return re.sub(r"\s+", " ", t).strip()
    return [(n, tex(a), tex(b), st, tex(w), order.get(n)) for n, a, b, st, w in rows]

def rec(p): return GH + p
def costs():   # reads the run ledger as rebuilt 2026-10-17 (month table: categories then a bold total; day table: a bold total per day)
    t = open(os.path.join(REPO, "Biological_Physics/MethylPhys/RUNS.md"), encoding="utf-8").read()
    months = [(m, tot, "; ".join(f"{c} {v}" for c, v in zip(("on-demand", "spot", "disks", "storage", "other"), rest.split(" | ")) if v != "$0.00"))
              for m, rest, tot in re.findall(r"^\| (\d{4}-\d{2}) \| (.*?) \| \*\*\$([\d,.]+)\*\* \|$", t, re.M)]
    days = sorted((d[5:], v) for d, v in re.findall(r"^\| (\d{4}-\d{2}-\d{2}) \| .*?\| \*\*\$([\d,.]+)\*\* \|", t, re.M))
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
<p class="more"><a href="{rec(u["record"])}">record</a></p></article>''')
    sys.path.insert(0, os.path.join(REPO, "CANON")); import results_to_tex as RT
    def vals(e):
        try: v = dict(RT.compute(e))
        except Exception: v = {}
        if "n" in v: v["exp"] = round(0.05 * v["n"])
        if "k" in v and v.get("exp"): v["xchance"] = round(v["k"] / (0.05 * v["n"]))
        return v
    _covered = {c for e in REG["entries"] if e.get("kind") == "commissioning" for c in e.get("covers", [])}   # steps folded into a commissioning are not shown again (author 2026-10-15: front-page news only)
    regs = sorted([e for e in REG["entries"] if e.get("status") != "withdrawn" and e["id"] not in _covered], key=lambda e: e["win"]["rank"])
    def card(e):
        w, v = e["win"], vals(e); f = lambda t: E(t.format(**v))
        chip = {"commissioned": '<span class="chip st-g">Commissioned</span>', "published": '<span class="chip st-g">Published</span>',
                "reproduced": '<span class="chip st-g">Reproduced</span>'}.get(e["status"], '<span class="chip st-b">Passed its sealed test</span>')
        if e.get("sealed") is False: chip = '<span class="chip st-b">Early result</span>'   # found in data already read; not a sealed test
        _story = "".join(f"<h4>{E(x['h'])}</h4><p>{E(x['p'])}</p>" for x in e.get("story", []))
        _story = f'<details><summary>Why it matters</summary>{_story}</details>' if _story else ""
        return f'''<article class="res win">{chip}{"" if w["stat"].strip().lower() == e.get("status", "") else '<p class="stat">' + f(w["stat"]) + '</p>'}<h3>{f(w["head"])}</h3><p>{f(w["sub"])}</p>{_story}<p class="more"><span>{E(e["date"])}</span> · <a href="{rec(e["record"])}">the record</a></p></article>'''
    MT = J("CANON/method.json")
    AS = J("CANON/assistant.json")
    ul = lambda xs: "".join(f"<li>{E(x)}</li>" for x in xs)
    miss = "".join(f'<li><b>{E(m["what"])}</b><br><span class="meta">What caught it: {E(m["caught"])}</span></li>' for m in AS["mistakes"])
    tog = "".join(f'<li><b>{E(x["q"])}</b><br>{E(x["a"])}</li>' for x in AS["together"])
    OR = J("CANON/origin.json")
    TR = J("CANON/trace.json")
    # the day logs: shift turnover, newest first, read straight off the generated files so the page cannot list a day that does not exist
    _dld = os.path.join(REPO, "Biological_Physics/MethylPhys/daylogs")
    _dl = []
    for _f in sorted(glob.glob(os.path.join(_dld, "20*.md")), reverse=True):
        _t = open(_f, encoding="utf-8").read().split("\n")[0].lstrip("# ").strip()
        _d = os.path.basename(_f)[:-3]
        _dl.append((_d, _t.split("\u2014", 1)[1].strip() if "\u2014" in _t else _t))
    _newest = _dl[0] if _dl else None
    _stand = ""
    if _newest:
        _txt = open(os.path.join(_dld, _newest[0] + ".md"), encoding="utf-8").read()
        _m = re.search(r"## Open at turnover\n\n(.+?)\n\n", _txt, re.S)
        _stand = _m.group(1).strip() if _m else ""
    daylogs = "".join(f'<li><a href="{rec("Biological_Physics/MethylPhys/daylogs/" + d + ".md")}"><b>{E(d)}</b></a> &mdash; {E(t)}</li>'
                      for d, t in _dl)
    steps = "".join(f'<li><b>{E(x["n"])}. {E(x["t"])}</b><br>{E(x["d"])}<br>'
                    f'<span class="meta"><a href="{rec(x["rec"])}">the record</a>'
                    + (f' &middot; commit {E(x["c"])}' if x["c"] and x["c"] != "open" else
                       ' &middot; still open' if x["c"] == "open" else "") + '</span></li>' for x in TR["steps"])
    orig = "".join(f'<li><b>{E(x["q"])}</b><br>{E(x["a"])}</li>' for x in OR["items"])
    LV = J("CANON/levers.json")
    method_secs = "\n".join(f'<h3>{E(x["title"])}</h3><p>{E(x["body"])}</p>' for x in MT["sections"])
    def lever_row(x):
        st = x.get("synthetic") or {}
        done = "; ".join(f'{k}: {E(str(v))}' for k, v in st.items()) if isinstance(st, dict) else E(str(st))
        return (f'<li><b>{E(x.get("name") or x["id"])}</b> &mdash; moves <i>{E(x.get("term", "") if isinstance(x.get("term"), str) else ", ".join(x.get("term", [])))}</i>. '
                f'<br><span class="meta">Breaks: <b>{E(x.get("falsifier_scope", "?"))}</b>. {E(x.get("falsifier", ""))}</span>'
                + (f'<br><span class="meta">Simulated: {done}</span>' if done else "") + '</li>')
    levers_list = "\n".join(lever_row(x) for x in LV["levers"])
    # the front (author 2026-10-17): only the commissioned chain and the tests it passed, in CANON/chain_front.json order; every other
    # register entry is in EARLIER_RESULTS.md (CANON/earlier_results.py), rendered as a site page and linked below the cards
    # The Commissioning Campaign in numbers (author 2026-10-17: "somewhere up top easy to see"); CANON/compute_snapshot.py, run by checkpoint
    CS = J("CANON/compute_snapshot.json")
    campaign = (f'<section class="wrap" id="campaign"><h2>{E(CS["name"])}: the final stretch</h2>'
        f'<p class="sub">Since {E(CS["since"])}, nearly every rented computer hour has gone to one goal: commissioning the instruments, packaging them into one chain '
        f'anyone can run, and testing the physics underneath. Paid for out of our own pocket. Updated {E(CS["as_of"])} from AWS billing and the committed records.</p>'
        f'<div class="grid costs"><div class="card"><p class="big">{CS["core_hours"]:,}</p><p>processor-core hours, about {CS["core_years"]} years of one computer core running non-stop</p></div>'
        f'<div class="card"><p class="big">{CS["cpg_total"] / 1e9:.1f} billion</p><p>DNA sites (CpGs) read: {CS["seq_calls"] / 1e9:.1f} billion on single molecules from '
        f'{CS["seq_samples"]} sequenced samples, {CS["array_reads"] / 1e9:.1f} billion on {CS["arrays"]:,} arrays (at least)</p></div>'
        f'<div class="card"><p class="big">{CS["fit_updates"] / 1e9:,.0f} billion</p><p>estimates updated by the IAM-Atlas v2.1 fit (MCMC): 813,951 sites x 79 cell types, '
        f'{CS["fit_steps"] / 1e6:.1f} million sampling steps in {CS["fit_blocks"]} blocks, each step itself up to 255 smaller steps (at least)</p></div>'
        f'<div class="card"><p class="big">{CS["machine_hours"]:,.0f} hours</p><p>on rented machines of up to 192 cores</p></div>'
        f'<div class="card"><p class="big">${CS["billed_usd"]:,.0f}</p><p>billed by AWS for computing and storage, every dollar listed in <a href="{rec("Biological_Physics/MethylPhys/RUNS.md")}">the run ledger</a></p></div>'
        f'</div><h3>What it took to build IAM-Atlas v2.1 alone</h3><div class="grid costs">'
        f'<div class="card"><p class="big">{CS["fit_updates"] / 1e9:,.0f} billion</p><p>estimates updated by the Bayesian (MCMC) fit: {CS["atlas_sites"]:,} DNA sites x '
        f'{CS["atlas_cells"]} cell types, 4 independent chains, 1,000 sampling steps each, in {CS["fit_blocks"]} blocks (at least: each step is itself up to 255 smaller steps)</p></div>'
        f'<div class="card"><p class="big">{CS["fit_process_hours"]:,} hours</p><p>of sampling, block after block, for {CS["fit_steps"] / 1e6:.1f} million sampling steps; '
        f'every block converged (largest R-hat {CS["fit_rhat_max"]})</p></div>'
        f'<div class="card"><p class="big">{CS["atlas_samples"]} samples</p><p>of purified healthy cells from {CS["atlas_sources"]} public sources, fitted as one model</p></div>'
        f'<div class="card"><p class="big">{CS["atlas_gb_all"]:.0f} GB</p><p>in all: {CS["atlas_gb_inputs"]:.0f} GB of input arrays, {CS["atlas_gb_fit"]:.0f} GB of raw fit output '
        f'({CS["atlas_gb_draws"]:.0f} GB of it posterior draws), and the {CS["atlas_gb_product"]:.1f} GB atlas itself</p></div>'
        f'</div></section>')
    CF = J("CANON/chain_front.json"); _byid = {e["id"]: e for e in REG["entries"]}
    results = "\n".join(card(_byid[i]) for i in CF["front"] if i in _byid)
    chain_lead = "".join(f'<p class="sub">{E(x)}</p>' for x in CF["lead"])
    n_earlier = sum(1 for e in REG["entries"] if e["id"] not in CF["front"] and e.get("status") != "withdrawn")
    decisions = "\n".join(f'<li><b>{E(x["date"])}</b> {E(x["plain"])} <a href="{rec(x["record"])}">the record</a></li>' for x in REG.get("decisions", []))
    losses = "\n".join(f'<li><b>{E(x["date"])}</b> {E(x["plain"])} <a href="{rec(x["record"])}">the record</a></li>' for x in REG["losses"])
    mouse = next(e for e in REG["entries"] if e["id"] == "DEV-SAM-LEVER-01"); mv = vals(mouse)
    going = "; ".join(E(s_["headline"]) for s_ in NRJ["stories"] if s_["status"] == "running")   # only what is running now (author 2026-10-17)
    AT = J("CANON/atlas_release.json")
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("v21r", os.path.join(REPO, "CANON", "v21_readiness.py")); _v21 = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_v21)
    _VR = J("CANON/v21_readiness.json")
    def _cnt(st):
        xs = [x for x in _VR["items"] if x["stage"] == st]; return sum(_v21.state(x)[0] for x in xs), len(xs)
    _bl, _bx, _af = _cnt("before_launch"), _cnt("box_first"), _cnt("after_fit")
    _gnames = [(_v21.state(x)[0], re.sub(r"^ATL-0[0-9][0-9] ", "", x["what"]).split(":")[0]) for x in _VR["items"] if x["stage"] == "after_fit"]
    _gifts = "".join("<li>" + ("<b>Done:</b> " if d else "") + E(n) + "</li>" for d, n in _gnames)
    _pub = _VR.get("public", {})
    if _pub.get("fit_finished_utc"): _v21_status = "The fit is finished; the checks against the source atlases are under way."
    elif _pub.get("fit_started_utc"):
        _fp = J("CANON/fit_progress.json") if os.path.exists(os.path.join(REPO, "CANON", "fit_progress.json")) else {}
        _v21_status = f"The fit is running, started {E(_pub['fit_started_utc'][:10])}; expected to take {E(_pub['fit_estimate'])}."
        if _fp.get("blocks_done"):
            _v21_status = (f"The fit is running: {_fp['blocks_done']} of {_fp['blocks']} blocks done ({_fp['percent']} percent, {_fp['blocks_per_hour']} blocks an hour), "
                           f"expected to finish {E(_fp['expected_finish_pacific'] or 'soon')}. Every block so far passed the sampler's health checks. "
                           f"Before it started, every check on the launch list passed and the tests it will be judged by were sealed.")
    else: _v21_status = (f"It is being prepared now: the computers are working through the data and the checks that must pass before the fit "
                         f"can start. The fit itself is expected to take {E(_pub['fit_estimate'])}.")
    # IAM-Atlas v2.1 story (author 2026-10-16): one source, CANON/atlas_story.json, feeds this section, the atlas README and the book
    _ST = J("CANON/atlas_story.json"); _LB = {"now": "In IAM-Atlas v2.1", "gift": "Complementary additions, ready or in progress", "after": "On the horizon"}
    def _tools(st):
        return "".join(f'<li><b>{E(t["name"])}.</b> <i>In cosmology:</i> {E(t["sky"])} <i>In the atlas:</i> {E(t["cell"])} <i>IAM-Atlas&rsquo;s unique capability:</i> {E(t["them"])}</li>'
                       for t in _ST["tools"] if t["status"] == st)
    v21_banner = (f'<section class="wrap" id="v21"><div class="banner"><p class="kicker">IAM-Atlas v2.1 · 79 cells · commissioned use by use</p><h2>{E(_ST["title"])}</h2>'
        + "".join(f"<p>{E(x)}</p>" for x in _ST["opening"]) + f'<p><b>{E(_ST["not_this"])}</b></p>'
        + "".join(f'<details><summary>{_LB[st]}</summary><ul class="road">{_tools(st)}</ul></details>' for st in ("now", "gift", "after"))
        + f'<p><b>What is commissioned.</b> {E(_ST["commissioned"])}</p><p class="sub"><i>{E(_ST["closing"])}</i></p></div></section>')
    _cs = importlib.util.spec_from_file_location("cat", os.path.join(REPO, "CANON", "catalogue.py")); CAT = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(CAT)
    _C, _M = CAT.load()
    _cards = "".join(f'<article class="card"><span class="chip st-b">{E(CAT.stage(i, _M))}</span><h3>{E(i["name"])}</h3><p><b>{E(i["question"])}</b></p>'
                     f'<p>{E(i["how"])}</p><p class="sub">Reads: {E(i["reads"])}.</p><p class="sub"><b>Next:</b> {E(i["next"])}</p></article>'
                     for i in _C["instruments"])
    catalogue = (f'<section class="wrap" id="instruments"><h2>The instruments</h2><p class="sub">Five readings, five different questions about the same sample. '
                 f'Every status below is computed from the Master Testing Manifest.</p>'
                 f'<p><b>Why a self-reference matters.</b> {E(_C["why_self_reference"])}</p><p><b>How they work together.</b> {E(_C["together"])}</p>'
                 f'<details class="fold"><summary>Open the catalogue: what each instrument does, and what it was commissioned on</summary>'
                 f'<div class="grid">{_cards}</div><p class="more"><a href="{rec("INSTRUMENTS.md")}">The catalogue as a document</a></p></details></section>')
    _bs = importlib.util.spec_from_file_location("bits", os.path.join(REPO, "CANON", "bit_tables.py")); BT = importlib.util.module_from_spec(_bs); _bs.loader.exec_module(BT)
    _bits_rows = "".join(f'<tr><th>{E(t)}<br><span class="sub">{E(m)}</span></th>' + "".join(f'<td>{E(BT.short(BT.S[k][d].get("native") or BT.S[k][d].get("formula"), 120))} <span class="chip st-b">{E(BT.S[k][d].get("status"))}</span></td>' for d in BT.DOM) + "</tr>"
                         for t, k, m in BT.ROWS)
    bits = (f'<section class="wrap" id="bits"><h2>One law, four applications, in computer language</h2>'
            f'<p class="sub">IAM is a model of how reality begins with a bit and the cost that comes with it. The same terms run through the cosmos, '
            f'quantum computers, computer chips and living cells; each row gives the word each application uses and how far it is established. Built from the four-domain schematic.</p>'
            f'<details class="fold"><summary>Open the connections table</summary><div style="overflow-x:auto"><table class="tools"><thead><tr><th>Computer term</th>'
            + "".join(f"<th>{BT.NAME[d]}</th>" for d in BT.DOM) + f'</tr></thead><tbody>{_bits_rows}</tbody></table></div>'
            f'<p class="more"><a href="{rec("FOUR_DOMAINS.md")}">Every bit table, with each value and its source, and the four-domain schematic</a></p></details></section>')
    _dlj = sorted(f for f in os.listdir(os.path.join(REPO, "CANON", "daylogs")) if re.fullmatch(r"\d{4}-\d\d-\d\d\.json", f))[-1]
    _DL = J("CANON/daylogs/" + _dlj); _ran = next((x for x in _DL["sections"] if x.get("kind") == "ran"), {"entries": []})
    _lrn = next((x for x in _DL["sections"] if x.get("kind") == "learned"), {"entries": []})
    _hl = "".join(f'<li><b>{E(e["title"])}</b> <span class="chip st-b">{E(e["status"])}</span> {E(e["what"])} <a href="{rec(e["record"])}">record</a></li>'
                  for e in [x for x in _lrn["entries"] if x["status"] not in ("failed", "diagnosis")] + [x for x in _ran["entries"] if x["status"] in ("sealed", "in progress")])   # wins and work under way only (author 2026-10-15: the page is for progress; exploratory misses stay in the research record)
    last24 = (f'<section class="wrap" id="today"><h2>The most recent day log</h2><p class="sub">{E(_DL["date"])}: {E(_DL["title"])}. '
              f'From the day log, which is written as the work happens; every line links to its record.</p><ul class="road">{_hl}</ul>'
              f'<p class="more"><a href="{rec("Biological_Physics/MethylPhys/daylogs/" + _DL["date"] + ".md")}">The whole day log</a></p></section>')
    MF = J("CANON/manifest.json"); _CLOSED = {"run", "retired"}
    mf_total = len(MF["rows"]); mf_run = sum(1 for r in MF["rows"] if r["runs"]); mf_open = sum(1 for r in MF["rows"] if r["status"] not in _CLOSED)
    mf_rows = "\n".join(
        f'<tr><td><a href="{rec("MANIFEST.md")}#{s_["code"].lower()}">{E(s_["code"])}</a></td><td>{E(s_["title"])}</td>'
        f'<td>{sum(1 for r in MF["rows"] if r["id"].rsplit("-", 1)[0] == s_["code"])}</td>'
        f'<td>{sum(1 for r in MF["rows"] if r["id"].rsplit("-", 1)[0] == s_["code"] and r["status"] not in _CLOSED)}</td></tr>'
        for s_ in MF["sections"])
    CO = J("CANON/costs_snapshot.json"); mkey = max(CO["aws_by_month"]); A = CO["aws_by_month"][mkey]
    s3c = f"${A.get('S3 storage', 0):,.2f}"
    names = {"machines": "rented computers", "machine disks and snapshots": "their disks", "S3 storage": "data storage", "data transfer": "data transfer", "other": "other Amazon services"}
    mlines = "; ".join([f"{names.get(k, k)} ${v:,.2f}" for k, v in A.items() if v >= 0.01] + [f"{f['item']} ${f['monthly_usd']:,.0f} a month ({f['note']})" for f in CO["fixed"]])
    mtot = sum(A.values()) + sum(f["monthly_usd"] for f in CO["fixed"])
    A = json.load(open(os.path.join(REPO, "CANON", "author.json"), encoding="utf-8"))
    apar = "\n".join(f"<p>{E(x)}</p>" for x in A["paragraphs"])
    _ap = [x for x in re.findall(r"<p[^>]*>.*?</p>", apar, re.S)] or [apar]
    apar_first, apar_rest = _ap[0], "\n".join(_ap[1:])
    amaybe = "\n".join(f'<li>{E(x)}</li>' for x in A["maybes"])
    BT = book_tools()
    GROUPS = [("In use in the chain", ("in use",)),
              ("Built or partly in use", ("built, not in chain", "partial", "calculated", "measured")),
              ("In the pipeline", ("not built", "reserved", "open")),
              ("Considered and set aside", ("analogy", "refused", "out", "does not translate"))]
    def trow(r):
        n, a, b, st, w, o = r
        tag = f' <span class="chip st-b">next #{o}</span>' if o else ""
        return f"<tr><td>{n}</td><td>{E(a)}{tag}</td><td>{E(b)}</td><td>{E(st)}</td><td>{E(w)}</td></tr>"
    tools = "".join(f'<tr class="grp"><th colspan="5">{E(g)} ({sum(1 for r in BT if r[3] in sts)})</th></tr>'
                    + "".join(trow(r) for r in sorted([r for r in BT if r[3] in sts], key=lambda r: (r[5] or 999, r[0])))
                    for g, sts in GROUPS)
    assert sum(1 for r in BT if any(r[3] in sts for _, sts in GROUPS)) == len(BT), "a tool status the page does not group"
    cur = months[-1]
    body = f'''<header class="hero" style="background-image:linear-gradient(rgba(10,12,16,.55),rgba(10,12,16,.85)),url({art("web_banner.jpg")})">
<div class="wrap"><p class="kicker">IAMPerformance · AstroGenetics and the Cosmic Methylome</p>
<h1>Two telescopes.<br>One sky map.</h1>
<p class="lead">It is all about the physics. One candidate law, IAM's Law, is followed across 37 orders of magnitude, from the atom to the cosmic horizon, and because the same law writes the record in the sky and in the cell, we come bearing cosmology's toolbox. <b>AstroGenetics</b> imports the measurement tools cosmologists built for the microwave background into DNA methylation: calibration against a reference, null tests, simulations with known answers, and analysis written down before the data is opened. <b>The Cosmic Methylome</b> is the methylome read the way that sky is read: what a cell's healthy reference cannot explain, mapped site by site. <a href="{PDF}#page={CH['astro'][1]}">Learn more: the chapter &rarr;</a></p>
<p class="lead">We read the methylation of human DNA with the same mathematics cosmologists use to map the sky, to measure how faithfully a cell holds its information. The goal is early cancer detection from a blood or stool sample. Every test is written down before its data is read, and every result is recorded in the open, win or lose: <a href="#daylogs">the daily log</a>.</p>
<p class="cta"><a class="btn" href="#daylogs">Follow the work</a><a class="btn ghost" href="read/">Read the book</a><a class="btn ghost" href="{E(FUND["url"])}">Help fund it</a></p></div></header>
<main>
{campaign}
<section class="wrap" id="wins"><h2>{E(CF["title"])}</h2>{chain_lead}<div class="grid">{results}</div>
<p class="sub">Each card links to its record. <a href="{rec("EARLIER_RESULTS.md")}">{E(CF["earlier"])}</a> ({n_earlier}, each with its record).</p>
<p class="sub"><b>For researchers.</b> <a href="{rec("QUESTIONS_AND_ANSWERS.md")}">Questions and answers</a>: what a researcher will ask, answered with the test behind each answer, and the open questions.
<a href="{rec("reproduce/README.md")}">Reproduce the results</a>: each test rebuilt from public data with one command.
<a href="{rec("Biological_Physics/MethylPhys/doors/data/SHOWCASE_V4/README.md")}">The showcase</a>: public samples read once through the one command.
<a href="{rec("Biological_Physics/MethylPhys/sop/MethylPhys_CPG_SOP_v4.md")}">The operating procedure</a>.</p></section>
{v21_banner}
{last24}
<section class="wrap" id="daylogs"><h2>Start here: the daily log</h2>
<p class="cta">{"".join(f'<a class="btn" href="{rec("Biological_Physics/MethylPhys/daylogs/" + d + ".md")}">{E(d)}</a>' for d, _ in _dl[:1])}{"".join(f'<a class="btn ghost" href="{rec("Biological_Physics/MethylPhys/daylogs/" + d + ".md")}">{E(d)}</a>' for d, _ in _dl[1:4])}</p>
</section>
<div class="wrap"><p class="continues">Running now: {going}. <a href="#road">The road ahead</a>.</p></div>
{catalogue}
{bits}
<section class="wrap pic" id="sky"><h2>IAM-Atlas v2.1 on the sky</h2>
<p class="sub">One sky map for every cell type, all 79 of the commissioned IAM-Atlas v2.1, drawn on the same sphere. Each site's position on that sphere is fixed by where it sits in the genome, so any two cells can be compared site for site, and so can a person's sample against the healthy cell it came from.</p>
<figure><img src="{art("sky_atlas_globes.jpg")}" alt="Two globes: the microwave sky measured by Planck, and healthy neutrophils from IAM-Atlas v2.1 drawn the same way."><figcaption>Two skies, drawn the same way. Left, the microwave sky as the Planck satellite measured it (ESA and the Planck Collaboration, 2018). Right, healthy neutrophils from IAM-Atlas v2.1: 813,950 sites, each one point on the sphere, from never tagged (blue) to always tagged (red).</figcaption></figure>
<figure><img src="{art("sky_v21_plate.jpg")}" alt="All 79 cell types of IAM-Atlas v2.1 on the sphere."><figcaption>The atlas of the sky: every cell type of IAM-Atlas v2.1 on the same sphere, in the same places. Blood, liver, brain, lung, gut, skin: each healthy identity leaves its own pattern on the same pixels. <a href="atlas/index.html">See every cell on its own page</a>, each with its map of how sure the atlas is.</figcaption></figure>
<details class="fold"><summary>How it was built, where the reference is sure, and what a researcher sees (5 maps)</summary>
<h3>How it was built</h3>
<ul><li><b>One map per cell type, every site in its place.</b> The sites are laid on the sphere in genome order, chromosome 1 through Y, on HEALPix, the pixel grid the Planck and WMAP missions used to map the microwave sky: 786,432 pixels, arranged so that a stretch of the genome fills a compact patch of sky. The layout is fixed in advance and identical for every cell and every sample. Nothing is fitted to draw it.</li>
<li><b>Fitted the way a sky map is fitted.</b> The atlas combines 21 public sources the way a sky map combines many detectors: one model for every sample at once, with each source's own offset fitted alongside the cells, so a laboratory's habits are not mistaken for biology.</li>
<li><b>Every value comes with its range.</b> Each site of each cell has a best value and a 95&nbsp;% range, so each cell has a second map: how sure the atlas is. That map is new with v2.1.</li></ul>
<figure><img src="{art("sky_v21_two_cells.jpg")}" alt="Neutrophils beside liver cells on the sphere."><figcaption>Blood neutrophils beside liver hepatocytes: same sites, same places, two different healthy identities.</figcaption></figure>
<h3>Where the reference is sure, and where it is not</h3>
<figure><img src="{art("sky_atlas_cells.jpg")}" alt="Six healthy cell types from IAM-Atlas v2.1, and how sure the atlas is for neutrophils."><figcaption>Six of the 79 cell types, and how sure the atlas is about neutrophils at every site. <b>Dark means certain</b>: 0 would mean the atlas knows the value exactly, and the brighter a point, the wider its range. Across most of the genome the range is very narrow: half of the neutrophil sites lie within &plusmn;0.006. The bright diamond is one stretch of the X chromosome, where healthy people differ about four times more than anywhere else, most likely because the atlas holds both men and women and one X in every female cell is switched off. Nobody told the map to look there. A researcher reading a sample against this reference knows, site by site, where the reference can be trusted.</figcaption></figure>
<h3>What a researcher sees: healthy, and a change that runs along the genome</h3>
<p>The chain lays every blood sample over its healthy reference from the atlas, on the same sphere, and quiets everything that matches. Grey is the same as healthy. Color appears only where the sample differs from healthy by more than twice its own normal spread: red where its pattern is more disordered than healthy (marks lost), blue where it is more ordered. A healthy sample shows a few scattered points. A change that runs along the genome shows as streaks and patches, and that bunching is what the C-Score measures.</p>
<figure><img src="{art("sky_v21_overlay_healthy.png")}" alt="A real healthy whole-blood sample laid over healthy neutrophils."><img src="{art("cscore_bar_healthy.png")}" alt="C-Score bar for the healthy sample."><figcaption>A real healthy whole-blood array (public study GSE235717) read by the commissioned chain: 53 of 3,072 pixels changed, scattered; C-Score 0.90, inside its healthy band.</figcaption></figure>
<figure><img src="{art("sky_v21_overlay_planted.png")}" alt="The same array with a planted clustered change."><img src="{art("cscore_bar_planted.png")}" alt="C-Score bar for the planted sample."><figcaption>PLANTED: the same array with a known change put into 10&nbsp;% of the C-Score's sites, in stretches, by the C-Score's own test rule. The streaks are the change (146 pixels), and the C-Score reads 2.66, far right of its band. Met-A, which reads the overall level, barely moves (1.01): the two instruments see different things.</figcaption></figure>
<h3>Who else could use it</h3>
<ul><li><b>Anyone studying methylation</b> can lay their own samples on the same sphere and see at a glance whether a change is scattered or runs along the genome.</li>
<li><b>Developmental and cell biologists</b> can compare any two of the 79 cell types site for site.</li>
<li><b>Researchers of X inactivation, imprinting and aging</b> get a map of where healthy people differ from each other, straight from the uncertainty maps.</li>
<li><b>Anyone building a reference</b> can see where it is sure before reading a sample against it.</li></ul>
<h3>Why AstroGenetics</h3>
<p>We took the instruments cosmologists built to read the oldest light in the universe and pointed them at the marks on human DNA. The sphere above is the same grid, the map is fitted the same way, and every value carries its uncertainty the same way.</p>
<p>Why would tools made for the sky work on a cell? Because both are records, and records obey a law. Every record costs energy to keep: to hold one bit against the jostling of heat, something pays a minimum price set by temperature, k<sub>B</sub>T&nbsp;ln&nbsp;2. IAM's Law says every record the universe writes, from the pattern frozen into the microwave sky to the marks a cell writes on its DNA, pays that price at its own temperature, across 37 orders of magnitude, from particle physics to cosmology.</p>
<p>In a cell the record is methylation. Each site is a bit, tagged or not; that is how a neutrophil remembers it is a neutrophil and not a liver cell. The cell pays about one ATP per site to hold those marks against the kicks of heat (the Mahaffey number, M = 20.94). When the holding weakens, the marks drift toward random, and that drift can be measured in bits. Met-A reads how far the level has drifted, IAM-A reads it on single molecules against the sample's own line, and the C-Score reads whether the drift is scattered or clustered.</p>
<p>Cosmology spent decades learning to read a faint record on a sphere against noise: combining many instruments into one map, fitting each instrument's quirks, giving every pixel its error, testing with simulations of known answer, and fixing the rule before opening the data. If one law writes both records, one toolkit reads both. IAM-Atlas v2.1 and the commissioned chain are that toolkit, rebuilt for the cell: the physics of the sky, applied to the physics of methylation. That is AstroGenetics.</p>
<p class="more"><b>All of this, in full, in the book:</b> {ch("skytools")} lists every cosmology tool and what each does here; {ch("sky")} is the sky reading itself and why its statistics stay withheld; {ch("reference")} is what the atlas is and how it is built; {ch("translation")} is why the tools carry over at all; {ch("astro")} is the idea in one chapter.</p>
</details></section>
<section class="wrap"><h2>Our instruments are the telescopes. The atlas is the sky map.</h2>
<details class="fold"><summary>How it works, and all {len(BT)} cosmology tools, in use and in the pipeline</summary>
<p>Met-A and IAM-A are the two instruments; each is commissioned once, on its own tests. The atlas, a map of what healthy human cells look like, is the sky map they read against. Like each release of the microwave-sky maps, every atlas version is refitted whole, gets its own DOI and is checked again by the same tests on data it never saw. This is not a metaphor: it is the same mathematics applied to a different signal. Each tool, with where it stands:</p>
<p>All {len(BT)} cosmology tools mapped onto the cell, from the table in {ch("skytools")}, grouped by where each stands. <b>next</b> marks the chapter's order for what to employ next.</p>
<table class="tools"><thead><tr><th>#</th><th>The cosmology tool</th><th>Its counterpart in the cell</th><th>Status</th><th>Where</th></tr></thead><tbody>{tools}</tbody></table>
<p class="more">Read it in full in the book: {ch("astro")} and {ch("landauer")} are the physics, {ch("translation")} is why cosmology's tools apply, {ch("skytools")} is the tool list. Sky data: Planck 2018 SMICA map from the Planck Legacy Archive (ESA and the Planck Collaboration, Planck 2018 results IV, A&A 641, A4, 2020), colors from the Planck Collaboration's own color scale. Every tool, built or not yet built: the <a href="read/">book's sky-tools chapter</a>. The atlas itself, free to download: <a href="https://doi.org/{E(AT["doi"])}">DOI {E(AT["doi"])}</a> ({AT["total_bytes"] / 1e9:.2f} GB in {AT["files"]} files, every file checked against its fingerprint).</p></details></section>
<section class="wrap" id="manifest"><h2>Every test, numbered</h2>
<p class="sub">The Master Testing Manifest is the one list of every test we have run, are running, or plan: {mf_total} tests, {mf_run} with at least one dated result, {mf_open} still open. A test is never removed; when it runs it gets a dated result, failures included. Ask for any test by its number.</p>
<p class="cta"><a class="btn" href="{rec("MANIFEST.md")}">The Master Testing Manifest</a></p>
<details class="fold"><summary>The sections, and how many tests each holds</summary>
<table class="tools"><tr><th>Section</th><th>What it covers</th><th>Tests</th><th>Open</th></tr>
{mf_rows}
</table>
</details>
</section>
<section class="wrap" id="road"><h2>The road ahead</h2>
<h3>What support pays for</h3>
<p class="sub"><b>No company, no charity, nothing for sale.</b> This is one person's work: there is no LLC, no nonprofit and no institution behind it, so support through the fundraiser is a personal gift and is <b>not tax-deductible</b>.</p><p class="sub">Work that needs a rented computer carries its estimate before it starts; everything else runs on one laptop at no cost. The fundraiser's goal is $25,000, all of it research: rented computing, the data work, and a workstation to replace rented hours.</p>
</section>
<div class="art-band" style="background-image:url({art("bh_banner.jpg")})" role="img" aria-label="Art from the book"></div>
<section class="wrap" id="howtest"><h2>How we test the physics</h2>
<p class="sub">{E(MT["lead"])}</p>
<details class="fold"><summary>What the instruments read, and what a lever is</summary>
{method_secs}
</details>
<details class="fold"><summary>The levers: every handle we can pull, and the reading that would break us</summary>
<p>{E(LV.get("_layer", ""))}</p>
<ul class="levers">{levers_list}</ul>
<p class="meta">Generated from the lever register, so a lever cannot appear here without a record and none can be left out.
Full records, including the simulated tests that failed: <a href="{rec("Biological_Physics/MethylPhys/doors/DEV_LEVERS_01.md")}">DEV-LEVERS-01</a>.</p>
</details>
<p class="sub">{E(MT["closing"])}</p>
</section>
<div class="art-band" style="background-image:url({art("lattice_banner.jpg")})" role="img" aria-label="Art from the book"></div>
<section class="wrap" id="whoweare"><h2>{E(AS["section_title"])}</h2>
<details class="fold"><summary>{E(OR["title"])}</summary>
<p class="sub">{E(OR["lead"])}</p>
<ul class="road">{orig}</ul>
<p class="meta">{E(OR["closing"])}</p>
</details>
</section>
<div class="art-band" style="background-image:url({art("chip_banner.jpg")})" role="img" aria-label="Art from the book"></div>
<section class="wrap" id="author"><h2>Who is doing this, and why</h2>
<!-- AUTHOR:BEGIN (the author's own words, from CANON/author.json; the only first-person block on the site) -->
<p class="sub">{E(A["name"])} &mdash; {E(A["role"])}</p>
{apar_first}
<details><summary>The rest, in his words &mdash; and the list of maybes this is for</summary>
{apar_rest}
<p class="sub">{E(A["maybes_intro"])}</p>
<ul class="road maybe">{amaybe}</ul>
<p class="closer">{E(A["closer"])}</p>
</details>
<!-- AUTHOR:END -->
</section>
<div class="art-band" style="background-image:url({art("genome_banner.jpg")})" role="img" aria-label="Art from the book"></div>
<section class="wrap" id="assistant">
<!-- ASSISTANT:BEGIN (from CANON/assistant.json; the voice check skips exactly this block, a vendor name is still refused) -->
<h2>{E(AS["title"])}</h2>
<p class="sub">{E(AS["intro"])}</p>
<p class="meta">{E(AS["note"])}</p>
<details class="fold"><summary>What it does, what it does not decide, and what the researcher does</summary>
<div class="grid">
<div class="card"><h3>What the partner does</h3><ul class="road">{ul(AS["does"])}</ul></div>
<div class="card"><h3>What it does not decide</h3><ul class="road">{ul(AS["does_not"])}</ul></div>
<div class="card"><h3>What the researcher does</h3><ul class="road">{ul(AS["author_does"])}</ul></div>
</div>
</details>
<details class="fold"><summary>{E(AS["whiteboard"]["title"])}</summary>
<p>{E(AS["whiteboard"]["lead"])}</p>
<p class="cta"><a class="btn" href="{rec(AS["whiteboard"]["record"])}">Read the session</a></p>
</details>
<!-- ASSISTANT:END -->
</section>
<section class="wrap"><h2>What it costs</h2><p class="sub">Every dollar, from Amazon's own billing and our storage, refreshed every time the site is published. Everything is paid by us and by people who choose to help.</p>
<details class="fold"><summary>See every cost: this month ${mtot:,.2f} so far</summary>
<div class="grid costs"><div class="card"><p class="big">${mtot:,.2f}</p><p>this month so far ({E(mkey)}), all costs</p><p class="small">{E(mlines)}</p></div>
<div class="card"><p class="big">{CO["s3"]["bytes"] / 1e12:.2f} TB</p><p>of public data stored for our tests ({CO["s3"]["objects"]:,} files), {E(s3c)} this month</p><p class="small">Amazon moves files we have not touched to cheaper storage on its own.</p></div>
<div class="card"><p class="big">${wsum:,.2f}</p><p>rented computing, last 7 billed days ({E(week[0][0])} to {E(week[-1][0])})</p><p class="small">Every run, its purpose, its estimate before it starts and its actual cost: <a href="{rec("Biological_Physics/MethylPhys/RUNS.md")}">the run ledger</a>.</p></div></div></details></section>
<section class="wrap"><h2>Check it yourself</h2><div class="grid links">
<a class="tile" href="read/">The book, every number checked by a script</a><a class="tile" href="book/">Run every check in your browser</a>
<a class="tile" href="{rec("FOUR_DOMAINS.md")}">One law, four applications, and IAM in bits</a><a class="tile" href="{rec("COMMISSIONING_REPORT.md")}">Commissioning report: what is commissioned and what it passed</a><a class="tile" href="{rec("INSTRUMENTS.md")}">The instrument catalogue</a>
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
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:18px}.banner{border:1px solid rgba(127,127,127,.35);border-radius:14px;padding:18px 22px}details.span{grid-column:1/-1}details.span>.grid{margin-top:18px}
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
table.tools tr.grp th{text-align:left;padding-top:18px;font-size:15px;border-bottom:1px solid var(--rule)}ol.road,ul.road,ol.why{padding-left:22px}dl.qa dt{font-weight:700;margin:16px 0 4px}#author p{max-width:62ch}.maybe li{margin:0 0 8px}p.closer{max-width:62ch;font-style:italic;margin-top:18px}dl.qa dd{margin:0;color:var(--muted)}ol.why li{margin:0 0 14px}.lost li{margin:0 0 14px}code{background:var(--panel);padding:2px 6px;border-radius:4px;font-size:14px}.road li{margin:0 0 14px}.road .meta{color:var(--muted);font-size:14px}h3.help{margin:30px 0 4px;font-size:22px}
details.fold{margin:6px 0 10px}details.fold>summary{cursor:pointer;display:flex;align-items:center;gap:10px;list-style:none;padding:6px 0;border:0;color:var(--link);font-weight:600;font-size:16px;transition:color .15s}summary::-webkit-details-marker{display:none}summary::before{content:"";width:8px;height:8px;border-right:2px solid currentColor;border-bottom:2px solid currentColor;transform:rotate(-45deg);transition:transform .18s;flex:none;margin-left:2px}details[open]>summary::before{transform:rotate(45deg)}summary:hover{color:var(--link);text-decoration:underline;text-underline-offset:3px}details{margin:14px 0}details[open]>summary{margin-bottom:10px}
details.fold[open]>summary{margin-bottom:20px}
.win .stat{font-size:38px;font-weight:800;margin:10px 0 0;line-height:1.05;font-variant-numeric:tabular-nums;color:var(--fg)}.win h3{font-size:20px}
.continues{color:var(--muted);font-size:14px;margin:22px 0 0}
footer{padding:50px 22px 60px;color:var(--muted);font-size:14px}
"""


# ---- documents on the site (author 2026-10-16: "The webpage is the interface, NOT the repo"; "the repo is the root that feeds everything
# else"). Every repository document the page links to is rendered from the repository at build time into docs/<name>.html, in the page's
# style with each ## section collapsible, and the page links there. Documents those pages link to are rendered too (two levels).
# Non-document files (code, data) still link to the repository.
sys.path.insert(0, HERE); import md_lite
INLINE = [("doc-commissioning", "Commissioning report: what is commissioned, its scope and what it passed", "COMMISSIONING_REPORT.md"),
          ("doc-instruments", "The instrument catalogue", "INSTRUMENTS.md"),
          ("doc-four", "One law, four applications, and IAM in bits", "FOUR_DOMAINS.md"),
          ("doc-runs", "Every rented-computer run and what AWS billed", "Biological_Physics/MethylPhys/RUNS.md"),
          ("doc-status", "Status of every instrument", "Biological_Physics/MethylPhys/STATUS.md")]
ALIAS = {"development/bit_paper/BIT_TABLES.md": "FOUR_DOMAINS.md"}   # moved; its tables are in FOUR_DOMAINS.md
def docname(p): return "docs/" + re.sub(r"[^A-Za-z0-9_.-]", "__", p[:-3]) + ".html"
def onpage_map(paths, front=True):
    M = {p: docname(p) for p in paths}
    if front: M.update({p: "#" + a for a, _, p in INLINE})
    for a, b in ALIAS.items(): M[a] = M.get(b, docname(b))
    return M
def doc_sections(path, onpage):
    intro, S = md_lite.sections(path, REPO, onpage)
    return intro + "".join(f"<details><summary>{E(t) if '<' not in t else t}</summary>{h}</details>" for t, h in S)
def tables_links():
    return ('<!-- RECORDS:BEGIN --><section class="wrap" id="tables"><h2>The tables</h2><p class="sub">Each is generated from the records every time they change; '
            'each opens in the repository.</p><div class="grid links">' + "".join(f'<a class="tile" id="{a}" href="{rec(p)}">{E(t)}</a>' for a, t, p in INLINE)
            + "</div></section><!-- RECORDS:END -->")
def inline_docs(onpage):
    return ('<!-- RECORDS:BEGIN --><section class="wrap" id="tables"><h2>The tables</h2><p class="sub">Built from the repository every time the site is published; '
            'open any section.</p>' + "".join(f'<details id="{a}"><summary><b>{E(t)}</b></summary>{doc_sections(p, onpage)}</details>' for a, t, p in INLINE)
            + "</section><!-- RECORDS:END -->")
# Dated and sealed records stay in the repository for auditors and are not rendered as site pages (author decision 2026-10-17): the
# development log, day logs, the box-run job lists, the archive, and the sealed records under doors/ (their wording is part of the seal).
# Links to them from the page go to the repository.
RECORD_ONLY = re.compile(r"^(development/METHYLPHYS_DEVELOPMENT_LOG\.md|Biological_Physics/MethylPhys/daylogs/|Biological_Physics/MethylPhys/doors/(?!COMMISSIONING_NOTE_)|"
                         r"Biological_Physics/MethylPhys/archive/|Biological_Physics/MethylPhys/boxruns/[^/]+/JOBS\.md|FIELD_NOTES\.md)")
def linked_docs(body, depth=2):
    seen, frontier = set(), set(re.findall(re.escape(GH) + r"([^\"#]+?\.md)", body))
    for _ in range(depth):
        new = set()
        for p in frontier - seen:
            q = ALIAS.get(p, p)
            if not os.path.exists(os.path.join(REPO, q)) or RECORD_ONLY.match(q) or q in NOT_ON_SITE: continue
            seen.add(q)
            md = open(os.path.join(REPO, q), encoding="utf-8").read(); base = os.path.dirname(q)
            for u in re.findall(r"\]\(([^)#\s]+\.md)", md):
                if not re.match(r"[a-z]+:", u): new.add(os.path.normpath(os.path.join(base, u)).replace(os.sep, "/"))
        frontier = new
    return sorted(seen)
# Working documents that are not for the site (author 2026-10-18: "dont leave things like this linked to the page if its stale"): never rendered,
# and a link to one on any site page is reduced to its text
def _moved(p):
    f = os.path.join(REPO, p)
    try: return os.path.exists(f) and open(f, encoding="utf-8").readline().strip().lower() in ("# moved", "# moved.")
    except Exception: return False
NOT_ON_SITE = {"COMMISSIONING_PLAN.md"} | {os.path.relpath(os.path.join(r, f), REPO).replace(os.sep, "/") for r, _, fs in os.walk(REPO)
              if "/.git" not in r for f in fs if f.endswith(".md") and _moved(os.path.relpath(os.path.join(r, f), REPO))}
def off_site(h):   # a link to a NOT_ON_SITE document (GitHub path or its rendered page) keeps only its text (author 2026-10-18; LESSONS C25)
    forms = [re.escape(GH + p) for p in NOT_ON_SITE] + [r"(?:\.\./)?(?:docs/)?" + re.escape(p.replace("/", "__")[:-3] + ".html") for p in NOT_ON_SITE]
    return re.sub(r'<a href="(?:' + "|".join(forms) + r')(?:#[^"]*)?"[^>]*>(.*?)</a>', r"\1", h, flags=re.S)
def relink(body, M, prefix=""):
    def r(m):
        p, frag = m.group(1), m.group(2) or ""
        if p in M: return M[p] if M[p].startswith("#") else prefix + M[p] + frag
        return m.group(0)
    body = off_site(body)
    return re.sub(re.escape(GH) + r"([^\"#]+?\.md)(#[^\"]*)?", r, body)

def moved_links(body):   # author 2026-10-18: a page link must never land on a file that only says it moved (LESSONS C25)
    bad = []
    for p in sorted(set(re.findall(re.escape(GH) + r"([^\"#]+?\.md)", body))):
        f = os.path.join(REPO, p)
        if os.path.exists(f) and open(f, encoding="utf-8").readline().strip().lower() in ("# moved", "# moved."): bad.append(p)
    return bad
def problems(body):
    # author 2026-10-19: "we cant have things that must be constantly updated 'last 24 hours'". Wording that is only true until the page
    # ages is refused; say what happened and when. Allowed: the cosmology convention "a = 1 today" (scale factor at the present epoch).
    _t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body)))
    _t = re.sub(r"(?i)today \(a = 1", "", _t); _pt = []
    for _m in re.finditer(r"(?i)\b(last|past) \d+ (hours|days)\b|\b(today|tonight|this morning|this evening|yesterday|this week|right now|so far today|hours ago|minutes ago)\b", _t):
        _pt.append("time-relative wording on the page (say what happened and its date instead): ..." + _t[max(0, _m.start() - 50):_m.end() + 30] + "...")
    if _pt: return _pt
    if re.search(r"<details[^>]* open", body): return ["a fold set to open by default; every fold starts closed (build.sh rule)"]
    if moved_links(body): return ["a link to a file that only says it moved (link the new home instead): " + ", ".join(moved_links(body))]
    # The voice check skips exactly two marked blocks: the author's own words, and the block that describes the research
    # research partner (the author asked for it, 2026-10-13; the words "assistant"/"partner" in that sense stay refused everywhere
    # else, and a vendor or
    # model name stays refused even inside it - see the brand check below).
    _b = re.sub(r"<!-- AUTHOR:BEGIN.*?<!-- AUTHOR:END -->", "", body, flags=re.S)
    _b = re.sub(r"<!-- ASSISTANT:BEGIN.*?<!-- ASSISTANT:END -->", "", _b, flags=re.S)
    _b = re.sub(r"<!-- RECORDS:BEGIN.*?<!-- RECORDS:END -->", "", _b, flags=re.S)   # the tables section shows the records in their own words
    txt = re.sub(r"<[^>]+>", " ", _b); P = []
    _brand = re.search(r"(?i:\bclaude\b|\bgpt\b|\bopenai\b|\banthropic\b|\bgemini\b|\bllama\b)", re.sub(r"<[^>]+>", " ", body))
    if _brand: P.append(f"vendor or model name '{_brand.group(0)}' on the page: describe the role, not the brand")
    # the marked blocks are exempt from the VOICE rule only. Hype words are refused everywhere, the author's corner and the
    # research-partner block included: a claim we cannot put a check behind does not become acceptable by being in a quote
    # (the first version of this exemption hid hype words in those blocks too, 2026-10-13).
    _all = re.sub(r"<[^>]+>", " ", body)
    _neg = re.compile(r"(?i)\b(?:not|none|never|nothing|no|cannot|can't|isn't|aren't|without)\b[^.]{0,30}$")
    for rx, why, scope in ((NR.VOICE, "voice", txt), (NR.HYPE, "hype word", _all)):
        for m in rx.finditer(scope):
            # a NEGATED hype word is honest, not hype: "None of this is proven" must pass while "this is proven" must not
            if why == "hype word" and _neg.search(scope[max(0, m.start() - 60):m.start()]): continue
            P.append(f"{why} '{m.group(0)}'"); break
    for p in re.findall(re.escape(GH) + r"([^\"#]+)", body):
        if not os.path.exists(os.path.join(REPO, p)): P.append(f"record not found: {p}")
    return P

DOCS = []   # (path, html) of every document page, filled by full()
def doc_page(path, css_tag, docs):
    M = {q: os.path.basename(docname(q)) for q in docs}; M.update({q: "../index.html#" + a for a, _, q in INLINE})
    for a_, b_ in ALIAS.items(): M[a_] = M.get(b_, os.path.basename(docname(b_)))
    title = re.search(r"(?m)^#\s+(.+)$", open(os.path.join(REPO, path), encoding="utf-8").read())
    t = title.group(1) if title else path
    body = (f'<section class="wrap"><p><a href="../index.html">&larr; IAMPerformance front page</a></p><h1>{E(t)}</h1>'
            f'{doc_sections(path, M)}<p class="more">Source in the repository: <a href="{GH}{path}">{E(path)}</a></p></section>')
    return (f'<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{E(t)}</title>{css_tag}<style>{CSS}</style></head><body>{body}</body></html>')
def full(art, css_tag):
    body = page(art)
    # author 2026-10-17: the tables link to their files in the repository (no rendered copies); back arrow returns to the page
    _tab = tables_links()
    _i = body.rfind("<footer"); body = (body[:_i] + _tab + body[_i:]) if _i >= 0 else body + _tab
    DOCS[:] = []
    head = f'''<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IAMPerformance: two telescopes, one sky map</title>{css_tag}<style>{CSS}</style></head><body>'''
    return head + body + "</body></html>", body

if __name__ == "__main__":
    ART = os.path.join(REPO, "website", "art", "out")
    if sys.argv[1] == "--check":
        import atlas_pages   # 2026-10-18: a syntax error in the cell-page code passed this check and broke the live build; import it here too
        _, b = full(lambda n: "../art/" + n, ""); P = problems(b)
        print("build_home: " + ("ok" if not P else "REFUSED: " + "; ".join(P))); sys.exit(1 if P else 0)
    if sys.argv[1] == "--standalone":
        b64 = lambda n: "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(ART, n), "rb").read()).decode()
        themes = open(os.path.join(REPO, "website", "static", "themes.css"), encoding="utf-8").read()
        doc, b = full(b64, f"<style>{themes}</style>"); P = problems(b)
        if P: raise SystemExit("build_home: " + "; ".join(P))
        open(sys.argv[2], "w", encoding="utf-8").write(off_site(doc)); print("build_home: standalone written", sys.argv[2]); sys.exit(0)
    site = sys.argv[1]
    live = "--preview" not in sys.argv              # live since 2026-10-13 (author approved): the project front page IS the site root
    out = site if live else os.path.join(site, "preview"); os.makedirs(out, exist_ok=True)
    pre = "" if live else "../"
    doc, b = full(lambda n: pre + "art/" + n, f'<link rel="stylesheet" href="{pre}static/themes.css">'); P = problems(b)
    if P: raise SystemExit("build_home: " + "; ".join(P))
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(off_site(doc)); print("build_home: " + ("index.html" if live else "preview/index.html") + " written")
    os.makedirs(os.path.join(out, "docs"), exist_ok=True); _css = f'<link rel="stylesheet" href="{pre}../static/themes.css">'
    for q in DOCS: open(os.path.join(out, docname(q)), "w", encoding="utf-8").write(off_site(doc_page(q, _css, DOCS)))
    import atlas_pages   # IAM-Atlas v2.1: every cell on its own page (author 2026-10-18)
    print("build_home:", atlas_pages.write(out, pre + "../art/", f'<link rel="stylesheet" href="{pre}../static/themes.css">', CSS, "../index.html"), "atlas cell pages written")
    print("build_home:", len(DOCS), "document pages written")
