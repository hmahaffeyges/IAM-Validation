#!/usr/bin/env python3
"""Assemble the website from LaTeXML's chapter pages.

  python3 website/build/build_site.py <build_dir> <site_dir>

Reads <build_dir>/html/ (latexmlpost output, one page per chapter), <build_dir>/checks_index.json (prepare_sources.py) and the repository's
verify_book.py, and writes <site_dir>/:
  index.html                      front page (title, subtitle, author, the seven Parts as tiles, PDF, cite box, run every check)
  part-1/ ... part-7/index.html   one landing page per Part (summary from website/build/parts.json, chapter list, Part PDF, run all)
  book/*.html                     the chapter pages, with the site bar, the Part banner, status chips, "Run this check" buttons,
                                  Hypothesis and a "Discuss this section" link
  checks/                         verify_book.py, the files it reads (DATA_FILES), checks.json (per label: Part, heavy, committed result)
  static/, art/                   CSS, JS, banner art
The book's text is not changed: only page chrome is added around LaTeXML's output.
"""
import sys, json, re, shutil, pathlib, html, importlib.util, dataclasses
from bs4 import BeautifulSoup

REPO = pathlib.Path(__file__).resolve().parents[2]
WEB = REPO / "website"
SITE_URL = "https://hmahaffeyges.github.io/IAM-Validation/"
REPO_URL = "https://github.com/hmahaffeyges/IAM-Validation"
TITLE = "IAM's Law and Order"
SUBTITLE = "The Actualization of Reality"
SUBSUB = "The Cost of Recording It, and the Price of Maintaining It"
AUTHOR = "Heath W. Mahaffey"
MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/mml-chtml.js"
HYPOTHESIS = "https://hypothes.is/embed.js"


def load_vb():
    spec = importlib.util.spec_from_file_location("verify_book", REPO / "docs" / "book" / "verify_book.py")
    vb = importlib.util.module_from_spec(spec)
    sys.modules["verify_book"] = vb
    spec.loader.exec_module(vb)
    return vb


def chapter_parts():
    """Chapter file -> Part number (1..7), 0 front matter, 8 appendices, in main.tex order."""
    part, out, order = 0, {}, []
    for line in (REPO / "docs/book/main.tex").read_text().split("\n"):
        line = line.split("%")[0]
        if line.strip().startswith("\\part{"):
            part += 1
        if "\\appendix" in line:
            part = 8
        m = re.search(r"\\input\{([^}]+)\}", line)
        if m and m.group(1) != "preamble":
            out[m.group(1)] = part
            order.append(m.group(1))
    return out, order


def head_html(root, theme_default, title, extra=""):
    return f"""<!DOCTYPE html>
<html lang="en" data-root="{root}" data-default-theme="{theme_default}">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{root}static/site.css">
<script src="{root}static/site.js"></script>
<script defer src="{root}static/checks.js"></script>
<script async src="{HYPOTHESIS}"></script>
{extra}
</head>"""


def topbar(root, parts):
    links = " ".join(f'<a href="{root}part-{p["n"]}/">{p["roman"]}</a>' for p in parts)
    return (f'<a class="skip" href="#content">Skip to the text</a><header class="iam-top"><a class="home" href="{root}">{TITLE}</a>'
            f'<nav aria-label="Parts">{links}<a href="{root}book/">Contents</a><a href="{root}pdf/IAMs_Law_and_Order.pdf">PDF</a></nav>'
            f'<button class="iam-toggle" id="iam-toggle" type="button" aria-label="Switch light or dark">Dark</button></header>')


def front_page(site, parts):
    tiles = "\n".join(
        f'<a class="iam-tile" href="part-{p["n"]}/"><div class="art" style="background-image:url(art/{p["opening"]})" role="img" '
        f'aria-label="Part {p["roman"]} art"></div><div class="t"><b>PART {p["roman"]}</b>{html.escape(p["title"])}</div></a>' for p in parts)
    cite = (f"{AUTHOR}. {TITLE}: {SUBTITLE}. {SUBSUB}. Working edition, October 2026.\n"
            f"DOI: DOI-TO-BE-MINTED\n{SITE_URL}")
    bib = ("@book{Mahaffey2026IAM,\n  author = {Mahaffey, Heath W.},\n  title  = {IAM's Law and Order: The Actualization of Reality},\n"
           "  year   = {2026},\n  doi    = {DOI-TO-BE-MINTED},\n  url    = {" + SITE_URL + "}\n}")
    body = f"""<body class="front">
{topbar('', parts)}
<div class="iam-banner" style="background-image:url(art/web_banner.jpg)" role="img" aria-label="The four skies of the book joined">
<div class="cap"><h1>{TITLE}</h1><p>{SUBTITLE} &mdash; {SUBSUB}</p></div></div>
<main class="iam-main iam-front" id="content">
<p><b>{AUTHOR}</b></p>
<p>IAM, the Informational Actualization Model</p>
<p><a class="iam-btn" href="pdf/IAMs_Law_and_Order.pdf">Download the PDF</a><a class="iam-btn secondary" href="book/">Read online</a></p>
<h2>The seven Parts</h2>
<div class="iam-tiles">{tiles}</div>
<h2>Cite this book</h2>
<div class="iam-cite">{html.escape(cite)}</div>
<div class="iam-cite">{html.escape(bib)}</div>
<h2>Check the book</h2>
<p>Every derivation and number the book checks is one function of <a href="{REPO_URL}/blob/main/docs/book/verify_book.py">verify_book.py</a>.
They run here, in this browser. Checks that need the Planck chains, CAMB or the methylation chain show their committed result and the command that reruns them.</p>
<p><button class="iam-run iam-btn" type="button" data-part="all">Run every check</button></p>
</main></body></html>"""
    (site / "index.html").write_text(head_html("", "dark", TITLE) + body)


def part_page(site, p, parts, chapters):
    items = "\n".join(f'<li><a href="../book/{c["href"]}">{c["title"]}</a></li>' for c in chapters)
    body = f"""<body class="part-{p['n']}">
{topbar('../', parts)}
<div class="iam-banner" style="background-image:url(../art/{p['banner']})" role="img" aria-label="Part {p['roman']} banner">
<div class="cap"><h1>Part {p['roman']}: {html.escape(p['title'])}</h1></div></div>
<main class="iam-main iam-front" id="content">
<p>{html.escape(p['summary'])}</p>
<p><b>For a referee in this field:</b> {html.escape(p['referee'])}</p>
<p><a class="iam-btn" href="../pdf/part-{p['n']}.pdf">Download Part {p['roman']} (PDF)</a></p>
<h2>Chapters</h2>
<ul class="iam-chlist">{items}</ul>
<h2>Checks</h2>
<p><button class="iam-run iam-btn" type="button" data-part="{p['n']}">Run all checks in this Part</button></p>
</main></body></html>"""
    d = site / f"part-{p['n']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.html").write_text(head_html("../", "dark", f"Part {p['roman']} - {TITLE}") + body)


def checks_bundle(site, vb, index):
    d = site / "checks"
    (d / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "docs" / "book" / "verify_book.py", d / "verify_book.py")
    for rel in vb.DATA_FILES:
        dst = d / "data" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
    results = {r.label: dataclasses.asdict(r) for r in vb.run(timeout_s=120)}
    checks = {}
    for c in vb.CHECKS:
        r = results.get(c.label, {})
        checks[c.label] = dict(part=c.meta["part"], heavy=bool(c.meta["heavy"]), rerun=c.meta["rerun"], title=c.meta["title"],
                               committed=dict(passed=r.get("passed"), printed=r.get("printed"), recomputed=r.get("recomputed"), tol=r.get("tol")))
    json.dump(dict(data_files=sorted(vb.DATA_FILES), checks=checks), open(d / "checks.json", "w"))
    return results


def decorate_chapter(path, root, part, parts_by_n, index, results, discuss_cat):
    soup = BeautifulSoup(path.read_text(), "html.parser")
    htmltag = soup.find("html")
    htmltag["data-root"] = root
    htmltag["data-default-theme"] = "light"
    head = soup.find("head")
    for tag in (f'<link rel="stylesheet" href="{root}static/site.css">', f'<script src="{root}static/site.js"></script>',
                f'<script defer src="{root}static/checks.js"></script>', f'<script async src="{HYPOTHESIS}"></script>',
                f'<script async src="{MATHJAX}"></script>', '<meta name="viewport" content="width=device-width, initial-scale=1">'):
        head.append(BeautifulSoup(tag, "html.parser"))
    body = soup.find("body")
    body["class"] = body.get("class", []) + [f"part-{part}"]
    parts = [parts_by_n[k] for k in sorted(parts_by_n)]
    body.insert(0, BeautifulSoup(topbar(root, parts), "html.parser"))
    nav = soup.find(class_="ltx_page_navbar")
    if nav:
        det = soup.new_tag("details", attrs={"class": "iam-toc"})
        summ = soup.new_tag("summary"); summ.string = "Contents"
        det.append(summ)
        nav.wrap(det)
        det.extract()
        (soup.find(class_="ltx_page_main") or body).insert(0, det)
    main = soup.find(class_="ltx_page_main") or body
    main["id"] = "content"
    if part in parts_by_n:
        p = parts_by_n[part]
        main.insert(0, BeautifulSoup(f'<div class="iam-banner" style="background-image:url({root}art/{p["banner"]})" role="img" '
                                     f'aria-label="Part {p["roman"]} banner"></div>', "html.parser"))
    n_buttons = 0
    for span in soup.select(".iam-checkmark"):
        ids = [int(x) for x in span.get_text().split()]
        frag = []
        for i in ids:
            it = index[i]
            lab = html.escape(it["label"], quote=True)
            what = html.escape(f'{it["title"]} (book: {it["printed"] or "algebra"})', quote=True)
            frag.append(f'<button class="iam-run" type="button" data-label="{lab}" title="{what}">Run this check</button>')
            n_buttons += 1
        span.replace_with(BeautifulSoup(" ".join(frag), "html.parser"))
    title = soup.find("title").get_text() if soup.find("title") else ""
    q = re.sub(r"\s+", " ", title)[:90]
    content = soup.find(class_="ltx_page_content") or main
    content.append(BeautifulSoup(
        f'<p class="iam-discuss"><a href="{REPO_URL}/discussions/new?category={discuss_cat}&amp;title={html.escape(q, quote=True)}">'
        f'Discuss this section</a> (GitHub Discussions) &middot; Public, dated comments: select any text to annotate it with Hypothesis.</p>',
        "html.parser"))
    for img in soup.find_all("img"):
        if not img.get("alt"):
            fig = img.find_parent(class_="ltx_figure")
            cap = fig.find(class_="ltx_caption") if fig else None
            img["alt"] = re.sub(r"\s+", " ", cap.get_text()).strip()[:400] if cap else "Figure"
    path.write_text(str(soup))
    return n_buttons


def _letters(t, n=40):
    return re.sub(r"[^a-z]", "", t.lower())[:n]


def fix_longtable_refs(site):
    """LaTeXML numbers a longtable's caption but does not register a \\label placed in its caption row, so references to it stay
    unresolved ("LABEL:tab:..."). Match each such label to its numbered caption by the caption's opening words and link it."""
    targets = {}
    for tex in (REPO / "docs/book").rglob("*.tex"):
        t = tex.read_text(errors="replace")
        for m in re.finditer(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}\s*\\label\{(tab:[^}]+)\}", t):
            cap = re.sub(r"\\[a-zA-Z]+\*?|[{}$~]", " ", m.group(1))
            targets[m.group(2)] = _letters(cap)
    where = {}
    pages = sorted((site / "book").glob("*.html"))
    for page in pages:
        soup = BeautifulSoup(page.read_text(), "html.parser")
        for cap in soup.select(".ltx_caption"):
            tag = cap.find(class_="ltx_tag_table")
            if not tag:
                continue
            num = re.sub(r"[^0-9.]", "", tag.get_text()).strip(".")
            body = _letters(cap.get_text().replace(tag.get_text(), "", 1))
            holder = cap.find_parent(id=True)
            for lab, key in targets.items():
                if key and body.startswith(key[:30]) and lab not in where:
                    where[lab] = (page.name, holder["id"] if holder else "", num)
    n = 0
    for page in pages:
        txt = page.read_text()
        if "ltx_missing_label" not in txt:
            continue
        soup = BeautifulSoup(txt, "html.parser")
        changed = False
        for sp in soup.select("span.ltx_missing_label"):
            lab = sp.get_text().replace("LABEL:", "")
            if lab in where:
                pg, anchor, num = where[lab]
                a = soup.new_tag("a", attrs={"class": "ltx_ref", "href": (pg if pg != page.name else "") + "#" + anchor})
                a.string = num
                sp.replace_with(a)
                n += 1
                changed = True
        if changed:
            page.write_text(str(soup))
    return n


def short_links(site):
    """/go/<file name without extension> -> the file's current GitHub path, for every file in the repository (git ls-files).
    Where several files share a name, each gets /go/<parent folder>/<name> instead; if that is still shared, the full folder path
    is used, and where one folder holds the name in several formats (a figure's .pdf and .png) the file name keeps its extension.
    Files at the repository root use the repository name as their parent folder. Writes go/index.html (the full list) and go/clashes.json (every shared name and where its files went)."""
    import subprocess, collections, urllib.parse
    files = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines()
    files = [f for f in files if not f.startswith("website/_")]
    def stem(f):
        n = f.rsplit("/", 1)[-1]
        return n.rsplit(".", 1)[0] if "." in n.lstrip(".") else n
    by = collections.defaultdict(list)
    for f in files:
        by[stem(f)].append(f)
    def parent(f):
        return f.rsplit("/", 2)[-2] if "/" in f else REPO_URL.rsplit("/", 1)[-1]
    def folder(f):
        return f.rsplit("/", 1)[0] if "/" in f else REPO_URL.rsplit("/", 1)[-1]
    links, clashes = {}, {}
    for name, fs in by.items():
        if len(fs) == 1:
            links[name] = fs[0]
            continue
        placed = {}
        g1 = collections.defaultdict(list)
        for f in fs:
            g1[f"{parent(f)}/{name}"].append(f)
        for key, g in g1.items():
            if len(g) == 1:
                placed[key] = g[0]
                continue
            g2 = collections.defaultdict(list)
            for f in g:
                g2[f"{folder(f)}/{name}"].append(f)
            if all(len(h) == 1 for h in g2.values()):          # same parent-folder name, different folders: full folder path
                for key2, h in g2.items():
                    placed[key2] = h[0]
                continue
            # one folder holds this name in several formats (a figure's .pdf and .png): the file name keeps its extension
            withext = collections.Counter(f"{parent(f)}/{f.rsplit('/', 1)[-1]}" for f in g)
            for f in g:
                k = f"{parent(f)}/{f.rsplit('/', 1)[-1]}"
                placed[k if withext[k] == 1 else f"{folder(f)}/{f.rsplit('/', 1)[-1]}"] = f
        assert len(placed) == len(fs), name
        links.update(placed)
        clashes[name] = {k: v for k, v in sorted(placed.items())}
    assert len(links) == len(files)
    go = site / "go"
    for key, f in links.items():
        d = go / key
        d.mkdir(parents=True, exist_ok=True)
        url = f"{REPO_URL}/blob/main/" + urllib.parse.quote(f)
        (d / "index.html").write_text(
            f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(f)}</title>'
            f'<meta http-equiv="refresh" content="0; url={html.escape(url, quote=True)}"><link rel="canonical" href="{html.escape(url, quote=True)}">'
            f'<meta name="robots" content="noindex"></head><body><p>Redirecting to <a href="{html.escape(url, quote=True)}">{html.escape(f)}</a>.</p></body></html>')
    rows = "\n".join(f'<li><a href="{urllib.parse.quote(k)}/"><code>/go/{html.escape(k)}</code></a> &rarr; {html.escape(v)}</li>'
                     for k, v in sorted(links.items(), key=lambda kv: kv[0].lower()))
    (go / "index.html").write_text(head_html("../", "light", f"Short links - {TITLE}") +
        f'<body><main class="iam-main iam-front" id="content"><h1>Short links</h1><p>Every file of the repository at '
        f'<code>/go/&lt;file name without extension&gt;</code>; where two files share a name, <code>/go/&lt;parent folder&gt;/&lt;name&gt;</code>. '
        f'Each link opens the file on GitHub at its current path.</p><ul>{rows}</ul></main></body></html>')
    json.dump(clashes, open(go / "clashes.json", "w"), indent=1, sort_keys=True)
    return len(links), clashes


def main(build, site):
    build, site = pathlib.Path(build), pathlib.Path(site)
    pj = json.load(open(WEB / "build/parts.json"))["parts"]
    titles = {}
    for line in (REPO / "docs/book/main.tex").read_text().split("\n"):
        m = re.match(r"\\part\{(.+)\}\\label\{([^}]+)\}", line.strip())
        if m:
            titles[m.group(2)] = m.group(1).replace("\\'", "'")
    for p in pj:
        p["title"] = titles[p["label"]]
    parts_by_n = {p["n"]: p for p in pj}
    if site.exists():
        shutil.rmtree(site)
    shutil.copytree(build / "html", site / "book")
    for sub in ("static", "art/out"):
        shutil.copytree(WEB / sub, site / sub.split("/")[0], dirs_exist_ok=True)
    vb = load_vb()
    index = json.load(open(build / "checks_index.json"))
    results = checks_bundle(site, vb, index)
    chap_part, order = chapter_parts()
    # which chapter file each LaTeXML page comes from: LaTeXML names pages by their \label (splitnaming=label)
    label_of = {}
    for f in order:
        t = (REPO / "docs/book" / (f + ".tex")).read_text()
        m = re.search(r"\\(?:chapter|part)\*?\{.*?\}\s*\\label\{([^}]+)\}", t, re.S)
        if m:
            label_of[m.group(1)] = f
    pages = {}
    total = 0
    for page in sorted((site / "book").glob("*.html")):
        stem = page.stem
        f = next((label_of[l] for l in label_of if re.sub(r"[^A-Za-z0-9]", "_", l) == re.sub(r"[^A-Za-z0-9]", "_", stem)), None)
        part = chap_part.get(f, 0) if f else 0
        if stem.startswith("part_") or stem.startswith("Pt"):
            m = re.search(r"(\d+)", stem)
        total += decorate_chapter(page, "../", part, parts_by_n, index, results,
                                  f"part-{part}" if 1 <= part <= 7 else "general")
        if f and 1 <= part <= 7:
            ps = BeautifulSoup(page.read_text(), "html.parser")
            h = ps.find(class_="ltx_title_chapter")
            if h:
                for tag in h.select(".ltx_tag"):
                    tag.decompose()
                ttl = "".join(str(x) for x in h.contents).strip()
            else:
                t = ps.find("title")
                ttl = html.escape(t.get_text().split("‣")[0].strip() if t else stem)
            pages.setdefault(part, []).append((order.index(f), dict(href=page.name, title=ttl)))
    fixed = fix_longtable_refs(site)
    for p in pj:
        part_page(site, p, pj, [c for _, c in sorted(pages.get(p["n"], []))])
    front_page(site, pj)
    n_go, clashes = short_links(site)
    (site / ".nojekyll").write_text("")
    print(f"site: {site}; chapter pages: {len(list((site / 'book').glob('*.html')))}; check buttons: {total}; "
          f"checks: {len(vb.CHECKS)}; data files: {len(vb.DATA_FILES)}; long-table references linked: {fixed}; "
          f"short links: {n_go} ({len(clashes)} shared names)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
