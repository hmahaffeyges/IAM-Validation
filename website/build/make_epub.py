#!/usr/bin/env python3
"""Package the EPUB 3 edition (for Apple Books and other e-readers) from LaTeXML's EPUB pages.

  python3 website/build/make_epub.py <pages dir> <out.epub>

<pages dir> is what build.sh has latexmlpost write from the same LaTeXML XML as the web edition, with LaTeXML's own EPUB 3
stylesheet (one XHTML page per chapter, math as MathML, figures as the PNGs of the build copy). This script adds what an EPUB
needs around those pages and zips them:
  - the cover: the four-skies banner (website/art/out/web_banner.jpg) on a portrait page with the book's title and author;
  - the package file (content.opf: metadata, every file, the reading order) and the navigation document (nav.xhtml);
  - the build's invisible check markers are removed (they are for the web edition's check buttons);
  - two fixes for EPUB's stricter XHTML, which the web pages do not need (epubcheck reports them otherwise): an inline <span>
    that LaTeXML wraps around a paragraph or table (\resizebox, an equation inside a table cell) becomes a <div>, and the rowspan
    LaTeXML writes on such spans is dropped; a link to an anchor LaTeXML did not write (an id inside a formula) points to the
    nearest enclosing anchor that exists, i.e. the same equation.
The reading order follows LaTeXML's own rel="next" links from the title page. epubcheck runs on the result in build.sh.
"""
import sys, re, shutil, pathlib, zipfile, uuid, datetime, html, mimetypes
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "website" / "build"))
from build_site import TITLE, SUBTITLE, SUBSUB, AUTHOR, SITE_URL  # noqa: E402

TYPES = {".xhtml": "application/xhtml+xml", ".css": "text/css", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".svg": "image/svg+xml", ".gif": "image/gif", ".js": "application/javascript"}
CHECKMARK = re.compile(r'<span class="iam-checkmark">[\d ]*</span>')


TAG = re.compile(r"<(/?)([A-Za-z][\w:.-]*)([^>]*?)(/?)>")
BLOCK = {"p", "div", "table", "figure", "ul", "ol", "dl", "blockquote", "section", "pre", "h1", "h2", "h3", "h4", "h5", "h6"}


def spans_to_divs(t):
    """Rename each <span> that contains a block element (and every <span> between it and that block) to <div>."""
    stack, rename = [], set()           # stack of (name, start, end) of open tags; rename = start offsets of open tags to rename
    pairs = {}
    for m in TAG.finditer(t):
        close, name, _, selfclose = m.group(1), m.group(2), m.group(3), m.group(4)
        if close:
            while stack:
                n, st, en = stack.pop()
                if n == name:
                    pairs[st] = (m.start(), m.end())
                    break
            continue
        if selfclose:
            continue
        if name in BLOCK:
            for n, st, en in reversed(stack):
                if n != "span":
                    break
                rename.add(st)
        stack.append((name, m.start(), m.end()))
    if not rename:
        return t
    edits = []
    for st in rename:
        edits.append((st, st + len("<span"), "<div"))
        if st in pairs:
            cs, ce = pairs[st]
            edits.append((cs, ce, "</div>"))
    out, last = [], 0
    for a, b, rep_ in sorted(edits):
        out.append(t[last:a]); out.append(rep_); last = b
    out.append(t[last:])
    return "".join(out)


def font(size, bold=False):
    for f in ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",):
        for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
            p = pathlib.Path(d) / f
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + [cur]


def cover(path):
    """1600 x 2400 portrait cover: the four-skies banner across the middle, the title above, the author below."""
    W, H = 1600, 2400
    im = Image.new("RGB", (W, H), (8, 10, 18))
    ban = Image.open(REPO / "website/art/out/web_banner.jpg").convert("RGB")
    bh = int(W * ban.height / ban.width)
    ban = ban.resize((W, bh))
    im.paste(ban, (0, (H - bh) // 2))
    d = ImageDraw.Draw(im)
    y = 330
    for line in wrap(d, TITLE, font(132, True), W - 200):
        d.text((W // 2, y), line, font=font(132, True), fill=(245, 242, 235), anchor="mm"); y += 160
    y += 40
    for text, size in ((SUBTITLE, 64), (SUBSUB, 48)):
        for line in wrap(d, text, font(size), W - 240):
            d.text((W // 2, y), line, font=font(size), fill=(220, 214, 200), anchor="mm"); y += int(size * 1.35)
        y += 20
    d.text((W // 2, H - 360), AUTHOR, font=font(80, True), fill=(245, 242, 235), anchor="mm")
    im.save(path, quality=88, optimize=True)


def title_of(text):
    m = re.search(r"<title>(.*?)</title>", text, re.S)
    t = html.unescape(m.group(1)) if m else ""
    return re.sub(r"\s+", " ", t.split("‣")[0]).strip()


def main(src, out):
    src, out = pathlib.Path(src), pathlib.Path(out)
    work = out.parent / (out.stem + "_pkg")
    if work.exists():
        shutil.rmtree(work)
    oebps = work / "OEBPS"
    shutil.copytree(src, oebps, ignore=shutil.ignore_patterns("LaTeXML.cache", "*.log"))
    (work / "META-INF").mkdir()
    (work / "mimetype").write_text("application/epub+zip")
    (work / "META-INF" / "container.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>\n</container>\n')

    # pages: strip the check markers; reading order from rel="next"
    pages = {}
    for p in sorted(oebps.glob("*.xhtml")):
        t = CHECKMARK.sub("", p.read_text())
        t = re.sub(r'(<span\b[^>]*?)\s(?:rowspan|colspan)="\d+"', r"\1", t)
        t = spans_to_divs(t)
        p.write_text(t)
        nxt = re.search(r'<link rel="next" href="([^"#]+)', t)
        up = re.search(r'<link rel="up" href="([^"#]+)', t)
        pages[p.name] = dict(text=t, next=nxt.group(1) if nxt else None, up=up.group(1) if up else None, title=title_of(t))
    # links to anchors that do not exist: fall back to the nearest enclosing id (Ex262X.m7.1 -> Ex262X.m7 -> Ex262X)
    ids_on = {n: set(re.findall(r'\sid="([^"]+)"', v["text"])) for n, v in pages.items()}

    def nearest(page, frag):
        known = ids_on[page]
        while frag not in known and "." in frag:
            frag = frag.rsplit(".", 1)[0]
        return frag if frag in known else None

    for n, v in pages.items():
        def fix(m):
            target, frag = (m.group(1) or n), m.group(2)
            if target not in ids_on or frag in ids_on[target]:
                return m.group(0)
            f = nearest(target, frag)
            return f'href="{m.group(1) or ""}' + (f"#{f}" if f else "") + '"'
        t = re.sub(r'href="([^"#:]+\.xhtml)?#([^"]+)"', fix, v["text"])
        if t != v["text"]:
            v["text"] = t
            (oebps / n).write_text(t)
    order, cur = [], "index.xhtml"
    while cur and cur in pages and cur not in order:
        order.append(cur); cur = pages[cur]["next"]
    order += [n for n in sorted(pages) if n not in order]          # anything not linked (none expected) still ships

    cover(oebps / "cover.jpg")
    (oebps / "cover.xhtml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" '
        'xml:lang="en" lang="en">\n<head><title>Cover</title>\n<style>html,body{margin:0;padding:0;height:100%;text-align:center}'
        'img{max-width:100%;max-height:100%}</style></head>\n'
        f'<body epub:type="cover"><img src="cover.jpg" alt="{html.escape(TITLE)}, {html.escape(AUTHOR)}"/></body>\n</html>\n')

    # navigation: Parts hold their chapters (LaTeXML's rel="up"); everything else at the top level
    def li(n):
        return f'<a href="{n}">{html.escape(pages[n]["title"] or n)}</a>'
    top, kids = [], {}
    for n in order:
        up = pages[n]["up"]
        if up and up != "index.xhtml" and up in pages:
            kids.setdefault(up, []).append(n)
        elif n != "index.xhtml":
            top.append(n)
    items = ['<li><a href="index.xhtml">Title page</a></li>']
    for n in top:
        sub = "".join(f"<li>{li(k)}</li>" for k in kids.get(n, []))
        items.append(f"<li>{li(n)}" + (f"<ol>{sub}</ol>" if sub else "") + "</li>")
    (oebps / "nav.xhtml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" '
        'xml:lang="en" lang="en">\n<head><title>Contents</title></head>\n<body>\n<nav epub:type="toc" id="toc"><h1>Contents</h1>\n<ol>\n'
        + "\n".join(items) + '\n</ol></nav>\n<nav epub:type="landmarks" hidden="hidden"><ol>'
        '<li><a epub:type="cover" href="cover.xhtml">Cover</a></li><li><a epub:type="toc" href="nav.xhtml">Contents</a></li>'
        '<li><a epub:type="bodymatter" href="index.xhtml">Start</a></li></ol></nav>\n</body>\n</html>\n')

    # package file
    ids, manifest = {}, []
    for f in sorted(x for x in oebps.rglob("*") if x.is_file() and x.name != "content.opf"):
        rel = f.relative_to(oebps).as_posix()
        mt = TYPES.get(f.suffix.lower()) or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        iid = "i" + re.sub(r"[^A-Za-z0-9_.-]", "_", rel)
        ids[rel] = iid
        props = []
        if f.suffix == ".xhtml":
            t = f.read_text()
            if "<math" in t:
                props.append("mathml")
            if "<svg" in t:
                props.append("svg")
        if rel == "nav.xhtml":
            props.append("nav")
        if rel == "cover.jpg":
            props.append("cover-image")
        manifest.append(f'<item id="{iid}" href="{html.escape(rel)}" media-type="{mt}"' + (f' properties="{" ".join(props)}"' if props else "") + "/>")
    spine = ['<itemref idref="icover.xhtml" linear="yes"/>', '<itemref idref="inav.xhtml" linear="yes"/>'] + \
            [f'<itemref idref="{ids[n]}"/>' for n in order]
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    book_id = uuid.uuid5(uuid.NAMESPACE_URL, SITE_URL + "epub")
    (oebps / "content.opf").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="en">\n'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        f'  <dc:identifier id="bookid">urn:uuid:{book_id}</dc:identifier>\n'
        f'  <dc:title>{html.escape(TITLE)}: {html.escape(SUBTITLE)}</dc:title>\n'
        f'  <dc:creator>{html.escape(AUTHOR)}</dc:creator>\n'
        '  <dc:language>en</dc:language>\n'
        f'  <dc:source>{SITE_URL}</dc:source>\n'
        f'  <meta property="dcterms:modified">{now}</meta>\n'
        f'  <meta name="cover" content="{ids["cover.jpg"]}"/>\n'
        '</metadata>\n<manifest>\n  ' + "\n  ".join(manifest) + '\n</manifest>\n<spine>\n  ' + "\n  ".join(spine) + '\n</spine>\n</package>\n')

    # zip: mimetype first and stored, then everything else compressed
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w") as z:
        z.write(work / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
        for f in sorted(x for x in work.rglob("*") if x.is_file() and x.name != "mimetype"):
            z.write(f, f.relative_to(work).as_posix(), compress_type=zipfile.ZIP_DEFLATED)
    shutil.rmtree(work)
    print(f"epub: {out} ({len(order)} pages, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
