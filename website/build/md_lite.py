"""Small markdown renderer for the front page (2026-10-16, author: "The webpage is the interface, NOT the repo"). Renders the generated
documents (tables, headings, lists, paragraphs, bold, italics, code, links, <br>) into HTML, each ## heading as its own collapsible section.
No dependency. Relative links resolve to the repository on GitHub, or to an in-page anchor when the target document is on the page."""
import html, os, re, posixpath
GH = "https://github.com/hmahaffeyges/IAM-Validation/blob/main/"
def _inline(t, base, onpage):
    links = []
    def lk(m):
        txt, url = m.group(1), m.group(2).strip()
        if not re.match(r"[a-z]+:", url) and not url.startswith("#"):
            path, _, frag = url.partition("#"); path = posixpath.normpath(posixpath.join(base, path)) if path else ""
            url = onpage[path] if path in onpage else GH + path + (("#" + frag) if frag else "")
        links.append((txt, url)); return f"\x01{len(links) - 1}\x01"
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lk, t)
    t = re.sub(r"<(https?://[^>\s]+)>", lambda m: links.append((m.group(1), m.group(1))) or f"\x01{len(links) - 1}\x01", t)
    code = []   # after links, so a link's own text keeps its own code spans
    t = re.sub(r"`([^`]+)`", lambda m: code.append(m.group(1)) or f"\x00{len(code) - 1}\x00", t)
    t = html.escape(t, quote=False).replace("&lt;br&gt;", "<br>")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t); t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\x01(\d+)\x01", lambda m: f'<a href="{html.escape(links[int(m.group(1))][1])}">{_inline(links[int(m.group(1))][0], base, onpage)}</a>', t)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{html.escape(code[int(m.group(1))])}</code>", t)
def _cells(l): return [c.strip() for c in re.split(r"(?<!\\)\|", l.strip().strip("|"))]
def render(md, base="", onpage=None):
    """Markdown -> HTML blocks (no section wrapping)."""
    onpage = onpage or {}; out = []; L = md.split("\n"); i = 0
    while i < len(L):
        l = L[i]
        if l.startswith("|") and i + 1 < len(L) and re.match(r"^\|?\s*:?-{2,}", L[i + 1]):
            hd = _cells(l); i += 2; rows = []
            while i < len(L) and L[i].startswith("|"): rows.append(_cells(L[i])); i += 1
            out.append('<div class="tablewrap" style="overflow-x:auto"><table class="tools"><thead><tr>' + "".join(f"<th>{_inline(c, base, onpage)}</th>" for c in hd)
                       + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c, base, onpage)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>")
            continue
        if l.startswith(">"):
            q = []
            while i < len(L) and L[i].startswith(">"): q.append(L[i][1:].lstrip()); i += 1
            out.append('<blockquote style="border-left:3px solid currentColor;margin:1em 0;padding:.2em 1em;opacity:.9">' + render("\n".join(q), base, onpage) + "</blockquote>"); continue
        m = re.match(r"^(#{3,6})\s+(.*)", l)
        if m: out.append(f"<h4>{_inline(m.group(2), base, onpage)}</h4>"); i += 1; continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", l):
            tag = "ol" if re.match(r"^\s*\d+\.", l) else "ul"; items = []
            while i < len(L) and (re.match(r"^\s*([-*]|\d+\.)\s+", L[i]) or (L[i].startswith("  ") and L[i].strip() and items)):
                if re.match(r"^\s*([-*]|\d+\.)\s+", L[i]): items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", L[i]))
                else: items[-1] += " " + L[i].strip()
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x, base, onpage)}</li>" for x in items) + f"</{tag}>"); continue
        if not l.strip() or l.strip().startswith("<!--") or re.match(r"^#{1,2}\s", l): i += 1; continue
        p = [l]; i += 1
        while i < len(L) and L[i].strip() and not L[i].startswith(("|", "#")) and not re.match(r"^\s*([-*]|\d+\.)\s+", L[i]): p.append(L[i]); i += 1
        out.append(f"<p>{_inline(' '.join(x.strip() for x in p), base, onpage)}</p>")
    return "\n".join(out)
def sections(path, repo, onpage=None, skip=()):
    """A generated document as (intro html, [(title, html)]) split at its ## headings."""
    md = open(os.path.join(repo, path), encoding="utf-8").read(); base = posixpath.dirname(path)
    parts = re.split(r"(?m)^##\s+(.+)$", md); intro = re.sub(r"(?m)^#\s.*$", "", parts[0])
    S = [(parts[k].strip(), render(parts[k + 1], base, onpage)) for k in range(1, len(parts), 2) if not any(s in parts[k] for s in skip)]
    return render(intro, base, onpage), S
