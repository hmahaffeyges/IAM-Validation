"""Appendix E, the formula sheet: every displayed equation of the seven parts, in book order.

Run from any directory:
    python docs/book/figscripts/make_app_E.py                 # writes docs/book/appendices/app_E_formulas.tex
    python docs/book/figscripts/make_app_E.py --out FILE      # writes FILE instead (the sheet is not touched)
    python docs/book/figscripts/make_app_E.py --check         # writes nothing; exit 1 with a diff if the sheet differs
Reads docs/book/main.tex for the order of parts and chapters (each \\input followed recursively), and
docs/book/figscripts/app_E_overrides.json for what the chapters do not hold: the one-line descriptions and the hand edits.
The script does not read the sheet it writes (except to compare, with --check).

Rules:
  * every display (equation, align, gather, multline, eqnarray and their starred forms, \\[ \\]) is an entry, in book order;
  * the body is copied exactly, with \\label, \\nonumber, \\notag and \\tag removed; multi-line environments are written as
    aligned (align, eqnarray, flalign) or gathered (gather, multline) inside the entry's own equation*;
  * status: the label the chapter attaches directly after the display (with its parenthesis, if any); otherwise the first label
    later in the same paragraph (up to the next blank line, past any further displays); otherwise the last label earlier in the
    same paragraph (back to the previous display); a paragraph label is written \\derived{}. The statuses written by hand for
    the sheet (the more specific wordings of the earlier selection, and labels set by hand) are overrides;
  * description: from the overrides file (written for the earlier selection); otherwise none;
  * the same relation in more than one place is listed each time, with pointers to its other entries.

Overrides (app_E_overrides.json) are keyed by chapter label and the display's first equation label, or, for a display with
no label, by chapter label and the display's body with white space removed. Each override records what the chapter gave when it
was written ("chapter"); if the chapter now gives something else, the script warns and uses the chapter, so an override never
hides a chapter change. An override whose key no longer matches any display is reported too.
"""
import re, sys, json, collections, difflib, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent
OUT = BOOK / "appendices" / "app_E_formulas.tex"
OVR = HERE / "app_E_overrides.json"
STAT = ("derived", "calc", "calibrated", "measured", "observed", "fitted", "conjecture", "prediction", "openprob", "interp", "analogy")
ENVS = ("equation", "align", "gather", "multline", "eqnarray", "flalign")
WRAP = {"align": "aligned", "eqnarray": "aligned", "flalign": "aligned", "gather": "gathered", "multline": "gathered"}
KINDS = ("omit", "body", "status", "intext", "desc")


def strip_comments(t):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in t.split("\n"))


def group(s, i, o="{", c="}"):
    """s[i] == o; return (content, index after the closing bracket)."""
    assert s[i] == o, (s[i:i + 40])
    d, j = 0, i
    while j < len(s):
        ch = s[j]
        if ch == "\\":
            j += 2; continue
        if ch == o:
            d += 1
        elif ch == c:
            d -= 1
            if d == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced: " + s[i:i + 80])


def expand(rel, seen=()):
    p = BOOK / (rel if rel.endswith(".tex") else rel + ".tex")
    t = strip_comments(p.read_text())
    out, k = [], 0
    for m in re.finditer(r"\\input\{([^}]+)\}", t):
        out.append(t[k:m.start()]); out.append(expand(m.group(1), seen + (rel,))); k = m.end()
    out.append(t[k:])
    return "".join(out)


def norm(b):
    b = re.sub(r"\\label\{[^}]*\}|\\nonumber|\\notag", "", b)
    b = re.sub(r"\\begin\{(aligned|gathered)\}|\\end\{(aligned|gathered)\}", "", b)
    return re.sub(r"\\[,;:! ]|~|\s|\\left|\\right|&|\\\\|[.,;]$", "", b)


DISP = re.compile(r"\\begin\{(" + "|".join(ENVS) + r")(\*?)\}|(?<!\\)\\\[")
HEAD = re.compile(r"\\(chapter|section|subsection)\*?(\[[^\]]*\])?\{")
STATRX = re.compile(r"\\(" + "|".join(STAT) + r")\b(\{\})?")


def attached_status(t, k):
    """A status label directly after a display ends at t[k]: skip punctuation and space, then the macro and its parenthesis."""
    j = k
    while j < len(t) and t[j] in " \t\n.,;:":
        if t[j] == "\n" and t[j:j + 2] == "\n\n":
            return None
        j += 1
    m = STATRX.match(t, j)
    if not m:
        return None
    s, j = "\\" + m.group(1) + "{}", m.end()
    while j < len(t) and t[j] == " ": j += 1
    if t.startswith("\\ ", j): j += 2
    while j < len(t) and t[j] == " ": j += 1
    if j < len(t) and t[j] == "(":
        par, _ = group(t, j, "(", ")")
        if len(par) < 200 and "\n\n" not in par:
            s += " (" + " ".join(par.split()) + ")"
    return s


def paragraph_status(t, k):
    """The first label later in the paragraph (up to the next blank line; later displays of the paragraph included)."""
    end = t.find("\n\n", k); end = len(t) if end < 0 else end
    m = STATRX.search(t, k, end)
    return ("\\" + m.group(1) + "{}") if m else None


def before_status(t, start, prev_end):
    """The last label earlier in the paragraph (back to its start or the end of the previous display)."""
    b = max(t.rfind("\n\n", 0, start), prev_end, 0)
    ms = list(STATRX.finditer(t, b, start))
    return ("\\" + ms[-1].group(1) + "{}") if ms else None


def read_book():
    main = strip_comments((BOOK / "main.tex").read_text())
    main = main.split("\\mainmatter", 1)[1].split("\\appendix", 1)[0]
    parts = []
    for m in re.finditer(r"\\part\{([^}]*)\}\\label\{([^}]+)\}|\\input\{([^}]+)\}", main):
        if m.group(1):
            parts.append(dict(title=m.group(1), label=m.group(2), chapters=[]))
        else:
            parts[-1]["chapters"].append(m.group(3))
    for P in parts:
        P["chaps"] = []
        for rel in P["chapters"]:
            t = expand(rel)
            heads = []
            for m in HEAD.finditer(t):
                title, e = group(t, m.end() - 1)
                heads.append((m.start(), m.group(1), title, e))
            chap = [h for h in heads if h[1] == "chapter"]
            if not chap:
                continue
            lab = re.match(r"\s*\\label\{([^}]+)\}", t[chap[0][3]:])
            C = dict(file=rel, title=" ".join(chap[0][2].split()), label=lab.group(1) if lab else None, entries=[])
            P["chaps"].append(C)
            prev_end = 0
            for m in DISP.finditer(t):
                if m.group(1):
                    env, star = m.group(1), m.group(2)
                    endtag = "\\end{" + env + star + "}"
                    b0 = m.end(); b1 = t.index(endtag, b0); k = b1 + len(endtag)
                else:
                    env, star = "bracket", "*"
                    b0 = m.end(); b1 = t.index("\\]", b0); k = b1 + 2
                raw = t[b0:b1]
                labels = re.findall(r"\\label\{([^}]+)\}", raw)
                body = re.sub(r"\\label\{[^}]*\}|\\nonumber|\\notag", "", raw)
                body = re.sub(r"\\tag\*?\{[^}]*\}", "", body)
                body = "\n".join(l.rstrip() for l in body.strip().split("\n") if l.strip())
                body = re.sub(r"\\\\\s*$", "", body).strip()
                if env in WRAP:
                    body = "\\begin{" + WRAP[env] + "}" + body + "\\end{" + WRAP[env] + "}"
                sec = [h for h in heads if h[0] < m.start() and h[1] == "section"]
                sec = " ".join(sec[-1][2].split()) if sec and sec[-1][0] > chap[0][0] else None
                numbered = star == "" and env != "bracket"
                E = dict(body=body, labels=labels, numbered=numbered, chapter=C, section=sec)
                E["attached"] = attached_status(t, k)
                E["para"] = paragraph_status(t, k)
                E["before"] = before_status(t, m.start(), prev_end)
                E["status"] = E["attached"] or E["para"] or E["before"] or "no status label attached in the chapter"
                prev_end = k
                C["entries"].append(E)
    # override keys: chapter|first label, or chapter|body:<body without white space>; #2, #3 for a repeat in one chapter
    seen = collections.Counter()
    for P in parts:
        for C in P["chaps"]:
            for E in C["entries"]:
                k = f"{C['label']}|" + (E["labels"][0] if E["labels"] else "body:" + re.sub(r"\s", "", E["body"]))
                seen[k] += 1
                E["key"] = k + (f"#{seen[k]}" if seen[k] > 1 else "")
    return parts


def intext(E):
    C = E["chapter"]; s = []
    if E["numbered"] and E["labels"]:
        s.append(("Eq.~" if len(E["labels"]) == 1 else "Eqs.~") + ", ".join(f"\\eqref{{{l}}}" for l in E["labels"]))
    s.append(f"Chapter~\\ref{{{C['label']}}}" + (f", section ``{E['section']}''" if E["section"] else ""))
    return "; ".join(s)


def build(warn, ovr=None):
    """Return (sheet text, summary, chapters with no display). warn(msg) is called for every override that does not apply."""
    parts = read_book()
    ovr = json.loads(OVR.read_text()) if ovr is None else ovr
    used, counts = set(), collections.Counter()

    def take(kind, E, field):
        """Apply override 'kind' to E[field] if its recorded chapter value still holds; otherwise warn and keep the chapter."""
        o = ovr[kind].get(E["key"])
        if o is None:
            return
        used.add((kind, E["key"]))
        if o["chapter"] != E[field]:
            warn(f"override {kind} {E['key'][:90]!r}: the chapter has changed since the override was written; using the chapter.\n"
                 f"    override written for: {o['chapter']!r}\n    chapter now gives:    {E[field]!r}")
            return
        E[field] = o["sheet"]; counts[kind] += 1

    entries, chap_no_disp = [], []
    for P in parts:
        for C in P["chaps"]:
            keep = []
            for E in C["entries"]:
                o = ovr["omit"].get(E["key"])
                if o is not None:
                    used.add(("omit", E["key"]))
                    if o["chapter"] == E["body"]:
                        counts["omit"] += 1; continue
                    warn(f"override omit {E['key'][:90]!r}: the display has changed since the override was written; it is listed.")
                keep.append(E)
            C["entries"] = keep
            entries += keep
            if not keep:
                chap_no_disp.append(C)
    same = collections.defaultdict(list)
    for i, E in enumerate(entries):
        E["n"] = i
        same[norm(E["body"])].append(i)   # pointers follow the chapter bodies
    for E in entries:
        others = [n for n in same[norm(E["body"])] if n != E["n"]]
        E["intext"] = intext(E)
        take("body", E, "body")
        take("status", E, "status")
        if others:
            E["intext"] += "; see also " + ", ".join(f"(\\ref{{app:fs:{n}}})" for n in others)
        take("intext", E, "intext")
        E["desc"] = ovr["desc"].get(E["key"], {}).get("sheet", "")
        if E["key"] in ovr["desc"]:
            used.add(("desc", E["key"])); counts["desc"] += 1
    for kind in KINDS:
        for k in ovr[kind]:
            if (kind, k) not in used and not k.startswith("_"):
                warn(f"override {kind} {k[:90]!r} matches no display (the chapter changed or the display moved); not applied.")

    N = len(entries)
    nodisp = ", ".join(f"\\ref{{{C['label']}}}" for C in chap_no_disp)
    L = [r"% Generated by docs/book/figscripts/make_app_E.py from the chapters named in docs/book/main.tex. Do not edit by hand.",
         r"% Every displayed equation, in book order; bodies copied from the chapters with \label, \nonumber, \notag and \tag removed.",
         r"\chapter{Formula sheet}\label{app:formulas}",
         r"\newcounter{iamfs}\renewcommand{\theiamfs}{\thechapter.\arabic{iamfs}}",
         r"\newcommand{\iamfsentry}[5]{\refstepcounter{iamfs}\label{app:fs:#1}%",
         r"  \begin{equation*}#2\tag{\theiamfs}\end{equation*}\nopagebreak%",
         r"  \noindent\parbox{\linewidth}{\small\setlength{\baselineskip}{1.15em}\setlength{\parskip}{2pt}#3\par\emph{Status:} #4.\quad\emph{In the text:} #5.}\par\medskip}",
         f"This appendix collects every displayed equation of the seven parts ({N} displays), in the order in which they appear, grouped by part and",
         r"chapter. Each entry gives the equation as printed in the chapter, its status, and where it appears in the text; the entries that were in the",
         r"earlier selection of this sheet keep their line on what the equation is. The numbers in parentheses belong to this appendix; the chapter's",
         r"own equation number is given where the chapter numbers the display. The status is the label the chapter attaches to the display. Where the",
         r"chapter attaches none, the status written for the earlier selection is used, then the first label later in the same paragraph; an entry",
         r"with none of these says so. A relation that appears in more than one place is listed each time, with pointers to its other entries.",
         r"Inline formulas are not collected; symbols and units are in Appendix~\ref{app:notation} and terms in Appendix~\ref{app:glossary}.",
         f"Chapters with no displayed equation, and so no entry here: {nodisp}; the front matter has none either.", ""]
    for pi, P in enumerate(parts, 1):
        if not any(C["entries"] for C in P["chaps"]):
            continue
        L.append(f"\\section{{\\texorpdfstring{{Part~\\ref{{{P['label']}}}}}{{Part {pi}}}: {P['title']}}}")
        for C in P["chaps"]:
            if not C["entries"]:
                continue
            L.append(f"\\subsection*{{Chapter~\\ref{{{C['label']}}}: {C['title']}}}")
            for E in C["entries"]:
                L.append(f"\\iamfsentry{{{E['n']}}}{{{E['body']}}}{{{E['desc']}}}{{{E['status']}}}{{{E['intext']}}}")
    L.append("")
    txt = "\n".join(L)
    assert txt.count("{") - txt.count("\\{") == txt.count("}") - txt.count("\\}"), "unbalanced braces"
    assert not re.search(r"\n\s*\n", "\n".join(l for l in L if l.startswith("\\iamfsentry"))), "blank line inside an entry"
    info = (f"{N} entries from {sum(len(P['chaps']) for P in parts)} chapters; overrides applied: "
            + ", ".join(f"{k} {counts[k]}" for k in KINDS))
    return txt, info, chap_no_disp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="where to write the sheet (default: %(default)s)")
    ap.add_argument("--check", action="store_true", help="write nothing; compare with the committed sheet, exit 1 if it differs")
    a = ap.parse_args()
    nwarn = []
    txt, info, chap_no_disp = build(lambda s: (nwarn.append(s), print("WARNING:", s, file=sys.stderr)))
    if nwarn:
        info += f"; {len(nwarn)} warnings"
    if a.check:
        cur = OUT.read_text()
        if cur == txt:
            print(f"check passed: {OUT.relative_to(BOOK.parent.parent)} is what the generator writes (0 differences); {info}")
            return 0
        d = list(difflib.unified_diff(cur.splitlines(True), txt.splitlines(True), "committed", "generated"))
        sys.stdout.writelines(d)
        nd = sum(1 for l in d if l[:1] in "+-" and l[:3] not in ("+++", "---"))
        print(f"\ncheck FAILED: {nd} differing lines; {info}")
        return 1
    a.out.write_text(txt)
    print(f"wrote {a.out}: {info}")
    print("chapters with no display:", ", ".join(C["label"] for C in chap_no_disp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
