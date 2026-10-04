"""Appendix E, the formula sheet: every displayed equation of the seven parts, in book order.

Run from any directory:  python docs/book/figscripts/make_app_E.py
Reads docs/book/main.tex for the order of parts and chapters (each \\input followed recursively), and the current
docs/book/appendices/app_E_formulas.tex once, for the one-line descriptions written for the earlier selection.
Writes docs/book/appendices/app_E_formulas.tex.

Rules:
  * every display (equation, align, gather, multline, eqnarray and their starred forms, \\[ \\]) is an entry, in book order;
  * the body is copied exactly, with \\label, \\nonumber, \\notag and \\tag removed; multi-line environments are written as
    aligned (align, eqnarray, flalign) or gathered (gather, multline) inside the entry's own equation*;
  * status: the label the chapter attaches directly after the display (with its parenthesis, if any). If the chapter attaches
    none, the earlier entry's status is used when the same equation was in the earlier selection; then the first label later in
    the same paragraph; otherwise the entry says that the chapter attaches no label. If the chapter's label and the earlier
    entry's label are the same macro, the earlier, more specific wording is kept;
  * description: carried from the earlier selection when the entry matches it (same label, or same body); otherwise none;
  * the same relation in more than one place is listed each time, with pointers to its other entries.
"""
import re, sys, collections, difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent
OUT = BOOK / "appendices" / "app_E_formulas.tex"
STAT = ("derived", "calc", "calibrated", "measured", "observed", "fitted", "conjecture", "prediction", "openprob", "interp", "analogy")
ENVS = ("equation", "align", "gather", "multline", "eqnarray", "flalign")
WRAP = {"align": "aligned", "eqnarray": "aligned", "flalign": "aligned", "gather": "gathered", "multline": "gathered"}


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


# ---------------------------------------------------------------- the earlier selection (descriptions and statuses)
old = OUT.read_text()
OLD = []
for m in re.finditer(r"\\iamfsentry\{", old):
    i = m.end() - 1; args = []
    for _ in range(5):
        a, i = group(old, i); args.append(a)
        while i < len(old) and old[i] in " \t": i += 1
    OLD.append(dict(n=args[0], body=args[1], desc=args[2], status=args[3], intext=args[4],
                    labels=re.findall(r"\\eqref\{([^}]+)\}", args[4].split(";")[0])))
def norm(b):
    b = re.sub(r"\\label\{[^}]*\}|\\nonumber|\\notag", "", b)
    b = re.sub(r"\\begin\{(aligned|gathered)\}|\\end\{(aligned|gathered)\}", "", b)
    return re.sub(r"\\[,;:! ]|~|\s|\\left|\\right|&|\\\\|[.,;]$", "", b)
for e in OLD:
    m = re.search(r"Chapter~\\ref\{([^}]+)\}", e["intext"]); e["chap"] = m.group(1) if m else None
old_by_label = {l: e for e in OLD for l in e["labels"]}
old_by_body = collections.defaultdict(list)
for e in OLD:
    old_by_body[norm(e["body"])].append(e)

# ---------------------------------------------------------------- the book
main = strip_comments((BOOK / "main.tex").read_text())
main = main.split("\\mainmatter", 1)[1].split("\\appendix", 1)[0]
parts = []
for m in re.finditer(r"\\part\{([^}]*)\}\\label\{([^}]+)\}|\\input\{([^}]+)\}", main):
    if m.group(1):
        parts.append(dict(title=m.group(1), label=m.group(2), chapters=[]))
    else:
        parts[-1]["chapters"].append(m.group(3))

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
    end = t.find("\n\n", k); end = len(t) if end < 0 else end
    nxt = DISP.search(t, k); end = min(end, nxt.start()) if nxt else end
    m = STATRX.search(t, k, end)
    return ("\\" + m.group(1) + "{}") if m else None


entries, chap_no_disp = [], []
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
            C["entries"].append(E); entries.append(E)
        if not C["entries"]:
            chap_no_disp.append(C)

# ---------------------------------------------------------------- descriptions, statuses, pointers
lead = lambda s: (re.match(r"\\(\w+)", s.strip()) or [None, None])[1]
counts = collections.Counter(); used = set()
for i, E in enumerate(entries):
    E["n"] = i
    o = next((old_by_label[l] for l in E["labels"] if l in old_by_label), None)
    nb = norm(E["body"])
    if o is None:   # the same body: first in the same chapter, then anywhere, an earlier entry not yet used
        allb = old_by_body.get(nb, []); same_b = [e for e in allb if e["n"] not in used]
        o = next((e for e in same_b if e["chap"] == E["chapter"]["label"]), same_b[0] if same_b else (allb[0] if allb else None))
    if o is None:   # same chapter, nearly the same body (the earlier sheet shortened some displays)
        cand = [(difflib.SequenceMatcher(None, norm(e["body"]), norm(E["body"])).ratio(), e["n"], e) for e in OLD
                if e["chap"] == E["chapter"]["label"] and e["n"] not in used]
        cand = [c for c in cand if c[0] >= 0.8]
        if cand:
            o = max(cand, key=lambda c: c[0])[2]
    if o is not None:
        used.add(o["n"])
    E["desc"] = o["desc"] if o else ""
    if E["attached"]:
        st = o["status"] if (o and lead(o["status"]) == lead(E["attached"]) and "(" in o["status"] and "(" not in E["attached"]) else E["attached"]
        counts["attached"] += 1
    elif o:
        st = o["status"]; counts["earlier entry"] += 1
    elif E["para"]:
        st = E["para"]; counts["later in the paragraph"] += 1
    else:
        st = "no status label attached in the chapter"; counts["none"] += 1
    E["status"] = st
    counts["described"] += bool(E["desc"])
same = collections.defaultdict(list)
for E in entries:
    same[norm(E["body"])].append(E["n"])

def intext(E):
    C = E["chapter"]; s = []
    if E["numbered"] and E["labels"]:
        s.append(("Eq.~" if len(E["labels"]) == 1 else "Eqs.~") + ", ".join(f"\\eqref{{{l}}}" for l in E["labels"]))
    loc = f"Chapter~\\ref{{{C['label']}}}" + (f", section ``{E['section']}''" if E["section"] else "")
    s.append(loc)
    others = [n for n in same[norm(E["body"])] if n != E["n"]]
    if others:
        s.append("see also " + ", ".join(f"(\\ref{{app:fs:{n}}})" for n in others))
    return "; ".join(s)

# ---------------------------------------------------------------- write
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
            L.append(f"\\iamfsentry{{{E['n']}}}{{{E['body']}}}{{{E['desc']}}}{{{E['status']}}}{{{intext(E)}}}")
L.append("")
txt = "\n".join(L)
assert txt.count("{") - txt.count("\\{") == txt.count("}") - txt.count("\\}"), "unbalanced braces"
assert not re.search(r"\n\s*\n", "\n".join(l for l in L if l.startswith("\\iamfsentry"))), "blank line inside an entry"
OUT.write_text(txt)
print(f"wrote {OUT.relative_to(BOOK.parent.parent)}: {N} entries from {sum(len(P['chaps']) for P in parts)} chapters;",
      f"{len(OLD)} earlier entries read; " + "; ".join(f"{k} {v}" for k, v in counts.items()))
print("chapters with no display:", ", ".join(C["label"] for C in chap_no_disp))
