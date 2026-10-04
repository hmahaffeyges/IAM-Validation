"""Appendix I: figure and data provenance, generated from the book source and the repository.

Run from any directory:  python docs/book/figscripts/make_app_I.py
Writes docs/book/appendices/app_I_provenance.tex.

Figures: every \\includegraphics in the chapters that main.tex inputs, in book order (plus the new chapters listed in EXTRA
when main.tex does not yet input them). For each: the figure label, the chapter, the file, the script in the repository
that writes a file of that name (search of every .py file for the quoted base name), the lines of that script that draw
it, and the data the script reads (file names in the script, the chain helper _chains.py, the growth helper _cosmo.py,
or constants only).
Tables: every table, longtable and free-standing tabular that holds numbers. The source is the first of: a record or
script named inside the table; one named within 15 lines of it; a published value cited in it; the records and scripts
the chapter names. Each named file is checked against the repository.
"""
import os, re, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK, REPO = HERE.parent, HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
from _texify import esc, path, check  # noqa: E402

OUT = BOOK / "appendices" / "app_I_provenance.tex"
SKIP = {"appendices/app_B_errata_physics", "appendices/app_B2_errata_cells"}      # leave the book
EXTRA = ["part5/p5_11_status_all", "appendices/app_G_predictions_register"]        # new, not yet in main.tex
NOTSOURCE = re.compile(r"ERRATA|LEDGER|NOTE|PLAN\.md|MANIFEST|GLOSSARY|read_ledgers|/sop/|^sop/|manual/|docs/papers|RETIRED|TODO", re.I)


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def clean(s):
    return s.replace("\\_", "_").replace("\\allowbreak{}", "").replace("\\allowbreak", "").replace("{}", "")


# ---------------- book order ----------------
main = (BOOK / "main.tex").read_text()
order = [i for i in re.findall(r"\\input\{([^}]+)\}", strip_comments(main)) if i != "preamble" and i not in SKIP]
order += [e for e in EXTRA if e not in order and (BOOK / (e + ".tex")).exists()]
SRC = {i: (BOOK / (i + ".tex")).read_text() for i in order if (BOOK / (i + ".tex")).exists()}


def chapter_label(s):
    m = re.search(r"\\chapter\*?(?:\[[^\]]*\])?\{.+?\}\s*\\label\{([^}]+)\}", strip_comments(s))
    return m.group(1) if m else None


CH = {i: chapter_label(s) for i, s in SRC.items()}

# ---------------- repository index ----------------
files = []
for root, dirs, fs in os.walk(REPO):
    dirs[:] = [d for d in dirs if d != ".git"]
    for f in fs:
        files.append(os.path.relpath(os.path.join(root, f), REPO))
byname = collections.defaultdict(list)
for p in files:
    byname[os.path.basename(p)].append(p)
PY = {p: (REPO / p).read_text(errors="ignore") for p in files if p.endswith(".py") and "RETIRED" not in p}


def resolve(name):
    for cand in (name, "docs/verification/" + name, "docs/book/" + name, "Biological_Physics/MethylPhys/" + name):
        if (REPO / cand).is_file():
            return cand
    c = [p for p in byname.get(os.path.basename(name), []) if p.endswith(name) and "RETIRED" not in p]
    return c[0] if len(c) == 1 else (sorted(c)[0] if c else None)


# ---------------- figures ----------------
DATA = re.compile(r"[\"']([^\"'\n]*?\.(?:json|csv|txt|npz|npy|dat|tsv|parquet|pkl|h5|hdf5|fits))[\"']")
SAVE = re.compile(r"(?:S\.save|save)\(fig,\s*\"(part\d)\",\s*\"([^\"]+)\"\)|savefig\([\"'][^\"']*?([A-Za-z0-9_]+)\.(?:pdf|png)[\"']")


def blocks(text):
    """(name, first line, last line, text) for each figure a script saves; a block runs from the previous save."""
    lines, out, start = text.splitlines(), [], 0
    seen = set()
    for i, l in enumerate(lines):
        m = SAVE.search(l)
        if m:
            name = m.group(2) or m.group(3)
            if name in seen:
                continue
            seen.add(name)
            out.append((name, start + 1, i + 1, "\n".join(lines[start:i + 1])))
            start = i + 1
    return out


def direct_refs(line):
    """Data a single source line reads directly."""
    d = []
    for m in DATA.finditer(line):
        v = m.group(1)
        if "{" in v or v.startswith(".") or "Pantheon" in v:
            continue
        r = v if (REPO / v).is_file() else resolve(v.split("/")[-1])
        d.append(r if r else v + " (not in repository)")
    if re.search(r"\bCH\.(L1|MG)\b", line):
        d.append("Level 1 chains, Cosmological_Physics/mgcamb_validation/chains (via _chains.py)")
    if re.search(r"\bCH\.L2\b", line):
        d.append("Level 2 chains, Cosmological_Physics/camb_validation/chains (via _chains.py)")
    if re.search(r"\bK\.\w+", line):
        d.append("_cosmo.py (linear growth, no data file)")
    if "iam_canon" in line:
        d.append("CANON/iam_canon.json")
    if "DataRelease" in line or "PANTHEON_DIR" in line:
        d.append("Pantheon+SH0ES data release (downloaded by the script)")
    return d


def script_reads(text):
    """Data a script reads: every direct read in its code (docstring excluded), in order of first appearance."""
    code = re.sub(r'(?s)^\s*(\"\"\"|\'\'\').*?\1', "", text, count=1)
    seen = []
    for line in code.splitlines():
        for x in direct_refs(line):
            if x not in seen:
                seen.append(x)
    return seen


def docstring_inputs(text):
    m = re.match(r'\s*(\"\"\"|\'\'\')(.*?)\1', text, re.S)
    if not m:
        return []
    return [x for x in dict.fromkeys(re.findall(r"([A-Za-z0-9_\-]+\.(?:npz|csv|json|txt|dat))", m.group(2)))]


FIGROWS, NOSCRIPT = [], []
for inp, s in SRC.items():
    spans = [(m.start(), s.find("\\end{figure", m.end())) for m in re.finditer(r"\\begin\{figure\*?\}", s)]
    # a figure set in a minipage with \captionof{figure} (no figure float) is a figure too
    spans += [(m.start(), s.find("\\end{minipage}", m.end())) for m in re.finditer(r"\\begin\{minipage\}", s)
              if "\\captionof{figure}" in s[m.start():s.find("\\end{minipage}", m.end())]]
    for a, b in sorted(spans):
        blk = s[a:b]
        for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", strip_comments(blk)):
            lab = re.findall(r"\\label\{([^}]+)\}", strip_comments(blk))
            base = os.path.splitext(os.path.basename(g))[0]
            q = re.compile(r"[\"'/]" + re.escape(base) + r"(\.pdf|\.png)?[\"']")
            hits = [p for p, t in PY.items() if q.search(t)]
            hits.sort(key=lambda p: (not p.startswith("docs/book/figscripts"), p))
            row = dict(inp=inp, ch=CH[inp], file=g, label=lab[0] if lab else None, exists=(BOOK / g).exists(), blk=blk,
                       line=s.count("\n", 0, a) + 1, script=None, lines=None, data=[])
            if hits:
                sc = hits[0]
                text = PY[sc]
                bl = [x for x in blocks(text) if x[0] == base]
                l0, l1 = (bl[0][1], bl[0][2]) if bl else (1, len(text.splitlines()))
                dat = script_reads(text)
                if not dat and not sc.startswith("docs/book/figscripts"):
                    dat = [(resolve(x) + " (named in the script's docstring)") if resolve(x) else
                           (x + " (named in the script's docstring; not in repository)") for x in docstring_inputs(text)]
                row.update(script=sc, lines=(l0, l1) if bl else None, data=dat)
                row["others"] = hits[1:]
            else:
                NOSCRIPT.append(row)
            FIGROWS.append(row)

# ---------------- tables ----------------
NAMED = re.compile(r"((?:[A-Za-z0-9_\-]+/)*[A-Za-z0-9_\-]+\.(?:py|json|csv|txt|md|npz|dat|tsv))")


def names_in(t):
    out = []
    for n in NAMED.findall(clean(t)):
        if NOTSOURCE.search(n):
            continue
        r = resolve(n)
        key = r or n
        if key not in [o[0] for o in out]:
            out.append((key, bool(r)))
    return out


def numeric(t):
    body = re.sub(r"\\[A-Za-z]+", " ", strip_comments(t))
    return len(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", body)) >= 6


CHAPNAMES = {i: names_in(s) for i, s in SRC.items()}
TABROWS = []
for inp, s in SRC.items():
    spans = []
    for kind in ("table", "longtable"):
        for m in re.finditer(r"\\begin\{" + kind + r"\*?\}", s):
            e = s.find("\\end{" + kind, m.end())
            spans.append((m.start(), e if e > 0 else len(s), kind))
    # a longtable inside a table float is one table
    spans = [x for x in spans if not any(y[0] < x[0] and x[1] <= y[1] for y in spans if y != x)]
    for m in re.finditer(r"\\begin\{tabular\*?\}", s):
        if not any(a <= m.start() <= b for a, b, _ in spans):
            e = s.find("\\end{tabular", m.end())
            spans.append((m.start(), e if e > 0 else len(s), "tabular"))
    for a, b, kind in sorted(spans):
        blk = s[a:b]
        if not numeric(blk):
            continue
        l0 = s.count("\n", 0, a) + 1
        lines = s.splitlines()
        l1 = s.count("\n", 0, b) + 1
        near = "\n".join(lines[max(0, l0 - 16):l0 - 1] + lines[l1:l1 + 15])
        lab = re.findall(r"\\label\{(tab:[^}]+)\}", strip_comments(blk))
        cites = [k for c in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", strip_comments(blk)) for k in c.split(",")]
        cites = list(dict.fromkeys(k.strip() for k in cites))
        own = [k for k in cites if k.lower().startswith("mahaffey") or re.search(r"(the quantum-processor report|the semiconductor report|the methylation report)", k)]
        cites = [k for k in cites if k not in own]
        src, how = names_in(blk), "named in the table"
        if not src:
            src, how = names_in(near), "named beside the table"
        if not src and cites:
            how = "published values cited in the table"
        if not src and not cites:
            src, how = CHAPNAMES[inp], "records the chapter names"
        chs = list(dict.fromkeys(re.findall(r"\\ref\{((?:ch|app):[^}]+)\}", strip_comments(blk))))
        if not src and not cites and chs:
            how = "chapters cited in the table"
        if not src and not cites and not chs:
            how = "none named"
        TABROWS.append(dict(inp=inp, ch=CH[inp], line=l0, kind=kind, label=lab[0] if lab else None, src=src, how=how,
                            cites=cites, own=own, chs=chs))

# ---------------- typeset (rewritten 2026-10-03: grouped by Part, what each item shows, clickable links, capped lists) ----------------
from urllib.parse import quote
GH = "https://github.com/hmahaffeyges/IAM-Validation/blob/main/"
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}

# which Part each input belongs to, from the \part headings of main.tex
PART, PNAME, cur, curname = {}, {}, 0, "Front matter"
for line in strip_comments(main).splitlines():
    m = re.match(r"\s*\\part\{(.+?)\}", line)
    if m:
        cur += 1; curname = f"Part {ROMAN[cur]}: " + m.group(1)
    if re.match(r"\s*\\appendix", line):
        cur, curname = 99, "Appendices"
    m = re.search(r"\\input\{([^}]+)\}", line)
    if m:
        PART[m.group(1)] = cur; PNAME[cur] = curname
for e in EXTRA:
    PART.setdefault(e, 99 if e.startswith("appendices") else max(v for v in PART.values() if v < 99)); PNAME.setdefault(99, "Appendices")


def chref(lab):
    if not lab:
        return "---"
    return (r"App.~\ref{" if lab.startswith("app") else r"Ch.~\ref{") + lab + "}"


def href(pth, text=None, lines=None):
    """A clickable link to a file on GitHub, shown by its file name."""
    url = GH + quote(pth) + (f"#L{lines[0]}-L{lines[1]}" if lines else "")
    url = url.replace("%", "\\%").replace("#", "\\#")
    t = esc(text or os.path.basename(pth)).replace("\\_", "\\_\\allowbreak{}")
    return r"\href{" + url + r"}{\texttt{" + t + "}}"


def brace(s, i):
    """Return the contents of the brace group that opens at s[i] == '{'."""
    d = 0
    for k in range(i, len(s)):
        if s[k] == "{": d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0: return s[i + 1:k]
    return s[i + 1:]


STATUS = r"\\(?:derived|calc|calibrated|measured|observed|fitted|conjecture|analogy|prediction|openprob|interp)\b(?:\{\})?"


def shows(blk, words=16):
    """First sentence of the caption, short, safe to typeset."""
    m = re.search(r"\\caption(?:of\{[a-z]+\})?(?:\[[^\]]*\])?\{", strip_comments(blk))
    if not m:
        return None
    c = brace(strip_comments(blk), m.end() - 1)
    c = re.sub(r"\\label\{[^}]*\}|~?\\cite[pt]?\*?(?:\[[^\]]*\])?\{[^}]*\}|" + STATUS, "", c)
    _seg = re.split(r"((?<!\\)\$[^$]*(?<!\\)\$)", c)          # panel letters, outside math only
    c = "".join(x if x.startswith("$") else re.sub(r"\((?:[a-z])\)\s*", "", x) for x in _seg)
    c = re.sub(r"\\(?:textbf|emph|textit|mathrm)\{([^{}]*)\}", r"\1", c)
    c = re.sub(r"\(\s*\)|\[\s*\]", "", c)                     # empty brackets
    # cross-references are kept: every \\ref resolves book-wide, and removing them left broken sentences
    c = re.sub(r"\s+([,.;:])", r"\1", c)
    c = re.sub(r"\s+", " ", c).strip(" ,;:")
    # first sentence outside math
    out, inm = "", False
    for k, ch in enumerate(c):
        if ch == "$": inm = not inm
        out += ch
        if ch == "." and not inm and (k + 1 == len(c) or c[k + 1] == " ") and len(out) > 25:
            break
    w = out.split()
    if len(w) > words:
        out = " ".join(w[:words]); out = out if out.count("$") % 2 == 0 else re.sub(r"\$[^$]*$", "", out)
        out = out.rstrip(" ,;:.") + "\\ldots"
    if check(out) is not None or out.count("{") != out.count("}"):
        out = re.sub(r"\$[^$]*\$|\\[A-Za-z]+|[{}^_]", " ", out); out = esc(re.sub(r"\s+", " ", out).strip())
    return out or None


# caption text for each figure and table row
for r in FIGROWS:
    r["shows"] = shows(r["blk"])
for r in TABROWS:
    s_ = SRC[r["inp"]]; lines_ = s_.splitlines(); a = len("\n".join(lines_[:r["line"] - 1]))
    e = min([x for x in (s_.find("\\end{table", a), s_.find("\\end{longtable", a), s_.find("\\end{tabular", a)) if x > 0] or [len(s_)])
    r["shows"] = shows(s_[a:e], 14)
    if not r["shows"]:
        # no caption: name the section the table sits in and its column headings
        secs = list(re.finditer(r"\\(?:sub)*section\*?\{([^{}]*)\}", strip_comments(s_[:a])))
        sec = secs[-1].group(1) if secs else None
        blk = strip_comments(s_[a:e])
        hm = re.search(r"\\toprule\s*(.*?)\\\\", blk, re.S) or re.search(r"\}\s*\n(.*?)\\\\", blk, re.S)
        head = None
        if hm:
            cells = [re.sub(r"\$[^$]*\$", "", c).replace("\\%", " percent ") for c in hm.group(1).split("&")]
            cells = [re.sub(r"\\[A-Za-z]+\*?|[{}~^_\\]", " ", c) for c in cells]
            cells = [re.sub(r"\s+", " ", c).strip().replace(" percent ", "\\,\\% ").replace(" percent", "\\,\\%") for c in cells]
            cells = [re.sub(r"\(\s*\)", "", c).strip(" ,") for c in cells]
            cells = [c for c in cells if c and c not in (",", "(", ")") and not re.fullmatch(r"\(.*\)", c)]
            cells = [c for c in cells if c and len(c) < 40 and not re.fullmatch(r"[-+0-9.,()\s]+", c)][:5]
            head = ", ".join(cells) if cells else None
        txt = (f"in \u201c{sec}\u201d" if sec else "") + (f": {head}" if head else "")
        txt = txt.strip(": ")
        r["shows"] = (txt if check(txt) is None and txt.count("{") == txt.count("}") else esc(re.sub(r"\$[^$]*\$|\\[A-Za-z]+", "", txt))) if txt else None


def data_cell(items, cap=3):
    parts = []
    for x in items[:cap]:
        note = ""
        if " (" in x:
            x, note = x.split(" (", 1); note = " (" + esc(note)
        parts.append((href(x) if (REPO / x).is_file() else esc(x)) + note)
    more = len(items) - cap
    return "; ".join(parts) + (f"; and {more} more" if more > 0 else "")


L = [r"% Generated by docs/book/figscripts/make_app_I.py from the book source and the repository. Do not edit by hand.",
     r"\chapter{Figure and data provenance}\label{app:provenance}", "",
     r"Every figure and every table of numbers in the book, Part by Part, with what it shows and where it comes from. Each file",
     r"name is a link to that file in the public repository \url{https://github.com/hmahaffeyges/IAM-Validation}; a script link",
     r"opens at the lines that draw the figure. To redraw a figure, run its script from the repository root. Figures computed",
     r"in the script read no data file: their numbers are physical constants and the values stated in the chapter. Chain",
     r"figures read the Cobaya chains through \texttt{\_chains.py} (30\,\% burn-in, weighted statistics), and growth figures use",
     r"\texttt{\_cosmo.py}, which solves the linear growth equation for $\Lambda$CDM and for IAM from the same early amplitude.", ""]

FC = (r"{@{}>{\raggedright\arraybackslash}p{0.07\textwidth}>{\raggedright\arraybackslash}p{0.40\textwidth}"
      r">{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.24\textwidth}@{}}")
FH = r"Fig. & what it shows & script & data it reads\\\midrule"
TC = (r"{@{}>{\raggedright\arraybackslash}p{0.10\textwidth}>{\raggedright\arraybackslash}p{0.45\textwidth}"
      r">{\raggedright\arraybackslash}p{0.39\textwidth}@{}}")
TH = r"Table & what it shows & where the numbers come from\\\midrule"


def table_source(r):
    if r["how"] == "published values cited in the table":
        return "published values: " + r"\cite{" + ",".join(r["cites"]) + "}"
    if r["how"] == "chapters cited in the table":
        return f"derived in the chapters each row names ({len(r['chs'])})"
    if r["how"] == "none named":
        return "values stated in the chapter text"
    ok = [n for n, good in r["src"] if good]
    if r["how"] == "records the chapter names":
        if not ok:
            return "values stated in the chapter text"
        return "records named in the chapter: " + data_cell(ok, 2)
    return data_cell(ok or [n for n, _ in r["src"]], 3)


for kind in ("Figures", "Tables of numbers"):
    L.append(r"\section{" + kind + "}")
    rows = FIGROWS if kind == "Figures" else TABROWS
    for pn in sorted({PART.get(r["inp"], 0) for r in rows}):
        sub = [r for r in rows if PART.get(r["inp"], 0) == pn]
        if not sub:
            continue
        L.append(r"\subsection*{" + esc(PNAME.get(pn, "Front matter")) + "}")
        L += [r"{\footnotesize\setlength{\tabcolsep}{3pt}\begin{longtable}" + (FC if kind == "Figures" else TC),
              r"\toprule " + (FH if kind == "Figures" else TH) + r"\endfirsthead",
              r"\toprule " + (FH if kind == "Figures" else TH) + r"\endhead", r"\bottomrule\endfoot"]
        for r in sub:
            what = r.get("shows") or ("see " + chref(r["ch"]) if r["ch"] else "frontispiece")
            where = f" ({chref(r['ch'])})" if r["ch"] else ""
            if kind == "Figures":
                ref = r"\ref{" + r["label"] + "}" if r["label"] else "---"
                if r["script"]:
                    sc = href(r["script"], lines=r["lines"]) + (f" (lines {r['lines'][0]}--{r['lines'][1]})" if r["lines"] else "")
                    nfig = sum(1 for x in FIGROWS if x["script"] == r["script"])
                    if r["data"] and nfig > 1 and len(r["data"]) > 3:
                        dat = f"files read by the script ({len(r['data'])}, shared by its {nfig} figures)"
                    else:
                        dat = data_cell(r["data"]) if r["data"] else "computed in the script"
                else:
                    sc, dat = "script to be added", "---"
                line = f"{ref} & {what}{where} & {sc} & {dat}\\\\"
            else:
                ref = r"\ref{" + r["label"] + "}" if r["label"] else f"{chref(r['ch'])}, unnumbered"
                line = f"{ref} & {what}{'' if not r['label'] else ' (' + chref(r['ch']) + ')'} & {table_source(r)}\\\\"
            assert check(line) is None, (check(line), line)
            L.append(line)
        L += [r"\end{longtable}}", ""]
    if kind == "Figures":
        L.append(f"{len(FIGROWS) - len(NOSCRIPT)} of the {len(FIGROWS)} figures are drawn by a script in the repository.")
        if NOSCRIPT:
            L.append(f" The scripts for the other {len(NOSCRIPT)} are being added:")
            L.append(r"\begin{itemize}\setlength\itemsep{0pt}\small")
            for r in NOSCRIPT:
                L.append(r"\item Figure~\ref{" + r["label"] + "}" + f", {chref(r['ch'])}" if r["label"] else r"\item an unlabelled figure")
            L.append(r"\end{itemize}")
        L.append("")
nn = [r for r in TABROWS if r["how"] == "none named"]
tex = "\n".join(L) + "\n"
assert check(tex) is None, check(tex)
OUT.write_text(tex)

# report for the manifest
print(f"wrote {OUT.relative_to(REPO)}: {len(FIGROWS)} figures ({len(NOSCRIPT)} without script), {len(TABROWS)} tables "
      f"({len(nn)} with no source named)")
for r in NOSCRIPT:
    print("NOSCRIPT", r["inp"], r["file"], r["label"])
for r in FIGROWS:
    if r["script"] and not r["script"].startswith("docs/book/figscripts"):
        print("OUTSIDE", r["file"], r["script"], r["lines"])
    if not r["label"]:
        print("NOLABEL", r["inp"], r["file"])
for r in TABROWS:
    if r["own"]:
        print("OWNCITE", r["inp"], r["line"], r["label"], r["own"])
    for n, ok in r["src"]:
        if not ok:
            print("MISSING", r["inp"], r["line"], n)
for r in nn:
    print("NONE", r["inp"], r["line"], r["label"])


