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
NOTSOURCE = re.compile(r"LEDGER|NOTE|PLAN\.md|MANIFEST|GLOSSARY|read_ledgers|/sop/|^sop/|manual/|docs/papers|RETIRED|TODO", re.I)


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
        d.append("Level 1 chains, mgcamb_validation/chains (via _chains.py)")
    if re.search(r"\bCH\.L2\b", line):
        d.append("Level 2 chains, camb_validation/chains (via _chains.py)")
    if re.search(r"\bK\.\w+", line):
        d.append("linear growth computed in _cosmo.py (no data file)")
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
    for a, b in ((m.start(), s.find("\\end{figure", m.end())) for m in re.finditer(r"\\begin\{figure\*?\}", s)):
        blk = s[a:b]
        for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", strip_comments(blk)):
            lab = re.findall(r"\\label\{([^}]+)\}", strip_comments(blk))
            base = os.path.splitext(os.path.basename(g))[0]
            q = re.compile(r"[\"'/]" + re.escape(base) + r"(\.pdf|\.png)?[\"']")
            hits = [p for p, t in PY.items() if q.search(t)]
            hits.sort(key=lambda p: (not p.startswith("docs/book/figscripts"), p))
            row = dict(inp=inp, ch=CH[inp], file=g, label=lab[0] if lab else None, exists=(BOOK / g).exists(),
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

# ---------------- typeset ----------------
def fmt_data(x):
    if " (" in x:
        p, rest = x.split(" (", 1)
        return (path(p) if ("/" in p or "." in p) and " " not in p else esc(p)) + " (" + esc(rest)
    return path(x) if ("/" in x or "." in x) and " " not in x else esc(x)


def chref(lab):
    if not lab:
        return "---"
    return (r"App.~\ref{" if lab.startswith("app") else r"Ch.~\ref{") + lab + "}"


L = [r"% Generated by docs/book/figscripts/make_app_I.py from the book source and the repository. Do not edit by hand.",
     r"\chapter{Figure and data provenance}\label{app:provenance}", "",
     r"This appendix lists, in the order of the book, the script in the repository \url{https://github.com/hmahaffeyges/IAM-Validation}",
     r"that draws each figure and the data that script reads, and the record or script behind each table of numbers, so that a",
     r"reader can rerun each figure and trace each number. Paths are relative to the repository root. The chain",
     r"helper \texttt{docs/\allowbreak{}book/\allowbreak{}figscripts/\allowbreak{}\_chains.py} reads the Cobaya chains with",
     r"a 30\,\% burn-in and weighted statistics; the growth helper \texttt{\_cosmo.py} in the same folder solves the linear",
     r"growth equation with the same early amplitude for $\Lambda$CDM and for IAM. A figure whose script reads no file is",
     r"computed from constants and from the values written in the script, which are those of the chapter that shows it.",
     r"Where one script draws several figures, the last column lists every file that script reads.",
     r"This list is generated by \texttt{docs/\allowbreak{}book/\allowbreak{}figscripts/\allowbreak{}make\_app\_I.py}.", ""]

L.append(r"\section{Figures}")
COLS = (r"{@{}>{\raggedright\arraybackslash}p{0.07\textwidth}>{\raggedright\arraybackslash}p{0.08\textwidth}"
        r">{\raggedright\arraybackslash}p{0.21\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}"
        r">{\raggedright\arraybackslash}p{0.29\textwidth}@{}}")
H = r"figure & chapter & file & script (lines) & data the script reads\\\midrule"
L += [r"{\footnotesize\begin{longtable}" + COLS, r"\toprule " + H + r"\endfirsthead", r"\toprule " + H + r"\endhead",
      r"\bottomrule\endfoot"]
for r in FIGROWS:
    fig = r"\ref{" + r["label"] + "}" if r["label"] else "unlabelled"
    f = path(r["file"].replace("figures/", "")) + ("" if r["exists"] else r" (file missing)")
    if r["script"]:
        sc = path(r["script"]) + (f" ({r['lines'][0]}--{r['lines'][1]})" if r["lines"] else "")
        if not r["script"].startswith("docs/book/figscripts"):
            sc += "; writes outside the book tree"
        dat = "; ".join(fmt_data(x) for x in r["data"]) or "constants and values in the script"
    else:
        sc, dat = r"\emph{no script found}", "---"
    line = f"{fig} & {chref(r['ch'])} & {f} & {sc} & {dat}\\\\"
    assert check(line) is None, (check(line), line)
    L.append(line)
L += [r"\end{longtable}}", ""]
L.append(f"Of the {len(FIGROWS)} figures, {len(FIGROWS) - len(NOSCRIPT)} have a script in the repository that writes a"
         f" file of that name; {sum(1 for r in FIGROWS if r['script'] and r['script'].startswith('docs/book/figscripts'))}"
         r" of these scripts are in \texttt{docs/\allowbreak{}book/\allowbreak{}figscripts}, which writes straight into"
         r" the figure folders of the book.")
if NOSCRIPT:
    L.append(f" No script was found for the following {len(NOSCRIPT)}; until one is added, these figures cannot be"
             r" regenerated from the repository:")
    L.append(r"\begin{itemize}\setlength\itemsep{0pt}\small")
    for r in NOSCRIPT:
        fig = r"Figure~\ref{" + r["label"] + "}" if r["label"] else "an unlabelled figure"
        L.append(r"\item " + fig + f", {chref(r['ch'])}: " + path(r["file"]))
    L.append(r"\end{itemize}")
L.append("")

L.append(r"\section{Tables of numbers}")
L.append(r"A table without a label is identified by its chapter and its line in the chapter source file. The column")
L.append(r"`how' says where the source was found: inside the table, within fifteen lines of it, as a published value cited in")
L.append(r"it, in the chapters its rows cite, or among the records and scripts the chapter names.")
COLS2 = (r"{@{}>{\raggedright\arraybackslash}p{0.12\textwidth}>{\raggedright\arraybackslash}p{0.09\textwidth}"
         r">{\raggedright\arraybackslash}p{0.15\textwidth}>{\raggedright\arraybackslash}p{0.58\textwidth}@{}}")
H2 = r"table & chapter & how & record or script\\\midrule"
L += [r"{\footnotesize\begin{longtable}" + COLS2, r"\toprule " + H2 + r"\endfirsthead", r"\toprule " + H2 + r"\endhead",
      r"\bottomrule\endfoot"]
for r in TABROWS:
    t = r"\ref{" + r["label"] + "}" if r["label"] else path(r["inp"].split("/")[-1] + ".tex") + f", line {r['line']}"
    parts = [path(n) + ("" if ok else " (not in repository)") for n, ok in r["src"]]
    if r["cites"] and r["how"] == "published values cited in the table":
        parts.append(r"\cite{" + ",".join(r["cites"]) + "}")
    if r["how"] == "chapters cited in the table":
        parts.append(f"{len(r['chs'])} chapters and appendices, each named in its row")
    line = f"{t} & {chref(r['ch'])} & {r['how']} & {'; '.join(parts) or '---'}\\\\"
    assert check(line) is None, (check(line), line)
    L.append(line)
L += [r"\end{longtable}}", ""]
nn = [r for r in TABROWS if r["how"] == "none named"]
L.append(f"Of the {len(TABROWS)} tables of numbers, {len(TABROWS) - len(nn)} have a source named in the book;"
         f" {sum(1 for r in TABROWS if r['how'] == 'records the chapter names')} of these are traced only through the"
         r" records their chapter names.")
L.append("")
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


# --- post-process (2026-10-02): keep the generated table inside the text width
import re as _re
_f = OUT
_s = open(_f, encoding="utf-8").read()
_s = _re.sub(r"/(?!\\allowbreak)", "/\\\\allowbreak{}", _s).replace("Runtime Matrices", "Runtime\\ Matrices")
_s = _s.replace("{\\footnotesize\\begin{longtable}", "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{longtable}")
_s = _s.replace("\\_", "\\_\\allowbreak{}")
open(_f, "w", encoding="utf-8").write(_s)
