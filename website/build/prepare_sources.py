#!/usr/bin/env python3
"""Copy docs/book/ to a build directory and mark where verify_book.py has a check, for the "Run this check" buttons.

  python3 website/build/prepare_sources.py <build_dir>

docs/book/ itself is never written. In the copy, each line that verify_book.py checks gets an invisible marker \\iamcheck{i j ...}
(indices into <build_dir>/checks_index.json), which the LaTeXML binding (website/latexml/iam.ltxml) turns into a span the site build
replaces with buttons. A marker never goes inside display math, a table, a figure or float, verbatim text, or a comment: if the
checked line is inside one of these, the marker goes just after the environment closes (so the button sits under the equation,
table or figure). In running text it goes at the end of the line, or at the last point of the line outside inline math.
"""
import sys, json, re, shutil, pathlib, importlib.util

REPO = pathlib.Path(__file__).resolve().parents[2]
BOOK = REPO / "docs" / "book"
PROTECT = {"equation", "equation*", "align", "align*", "alignat", "alignat*", "gather", "gather*", "multline", "multline*", "eqnarray",
           "eqnarray*", "displaymath", "math", "split", "aligned", "cases", "array", "matrix", "pmatrix", "bmatrix",
           "tabular", "tabular*", "tabularx", "longtable", "table", "table*", "figure", "figure*", "verbatim", "Verbatim", "lstlisting",
           "minipage", "subequations", "thebibliography", "tikzpicture", "picture"}
BEGIN = re.compile(r"\\begin\{([^}]+)\}")
END = re.compile(r"\\end\{([^}]+)\}")


def load_checks():
    spec = importlib.util.spec_from_file_location("verify_book", REPO / "verify_book.py")
    vb = importlib.util.module_from_spec(spec)
    sys.modules["verify_book"] = vb
    spec.loader.exec_module(vb)
    return vb


def comment_pos(line):
    """Index of the first unescaped % (or len(line))."""
    i = 0
    while i < len(line):
        if line[i] == "\\":
            i += 2
            continue
        if line[i] == "%":
            return i
        i += 1
    return len(line)


def scan(lines):
    """Per line: (protected_depth_after_line, display_math_open_after_line, end_line_of_outermost_protected_env_for_each_line)."""
    stack, disp = [], False
    info = []
    open_at = None
    close_of = {}
    for n, raw in enumerate(lines):
        line = raw[:comment_pos(raw)]
        start_prot = bool(stack) or disp
        if start_prot and open_at is None:
            open_at = n
        events = []
        for m in BEGIN.finditer(line):
            events.append((m.start(), "b", m.group(1)))
        for m in END.finditer(line):
            events.append((m.start(), "e", m.group(1)))
        for m in re.finditer(r"(?<!\\)\\\[|(?<!\\)\\\]|\$\$", line):
            events.append((m.start(), "d", m.group(0)))
        events.sort()
        for _, kind, name in events:
            if kind == "b" and name in PROTECT:
                if not stack and not disp:
                    open_at = n
                stack.append(name)
            elif kind == "e" and name in PROTECT and stack:
                if name in stack:
                    while stack and stack.pop() != name:
                        pass
                if not stack and not disp:
                    for k in range(open_at, n + 1):
                        close_of[k] = n
                    open_at = None
            elif kind == "d":
                if name == "\\[" or (name == "$$" and not disp):
                    if not stack and not disp:
                        open_at = n
                    disp = True
                else:
                    disp = False
                    if not stack:
                        for k in range(open_at if open_at is not None else n, n + 1):
                            close_of[k] = n
                        open_at = None
        info.append(bool(stack) or disp)
    return info, close_of


def brace_depths(lines):
    """Brace depth at the end of each line (comments and escaped braces ignored)."""
    d, out = 0, []
    for raw in lines:
        line = raw[:comment_pos(raw)]
        i = 0
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == "{":
                d += 1
            elif line[i] == "}":
                d -= 1
            i += 1
        out.append(d)
    return out


def text_point(line):
    """Column where a text-mode marker can go on this line: before any comment, outside inline math; None if impossible."""
    cut = comment_pos(line)
    body = line[:cut]
    depth, last_ok, i = 0, None, 0
    in_math = False
    while i < len(body):
        c = body[i]
        if c == "\\":
            if body[i + 1:i + 2] in ("(",):
                in_math = True
            elif body[i + 1:i + 2] in (")",):
                in_math = False
            i += 2
            continue
        if c == "$":
            in_math = not in_math
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if not in_math:
        end = len(body.rstrip())
        k = 0
        while k < end and body[end - 1 - k] == "\\":
            k += 1
        if k % 2 == 1:                      # the line ends in a control space "\\": the marker goes before it
            end -= 1
        return end
    return None


def web_figures(src, dpi=150):
    """Render every PDF figure the chapters include to a PNG next to it, and point \\includegraphics at the PNG (build copy only)."""
    import subprocess
    inc = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)\.pdf\}")
    done = set()
    for tex in src.rglob("*.tex"):
        t = tex.read_text()
        for m in inc.finditer(t):
            rel = m.group(2)
            pdf = src / (rel + ".pdf")
            png = src / (rel + ".png")
            if rel not in done and pdf.exists():
                subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(src / rel)], check=True)
                done.add(rel)
        t2 = inc.sub(lambda m: m.group(1) + m.group(2) + ".png}", t)
        if t2 != t:
            tex.write_text(t2)
    return len(done)


def main(build):
    build = pathlib.Path(build)
    src = build / "src"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(BOOK, src)
    vb = load_checks()
    index = []
    per_file = {}
    for c in vb.CHECKS:
        m = c.meta
        if not m["file"] or not m["line"]:
            continue
        index.append(dict(label=c.label, part=m["part"], chapter=m["chapter"], file=m["file"], line=m["line"], printed=m["printed"],
                          title=m["title"], heavy=bool(m["heavy"])))
        per_file.setdefault(m["file"], []).append(len(index) - 1)
    placed, moved, skipped = 0, 0, []
    for f, idxs in per_file.items():
        p = src / (f + ".tex")
        if not p.exists():
            skipped += idxs
            continue
        lines = p.read_text().split("\n")
        prot, close_of = scan(lines)
        depth = brace_depths(lines)
        marks = {}
        for i in idxs:
            n = index[i]["line"] - 1
            if n >= len(lines):
                skipped.append(i)
                continue
            target = n
            if (prot[n] or n in close_of) and n in close_of:
                target = close_of[n]
                moved += 1
            elif prot[n]:
                skipped.append(i)
                continue
            while target < len(lines) and depth[target] != 0:     # inside a macro argument: after the argument closes
                target += 1
            if target >= len(lines):
                skipped.append(i)
                continue
            marks.setdefault(target, []).append(i)
        for n, ids in sorted(marks.items()):
            line = lines[n]
            col = None if n not in close_of or close_of.get(n) != n else None
            if re.search(r"\\end\{(Verbatim|verbatim|lstlisting)\}", line):   # verbatim swallows the rest of its closing line
                if n + 1 < len(lines):
                    lines[n + 1] = "\\iamcheck{" + " ".join(map(str, ids)) + "}" + lines[n + 1]
                    placed += len(ids)
                    continue
            if prot[n] and n in close_of:          # the line that closes a protected environment: after the \end{...}
                ends = [m.end() for m in END.finditer(line[:comment_pos(line)])] + \
                       [m.end() for m in re.finditer(r"(?<!\\)\\\]|\$\$", line[:comment_pos(line)])]
                col = max(ends) if ends else None
            if col is None:
                col = text_point(line)
            if col is None:                       # line ends inside inline math: put it at the start of the next line
                n2 = n + 1
                while n2 < len(lines) and text_point(lines[n2]) is None:
                    n2 += 1
                if n2 >= len(lines):
                    skipped += ids
                    continue
                lines[n2] = "\\iamcheck{" + " ".join(map(str, ids)) + "}" + lines[n2]
            else:
                lines[n] = line[:col] + "\\iamcheck{" + " ".join(map(str, ids)) + "}" + line[col:]
            placed += len(ids)
        p.write_text("\n".join(lines))
    json.dump(index, open(build / "checks_index.json", "w"))
    figs = web_figures(src)
    print(f"figures: {figs} PDF figures rendered to PNG for the web (the PDF build keeps the PDFs)")
    print(f"checks: {len(index)}; markers placed: {placed} ({moved} moved after an equation, table or figure); not placed: {len(skipped)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "website/_build")
