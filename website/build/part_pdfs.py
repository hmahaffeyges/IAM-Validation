#!/usr/bin/env python3
"""Write the LaTeX drivers for the whole-book PDF and one PDF per Part (front matter + that Part + bibliography).

  python3 website/build/part_pdfs.py <dir>     copies docs/book/ to <dir> and writes part-1.tex ... part-7.tex next to main.tex

The drivers reuse main.tex line for line: the front matter (everything before \\mainmatter), then only the chosen \\part and its
\\input lines, then the bibliography. References to chapters in other Parts print as "??" in a Part PDF; the whole-book PDF has them all.
"""
import sys, re, shutil, pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]


def main(out):
    out = pathlib.Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(REPO / "docs/book", out)
    lines = (out / "main.tex").read_text().split("\n")
    i_main = next(i for i, l in enumerate(lines) if l.strip().startswith("\\mainmatter"))
    i_app = next(i for i, l in enumerate(lines) if l.strip().startswith("\\appendix"))
    i_back = next(i for i, l in enumerate(lines) if l.strip().startswith("\\backmatter"))
    front, body, back = lines[:i_main + 1], lines[i_main + 1:i_app], lines[i_back:]
    starts = [i for i, l in enumerate(body) if l.strip().startswith("\\part{")]
    for n, s in enumerate(starts, 1):
        e = starts[n] if n < len(starts) else len(body)
        (out / f"part-{n}.tex").write_text("\n".join(front + body[s:e] + back) + "\n")
    print(f"{len(starts)} Part drivers in {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "website/_build/pdf_src")
