"""Chapter start pages, read from the compiled book itself (2026-10-13): the front page's chapter links must open the PDF on the
right page, and hand-typed page numbers drifted by two pages when chapter 48 grew. Reads the PDF's text page by page (pdftotext,
the poppler tool the build already uses) and finds the page where each chapter's opening line is printed.
Usage: python3 chapter_pages.py BOOK.pdf  -> prints JSON {key: [chapter number, page]}; exits 1 if any chapter in build_home.CH is not found."""
import json, os, re, subprocess, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
def chapter_table():
    spec = importlib.util.spec_from_file_location("bh", os.path.join(HERE, "build_home.py")); src = open(spec.origin, encoding="utf-8").read()
    import ast; return ast.literal_eval(re.search(r"CH = (\{.*?\})\n", src, re.S).group(1))
def pages(pdf):
    """2026-11-05: matched by TITLE, and the chapter number read from the book too: chapters added before these ones had shifted every
    number, and matching 'chapter N' opened the wrong chapter (the label said one chapter, the page another)."""
    t = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout.split("\f")
    norm = lambda s: re.sub(r"\s+", " ", s.replace("\u2019", "'").replace("\u2018", "'")).strip().lower()
    T = [norm(x) for x in t]
    out, miss = {}, []
    for k, (n, _, title) in chapter_table().items():
        head = norm(title).split(":")[0][:30]
        hit = [(j + 1, int(m.group(1))) for j, x in enumerate(T) for m in [re.match(r"chapter (\d+) (.*)", x[:300])] if m and m.group(2).startswith(head)]   # the title opens the chapter page
        if hit: out[k] = [hit[0][1], hit[0][0]]
        else: miss.append(f"{k} ({title})")
    return out, miss
if __name__ == "__main__":
    out, miss = pages(sys.argv[1])
    if miss: print("chapter_pages: chapters not found in the compiled book: " + "; ".join(miss), file=sys.stderr); sys.exit(1)
    print(json.dumps(out))
