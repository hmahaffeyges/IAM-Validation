"""Chapter start pages, read from the compiled book itself (2026-10-13): the front page's chapter links must open the PDF on the
right page, and hand-typed page numbers drifted by two pages when chapter 48 grew. Reads the PDF's text page by page (pdftotext,
the poppler tool the build already uses) and finds the page where each chapter's opening line is printed.
Usage: python3 chapter_pages.py BOOK.pdf  -> prints JSON {key: page}; exits 1 if any chapter in build_home.CH is not found."""
import json, os, re, subprocess, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
def chapter_table():
    spec = importlib.util.spec_from_file_location("bh", os.path.join(HERE, "build_home.py")); src = open(spec.origin, encoding="utf-8").read()
    import ast; return ast.literal_eval(re.search(r"CH = (\{.*?\})\n", src, re.S).group(1))
def pages(pdf):
    t = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout.split("\f")
    T = [re.sub(r"\s+", " ", x).strip().lower() for x in t]
    out, miss = {}, []
    for k, (n, _, title) in chapter_table().items():
        head = title.split(":")[0][:30].lower()
        hit = [j + 1 for j, x in enumerate(T) if x.startswith(f"chapter {n} ") or (f"chapter {n} " in x[:60] and head in x[:300])]
        if hit: out[k] = hit[0]
        else: miss.append(f"{k} (chapter {n}, {title})")
    return out, miss
if __name__ == "__main__":
    out, miss = pages(sys.argv[1])
    if miss: print("chapter_pages: chapters not found in the compiled book: " + "; ".join(miss), file=sys.stderr); sys.exit(1)
    print(json.dumps(out))
