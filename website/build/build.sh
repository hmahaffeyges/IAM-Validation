#!/usr/bin/env bash
# Build the website (and the PDFs) from docs/book/. Used by .github/workflows/website.yml; runs the same locally:
#   bash website/build/build.sh            -> website/_site/ (open website/_site/index.html, or: python3 -m http.server -d website/_site)
#   SKIP_PDF=1 bash website/build/build.sh -> HTML only
# Needs: latexml, a TeX Live with the book's packages (for LaTeXML), tectonic (for the PDFs), epubcheck and java (for the EPUB), poppler-utils (pdftoppm, pdfinfo), python3 with numpy scipy sympy beautifulsoup4 matplotlib pillow.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
B="$REPO/website/_build"
SITE="$REPO/website/_site"
rm -rf "$B"; mkdir -p "$B"

# 1. build copy of the sources with check markers and web figures (docs/book/ is never written)
python3 "$REPO/website/build/prepare_sources.py" "$B"
cp "$REPO"/website/latexml/*.ltxml "$B/src/"

# 2. LaTeX -> XML -> one HTML page per chapter
cd "$B/src"
latexml --preload=iam.ltxml --path=. --dest="$B/book.xml" --log="$B/latexml.log" main.tex
latexml --preload=amssymb.sty --preload=url.sty --dest="$B/iam.bib.xml" --log="$B/bib.log" iam.bib
latexmlpost --dest="$B/html/index.html" --format=html5 --splitat=chapter --splitnaming=label \
  --bibliography="$B/iam.bib.xml" --navigationtoc=context --urlstyle=file --sourcedirectory="$B/src" \
  --log="$B/latexmlpost.log" "$B/book.xml"

# 3. banner art (scripts committed under website/art/)
python3 "$REPO/website/art/make_banners.py" > /dev/null

# 3b. reading themes (Paper, Dark, Sepia): writes website/static/themes.css; fails the build if a text colour misses WCAG AA
python3 "$REPO/website/build/themes.py"

# 4. site: front page, Part pages, chapter chrome, check buttons, verify_book.py + data for the browser
python3 "$REPO/website/build/build_site.py" "$B" "$SITE"

# 4b. EPUB edition (Apple Books and other e-readers) from the same LaTeXML XML: LaTeXML's EPUB 3 pages (math as MathML, figures as
#     the build's PNGs), packaged with the four-skies cover by make_epub.py, then checked with epubcheck: any epubcheck error fails
#     the build. EPUBCHECK_JAR names the checker (default: the Ubuntu epubcheck package's jar).
cd "$B/src"
latexmlpost --dest="$B/epub_pages/index.xhtml" --format=xhtml --stylesheet=LaTeXML-epub3.xsl --splitat=chapter --splitnaming=label \
  --bibliography="$B/iam.bib.xml" --urlstyle=file --sourcedirectory="$B/src" --log="$B/epubpost.log" "$B/book.xml" > /dev/null 2>&1 \
  || { echo "EPUB pages FAILED (latexmlpost); end of $B/epubpost.log:"; tail -n 40 "$B/epubpost.log"; exit 1; }
python3 "$REPO/website/build/make_epub.py" "$B/epub_pages" "$SITE/epub/IAMs_Law_and_Order.epub"
java -jar "${EPUBCHECK_JAR:-/usr/share/java/epubcheck.jar}" "$SITE/epub/IAMs_Law_and_Order.epub" > "$B/epubcheck.txt" 2>&1 \
  || { echo "epubcheck FAILED:"; grep -E "^(FATAL|ERROR)" "$B/epubcheck.txt" | head -n 40; tail -n 3 "$B/epubcheck.txt"; exit 1; }
echo "epubcheck: $(grep -E '^Messages:' "$B/epubcheck.txt")"

# 5. PDFs with tectonic, the engine the book is compiled with: the whole book and one per Part (front matter + that Part +
#    bibliography). A failed compile prints the end of its log and stops the build, and the build fails if any PDF is missing,
#    so the site can never deploy without them. TECTONIC names the binary (default: tectonic on PATH).
if [ -z "${SKIP_PDF:-}" ]; then
  TECTONIC="${TECTONIC:-tectonic}"
  python3 "$REPO/website/build/part_pdfs.py" "$B/pdf_src"
  mkdir -p "$SITE/pdf"
  cd "$B/pdf_src"
  compile() {   # $1 = driver without .tex
    if ! "$TECTONIC" -X compile --keep-logs "$1.tex" > "$B/pdf-$1.out" 2>&1; then
      echo "PDF FAILED: $1.tex (tectonic). End of its output:"; tail -n 80 "$B/pdf-$1.out"; return 1
    fi
    echo "pdf: $1.pdf ($(pdfinfo "$1.pdf" 2>/dev/null | awk '/^Pages:/{print $2}') pages)"
  }
  compile main                                  # first: fills tectonic's package cache for the Part builds
  pids=()
  for n in 1 2 3 4 5 6 7; do compile "part-$n" & pids+=($!); done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [ "$failed" = 0 ] || { echo "Build failed: a Part PDF did not compile."; exit 1; }
  cp main.pdf "$SITE/pdf/IAMs_Law_and_Order.pdf"
  for n in 1 2 3 4 5 6 7; do cp "part-$n.pdf" "$SITE/pdf/"; done
fi

# 6. Never publish a site without its EPUB and its PDFs (the PDF check is skipped only for a local HTML-only build: SKIP_PDF=1)
[ -s "$SITE/epub/IAMs_Law_and_Order.epub" ] || { echo "MISSING: epub/IAMs_Law_and_Order.epub"; echo "Build failed: the site is missing its EPUB."; exit 1; }
if [ -z "${SKIP_PDF:-}" ]; then
  missing=0
  for f in IAMs_Law_and_Order.pdf part-1.pdf part-2.pdf part-3.pdf part-4.pdf part-5.pdf part-6.pdf part-7.pdf; do
    if [ ! -s "$SITE/pdf/$f" ]; then echo "MISSING: pdf/$f"; missing=1; fi
  done
  [ "$missing" = 0 ] || { echo "Build failed: the site is missing PDFs."; exit 1; }
  echo "pdfs: all 8 present in $SITE/pdf"
fi
echo "built: $SITE"
