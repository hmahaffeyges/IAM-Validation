#!/usr/bin/env bash
# Build the website (and the PDFs) from docs/book/. Used by .github/workflows/website.yml; runs the same locally:
#   bash website/build/build.sh            -> website/_site/ (open website/_site/index.html, or: python3 -m http.server -d website/_site)
#   SKIP_PDF=1 bash website/build/build.sh -> HTML only
# Needs: latexml, a TeX Live with the book's packages, latexmk, poppler-utils (pdftoppm), python3 with numpy scipy sympy beautifulsoup4 matplotlib pillow.
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

# 5. PDFs: the whole book and one per Part (front matter + that Part + bibliography)
if [ -z "${SKIP_PDF:-}" ]; then
  python3 "$REPO/website/build/part_pdfs.py" "$B/pdf_src"
  mkdir -p "$SITE/pdf"
  cd "$B/pdf_src"
  latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet main.tex > /dev/null && cp main.pdf "$SITE/pdf/IAMs_Law_and_Order.pdf"
  for n in 1 2 3 4 5 6 7; do
    latexmk -pdf -interaction=nonstopmode -quiet "part-$n.tex" > /dev/null || true
    [ -f "part-$n.pdf" ] && cp "part-$n.pdf" "$SITE/pdf/"
  done
fi
echo "built: $SITE"
