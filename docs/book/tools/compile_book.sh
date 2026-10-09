set -u
W=$(pwd); rm -rf src && mkdir src && tar xzf book.tgz -C src && cd src/IAMs_Law_and_Order
which pdflatex >/dev/null || { echo "no pdflatex on this host"; exit 2; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null 2>&1; bibtex main > bibtex.log 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
cp main.pdf main.log "$W"/ 2>/dev/null
echo "errors $(grep -c '^! ' main.log)"; grep -E "Warning: (Citation|Reference) .* undefined" main.log | sort -u | head -5
python3 -c "import re;t=open('main.log',errors='replace').read();m=re.findall(r'Output written on main.pdf \((\d+) pages',t);print('pages',m[-1] if m else None)"
