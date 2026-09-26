#!/bin/sh
# Two passes. The first renders the document with the contents page numbers blank; om_part3 then locates each
# chapter in the rendered PDF and writes toc_pages.json; the second pass renders the contents with real numbers.
# A page number in the contents is therefore a measurement of the document, not a hand-maintained list.
set -e
export IAM_TWOPASS=1   # the single-pass build refuses to run without this: see the gate in build_operations_manual.py
OUT="$1"
python3 build_operations_manual.py "$OUT" >/dev/null
python3 - "$OUT" <<'PY'
import sys
sys.path.insert(0, ".")
import gape002_lib as L, om_data as D, om_part3 as P3
L.CARD_NUMBER_BY_POSITION = True
pages, missing = P3.collect_toc_pages(sys.argv[1], L, D)
print("toc: %d of %d chapters located%s" % (len(pages), len(pages) + len(missing),
      ("; NOT FOUND: " + ", ".join(missing)) if missing else ""))
PY
python3 build_operations_manual.py "$OUT" | tail -1
