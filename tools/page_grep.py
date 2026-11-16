#!/usr/bin/env python3
"""Read an open web page or PDF ON A GITHUB MACHINE and report only the sentences matching a pattern (2026-11-15; PDF work is banned on the
owner's laptop and results must stay a few KB over the hotspot). Usage (explore.yml): tools/page_grep.py <url> <regex> [max sentences, 12]"""
import sys, re, io, subprocess, requests
url, pat = sys.argv[1], re.compile(sys.argv[2], re.I); n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
r = requests.get(url, timeout=120); b = r.content
if b[:4] == b"%PDF":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypdf"], check=True)
    from pypdf import PdfReader
    t = " ".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(b)).pages)
else:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", r.text, flags=re.S); t = re.sub(r"<[^>]+>", " ", t)
t = re.sub(r"\s+", " ", t); S = [s for s in re.split(r"(?<=[.;])\s+", t) if pat.search(s)]
print(f"::notice::{url[:80]} status {r.status_code}, {len(t)} chars, {len(S)} matching sentences")
for s in S[:n]: print("::notice::" + s[:290])
