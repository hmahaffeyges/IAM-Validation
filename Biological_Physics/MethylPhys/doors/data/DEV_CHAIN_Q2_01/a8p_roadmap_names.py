#!/usr/bin/env python3
"""A8P step 1 (register A8, packing lead; exploration, 2026-11-16). Runs ON A GITHUB MACHINE. Lists the Roadmap Epigenomics 15-state ChromHMM
segmentation files (open) and the epigenome names, so the names table is written BEFORE any state share is read. Prints names only."""
import re, json, urllib.request
B = "https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/final/"
html = urllib.request.urlopen(B, timeout=120).read().decode("utf-8", "replace")
files = sorted(set(re.findall(r'href="(E\d{3}_15_coreMarks_mnemonics\.bed\.gz)"', html)))
print(f"::notice::A8P files: {len(files)} 15-state mnemonics files; first {files[:3]}")
names = {}
for u in ("https://egg2.wustl.edu/roadmap/data/byFileType/metadata/EID_metadata.tab",
          "https://raw.githubusercontent.com/nboley/roadmap_metadata/master/EID_metadata.tab"):
    try:
        t = urllib.request.urlopen(u, timeout=60).read().decode("utf-8", "replace"); print(f"::notice::A8P metadata from {u}: {len(t.splitlines())} lines")
        for l in t.splitlines()[1:]:
            p = l.split("\t")
            if p and re.match(r"E\d{3}$", p[0]): names[p[0]] = " | ".join(x for x in p[1:] if x)[:160]
        if names: break
    except Exception as e:
        print(f"::notice::A8P metadata {u}: {type(e).__name__} {str(e)[:80]}")
if not names:
    ht = urllib.request.urlopen("https://egg2.wustl.edu/roadmap/web_portal/meta.html", timeout=60).read().decode("utf-8", "replace")
    for m in re.finditer(r"(E\d{3})</t[dh]>\s*<t[dh][^>]*>([^<]{3,120})", ht): names.setdefault(m.group(1), m.group(2).strip())
keep = [f[:4] for f in files]
s = json.dumps({e: names.get(e, "?") for e in keep})
for i in range(0, len(s), 1800): print("::notice::A8P names part " + str(i // 1800 + 1) + ": " + s[i:i + 1800])
