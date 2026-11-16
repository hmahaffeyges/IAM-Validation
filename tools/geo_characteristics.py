#!/usr/bin/env python3
"""GEO sample characteristics on a GitHub machine (2026-11-15; register A8, the history test). For each series given, reads GEO's brief
sample text and writes one row per sample: accession, title, source, and every 'characteristics' field (age, sex, cell type, donor...) to
xout/geo_chars_<GSE>.csv, a few KB, so the owner's session fetches only that table. Prints one notice per series with the fields found.
Usage (explore.yml): tools/geo_characteristics.py GSE173787 GSE186458"""
import sys, os, re, csv, time, urllib.request
os.makedirs("xout", exist_ok=True)
for g in sys.argv[1:]:
    t = urllib.request.urlopen(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={g}&targ=gsm&form=text&view=brief", timeout=180).read().decode(errors="replace")
    rows, keys = [], []
    for b in t.split("^SAMPLE")[1:]:
        r = {"gsm": re.search(r"!Sample_geo_accession = (\S+)", b).group(1),
             "title": (re.search(r"!Sample_title = (.*)", b) or [None, ""])[1].strip(),
             "source": (re.search(r"!Sample_source_name_ch1 = (.*)", b) or [None, ""])[1].strip(),
             "strategy": (re.search(r"!Sample_library_strategy = (.*)", b) or [None, ""])[1].strip()}
        for ch in re.findall(r"!Sample_characteristics_ch1 = (.*)", b):
            k, _, v = ch.partition(":"); k = k.strip().lower().replace(" ", "_")[:30]
            r[k] = v.strip()
            if k not in keys: keys.append(k)
        rows.append(r)
    with open(f"xout/geo_chars_{g}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gsm", "title", "source", "strategy"] + keys, extrasaction="ignore"); w.writeheader(); [w.writerow(x) for x in rows]
    print(f"::notice::{g}: {len(rows)} samples; fields {keys[:12]}")
    time.sleep(2)
