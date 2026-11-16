#!/usr/bin/env python3
"""GEO lookup on a GitHub machine (2026-11-15). Each argument is an Entrez query on db=gds; prints accession, sample count, date and title of
up to 10 series per query as notices. Usage (explore.yml): tools/geo_lookup.py "query one" "query two" """
import sys, json, time, urllib.request, urllib.parse
for q in sys.argv[1:]:
    r = json.load(urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gds&retmax=10&retmode=json&term=" + urllib.parse.quote(q), timeout=60))
    ids = r["esearchresult"]["idlist"]; print(f"::notice::query [{q}] {len(ids)} hits")
    if ids:
        s = json.load(urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gds&retmode=json&id=" + ",".join(ids), timeout=60))["result"]
        for i in ids: x = s[i]; print(f"::notice::{x.get('accession')} n={x.get('n_samples')} {x.get('pdat')} {x.get('gpl')} | {x.get('title', '')[:150]}")
    time.sleep(1)
