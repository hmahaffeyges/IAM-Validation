#!/usr/bin/env python3
"""Summarize a geo_characteristics run ON A GITHUB MACHINE (2026-11-15; owner's hotspot: only a few KB come back). Downloads the run's
explore-out artifact here, reads every geo_chars_<GSE>.csv, and writes ONLY xout/geo_summary.txt: one line per series with its sample count,
library strategies, the commonest cell-type / tissue / source values, and how many samples look healthy (control / healthy / normal).
Usage (explore.yml): tools/geo_summary.py <run id of a geo_characteristics job>"""
import os, sys, io, zipfile, collections, requests, pandas as pd
H = {"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"}; R = "https://api.github.com/repos/" + os.environ["GITHUB_REPOSITORY"]
A = requests.get(f"{R}/actions/runs/{sys.argv[1]}/artifacts", headers=H, timeout=60).json()["artifacts"]
z = zipfile.ZipFile(io.BytesIO(requests.get(A[0]["archive_download_url"], headers=H, timeout=300).content))
L = []
for n in sorted(m for m in z.namelist() if os.path.basename(m).startswith("geo_chars_")):
    d = pd.read_csv(io.BytesIO(z.read(n)), dtype=str).fillna(""); g = os.path.basename(n)[10:-4]
    strat = dict(collections.Counter(d.strategy).most_common(3))
    kind = next((c for c in ("cell_type", "cell.type", "cell_types", "tissue", "source") if c in d.columns and d[c].str.len().sum()), "source")
    top = "; ".join(f"{k} {v}" for k, v in collections.Counter(d[kind]).most_common(5))
    blob = d.astype(str).agg(" ".join, axis=1).str.lower()
    healthy = int(blob.str.contains(r"healthy|control|normal|\bhc\b").sum())
    L.append(f"{g} | n {len(d)} | {strat} | {kind}: {top[:220]} | healthy-like {healthy}")
os.makedirs("xout", exist_ok=True); open("xout/geo_summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
print(f"::notice::{len(L)} series summarized into xout/geo_summary.txt")
