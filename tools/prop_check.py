#!/usr/bin/env python3
"""Propagation check (2026-11-16; author: 'confirm everything is updated ... DO NOT simply check the definition is there ... THE MATH').
Runs ON A GITHUB MACHINE against main; reads text only; changes nothing. For each finding, searches each public place for its NUMBERS
(not its name) and prints one line per finding: the places where every number was found, partly found, or absent, plus the first
matching line in each. Writes xout/prop_check.txt (a few KB)."""
import os, re, glob
PLACES = {
    "canon": ["CANON/iam_canon.json"],
    "book": sorted(glob.glob("docs/book/part6/*.tex")) + sorted(glob.glob("docs/book/appendices/*.tex")),
    "decoder": ["Biological_Physics/MethylPhys/manual/decoder_text.md"],
    "devlog": ["Biological_Physics/MethylPhys/METHYLPHYS_DEVELOPMENT_LOG.md"],
    "four_domains": ["CANON/four_domains.json"],
    "simulator": sorted(glob.glob("Biological_Physics/MethylPhys/minicell/*.py")),
    "chain": sorted(glob.glob("Biological_Physics/MethylPhys/chain/*.py")) + sorted(glob.glob("Biological_Physics/MethylPhys/chain/MethylPhys_Interface/*.py")),
    "glossary": ["docs/book/appendices/app_F_glossary.tex"],
}
FIND = {   # numbers that carry the math of each finding
    "F006 phi derived + England bound": ["0.1464", "0.04454", "England", "3.11"],
    "F004 islands = meter": ["0.97", "0.0002", "0.0138"],
    "F005 meter in blood": ["conv_fail", "tc_err", "one-step"],
    "F022 clock sites writer": ["0.0289", "0.0311", "0.0164"],
    "F021 per-type A (one command)": ["0.948", "1.059", "0.77"],
    "F013 same-donor T cells": ["0.80", "1.06", "same donor"],
    "F023 Met-A array held height": ["0.9545", "0.959"],
    "F007 fish coin": ["55.6", "jeb.008763"],
    "F001 eraser drops out (mature blood)": ["0.96", "1.17", "drops out"],
    "F008 c is a fit": ["1.02", "0.9922"],
    "F009 fill history": ["0.159"],
    "edit019 phi_derived in simulator": ["phi_derived"],
    "edit036 site-set comma list": ["count_sites_all"],
}
txt = {}
for pl, fs in PLACES.items():
    t = []
    for f in fs:
        if os.path.exists(f):
            t.append((f, open(f, encoding="utf-8", errors="replace").read()))
    txt[pl] = t
out = []
for name, nums in FIND.items():
    row = []
    for pl, files in txt.items():
        hit = {n: next(((f, l.strip()[:120]) for f, s in files for l in s.split("\n") if n in l), None) for n in nums}
        k = sum(1 for v in hit.values() if v)
        if k == 0: continue
        first = next(v for v in hit.values() if v)
        row.append(f"{pl} {k}/{len(nums)} [{os.path.basename(first[0])}: {first[1][:90]}]")
    out.append(f"{name}: " + (" | ".join(row) if row else "ABSENT everywhere"))
os.makedirs("xout", exist_ok=True); open("xout/prop_check.txt", "w").write("\n".join(out) + "\n")
for l in out: print("::notice::" + l[:1000])
