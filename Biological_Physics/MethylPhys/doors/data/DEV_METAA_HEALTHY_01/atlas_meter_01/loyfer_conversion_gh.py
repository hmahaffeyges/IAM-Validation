#!/usr/bin/env python3
"""Register M2 (exploration; runs on GitHub explore.yml): the atlas laboratory's meter. The atlas (Loyfer et al. 2023, Nature 613:355,
doi 10.1038/s41586-022-05580-6; open access, PMC9811898) spiked lambda and pUC19 into every library (its methods: reads mapped to lambda and
pUC19 with bwa-meth). Its raw reads are gated (EGA), so they are not used. This reads the OPEN supplementary package from Europe PMC, opens every
spreadsheet, and writes every column whose name mentions conversion, lambda, pUC, non-CpG or CHH/CHG, with the sample identifiers, to
xout/loyfer_conversion.csv and a summary (medians by cell type where a type column exists). No chain reading is made; this is a meter input."""
import os, io, zipfile, re, urllib.request, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openpyxl", "pandas"], check=False)
import pandas as pd
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
u = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9811898/supplementaryFiles"
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(u, timeout=1800).read()))
L = [f"files in the open package: {[i.filename for i in z.infolist()]}"]
pat = re.compile(r"conver|lambda|puc|non.?cpg|chh|chg|bisulf", re.I); keep = []
for n in z.namelist():
    if n.lower().endswith(".zip"):
        L.append(f"{n}: nested archive, contents {zipfile.ZipFile(io.BytesIO(z.read(n))).namelist()[:12]}"); continue
    if not n.lower().endswith((".xlsx", ".xls", ".csv", ".tsv")): continue
    try:
        sheets = {"csv": pd.read_csv(io.BytesIO(z.read(n)), sep=None, engine="python")} if n.lower().endswith((".csv", ".tsv")) else pd.read_excel(io.BytesIO(z.read(n)), sheet_name=None)
    except Exception as e:
        L.append(f"{n}: could not open ({e})"); continue
    for sh, df in sheets.items():
        # 2026-11-15: the tables carry a title line above the real header; use the first row with at least 3 text cells as the header
        if all(str(c).startswith('Unnamed') for c in list(df.columns)[1:]):
            for i in range(min(6, len(df))):
                if sum(isinstance(v, str) for v in df.iloc[i]) >= 3:
                    df.columns = [str(v) for v in df.iloc[i]]; df = df.iloc[i + 1:].reset_index(drop=True); break
        hits = [c for c in df.columns if pat.search(str(c))]
        L.append(f"{n} / {sh}: {df.shape[0]} rows; columns {list(map(str, df.columns))[:40]}; meter columns {hits}")
        if sh == "Table S1": df.head(3).to_csv(os.path.join(OUT, "loyfer_tableS1_head.csv"), index=False)
        if hits:
            idc = [c for c in df.columns if re.search(r"sample|name|id|cell|type|gsm", str(c), re.I)][:4]
            t = df[idc + hits].copy(); t.insert(0, "source", f"{n}/{sh}"); keep.append(t)
            for c in hits:
                v = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(v): L.append(f"   {c}: n {len(v)}, median {v.median():.6g}, range {v.min():.6g} to {v.max():.6g}")
if keep: pd.concat(keep, ignore_index=True).to_csv(os.path.join(OUT, "loyfer_conversion.csv"), index=False)
L.append("Reading guide: the atlas meter (conversion failure per library) for M2's unmarked side, as read = 1/(1+D) + meter.")
open(os.path.join(OUT, "loyfer_conversion_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
