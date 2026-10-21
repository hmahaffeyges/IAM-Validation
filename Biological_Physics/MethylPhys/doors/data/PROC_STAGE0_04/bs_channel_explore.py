"""PROC-STAGE0-04, exploratory (2026-10-20, not sealed, no bar): is the bisulfite-conversion metric reading some of its control pairs in the
wrong color channel? Every healthy array so far sits at 0.64-0.83 against the SOP's 0.95 (731 healthy arrays, PROC-STAGE0-02; every
Group-A rehearsal array 0.66-0.70). The QC hand-off (chain/stage_0_1_qc_handoff.py) takes every BS Conversion I pair C_k / (C_k + U_k)
in the GREEN channel. Hypothesis (Chain Decoder finding F8): on Illumina arrays some of those pairs are read in RED (the manifest's
Color column says which), so in green they sit at background on both probes and give about 0.5, pulling the median down.
For each spent healthy array: every BS Conversion I control with its manifest color, its green and red intensity, then the efficiency
three ways: (a) as the chain does it, (b) each pair in its own manifest color, (c) the green pairs only. Uses the chain's own manifest
loader and IDAT reader. Writes bs_channel_explore.json and _summary.txt.
Usage: python3 bs_channel_explore.py OUTDIR N   (the first N arrays of the Group-A healthy list, plus GSE235717 GSM7508429)"""
import os, sys, json, re, urllib.request, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", ".."))
sys.path.insert(0, os.path.join(MP, "chain")); sys.path.insert(0, os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING"))
import stage_0_1_qc_handoff as QC
from read_groupa import fetch
out, N = sys.argv[1], int(sys.argv[2]); os.makedirs(out, exist_ok=True)
S = pd.read_csv(os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING", "groupa_samples.csv")).head(N)
arrays = [(r.gsm, r.grn, r.red) for r in S.itertuples()]
u = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM7508nnn/GSM7508429/suppl/"; L = urllib.request.urlopen(u).read().decode()
f = sorted(set(re.findall(r"GSM7508429_[^\"]*_(?:Grn|Red)\.idat\.gz", L))); arrays.append(("GSM7508429", [x for x in f if "Grn" in x][0], [x for x in f if "Red" in x][0]))
ctl = QC._manifest("EPIC_V1").control_data_frame
bs = ctl[ctl["Control_Type"].astype(str) == "BISULFITE CONVERSION I"].copy()
colcol = next((c for c in bs.columns if c.lower() in ("color", "colour")), None)
bs["key"] = bs["Extended_Type"].astype(str).str.replace("-", " ", regex=False).str.upper().str.replace("BS CONVERSION I", "", regex=False).str.strip()
rows, res = [], []
for gsm, gn, rn in arrays:
    g = fetch(gsm, gn, os.path.join(out, gn)); r = fetch(gsm, rn, os.path.join(out, rn))
    gm = QC._channel_means(g, "grn")["mean"]; rm = QC._channel_means(r, "red")["mean"]
    P = {}
    for addr, x in bs.iterrows():
        k = x["key"]; kind, num = k[:1], (k[1:].strip() or "1"); col = str(x[colcol]) if colcol else ""
        P.setdefault(num, {})[kind] = dict(addr=int(addr), color=col, grn=float(gm.get(addr, np.nan)), red=float(rm.get(addr, np.nan)))
        rows.append(dict(gsm=gsm, probe=x["Extended_Type"], color=col, grn=round(float(gm.get(addr, np.nan)), 1), red=round(float(rm.get(addr, np.nan)), 1)))
    a, b, c = [], [], []
    for num, p in sorted(P.items()):
        if "C" not in p or "U" not in p: continue
        C, U = p["C"], p["U"]; a.append(C["grn"] / (C["grn"] + U["grn"]))
        ch = "red" if "red" in C["color"].lower() else "grn"; b.append(C[ch] / (C[ch] + U[ch]))
        if ch == "grn": c.append(C["grn"] / (C["grn"] + U["grn"]))
    q = QC.decode_qc_inputs(g, r, "EPIC_V1")["control_summary"]
    res.append(dict(gsm=gsm, pairs=len(a), chain_as_is=round(float(np.median(a)), 4), chain_record=round(float(q["bisulfite_conversion_I_median"]), 4),
                    own_color=round(float(np.median(b)), 4), green_only=round(float(np.median(c)), 4) if c else None,
                    per_pair_as_is=[round(v, 3) for v in a], per_pair_own_color=[round(v, 3) for v in b]))
    for p_ in (g, r): os.remove(p_)
    print(res[-1], flush=True)
json.dump(dict(color_column=colcol, probes=rows, arrays=res), open(os.path.join(out, "bs_channel_explore.json"), "w"), indent=1)
T = pd.DataFrame(res)
L = [f"arrays {len(T)}; manifest color column: {colcol}",
     f"chain as it is (every pair in green): median {T.chain_as_is.median():.4f}, range {T.chain_as_is.min():.4f}-{T.chain_as_is.max():.4f}; equals the chain's record: {bool((T.chain_as_is - T.chain_record).abs().max() < 1e-4)}",
     f"each pair in its own manifest color: median {T.own_color.median():.4f}, range {T.own_color.min():.4f}-{T.own_color.max():.4f}",
     f"green pairs only: median {T.green_only.median():.4f}" if T.green_only.notna().any() else "green pairs only: none"]
Pr = pd.DataFrame(rows)   # per probe, the median over arrays of each channel (2026-10-20: the per-pair table, so the summary alone answers it)
L.append("per probe, median over arrays: probe | manifest color | green | red")
for (pr, col), g in Pr.groupby(["probe", "color"], sort=True): L.append(f"  {pr} | {col} | {g.grn.median():.0f} | {g.red.median():.0f}")
L.append("per pair, median over arrays of C/(C+U): pair | in green | in red")
for k in sorted({str(x)[-1] for x in Pr.probe}):
    c = Pr[Pr.probe.str.replace("-", " ").str.upper().str.endswith("C" + k)]; u = Pr[Pr.probe.str.replace("-", " ").str.upper().str.endswith("U" + k)]
    if len(c) and len(u):
        m = c.merge(u, on="gsm", suffixes=("_c", "_u"))
        L.append(f"  pair {k} | {(m.grn_c / (m.grn_c + m.grn_u)).median():.4f} | {(m.red_c / (m.red_c + m.red_u)).median():.4f}")
open(os.path.join(out, "bs_channel_explore_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
