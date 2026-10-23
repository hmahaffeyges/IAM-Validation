"""DEV-REPLICATE-01, replication R1 (written 2026-10-23 before it runs; tuning, nothing scored). The 11 arrays the earlier CPG engine was tested
on (its TEST_DATA_MANIFEST), each read by run_sample.py. This scorer only summarizes the bundles: each array's IDAT fingerprints against the
author's local copies (r1_local_idat_sha256.txt), its cell shares, Met-A for the whole sample (with its parts), each cell's Met-A, the
territories and the C-Scores.  Usage: python3 r1_score.py ROWS_CSV OUTDIR (the series workflow's scorer call)"""
import os, sys, json, glob
H = os.path.dirname(os.path.abspath(__file__)); src = sys.argv[1] if len(sys.argv) > 1 else "."; out = os.path.dirname(os.path.abspath(src)) if os.path.isfile(src) else src   # 2026-10-23: the series workflow passes its own arguments; write beside rows.csv
loc = {}
for l in open(os.path.join(H, "r1_local_idat_sha256.txt")):
    s, n = l.split(None, 1); loc[n.strip().replace(".idat.gz", "")] = s
def find(o, k):
    if isinstance(o, dict):
        for a, b in o.items():
            if a == k: return b
            r = find(b, k)
            if r is not None: return r
    if isinstance(o, list):
        for b in o:
            r = find(b, k)
            if r is not None: return r
    return None
L = ["DEV-REPLICATE-01 R1: the earlier CPG engine's 11 test arrays through run_sample.py (pass 1, untared)"]
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(src)) if os.path.isfile(src) else src, "**", "GSM*.json"), recursive=True)):
    if f.endswith(("_values.json", "_planted.json", "_info_sphere.json")): continue
    B = json.load(open(f)); g = os.path.basename(f).split("_")[0].split(".")[0]
    fp = [find(B, "grn_sha256"), find(B, "red_sha256")]; same = [fp[0] == loc.get(g + "_Grn"), fp[1] == loc.get(g + "_Red")]
    cs = sorted((B.get("cell_shares") or {}).get("cells") or [], key=lambda c: -(c.get("share") or 0))
    W = B.get("met_a_whole") or {}; T = (B.get("territories") or {}).get("sets") or {}
    ec = sorted([(c["cell"], c.get("A")) for c in (B.get("met_a_each_cell") or {}).get("cells", []) if c.get("A") is not None], key=lambda x: -(x[1] or 0))[:5]
    L.append(f"{g} | files match the author's copies: {same} | cells: " + ", ".join(f"{c['cell']} {c['share']:.3f}" for c in cs if (c.get('share') or 0) >= 0.03)
             + f" | whole A {W.get('A')} (" + ", ".join(f"{k} {v['A']}" for k, v in (W.get('by_place') or {}).items() if v) + ")"
             + " | highest cells " + ", ".join(f"{c} {a}" for c, a in ec) + " | territories " + ", ".join(f"{k} {v.get('A')}" for k, v in T.items())
             + f" | C whole genome {(((B.get('cscore_each_cell') or {}).get('whole_genome') or {}).get('C'))}" + (f" | refusal {B.get('refusal')}" if B.get("refusal") else ""))
os.makedirs(out, exist_ok=True); open(os.path.join(out, "r1_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
