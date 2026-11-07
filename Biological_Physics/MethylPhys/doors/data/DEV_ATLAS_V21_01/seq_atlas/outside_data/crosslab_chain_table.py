"""Development, after the fact (author: 'RUN THE FULL CHAIN ALWAYS AND LET THE CHAIN DECIDE'): every outside-laboratory sequencing sample the one
command has already read (89 bundles: BLUEPRINT 59, GSE245832 12, GSE98837 8, GSE134664 4, Johns Hopkins CD4 6), tabulated from what the chain
itself reported: the Deconvolver's top cell and share (not the label), whole-sample Met-A, Met-A at the identity sites of the cells found, Met-A of
the cell the Deconvolver found most of, depth (calls a site), and the library's false-call rate at the library-floor sites. Nothing is read again
and nothing is scored. Usage: python3 crosslab_chain_table.py BUNDLE_DIR OUT.csv (BUNDLE_DIR holds one folder per laboratory of *_bundle.json)."""
import json, glob, os, ast, sys, pandas as pd
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "*_bundle.json"))):
    B = json.load(open(f)); g = os.path.basename(f).split("_bundle")[0]; lab = os.path.basename(os.path.dirname(f))
    sh = sorted(((x["share"], x["cell"]) for x in (B.get("cell_shares") or {}).get("cells") or []), reverse=True)
    W = B.get("met_a_whole") or {}; bp = W.get("by_place"); bp = ast.literal_eval(bp) if isinstance(bp, str) else (bp or {})
    pc = {c["cell"]: (c["share"], c.get("A")) for c in ((B.get("met_a_each_cell") or {}).get("cells") or [])}
    lf = (B.get("development") or {}).get("library_floor") or {}; sv = B.get("sequencing_values") or {}
    rows.append(dict(lab=lab, sample=g, label=B.get("specimen"), top=sh[0][1] if sh else None, top_share=round(sh[0][0], 3) if sh else None,
                     second=f"{sh[1][1]} {sh[1][0]:.2f}" if len(sh) > 1 else "", whole=float(W["A"]) if W.get("A") not in (None, "None") else None,
                     id_sites=(bp.get("identity sites of the cells found") or {}).get("A"), top_cell_A=pc.get(sh[0][1], (None, None))[1] if sh else None,
                     depth=float(sv.get("calls_per_site_median") or "nan"), floor=lf.get("rate")))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
