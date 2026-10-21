"""DEV-MAPPABILITY-01, exploration (2026-10-21, not sealed): do the commissioned whole-blood readings lean on probes that cannot be placed
uniquely in the genome? Hard-to-map regions (near-identical copies) mix two places in one probe's signal. The 27,117 EPIC v1 CpG probes whose
own sequence maps with mapQ < 10 or not at all (epic_v1_poor_mapping_probes.txt.gz, from a public per-probe annotation) are removed from each
array after Stage 1, and the array is read again by the chain's conductor (conductor_v3.run_neutrophil) and Stage D21 (stage_v4.stage_d21),
unchanged. Healthy whole bloods of GSE318173 (Group-A's controls, by slide and position after the first 160).
Usage: python3 masked_reread_01.py OUTDIR [N]. Writes OUTDIR/masked_reread_01_rows.csv."""
import os, sys, gzip, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", "..")); CH = os.path.join(MP, "chain")
sys.path[:0] = [CH, os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING")]
import conductor_v3 as C, stage_v4 as V4, stage_1_idat_calibration as S1
from read_groupa import fetch
out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 60; os.makedirs(out, exist_ok=True)
MASK = pd.Index([l.strip() for l in gzip.open(os.path.join(H, "epic_v1_poor_mapping_probes.txt.gz"), "rt") if l.strip() and not l.startswith("#")])
S = pd.read_csv(os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING", "groupa_samples.csv")).sort_values(["slide", "position"]).iloc[160:160 + N]
rows = []
for r in S.itertuples():
    try:
        g = fetch(r.gsm, r.grn, os.path.join(out, r.grn)); d = fetch(r.gsm, r.red, os.path.join(out, r.red))
        beta, _ = S1.calibrate_idat_to_beta(g, d, verbose=False); os.remove(g); os.remove(d)
        beta = pd.Series(beta, dtype="float64"); beta.index = beta.index.astype(str)
        for case, b in (("as read", beta), ("poor-mapping probes removed", beta.drop(MASK.intersection(beta.index)))):
            o = C.run_neutrophil(b, specimen="whole blood"); m = o["met_a"]; c = o.get("met_a_cscore") or {}
            sh = {x["cell"]: x["share"] for x in (V4.stage_d21(b).get("cells") or [])}
            rows.append(dict(gsm=r.gsm, slide=str(r.slide), case=case, n_probes=int(b.notna().sum()), A=m.get("A"), sites=m.get("n_sites") or m.get("sites_measured"),
                             C=c.get("C") if isinstance(c, dict) else None, f_neu=(o.get("composition") or {}).get("fractions", {}).get("NEU"), **{f"share_{k}": v for k, v in sh.items()}))
    except Exception as e:
        rows.append(dict(gsm=r.gsm, slide=str(r.slide), case="ERROR", error=str(e)[:200]))
    print(r.gsm, rows[-1].get("case"), rows[-1].get("A"), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, "masked_reread_01_rows.csv"), index=False); print("rows", len(rows))
