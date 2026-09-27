#!/usr/bin/env python3
"""Build percell_reference_mcmc_v1.json - where each cell's OWN reference sits on the identity surface, and how well
the MCMC atlas knows it (author, 2026-09-26: 'are we using the CI we have available from the MCMC created atlas?').

For every cell with identity loci: draw the atlas mean at each locus from N(mu, sd) - the per-locus posterior the
G-002 MCMC left in IAMAtlasREBUILD.csv - 2,000 times, compute A = H(mean beta) / H_min[class] per draw, and record the
median and the 2.5 / 97.5 percentiles. This is a PHYSICS tolerance - the atlas's own uncertainty about the cell's
fixed point - not a population's spread. No specimen is read. The identity loci sit within +-0.05 of H_min_beta, so a
cell's reference lands a little under 1.00 by construction; the file records that offset so the report can say
'this cell's reference reads 0.9945 [0.9924-0.9964]' rather than assume 1.000.
"""
import json, math, os, sys
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); CH=os.path.join(os.path.dirname(HERE),"chain"); sys.path.insert(0,CH)
import cpg_conductor as C  # noqa
def H(b): b=min(max(b,1e-12),1-1e-12); return -b*math.log2(b)-(1-b)*math.log2(1-b)
def main(atlas_csv, n_draws=2000, seed=0):
    pci=json.load(open(C._find("iamatlas_percell_identity_loci_v1_0.json"))); pci=pci.get("cells",pci)
    ident=json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
    head=pd.read_csv(atlas_csv,nrows=0).columns.tolist(); cells=[c for c in pci if isinstance(pci[c],dict) and pci[c].get("loci") and c+"_mean" in head and c+"_sd" in head]
    cols=[head[0]]+[c+s for c in cells for s in ("_mean","_sd")]
    A=pd.read_csv(atlas_csv,usecols=cols,index_col=0); A.index=A.index.map(str); rng=np.random.default_rng(seed); out={}
    for c in cells:
        e=pci[c]; cl=e.get("class"); hm=(ident.get(cl) or {}).get("H_min")
        if hm is None: continue
        sub=A.loc[[l for l in map(str,e["loci"]) if l in A.index],[c+"_mean",c+"_sd"]].dropna()
        if len(sub)<50: out[c]={"class":cl,"n_loci":int(len(sub)),"status":"INSUFFICIENT_LOCI"}; continue
        mu=sub[c+"_mean"].values; sd=sub[c+"_sd"].values
        draws=np.array([H(float(np.mean(np.clip(rng.normal(mu,sd),0,1))))/hm for _ in range(n_draws)])
        out[c]={"class":cl,"H_min":hm,"n_loci":int(len(sub)),"ref_A":H(float(mu.mean()))/hm,"ref_A_median":float(np.median(draws)),
                "ci95_lo":float(np.percentile(draws,2.5)),"ci95_hi":float(np.percentile(draws,97.5)),"median_locus_sd":float(np.median(sd)),"status":"OK"}
    meta={"built":"2026-09-26","source":"IAMAtlasREBUILD.csv per-locus posterior mean and sd (G-002 MCMC); identity loci iamatlas_percell_identity_loci_v1_0.json; H_min iamatlas_gauge_identity_loci_v1_0.json",
          "definition":"reference A = H(mean over the cell's identity loci of the atlas mean)/H_min[class]; ci95 from 2,000 draws of the atlas means from N(mu, sd) per locus",
          "what_it_is":"the atlas's own uncertainty about where this cell's A = 1 reference sits - a physics tolerance, not a population spread","n_draws":n_draws,"seed":seed}
    p=os.path.join(CH,"Runtime Matrices","Percell_Reference","percell_reference_mcmc_v1.json"); os.makedirs(os.path.dirname(p),exist_ok=True)
    json.dump({"_meta":meta,"entries":out},open(p,"w"),indent=1)
    ok=[v for v in out.values() if v.get("status")=="OK"]; w=[v["ci95_hi"]-v["ci95_lo"] for v in ok]; r=[v["ref_A"] for v in ok]
    print(f"wrote {p}: {len(ok)} cells | ref A median {np.median(r):.4f} (min {min(r):.4f} max {max(r):.4f}) | ci95 width median {np.median(w):.4f} max {max(w):.4f}")
if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else str(C._find("IAMAtlasREBUILD.csv")))
