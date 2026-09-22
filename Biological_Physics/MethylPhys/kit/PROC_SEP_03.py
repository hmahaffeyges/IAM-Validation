#!/usr/bin/env python3
"""PROC-SEP-03 — Tool B (lineage splitter) on the 11 Stage-1 test samples. Expect: 7/7 whole blood SPLIT, kappa 3.9-5.2,
stem_adult = 0.000; tissue with no haematopoietic-progenitor mass -> COMPARTMENT_ONLY (kappa = inf)."""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpg_kit as K
sys.path.insert(0, K.ENGINE)
from walther_iam_deconvolver import WaltherIAMDeconvolver; from lineage_splitter import LineageSplitter
def main():
    A=os.path.join(K.DATA,"IAMAtlasREBUILD.csv"); W=WaltherIAMDeconvolver(A, celltype_class_map=os.path.join(K.ENGINE,"IAMAtlasREBUILD_celltype_to_class.json"), verbose=False)
    L=LineageSplitter(A, min_delta=0.2); cache=pickle.load(open(os.path.join(K.DATA,"betas_cache.pkl"),"rb"))
    X=np.array(list(L.ref.values())); print(f"  contrast reference: {X.shape[0]} CpGs, kappa {np.linalg.cond(X):.2f}, r {np.corrcoef(X.T)[0,1]:+.3f}")
    ok=0; n=0
    for gsm,v in cache.items():
        d={k:float(x) for k,x in (v.items() if hasattr(v,'items') else enumerate(v)) if x==x}
        fa=W.deconvolve(d).class_fractions; comp=fa.get("progenitor",0)+fa.get("stem_adult",0)
        bg={c: sum(fa.get(k,0)*W.class_ref[c][k] for k in fa if k not in ("progenitor","stem_adult") and k in W.class_ref[c]) for c in L.ref if c in W.class_ref}
        s=L.split(d, comp, background=bg if len(bg)>50 else None)
        blood = fa.get("immune",0)>0.5; n+=blood
        if blood and s.status=="SPLIT" and s.kappa<10 and s.fractions["stem_adult"]<0.005: ok+=1
        print(f"  {gsm} comp {comp:.3f} {s.status:<16} kappa {s.kappa:6.2f} " + (f"prog/sa {s.fractions['progenitor']:.3f}/{s.fractions['stem_adult']:.3f}" if s.status=="SPLIT" else s.reason[:50]))
    print(f"  verdict      {'PASS' if ok==n else 'FAIL'} ({ok}/{n} blood samples split with kappa<10 and stem_adult<0.005)")
if __name__=="__main__": main()
