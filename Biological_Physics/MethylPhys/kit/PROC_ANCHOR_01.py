#!/usr/bin/env python3
"""PROC-ANCHOR-01 — reproduce the sealed foundation-cohort per-cell anchors from raw GEO betas.

Input   : data/GSE51032_series_matrix.txt.gz (3.0 GB) and/or data/GSE51057_series_matrix.txt.gz (1.2 GB)
          anchors_v1/<GSE>_115celltype_ascores.csv   (sealed 2026-05-29, repo foundation_cohort/)
          anchors_v2/<GSE>_115celltype_ascores_v2_chrXremoved.csv (re-sealed 2026-09-19, RULING M1b)
Operation: per cell type, A = mean_i H(beta_i)/H_min(class) over the cell's discriminative markers (RULING A3 separation surface)
Expected : v1 anchor with repo-HEAD markers: r >= 0.9999, max|diff| < 1e-3
           v2 anchor with chrX-removed markers: same
Usage    : python PROC_ANCHOR_01.py [GSE51032] [GSE51057]
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpg_kit as K

def run(gse):
    ident = K.load_identity()
    out = []
    for which, anchor in [("repo_head", f"anchors_v1/{gse}_115celltype_ascores.csv"),
                          ("chrX_removed", f"anchors_v2/{gse}_115celltype_ascores_v2_chrXremoved.csv")]:
        markers, c2c = K.load_markers(which)
        sealed = pd.read_csv(os.path.join(K.ROOT, anchor))
        cells = [c for c in sealed.columns if c not in ("gsm", "arm")]
        need = set(); [need.update(markers.get(ct, [])) for ct in cells]
        gsms, B = K.stream_geo_matrix(os.path.join(K.DATA, f"{gse}_series_matrix.txt.gz"), keep_cpgs=need, keep_gsms=set(sealed.gsm))
        S = sealed.set_index("gsm").loc[gsms, cells].to_numpy(float)
        X = np.full_like(S, np.nan)
        for j, ct in enumerate(cells):
            loci = [c for c in markers.get(ct, []) if c in B]
            if loci: X[:, j] = np.nanmean(K.H(np.vstack([B[c] for c in loci])), axis=0) / ident[c2c[ct]]["H_min"]
        ok = np.isfinite(X) & np.isfinite(S)
        r = float(np.corrcoef(X[ok], S[ok])[0, 1]); mx = float(np.abs(X - S)[ok].max())
        good = sum(1 for j in range(len(cells)) if np.isfinite(X[:, j]).all() and np.corrcoef(X[:, j], S[:, j])[0, 1] > 0.999)
        verdict = "PASS" if (r >= 0.9999 and mx < 1e-3) else "FAIL"
        out.append((f"{gse} {which}", f"n={len(gsms)} r={r:.5f} max|diff|={mx:.5f} cells r>0.999: {good}/{len(cells)} -> {verdict}"))
    K.report(f"PROC-ANCHOR-01 {gse}", out, "PASS" if all(v.endswith("PASS") for _, v in out) else "FAIL",
             path=os.path.join(K.ROOT, "results", f"PROC_ANCHOR_01_{gse}.json"))

if __name__ == "__main__":
    os.makedirs(os.path.join(K.ROOT, "results"), exist_ok=True)
    for gse in (sys.argv[1:] or ["GSE51057", "GSE51032"]): run(gse)
