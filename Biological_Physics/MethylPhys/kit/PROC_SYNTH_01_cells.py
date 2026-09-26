#!/usr/bin/env python3
"""PROC-SYNTH-01 step 2: hand the chain a PURE CELL TYPE and see whether it names it and scores it.

This is the decisive test of the architecture the author specified - deconvolve to cell types, then score each
against its architecture class's H_min - because for a pure cell type the truth is exact and computable:

    fraction    the named cell must read 1.0
    A           mean_i H(mu_ct,i) / H_min[class(ct)]  over that cell's OWN markers
                (the per-cell formula is mean-of-per-CpG-H, NEVER H(mean beta) - LESSON-ASCORE-02)

Step 1 built pure specimens from CLASS means, which is not a real cell and so decomposes into a blend. That
was a flaw in the test, not the chain.

Through run_sample.py, as every procedure must be from 2026-09-26 on.
"""
import json
import math
import os
import subprocess
import sys

import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
RUN = f"{CH}/MethylPhys_Interface/run_sample.py"
OUT = "results/synth01"
ATLAS = "atlas_work/IAMAtlasREBUILD.csv"


def H(b):
    b = min(max(float(b), 1e-12), 1 - 1e-12)
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def chain_run(name, beta):
    os.makedirs(OUT, exist_ok=True)
    csv = f"{OUT}/{name}.csv"
    pd.Series(beta, name="beta").to_csv(csv, index_label="cpg")
    subprocess.run([sys.executable, RUN, "--betas", csv, "--age", "55", "--sex", "F",
                    "--lab", "SYNTH", "--specimen", "whole blood", "--no-intake",
                    "--out", f"{OUT}/{name}.html", "--id", name],
                   capture_output=True, text=True,
                   env={**os.environ, "HOME": os.path.abspath("stage1/mp_home")})
    b = f"{OUT}/{name}_bundle.json"
    return json.load(open(b)) if os.path.exists(b) else None


def main():
    sys.path.insert(0, CH)
    import cpg_conductor as C
    asc_path = C._find("iamatlas_a_scoring.py")
    sys.path.insert(0, os.path.dirname(str(asc_path)))
    import iamatlas_a_scoring as asc
    markers_path = C._find("iamatlas_celltype_markers_v0_2.json", required=False)
    if markers_path is None:
        import glob
        cands = glob.glob(f"{CH}/**/*celltype_marker*.json", recursive=True) + \
            glob.glob(f"{CH}/**/*markers*.json", recursive=True)
        markers_path = cands[0] if cands else None
    print("markers artifact:", str(markers_path).split("/")[-1], flush=True)
    _meta, ct_markers, c2c, h_min = asc.load_artifact(str(markers_path))
    print(f"cell types with markers: {len(ct_markers)} | H_min classes: {len(h_min)}", flush=True)

    head = pd.read_csv(ATLAS, nrows=0).columns.tolist()
    # the cells to test: one per architecture class where a marker set exists, plus the organs that matter
    want = ["CD4_T-cells", "Neutrophils_reinius", "HSC", "GMP", "Breast",
            "Colon_epithelial_cells", "fibroblast", "Cortical_neurons", "stem_pluri"]
    want = [w for w in want if w in ct_markers and f"{w}_mean" in head]
    print("testing:", want, flush=True)
    mu = pd.read_csv(ATLAS, usecols=[head[0]] + [f"{w}_mean" for w in want]).set_index(head[0])
    mu.columns = [c[:-5] for c in mu.columns]

    rows = []
    print(f"\n{'cell handed in':<24}{'named?':>8}{'f read':>9}{'A read':>10}{'A expected':>12}{'diff':>9}{'class':>12}",
          flush=True)
    for ct in want:
        beta = mu[ct].dropna()
        if len(beta) < 2000:
            print(f"{ct:<24}  only {len(beta)} loci in the atlas - skipped", flush=True)
            continue
        bun = chain_run(f"cell_{ct}", beta.to_dict())
        if bun is None:
            print(f"{ct:<24}  CHAIN PRODUCED NO BUNDLE", flush=True)
            continue
        cells = bun.get("cells_all") or {}
        rec = cells.get(ct) or {}
        f_read = rec.get("fraction")
        a_read = rec.get("A")
        cls = c2c.get(ct)
        mk = [m for m in ct_markers[ct] if m in beta.index]
        a_exp = (sum(H(beta[m]) for m in mk) / len(mk) / h_min[cls]) if mk and cls in h_min else None
        top = max(((k, v.get("fraction") or 0) for k, v in cells.items() if isinstance(v, dict)),
                  key=lambda kv: kv[1], default=("-", 0))
        rows.append({"cell": ct, "class": cls, "f_read": f_read, "A_read": a_read, "A_expected": a_exp,
                     "n_markers_present": len(mk), "top_called": list(top)})
        d = (a_read - a_exp) if (a_read is not None and a_exp is not None) else float("nan")
        print(f"{ct:<24}{('YES' if top[0] == ct else 'no:' + top[0][:9]):>8}"
              f"{(f_read if f_read is not None else float('nan')):>9.3f}"
              f"{(a_read if a_read is not None else float('nan')):>10.4f}"
              f"{(a_exp if a_exp is not None else float('nan')):>12.4f}{d:>9.4f}{str(cls):>12}", flush=True)

    json.dump(rows, open("handoff/synth01_cells.json", "w"), indent=1)
    bad = [r for r in rows if r["A_read"] is not None and r["A_expected"] is not None
           and abs(r["A_read"] - r["A_expected"]) > 1e-6]
    print(f"\ncells whose reported A differs from the formula by more than 1e-6: {len(bad)} of {len(rows)}",
          flush=True)
    for r in bad:
        print(f"   {r['cell']}: read {r['A_read']:.6f} vs expected {r['A_expected']:.6f}", flush=True)


if __name__ == "__main__":
    main()
