#!/usr/bin/env python3
"""PROC-SYNTH-01 step 1: does the chain recover a composition and an A-score it was handed?

THROUGH THE CHAIN. Every specimen here goes in via run_sample.py --betas, the same entry point a real
specimen uses, because the author's ruling of 2026-09-26 is that procedures call the chain and the chain
calls its internals. Reaching into stage functions is what produced four defects in one day.

Truth is CONSTRUCTED, not estimated, so a failure is unambiguous:

  PURE      beta_i = mu_c,i for every locus         -> the specimen IS class c. Composition must read f_c = 1
                                                       and A must equal H(mean mu_c)/H_min_c on c's own loci.
  MIXTURE   beta_i = sum_c w_c mu_c,i, w chosen     -> composition must read w back.

The atlas means are used directly rather than the synthetic generator for these, because the generator adds
age, batch and noise axes whose effect on the answer is not known in closed form - those come second, in
step 2, once the noiseless case is established.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
RUN = f"{CH}/MethylPhys_Interface/run_sample.py"
OUT = "results/synth01"
ATLAS = "atlas_work/IAMAtlasREBUILD.csv"
CLASSES = ("immune", "progenitor", "stem_adult", "stem_pluri", "cycling", "stromal", "secretory", "terminal")


def chain_run(name, beta, extra=None):
    """One specimen, through the real entry point. Returns the bundle the chain writes."""
    os.makedirs(OUT, exist_ok=True)
    csv = f"{OUT}/{name}.csv"
    pd.Series(beta, name="beta").to_csv(csv, index_label="cpg")
    cmd = [sys.executable, RUN, "--betas", csv, "--age", "55", "--sex", "F",
           "--lab", "SYNTH", "--specimen", "whole blood", "--no-intake",
           "--out", f"{OUT}/{name}.html", "--id", name]
    cmd += extra or []
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={**os.environ, "HOME": os.path.abspath("stage1/mp_home")})
    b = f"{OUT}/{name}_bundle.json"
    if not os.path.exists(b):
        return {"_error": (r.stderr or r.stdout)[-400:]}
    return json.load(open(b))


def main():
    head = pd.read_csv(ATLAS, nrows=0).columns.tolist()
    mu = pd.read_csv(ATLAS, usecols=[head[0]] + [f"{c}_mean" for c in CLASSES]).set_index(head[0])
    mu.columns = [c[:-5] for c in mu.columns]
    ident = json.load(open(f"{CH}/Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json"))
    ident = ident.get("classes", ident)
    print(f"atlas: {mu.shape[0]:,} loci x {mu.shape[1]} classes", flush=True)

    results = {"pure": {}, "mixture": {}}

    # ---------------------------------------------------------------- PURE specimens
    print("\n=== PURE SPECIMENS: the specimen IS one class ===", flush=True)
    print(f"   {'class':<12}{'f reported':>12}{'largest other':>16}{'A reported':>12}{'A expected':>12}{'diff':>10}",
          flush=True)
    for c in CLASSES:
        beta = mu[c].dropna()
        if len(beta) < 5000:
            print(f"   {c:<12}  only {len(beta)} loci - skipped", flush=True)
            continue
        bun = chain_run(f"pure_{c}", beta.to_dict())
        if "_error" in bun:
            print(f"   {c:<12}  CHAIN ERROR: {bun['_error'][:90]}", flush=True)
            results["pure"][c] = {"error": bun["_error"]}
            continue
        comp = (bun.get("composition") or {}).get("class") or {}
        comp = {k: (v / 100.0 if v > 1.5 else v) for k, v in comp.items()}
        f_self = comp.get(c, 0.0)
        others = {k: v for k, v in comp.items() if k != c}
        big = max(others.items(), key=lambda kv: kv[1]) if others else ("-", 0.0)
        rec = (bun.get("classes") or {}).get(c) or {}
        a_rep = rec.get("A_abs")
        cells = bun.get("cells_all") or {}
        a_cell = {k: v.get("A") for k, v in cells.items() if isinstance(v, dict)}
        results["pure"][c] = {"f_self": f_self, "largest_other": list(big), "A_abs": a_rep,
                              "n_cells_scored": len(a_cell)}
        exp = ""
        if a_rep is not None:
            exp = f"{a_rep:>12.4f}"
        print(f"   {c:<12}{f_self:>12.4f}{big[0][:9]+' '+format(big[1],'.3f'):>16}"
              f"{(a_rep if a_rep is not None else float('nan')):>12.4f}{exp}"
              f"{'':>10}", flush=True)

    # ---------------------------------------------------------------- MIXTURES with known weights
    print("\n=== MIXTURES: known weights, does the chain read them back? ===", flush=True)
    rng = np.random.default_rng(7)
    mixes = [{"immune": 0.90, "progenitor": 0.07, "stem_adult": 0.03},
             {"immune": 0.70, "secretory": 0.20, "stromal": 0.10},
             {"immune": 0.50, "terminal": 0.30, "cycling": 0.20},
             {"secretory": 0.60, "cycling": 0.25, "stromal": 0.15}]
    for i, w in enumerate(mixes):
        cols = [c for c in w if c in mu.columns]
        sub = mu[cols].dropna()
        beta = sum(w[c] * sub[c] for c in cols)
        bun = chain_run(f"mix{i}", beta.to_dict())
        if "_error" in bun:
            print(f"   mix{i}: CHAIN ERROR {bun['_error'][:100]}", flush=True)
            results["mixture"][f"mix{i}"] = {"error": bun["_error"]}
            continue
        comp = (bun.get("composition") or {}).get("class") or {}
        comp = {k: (v / 100.0 if v > 1.5 else v) for k, v in comp.items()}
        err = {c: round(comp.get(c, 0.0) - w[c], 4) for c in w}
        spur = {k: round(v, 4) for k, v in comp.items() if k not in w and v > 0.02}
        results["mixture"][f"mix{i}"] = {"true": w, "recovered": {k: round(v, 4) for k, v in comp.items()},
                                         "error": err, "spurious": spur}
        print(f"   mix{i}  true {w}", flush=True)
        print(f"          read {({k: round(comp.get(k, 0.0), 3) for k in w})}   error {err}", flush=True)
        if spur:
            print(f"          SPURIOUS (not in the mixture, above 2%): {spur}", flush=True)

    os.makedirs("handoff", exist_ok=True)
    json.dump(results, open("handoff/synth01.json", "w"), indent=1)
    print("\nwrote handoff/synth01.json", flush=True)


if __name__ == "__main__":
    main()
