#!/usr/bin/env python3
"""Per-cell healthy reference ON THE IDENTITY SURFACE - 2026-09-26.

Rebuilds what build_percell_reference.py built on 2026-09-22 (percell_reference_v0_3.json), with one change:
the A is computed the way the chain now computes it, on each cell's IDENTITY loci with the class gauge's
formula, A = H(mean beta over the loci) / H_min[class(cell)]. v0_3 was measured on the discriminative
MARKER surface, which the reference audit disqualified, so its bands (0.47-0.95) cannot judge readings that
now sit near 1.0.

Design, kept from v0_3 so the two are comparable:
  * per laboratory, a BUILD panel and a HELD-OUT set, disjoint; bands are p10-p90 of A on the build panel;
    the held-out set is scored against them and the in-band fraction reported (nominal 0.80)
  * four laboratories; Uppsala (GSE87571) now at full depth, 732 arrays, split 366/366 at random with a
    fixed seed; the other three at their published 80-array panels split 40/40 as before
  * pooled: median within-lab spread, between-lab range of medians, median held-out coverage

What is deliberately NOT done:
  * no age adjustment and no laboratory zero - these are raw per-cell A bands per laboratory, which is what
    v0_3 was. Age and zero belong to the class gauge, whose surface this is not.
  * no fraction conditioning. The author's ruling: fraction is a DETECTION gate, not a band conditioner.
    The fraction of every cell in every array is RECORDED beside its A so the gate can be set from this
    file later, but the band is the band.

Every array is scored through the chain's own scorer (iamatlas_a_scoring._score_one_identity), not a
re-implementation, so the band and the reading it will judge are the same arithmetic.
"""
import glob
import hashlib
import json
import lzma
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
FULL = "stage1_betas_GSE87571_FULL.parquet"
PANELS = "iamrepo/Biological_Physics/MethylPhys/reference_data/stage1_betas_{gse}.pkl.xz"
OUT = f"{CH}/Runtime Matrices/Percell_Reference/percell_reference_identity_v1_0.json"
SEED = 20260926


def main():
    sys.path.insert(0, CH)
    import cpg_conductor as C
    asc = C._load_module("iamatlas_a_scoring", C._find("iamatlas_a_scoring.py"))
    pci = asc.load_percell_identity(str(C._find("iamatlas_percell_identity_loci_v1_0.json")))
    cells = sorted(pci)
    print(f"cells with identity panels: {len(cells)}", flush=True)

    labs = {}
    t0 = time.time()
    df = pd.read_parquet(FULL)
    df.index = df.index.map(str)
    labs["GSE87571"] = df
    print(f"GSE87571 full: {df.shape[1]} arrays x {df.shape[0]:,} loci  ({time.time()-t0:.0f}s)", flush=True)
    for gse in ("GSE42861", "GSE111629", "GSE125105"):
        d = pickle.load(lzma.open(PANELS.format(gse=gse), "rb"))
        d.index = d.index.map(str)
        labs[gse] = d
        print(f"{gse} panel: {d.shape[1]} arrays x {d.shape[0]:,} loci", flush=True)

    rng = np.random.default_rng(SEED)
    JG = {}
    entries = {ct: {"class": pci[ct]["class"], "H_min": pci[ct]["H_min"], "n_loci_panel": pci[ct]["n_loci"],
                    "branch": pci[ct].get("branch"), "labs": {}} for ct in cells}
    splits = {}
    for gse, d in labs.items():
        cols = list(d.columns)
        perm = rng.permutation(len(cols))
        half = len(cols) // 2
        build = [cols[i] for i in perm[:half]]
        held = [cols[i] for i in perm[half:]]
        splits[gse] = {"build": build, "held_out": held}
        # score every array on every cell through the chain's scorer
        A = {ct: {} for ct in cells}
        nfound = {}
        for j, gsm in enumerate(cols):
            # SCALE-MAPPED, as the chain now scores (2026-09-26): raw stage-1 betas sit ~0.07 above the atlas
            # at the identity loci and read every present cell SUPPRESSED; the class gauge always mapped first.
            brm, _ = C.stage_1s_scale_map(d[gsm].dropna().to_dict(), "stage1_noob_450K")
            s = pd.Series(brm)
            for ct in cells:
                r = asc._score_one_identity(s, pci[ct]["loci"], float(pci[ct]["H_min"]))
                A[ct][gsm] = r["A"] if r["status"] == "OK" else np.nan
                nfound[ct] = r["n_markers_matched"]
                if r["status"] == "OK":
                    JG.setdefault(ct, []).append(r["jensen_gap"])
            if (j + 1) % 100 == 0:
                print(f"  {gse}: scored {j+1}/{len(cols)}  ({time.time()-t0:.0f}s)", flush=True)
        for ct in cells:
            b = np.array([A[ct][g] for g in build], float)
            b = b[~np.isnan(b)]
            h = np.array([A[ct][g] for g in held], float)
            h = h[~np.isnan(h)]
            if len(b) < 20:
                entries[ct]["labs"][gse] = {"status": "TOO FEW SCOREABLE", "n_build": int(len(b))}
                continue
            p10, p50, p90 = (float(np.percentile(b, q)) for q in (10, 50, 90))
            inb = float(np.mean((h >= p10) & (h <= p90))) if len(h) else None
            entries[ct]["labs"][gse] = {
                "n_loci_found": int(nfound[ct]), "n_build": int(len(b)), "n_held_out": int(len(h)),
                "A_p10": round(p10, 4), "A_p50": round(p50, 4), "A_p90": round(p90, 4),
                "held_out_p50": round(float(np.median(h)), 4) if len(h) else None,
                "held_out_frac_in_band": round(inb, 4) if inb is not None else None}
        print(f"{gse} done ({time.time()-t0:.0f}s)", flush=True)

    for ct, e in entries.items():
        ok = [v for v in e["labs"].values() if "A_p50" in v]
        if ok:
            e["pooled"] = {
                "n_labs": len(ok),
                "median_within_lab_spread": round(float(np.median([v["A_p90"] - v["A_p10"] for v in ok])), 4),
                "between_lab_median_range": round(float(max(v["A_p50"] for v in ok) - min(v["A_p50"] for v in ok)), 4),
                "held_out_coverage_median": round(float(np.median([v["held_out_frac_in_band"] for v in ok
                                                                    if v["held_out_frac_in_band"] is not None])), 4),
                "grand_median_A": round(float(np.median([v["A_p50"] for v in ok])), 4)}

    meta = {"built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "surface": "identity_loci (iamatlas_percell_identity_loci_v1_0.json)",
            "formula": "A = H(mean beta over the cell's identity loci) / H_min[class(cell)] - RULING A3, the commissioned identity form used by stage_b_identity; betas scale-mapped to the atlas (stage_1s_scale_map) before scoring, as the class gauge does",
            "jensen_gap_median_per_cell": {ct: round(float(np.median(v)), 5) for ct, v in JG.items()},
            "scorer": "iamatlas_a_scoring._score_one_identity - the chain's own, not a re-implementation",
            "supersedes_for_scoring": "percell_reference_v0_3.json, which is on the MARKER surface and cannot judge identity-surface readings",
            "design": "per laboratory: random disjoint build/held-out split (seed %d); band = p10-p90 of A on build; held-out scored against it, nominal in-band 0.80" % SEED,
            "not_done": ["age adjustment", "laboratory zero", "fraction conditioning (fraction is a detection gate, author's ruling)"],
            "cohorts": {gse: {"n_arrays": int(d.shape[1]), "n_build": len(splits[gse]["build"]),
                              "n_held_out": len(splits[gse]["held_out"]), "n_loci": int(d.shape[0])}
                        for gse, d in labs.items()},
            "splits": splits}
    json.dump({"_meta": meta, "entries": entries}, open(OUT, "w"), indent=1)
    raw = open(OUT, "rb").read()
    print(f"\nwrote {OUT.split('chain/')[-1]}  ({len(raw)/1e6:.1f} MB, sha256 {hashlib.sha256(raw).hexdigest()[:12]})")

    # summary
    gm = np.array([e["pooled"]["grand_median_A"] for e in entries.values() if "pooled" in e])
    sp = np.array([e["pooled"]["median_within_lab_spread"] for e in entries.values() if "pooled" in e])
    br = np.array([e["pooled"]["between_lab_median_range"] for e in entries.values() if "pooled" in e])
    cv = np.array([e["pooled"]["held_out_coverage_median"] for e in entries.values() if "pooled" in e])
    print(f"\n=== {len(gm)} cells banded on the identity surface ===")
    print(f"   grand median A across cells: min {gm.min():.4f}  median {np.median(gm):.4f}  max {gm.max():.4f}")
    print(f"   within-lab band width (p90-p10): median {np.median(sp):.4f}   [immune class gauge band: 0.0524]")
    print(f"   between-lab range of medians:    median {np.median(br):.4f}")
    print(f"   held-out in-band coverage:       median {np.median(cv):.3f}   [nominal 0.80; v0_3 achieved 0.75]")
    print(f"   cells with grand median A in NORMAL 0.95-1.04 (the SOP scale): {int(((gm>=0.95)&(gm<=1.04)).sum())} of {len(gm)}")
    jg = np.array([np.median(v) for v in JG.values()])
    print(f"   Jensen gap H(mean)-mean(H), median over cells: {np.median(jg):.5f}  max {jg.max():.5f}  [panel construction promises < 0.05]")


if __name__ == "__main__":
    main()
