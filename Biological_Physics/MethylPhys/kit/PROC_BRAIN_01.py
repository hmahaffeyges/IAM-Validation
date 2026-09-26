#!/usr/bin/env python3
"""PROC-BRAIN-01: can the chain find terminal-class cells in a liquid specimen from a CNS tumour patient?

THROUGH THE CHAIN. Every specimen enters via run_sample.py --betas, per the ruling of 2026-09-26 that a
procedure invokes the chain rather than reimplementing it. propagate.py rule 10 enforces this.

Bars are fixed in PROC_BRAIN_01_PREREG.md (amended before any array was scored, to record that the CSF arm
is 24 specimens and not 181):
  presence gate   terminal fraction > 0.0312, the maximum across 845 non-CNS whole bloods
  A bar           terminal A > 1.05, the author's own VAL-009 rule from April 2026
"""
import gzip
import json
import os
import subprocess
import sys

import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
RUN = f"{CH}/MethylPhys_Interface/run_sample.py"
OUT = "results/brain01_clean"
BETAS = "brain01/GSE292312_BetaValues.txt.gz"
GATE = 0.0312


def keep_loci():
    sys.path.insert(0, CH)
    import cpg_conductor as C
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
    ident = ident.get("classes", ident)
    keep = set()
    for v in ident.values():
        if isinstance(v, dict) and v.get("loci"):
            keep |= {str(x) for x in v["loci"]}
    # 2026-09-26 DEFECT, mine: this guessed the markers artifact's JSON shape ("celltype_markers" then
    # list-valued entries) and got ZERO markers, so the streaming pass below dropped every cell-type marker
    # CpG. The deconvolver still worked - it selects markers itself from the atlas - so the FRACTIONS were
    # right, but every per-cell A came back nan with status INSUFFICIENT_MARKERS and B3 was unscoreable.
    # The artifact is read with its own loader now, and the keep set is no longer reduced at all: the
    # submitted matrix carries 370,346 CpGs and passing all of them per specimen costs a few seconds, which
    # is less than the cost of being wrong about which ones matter.
    asc_path = C._find("iamatlas_a_scoring.py")
    sys.path.insert(0, os.path.dirname(str(asc_path)))
    import iamatlas_a_scoring as asc
    _m, ct_markers, _c2c, _h = asc.load_artifact(str(C._find("iamatlas_celltype_markers_v0_2.json")))
    for v in ct_markers.values():
        keep |= {str(x) for x in (v if isinstance(v, (list, tuple, set)) else [])}
    print(f"cell types with markers: {len(ct_markers)}", flush=True)
    maps = json.load(open(C._find("beta_scale_maps_v1.json")))
    names = list(maps.get("maps", maps))
    return keep, names


def main():
    global _RS
    sys.path.insert(0, os.path.join(CH, "MethylPhys_Interface"))
    sys.path.insert(0, CH)
    import run_sample as _RS
    os.makedirs(OUT, exist_ok=True)
    keep, pipelines = keep_loci()
    print(f"loci the chain reads: {len(keep):,}", flush=True)
    print(f"pipeline maps available: {pipelines}", flush=True)
    pipe = next((p for p in pipelines if "EPIC" in p.upper()), pipelines[0] if pipelines else "stage1_noob_450K")
    print(f"using pipeline map: {pipe}  (the per-cell A reads RAW betas, so this affects only the class gauge,"
          f" which is withheld here for want of a laboratory zero)", flush=True)

    cols = json.load(open("handoff/brain01_cols.json"))
    m = cols["map"]
    meta = {d["gsm"]: d for d in json.load(open("handoff/brain01_meta.json"))}
    # CSF ONLY in this pass. The amendment declares the 157 tissue arrays a reported arm and NOT a bar,
    # and each specimen costs about five minutes now that the locus set is no longer reduced - 157 of
    # them is sixteen hours. The bar arm is scored first and completely; the tissue arm is a separate
    # run, recorded as not yet done rather than partially done.
    want = cols["csf"]

    # one streaming pass; keep only the chain's loci
    print("streaming the 589 MB matrix once...", flush=True)
    rows = {}
    with gzip.open(BETAS, "rt", errors="replace") as f:
        hdr = [c.strip().strip('"') for c in f.readline().rstrip("\n").split("\t")]
        idx = {c: i for i, c in enumerate(hdr) if c}
        pos = {c: idx[c] for c in want if c in idx}
        for line in f:
            p = line.rstrip("\n").split("\t")
            cg = p[0].strip().strip('"')
            if True:            # keep everything; see the note in keep_loci
                rows[cg] = p
    print(f"kept {len(rows):,} loci", flush=True)

    out = []
    for n, c in enumerate(want, 1):
        gsm = m[c]
        md = meta.get(gsm, {})
        sub = md.get("tissue", "?")
        beta = {}
        j = pos[c]
        for cg, p in rows.items():
            try:
                v = float(p[j])
            except (ValueError, IndexError):
                continue
            if 0.0 <= v <= 1.0:
                beta[cg] = v
        name = f"{gsm}_{'csf' if sub == 'CSF' else 'tum'}"
        csv = f"{OUT}/{name}.csv"
        pd.Series(beta, name="beta").to_csv(csv, index_label="cpg")
        # run_sample.main() IN THIS PROCESS, not as a subprocess. Identical code path - every guard the
        # chain owns still runs - but the deconvolver cache (cpg_conductor._DEC_CACHE) now survives between
        # specimens instead of the 605 MB atlas being re-read 181 times. A subprocess per specimen gains
        # nothing from the cache, which is why the first attempt at this run was going to take six hours.
        argv = ["run_sample.py", "--betas", csv, "--age", "9", "--sex", "F",
                "--lab", "GSE292312", "--specimen", "CSF" if sub == "CSF" else "brain tumour tissue",
                "--pipeline", pipe, "--array-type", "EPIC_v1", "--no-intake",
                "--out", f"{OUT}/{name}.html", "--id", name]
        old_argv, old_out = sys.argv, sys.stdout
        try:
            sys.argv = argv
            sys.stdout = open(os.devnull, "w")
            _RS.main()
        except SystemExit:
            pass
        except Exception as e:
            sys.stdout = old_out
            print(f"  {name}: chain raised {type(e).__name__}: {str(e)[:110]}", flush=True)
        finally:
            if sys.stdout is not old_out:
                sys.stdout.close()
            sys.argv, sys.stdout = old_argv, old_out
        b = f"{OUT}/{name}_bundle.json"
        if not os.path.exists(b):
            out.append({"gsm": gsm, "substrate": sub, "error": "no bundle"})
            continue
        bun = json.load(open(b))
        comp = (bun.get("composition") or {}).get("class") or {}
        comp = {k: (v / 100.0 if v > 1.5 else v) for k, v in comp.items()}
        cells = bun.get("cells_all") or {}
        term = {k: v for k, v in cells.items() if isinstance(v, dict) and v.get("class") == "terminal"}
        f_cells = sum((v.get("fraction") or 0.0) for v in term.values())
        rec = (bun.get("classes") or {}).get("terminal") or {}
        out.append({"gsm": gsm, "substrate": sub, "genotype": md.get("genotype"),
                    "n_betas": len(beta),
                    "terminal_pooled": comp.get("terminal", 0.0),
                    "terminal_cells_sum": f_cells,
                    "terminal_A_class": rec.get("A_mapped"),
                    "terminal_cells": {k: {"A": v.get("A"), "f": v.get("fraction"),
                                           "present": v.get("present")} for k, v in term.items()},
                    "all_class_fractions": {k: round(v, 4) for k, v in comp.items()}})
        if n % 10 == 0 or n == len(want):
            print(f"  [{n}/{len(want)}] last: {gsm} {sub} terminal_cells={f_cells:.4f}", flush=True)
        json.dump(out, open("handoff/brain01_clean.json", "w"))

    json.dump(out, open("handoff/brain01_clean.json", "w"), indent=1)
    print(f"\nscored {len(out)} specimens", flush=True)


if __name__ == "__main__":
    main()
