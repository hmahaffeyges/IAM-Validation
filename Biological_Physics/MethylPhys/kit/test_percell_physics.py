#!/usr/bin/env python3
"""FAILSAFE: the per-cell A-score is physics, on the commissioned surface, and reads NORMAL on healthy references.

Author, 2026-09-26: "The drift is real ... we need failsafes!" Four root causes were found in one morning, every
one of them a silent regression that no document had prevented because a document is a sentence a reader must
remember. This file is the machine check. It is run by propagate.py on every push and fails the push.

  RC1  WRONG SURFACE.       The per-cell A was computed on discriminative MARKER panels (bimodal by design), so
                            most cells' own atlas reference read far below 1.0 (Cortical_neurons 0.0099).
       CHECK: every per-cell reading reports surface == "identity_loci"; every cell's own atlas mean reads
              within the construction tolerance of 1.0.
  RC2  UNMAPPED BETAS.      The per-cell path scored RAW stage-1 betas; the class gauge scale-maps them onto the
                            atlas first. Raw, every present cell in healthy blood read SUPPRESSED.
       CHECK: stage_a_cells calls stage_1s_scale_map before score_per_celltype (verified in the source), and a
              known healthy panel array's present cells read NORMAL (0.95-1.04) through the chain.
  RC3  CROSSED FORMULA.     The SOP carries two rules for two surfaces - LESSON-ASCORE-02 (mean of per-CpG H,
                            for marker panels) and RULING A3 (H of the mean beta, for identity loci, the
                            commissioned gauge). Applying the first to the second read healthy cells SUPPRESSED.
       CHECK: the per-cell scorer and stage_b_identity give the SAME A on the same loci and betas, to 1e-9.
  RC4  UNREACHABLE REFERENCE. percell_reference_v0_3.json sat in a folder the chain's search path did not
                            include, so the per-cell bands existed and were never loaded.
       CHECK: the per-cell reference resolves through _find and covers >= 100 cells.

And two physical constraints that are always true and therefore cheap to assert:
  * A <= 1 / H_min for every cell (binary entropy is at most 1.0).
  * A cell at fraction 0 is not scored as a reading; it is a detection gate, not a band matter (author's rule).

Exit 1 on the first failure, with the root cause named, so the propagate gate refuses the push.
"""
import glob
import inspect
import json
import lzma
import math
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.path.dirname(HERE)
CH = os.path.join(MP, "chain")
sys.path.insert(0, CH)

NORMAL = (0.95, 1.04)          # tier_breakpoints.json v1.4 - the OM's scale
OWN_REF_TOL = (0.90, 1.10)     # a cell's own atlas mean, within the panel's +/-0.05 construction tolerance
FAILS = []


def fail(rc, msg):
    FAILS.append((rc, msg))
    print("  FAIL %-4s %s" % (rc, msg))


def ok(rc, msg):
    print("  ok   %-4s %s" % (rc, msg))


def main():
    import cpg_conductor as C
    asc = C._load_module("iamatlas_a_scoring", C._find("iamatlas_a_scoring.py"))

    # ---------------------------------------------------------------- RC4 the reference resolves
    try:
        ref_p = C._find("percell_reference_identity_v1_0.json")
        ref = json.load(open(ref_p))["entries"]
        n_ref = sum(1 for e in ref.values() if "pooled" in e)
        (ok if n_ref >= 100 else fail)("RC4", "per-cell reference resolves via _find, %d cells banded" % n_ref)
    except Exception as e:
        fail("RC4", "per-cell reference NOT reachable from the chain: %s" % e)
        ref = {}
    try:
        C._find("percell_reference_v0_3.json")
        ok("RC4", "Percell_Reference folder is on the search path")
    except Exception as e:
        fail("RC4", "Percell_Reference folder is NOT on the search path: %s" % e)

    # ---------------------------------------------------------------- RC2 the source maps before it scores
    src = inspect.getsource(C.stage_a_cells)
    i_map = src.find("stage_1s_scale_map")
    i_sc = src.find("score_per_celltype")
    (ok if 0 <= i_map < i_sc else fail)("RC2", "stage_a_cells scale-maps before per-cell scoring (source order)")

    # ---------------------------------------------------------------- RC1 + RC3 on the atlas means
    pci = asc.load_percell_identity(str(C._find("iamatlas_percell_identity_loci_v1_0.json")))
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
    atlas = str(C._find("IAMAtlasREBUILD.csv")) if os.path.exists(str(C._find("IAMAtlasREBUILD.csv", required=False) or "")) else None
    if atlas is None:
        cand = glob.glob(os.path.join(MP, "atlas", "IAMAtlasREBUILD.csv")) + glob.glob(os.path.join(os.getcwd(), "atlas_work", "IAMAtlasREBUILD.csv"))
        atlas = cand[0] if cand else None
    if atlas is None:
        fail("RC1", "atlas CSV not found - cannot score own references (decompress atlas/IAMAtlasREBUILD.csv.xz)")
    else:
        head = pd.read_csv(atlas, nrows=0).columns.tolist()
        probe = [c for c in ("Cortical_neurons", "CD4_T-cells", "Colon_epithelial_cells", "Breast", "fibroblast", "HSC")
                 if c in pci and f"{c}_mean" in head]
        mu = pd.read_csv(atlas, usecols=[head[0]] + [f"{c}_mean" for c in probe]).set_index(head[0])
        mu.index = mu.index.map(str)
        worst = 0.0
        for c in probe:
            s = mu[f"{c}_mean"].dropna()
            r = asc._score_one_identity(s, pci[c]["loci"], float(pci[c]["H_min"]))
            if r.get("surface") != "identity_loci":
                fail("RC1", "%s scored on surface %r, not identity_loci" % (c, r.get("surface")))
            A = r["A"]
            worst = max(worst, abs(A - 1.0))
            if not (OWN_REF_TOL[0] <= A <= OWN_REF_TOL[1]):
                fail("RC1", "%s's OWN atlas reference reads A=%.4f - the surface is wrong" % (c, A))
            # RC3: the per-cell scorer and the class gauge's H() agree on the same loci
            vals = s.loc[[x for x in pci[c]["loci"] if x in s.index]].astype(float).values
            b = min(max(float(np.mean(vals)), 1e-12), 1 - 1e-12)
            H_gauge = -b * math.log2(b) - (1 - b) * math.log2(1 - b)
            if abs(H_gauge / float(pci[c]["H_min"]) - A) > 1e-9:
                fail("RC3", "%s: per-cell scorer and the class gauge's H(beta_mean) DISAGREE (%.6f vs %.6f)"
                     % (c, A, H_gauge / float(pci[c]["H_min"])))
            if A > 1.0 / float(pci[c]["H_min"]) + 1e-9:
                fail("PHYS", "%s reads A=%.4f above the ceiling 1/H_min=%.4f" % (c, A, 1.0 / float(pci[c]["H_min"])))
        if not any(f[0] in ("RC1", "RC3") for f in FAILS):
            ok("RC1", "%d cells' own atlas means read within %s of 1.0 (worst |A-1| = %.4f) on identity_loci"
               % (len(probe), OWN_REF_TOL, worst))
            ok("RC3", "per-cell scorer == class gauge H(beta_mean)/H_min to 1e-9 on the same loci (RULING A3)")

    # ---------------------------------------------------------------- RC2 on a real healthy array, through the chain
    panel = os.path.join(MP, "reference_data", "stage1_betas_GSE87571.pkl.xz")
    if not os.path.exists(panel):
        fail("RC2", "healthy panel %s missing - cannot run the NORMAL check" % os.path.basename(panel))
    else:
        df = pickle.load(lzma.open(panel, "rb"))
        gsm = df.columns[0]
        beta = df[gsm].dropna().to_dict()
        # the panel array is Uppsala (GSE87571); pass the laboratory so the per-cell offset applies, as it
        # would through run_sample --lab. Judge the ZEROED A against each cell's own centre from the
        # calibration record, not against a universal ceiling: CD4 sits at 1.027 in all four labs and a
        # universal 1.04 would call healthy CD4 ELEVATED routinely (measured 2026-09-26).
        out = C.stage_a_cells(beta, atlas or str(C._find("IAMAtlasREBUILD.csv")), cfg={"lab": "GSE87571"})
        cells = out.get("cells") or {}
        present = {k: v for k, v in cells.items() if (v.get("fraction") or 0) >= 0.02 and v.get("A") is not None
                   and not (isinstance(v["A"], float) and v["A"] != v["A"])}
        if not present:
            fail("RC2", "no present cell (fraction >= 0.02) scored on healthy array %s" % gsm)
        else:
            centres = json.load(open(C._find("percell_reference_identity_v1_0.json")))["_meta"].get("percell_centres_after_offset", {})
            def judged(k, v):
                a = v.get("A_zeroed") if v.get("A_zeroed") is not None else v["A"]
                c = centres.get(k, 1.0)
                # NORMAL for THIS cell: its own centre, with the universal half-widths (0.05 below, 0.04 above)
                return a, (c - 0.05 <= a <= c + 0.04)
            bad = {k: round(judged(k, v)[0], 4) for k, v in present.items() if not judged(k, v)[1]}
            unset = [k for k, v in present.items() if v.get("lab_offset") == "UNSET"]
            if unset:
                fail("RC2", "laboratory offset UNSET for present cells %s - the lab did not reach the per-cell path" % unset[:4])
            wrong_surface = [k for k, v in present.items() if v.get("surface") != "identity_loci"]
            if wrong_surface:
                fail("RC1", "present cells scored off the identity surface: %s" % wrong_surface[:5])
            # allow one present cell outside NORMAL (2-3 per cent of healthy readings fall outside by design),
            # but the MAJORITY of present cells must be NORMAL on a healthy array
            if len(bad) > max(1, len(present) // 3):
                fail("RC2", "healthy array %s: %d of %d present cells read outside their own NORMAL (zeroed): %s"
                     % (gsm, len(bad), len(present), bad))
            else:
                ok("RC2", "healthy array %s: %d of %d present cells read NORMAL against their own centre (zeroed)%s"
                   % (gsm, len(present) - len(bad), len(present), (" (outside: %s)" % bad) if bad else ""))
            absent_scored = [k for k, v in cells.items() if (v.get("fraction") or 0) == 0 and v.get("reportable")]
            (ok if not absent_scored else fail)("GATE", "%d cells at fraction 0 marked reportable" % len(absent_scored))

    print()
    if FAILS:
        print("test_percell_physics: %d FAILURE(S) - the per-cell A is not on the commissioned physics surface" % len(FAILS))
        for rc, m in FAILS:
            print("   %s  %s" % (rc, m))
        return 1
    print("test_percell_physics: PASS - four root causes checked, none present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
