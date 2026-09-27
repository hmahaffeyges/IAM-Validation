#!/usr/bin/env python3
"""Kit test for Stage 2d as ONE JOINT FIT (PROC-STAGE2D-03, adopted 2026-09-27). Contract:
  K1  the stage reads detection_panel_v3.json and nothing older (a v1 or v2 line anywhere is a failure)
  K2  on healthy arrays from the three admitted laboratories, no detectable template fires above twice its design rate
      (floor = 0.99 quantile -> 1 % design; allowance ceil(2 % of N)); UNSPECIFIC arrays <= ceil(4 % of N) (measured 1.9 %,
      concentrated in the oldest arrays - PROC_STAGE2D_03_OUTCOME.md)
  K3  the not-detectable template(s) print NOT DETECTABLE on every array and never a line
  K4  an honest spike (the cell's atlas profile at its defined loci, host beta elsewhere) of Glia at 5 % into a healthy array
      is detected and named Glia; the detector responds, it does not merely fail to fire
  K5  the record carries the noise floor's N and quantile and the foreign-like mass total
Reads N_PER_LAB arrays per laboratory from reference_data (default 6). Calls the chain's own stages in the chain's order.
"""
import glob, json, lzma, math, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(HERE); CH = os.path.join(MP, "chain")
sys.path.insert(0, CH); sys.path.insert(0, os.path.join(CH, "Synthetic_Patient_Generator"))
import cpg_conductor as C  # noqa: E402

N_PER_LAB = int(os.environ.get("N_PER_LAB", "6")); PIPE = "stage1_noob_450K"
LABS = ["GSE87571", "GSE42861", "GSE111629"]          # the admitted laboratories; GSE125105 is refused at intake


def main():
    atlas = os.path.join(MP, "atlas", "IAMAtlasREBUILD.csv")
    if not os.path.exists(atlas):
        atlas = os.path.join(os.getcwd(), "atlas_work", "IAMAtlasREBUILD.csv")
    fails = []
    # K1
    import inspect
    src = inspect.getsource(C.stage_2d_foreign_detection)
    if "detection_panel_v3.json" not in src or "detection_panel_v1" in src or "detection_panel_v2" in src:
        fails.append("K1: stage_2d_foreign_detection does not read detection_panel_v3.json alone")
    P = json.load(open(C._find("detection_panel_v3.json"))); nd = set(P["not_detectable"])
    dec_mod = C._load_module("legacy_iam_deconvolver", C._find("legacy_iam_deconvolver.py"))
    dec = dec_mod.legacyIAMDeconvolver(atlas, celltype_class_map=str(C._find("IAMAtlasREBUILD_celltype_to_class.json")), verbose=False)
    fired = {}; unspecific = 0; n_total = 0; first = None
    for lab in LABS:
        pk = glob.glob(os.path.join(MP, "reference_data", f"stage1_betas_{lab}*.pkl.xz"))
        if not pk:
            fails.append(f"{lab}: no reference_data panel on disk"); continue
        df = pickle.load(lzma.open(pk[0], "rb")); df.index = df.index.map(str)
        for gsm in list(df.columns)[:N_PER_LAB]:
            beta = df[gsm].dropna(); mapped, scale = C.stage_1s_scale_map(beta.to_dict(), PIPE)
            r = dec.deconvolve(mapped); sa = {"class_fractions": dict(r.class_fractions), "celltype_fractions": dict(r.celltype_fractions)}
            bi = C.stage_b_identity(mapped, sa, 60, scale, lab_zero=None)
            fd = C.stage_2d_foreign_detection(mapped, sa, lab, bi=bi); n_total += 1
            if first is None: first = (lab, beta, mapped, sa, bi)
            st = fd.get("status") or ""
            if not st.startswith("OK"):
                fails.append(f"{lab} {gsm}: status {st[:80]}"); continue
            if st.startswith("OK_BUT_UNSPECIFIC"): unspecific += 1
            for c in (fd.get("detected") or []): fired[c] = fired.get(c, 0) + 1
            # K3
            if set(fd.get("not_detectable") or {}) != nd or any(c in (fd.get("cells") or {}) for c in nd):
                fails.append(f"K3: {lab} {gsm}: not-detectable templates {sorted(nd)} not printed as such")
            # K5
            nf = fd.get("noise_floor") or {}
            if not (nf.get("n_arrays") and nf.get("quantile") and fd.get("foreign_mass_total") is not None):
                fails.append(f"K5: {lab} {gsm}: record lacks noise-floor N / quantile / foreign_mass_total")
    # K2
    allow = math.ceil(0.02 * n_total)
    for c, k in fired.items():
        if k > allow: fails.append(f"K2: {c} fired on {k} of {n_total} healthy arrays (allowed {allow})")
    if unspecific > math.ceil(0.04 * n_total):
        fails.append(f"K2: {unspecific} of {n_total} healthy arrays UNSPECIFIC (allowed {math.ceil(0.04 * n_total)})")
    # K4: honest Glia spike at 5 % into the first array
    if first is not None:
        import synthetic_patient_generator as SPG
        lab, beta, mapped, sa, bi = first
        mu = SPG._cell_means(atlas); mu.index = mu.index.map(str); col = mu["Glia"].dropna()
        loci = beta.index.intersection(col.index); s = beta.copy(); s.loc[loci] = 0.95 * beta.loc[loci] + 0.05 * col.loc[loci].values
        m2, _ = C.stage_1s_scale_map(s.to_dict(), PIPE); fd2 = C.stage_2d_foreign_detection(m2, sa, lab, bi=bi)
        cells = fd2.get("cells") or {}
        top = max(cells, key=lambda c: cells[c]["f_hat"]) if cells else None
        if "Glia" not in (fd2.get("detected") or []) or top != "Glia":
            fails.append(f"K4: 5 % Glia spike -> detected {fd2.get('detected')}, largest {top} (f_hat Glia {cells.get('Glia', {}).get('f_hat')})")
    print(json.dumps({"fired": fired, "unspecific": unspecific, "n_arrays": n_total, "not_detectable": sorted(nd)}, indent=1))
    if fails:
        print("FAIL"); [print("  ", f) for f in fails]; sys.exit(1)
    print(f"PASS: joint-fit detector quiet at the design rate on {n_total} healthy arrays; not-detectable printed; 5 % Glia spike detected and named")


if __name__ == "__main__":
    main()
