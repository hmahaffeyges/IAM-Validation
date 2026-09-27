#!/usr/bin/env python3
"""Kit test for Stage 2d (PLAN item 7): the four commissioned laboratories' own healthy arrays never fire
FOREIGN_CELL_DETECTED above the line's design rate, and a laboratory without a panel prints 'not commissioned' and
nothing else. Reads N_PER_LAB healthy arrays per laboratory from reference_data (default 6 - a kit test, not a
procedure; the full held-out measurement is PLAN item 6). Calls the chain's own stages in the chain's order.
"""
import glob, json, lzma, math, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(HERE); CH = os.path.join(MP, "chain")
sys.path.insert(0, CH)
import cpg_conductor as C  # noqa: E402

N_PER_LAB = int(os.environ.get("N_PER_LAB", "6")); PIPE = "stage1_noob_450K"


def main():
    atlas = os.path.join(MP, "atlas", "IAMAtlasREBUILD.csv")
    if not os.path.exists(atlas):
        atlas = os.path.join(os.getcwd(), "atlas_work", "IAMAtlasREBUILD.csv")
    P = json.load(open(C._find("detection_panel_v1.json"))); labs = sorted(P["laboratories"])
    dec_mod = C._load_module("legacy_iam_deconvolver", C._find("legacy_iam_deconvolver.py"))
    dec = dec_mod.legacyIAMDeconvolver(atlas, celltype_class_map=str(C._find("IAMAtlasREBUILD_celltype_to_class.json")), verbose=False)
    fails = []; fired = {}
    for lab in labs:
        pk = glob.glob(os.path.join(MP, "reference_data", f"stage1_betas_{lab}*.pkl.xz"))
        if not pk:
            fails.append(f"{lab}: no reference_data panel on disk"); continue
        df = pickle.load(lzma.open(pk[0], "rb")); df.index = df.index.map(str)
        n_panel = P["laboratories"][lab]["n_panel"]; allow = math.ceil(2.0 * N_PER_LAB / n_panel)   # twice the design rate, rounded up
        counts = {}
        for gsm in list(df.columns)[:N_PER_LAB]:
            mapped, scale = C.stage_1s_scale_map(df[gsm].dropna().to_dict(), PIPE)
            r = dec.deconvolve(mapped); sa = {"class_fractions": dict(r.class_fractions), "celltype_fractions": dict(r.celltype_fractions)}
            bi = C.stage_b_identity(mapped, sa, 60, scale, lab_zero=None)
            fd = C.stage_2d_foreign_detection(mapped, sa, lab, bi=bi)
            if not (fd.get("status") or "").startswith("OK"):
                fails.append(f"{lab} {gsm}: status {fd.get('status')}"); continue
            for c in (fd.get("detected") or []): counts[c] = counts.get(c, 0) + 1
        fired[lab] = counts
        for c, k in counts.items():
            if k > allow: fails.append(f"{lab}: {c} fired on {k} of {N_PER_LAB} healthy arrays (allowed {allow} at panel n={n_panel})")
    # a laboratory with no panel: no line, one sentence
    df = pickle.load(lzma.open(glob.glob(os.path.join(MP, "reference_data", "stage1_betas_*.pkl.xz"))[0], "rb")); df.index = df.index.map(str)
    mapped, scale = C.stage_1s_scale_map(df[df.columns[0]].dropna().to_dict(), PIPE)
    fd = C.stage_2d_foreign_detection(mapped, {}, "NO_SUCH_LAB", bi=None)
    if not (fd.get("status") or "").startswith("NOT_COMMISSIONED") or fd.get("cells") or fd.get("detected"):
        fails.append(f"uncommissioned laboratory returned {fd.get('status')!r} with cells={bool(fd.get('cells'))}")
    print(json.dumps({"fired": fired, "n_per_lab": N_PER_LAB}, indent=1))
    if fails:
        print("FAIL"); [print("  ", f) for f in fails]; sys.exit(1)
    print(f"PASS: {len(labs)} commissioned laboratories quiet at the design rate on {N_PER_LAB} arrays each; uncommissioned laboratory refused with one sentence")


if __name__ == "__main__":
    main()
